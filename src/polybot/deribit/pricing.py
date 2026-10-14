"""Task B. Options-implied probabilities for Polymarket price markets.

Polymarket expiries rarely match Deribit's 08:00 UTC settlement, so implied
variance must be interpolated in time between the two surrounding expiries,
and IV interpolated across strikes (the smile matters for far strikes).

Method
------
1. Smile per expiry. Deribit's mark IV at strike K, as a function of log-moneyness
   k = ln(K / F) with F the forward Deribit quotes for that expiry (`underlying_price`), is
   interpolated by a natural cubic spline. Call and put marks at one strike coincide, so they
   are averaged. Beyond the quoted strikes total variance w = sigma^2 * T is continued linearly
   (value and slope stay continuous; the slope is kept on the wing's side of zero and below
   Lee's moment bound of 2).
2. Time. For T1 < T < T2 total variance at fixed k is interpolated linearly in calendar time,
   w(k, T) = (1 - l) * w1(k) + l * w2(k) with l = (T - T1) / (T2 - T1); the forward is
   interpolated log-linearly. Before the first expiry w is scaled from zero at `now`
   (w = w1 * T / T1, i.e. the first expiry's IV is held). T is measured on Deribit's
   365-day year from `now`. Deribit's interest rate is 0, so there is no discounting.
3. Digital with smile correction. The risk-neutral P(S_T > K) = -dC/dK, where
   C(K) = Black76(F, K, sigma(K), T). The chain rule gives N(d2) - vega * dsigma/dK, which in
   terms of w is  N(d2) - exp(-k) * phi(d1) * w'(k) / (2 * sqrt(w)).  The slope term is not a
   detail: on real chains it moves the probability by 2-7 points away from plain N(d2).
4. Monotonicity. Marks that imply a (tiny) butterfly arbitrage make the raw P(k) wiggle in the
   wings. It is evaluated on a fine grid, projected onto non-increasing sequences (pool adjacent
   violators) and interpolated monotonically (PCHIP). So P is non-increasing in the strike
   and within [0, 1] by construction, RANGE = P(lo) - P(hi) is never negative, and the
   probabilities of a complete ladder of ABOVE/RANGE/BELOW markets telescope to 1.

The Deribit index is not what Polymarket resolves on (usually a Binance candle). If the caller
knows the premium of the resolution source over the Deribit index, `fair_probability` shifts the
strikes accordingly and records it in `meta`.

`FairProbability.meta` keys (all floats):
    tte_years            time to the market's expiry
    iv                   implied vol at the market's strike (RANGE: mean of both strikes)
    iv_low, iv_high      RANGE only
    forward              model forward of the Deribit index at expiry
    resolution_basis     premium of the resolution source over the index that was applied
    expiry_weight        interpolation weight of the later Deribit expiry (0 = exact/scaled)
    chain_age_seconds    now minus the chain's latest quote time: the forward is only as fresh
                         as this, and 0.1% of spot is ~2 points of probability for a daily market
    moneyness_low/high   ln(K_effective / forward), for the strikes the market uses
    p_above_low/high     P(S > strike) for the strikes the market uses
    p_flat_iv            the same market priced with plain N(d2) at each strike's own IV
    smile_adjustment     p_yes - p_flat_iv: what the slope correction added
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.special import ndtr

from polybot.models import CryptoMarket, FairProbability, MarketKind, OptionQuote

MODEL_NAME = "deribit_iv_v1"

SECONDS_PER_YEAR = 365.0 * 86400.0
MIN_STRIKES = 5  # an expiry with fewer distinct strikes is ignored
MIN_TTE = timedelta(minutes=5)  # closer to expiry the index/resolution gap dominates everything
COVERAGE_TOL = 1e-3  # slack, in log-moneyness, when checking that a strike is actually quoted
GRID_STDEVS = 8.0  # the probability grid spans +-8 at-the-money standard deviations
GRID_POINTS = 1601
MAX_WING_SLOPE = 2.0  # Roger Lee: dw/d|k| <= 2 in the wings
_SQRT_2PI = math.sqrt(2.0 * math.pi)


class ChainCoverageError(ValueError):
    """The option chain cannot price the requested expiry (or has no usable data)."""


def _as_utc(name: str, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware, got naive {value!r}")
    return value.astimezone(UTC)


# --- one expiry ------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class _Slice:
    """Mark-IV smile of one Deribit expiry, exposed as total variance w(k) and dw/dk."""

    expiry: datetime
    tau: float  # years from `now`
    forward: float
    k_min: float  # quoted range in log-moneyness
    k_max: float
    spline: CubicSpline  # sigma(k)
    left: tuple[float, float]  # (w, dw/dk) at k_min, slope clipped to a valid left wing
    right: tuple[float, float]  # same at k_max

    def total_variance(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inside = np.clip(k, self.k_min, self.k_max)
        sigma = self.spline(inside)
        w = sigma * sigma * self.tau
        dw = 2.0 * sigma * self.spline(inside, 1) * self.tau
        for mask, (w_edge, dw_edge), k_edge in (
            (k < self.k_min, self.left, self.k_min),
            (k > self.k_max, self.right, self.k_max),
        ):
            if mask.any():
                w = np.where(mask, w_edge + dw_edge * (k - k_edge), w)
                dw = np.where(mask, dw_edge, dw)
        return w, dw


def _build_slice(expiry: datetime, quotes: Sequence[OptionQuote], now: datetime) -> _Slice | None:
    tau = (expiry - now).total_seconds() / SECONDS_PER_YEAR
    if tau <= 0:
        return None
    forwards = [q.underlying_price for q in quotes if q.underlying_price > 0]
    if not forwards:
        return None
    forward = float(np.median(forwards))

    marks: dict[float, list[float]] = defaultdict(list)
    for q in quotes:
        if q.strike > 0 and q.mark_iv > 0 and math.isfinite(q.mark_iv):
            marks[q.strike].append(q.mark_iv)
    if len(marks) < MIN_STRIKES:
        return None
    strikes = sorted(marks)
    k = np.log(np.asarray(strikes) / forward)
    if not k[0] < 0.0 < k[-1]:
        return None  # strikes do not straddle the forward: stale or truncated data
    sigma = np.array([sum(marks[s]) / len(marks[s]) for s in strikes])
    spline = CubicSpline(k, sigma, bc_type="natural")

    def edge(k_edge: float, lowest_slope: float, highest_slope: float) -> tuple[float, float]:
        s = float(spline(k_edge))
        slope = 2.0 * s * float(spline(k_edge, 1)) * tau
        return s * s * tau, min(max(slope, lowest_slope), highest_slope)

    return _Slice(
        expiry=expiry,
        tau=tau,
        forward=forward,
        k_min=float(k[0]),
        k_max=float(k[-1]),
        spline=spline,
        left=edge(float(k[0]), -MAX_WING_SLOPE, 0.0),
        right=edge(float(k[-1]), 0.0, MAX_WING_SLOPE),
    )


# --- the target expiry -----------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class _Blend:
    """Total variance at the target expiry: weighted slices at fixed log-moneyness."""

    tau: float
    forward: float  # of the Deribit index, at the target expiry
    parts: tuple[tuple[float, _Slice], ...]
    k_min: float  # log-moneyness range quoted in every part
    k_max: float
    weight_next: float

    def total_variance(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w: np.ndarray | float = 0.0
        dw: np.ndarray | float = 0.0
        for weight, part in self.parts:
            part_w, part_dw = part.total_variance(k)
            w = w + weight * part_w
            dw = dw + weight * part_dw
        return np.asarray(w), np.asarray(dw)

    def iv(self, k: float) -> float:
        w, _ = self.total_variance(np.array([k]))
        return math.sqrt(float(w[0]) / self.tau)


def _isotonic_non_increasing(y: np.ndarray) -> np.ndarray:
    """L2 projection onto non-increasing sequences (pool adjacent violators)."""
    if not np.any(np.diff(y) > 0.0):
        return y
    levels: list[float] = []
    counts: list[int] = []
    for value in (-y).tolist():
        levels.append(value)
        counts.append(1)
        while len(levels) > 1 and levels[-2] > levels[-1]:
            last, last_n = levels.pop(), counts.pop()
            prev, prev_n = levels.pop(), counts.pop()
            levels.append((prev * prev_n + last * last_n) / (prev_n + last_n))
            counts.append(prev_n + last_n)
    return -np.repeat(levels, counts)


@dataclass(frozen=True, slots=True, eq=False)
class _Digital:
    """P(S_T > K) as a monotone function of k = ln(K / forward) for one target expiry.

    Inside the grid it is the PCHIP curve. Beyond the grid (more than `GRID_STDEVS` standard
    deviations out) the remaining probability mass decays exponentially, joined to the curve in
    value and slope, so P tends to 1 as K -> 0 and to 0 as K -> infinity.
    """

    blend: _Blend
    half_width: float
    curve: PchipInterpolator
    left_gap: float  # 1 - P at the left end of the grid
    left_scale: float  # e-folding length of that gap beyond the end
    right_p: float  # P at the right end of the grid
    right_scale: float

    @property
    def forward(self) -> float:
        return self.blend.forward

    def covers(self, k: float) -> bool:
        return self.blend.k_min - COVERAGE_TOL <= k <= self.blend.k_max + COVERAGE_TOL

    def p_above(self, k: float) -> float:
        if k < -self.half_width:
            beyond = -self.half_width - k
            return 1.0 - self.left_gap * math.exp(-beyond / self.left_scale)
        if k > self.half_width:
            beyond = k - self.half_width
            return self.right_p * math.exp(-beyond / self.right_scale)
        return min(max(float(self.curve(k)), 0.0), 1.0)

    def p_above_flat_iv(self, k: float) -> float:
        """Plain N(d2) at this strike's own IV: what the model gives without the slope term."""
        w, _ = self.blend.total_variance(np.array([k]))
        s = math.sqrt(float(w[0]))
        return float(ndtr((-k - 0.5 * s * s) / s))


def _build_digital(blend: _Blend) -> _Digital:
    w_atm, _ = blend.total_variance(np.zeros(1))
    half_width = GRID_STDEVS * math.sqrt(float(w_atm[0]))
    k = np.linspace(-half_width, half_width, GRID_POINTS)
    w, dw = blend.total_variance(k)
    s = np.sqrt(w)
    d1 = (-k + 0.5 * w) / s
    d2 = d1 - s
    phi_d1 = np.exp(-0.5 * d1 * d1) / _SQRT_2PI
    raw = ndtr(d2) - np.exp(-k) * phi_d1 * dw / (2.0 * s)
    monotone = _isotonic_non_increasing(np.clip(raw, 0.0, 1.0))
    curve = PchipInterpolator(k, monotone)
    left_gap = 1.0 - float(monotone[0])
    right_p = float(monotone[-1])
    one_stdev = half_width / GRID_STDEVS
    return _Digital(
        blend=blend,
        half_width=half_width,
        curve=curve,
        left_gap=left_gap,
        left_scale=_decay_length(left_gap, abs(float(curve(k[0], 1))), one_stdev),
        right_p=right_p,
        right_scale=_decay_length(right_p, abs(float(curve(k[-1], 1))), one_stdev),
    )


def _decay_length(mass: float, slope: float, cap: float) -> float:
    """Length of the exponential tail `mass * exp(-x / length)` that matches `slope` at x = 0.

    Capped (at one at-the-money standard deviation), so that a flat end of the curve, left by the
    monotone projection, still decays instead of staying constant forever.
    """
    if mass > 0.0 and slope > 0.0:
        return min(mass / slope, cap)
    return cap


class VolSurface:
    """Implied-vol surface of one chain snapshot, evaluated as of `now`.

    Slices (and the probability curve of each target expiry) are built lazily and cached, so
    pricing several strikes of one expiry costs one build.
    """

    def __init__(self, chain: Sequence[OptionQuote], now: datetime) -> None:
        self.now = _as_utc("now", now)
        by_expiry: dict[datetime, list[OptionQuote]] = defaultdict(list)
        for quote in chain:
            if quote.expiry > self.now:
                by_expiry[quote.expiry].append(quote)
        self._by_expiry = by_expiry
        self._expiries = sorted(by_expiry)
        self._slices: dict[datetime, _Slice | None] = {}
        self._digitals: dict[datetime, _Digital] = {}

    def _slice(self, expiry: datetime) -> _Slice | None:
        if expiry not in self._slices:
            self._slices[expiry] = _build_slice(expiry, self._by_expiry[expiry], self.now)
        return self._slices[expiry]

    def _usable_up_to(self, expiry: datetime) -> _Slice | None:
        for index in range(bisect_right(self._expiries, expiry) - 1, -1, -1):
            if (found := self._slice(self._expiries[index])) is not None:
                return found
        return None

    def _usable_from(self, expiry: datetime) -> _Slice | None:
        for index in range(bisect_left(self._expiries, expiry), len(self._expiries)):
            if (found := self._slice(self._expiries[index])) is not None:
                return found
        return None

    def nearest_forward(self) -> float | None:
        """Forward of the first usable expiry: within a few bp of the index for near expiries."""
        return next((s.forward for e in self._expiries if (s := self._slice(e)) is not None), None)

    def blend(self, expiry: datetime) -> _Blend:
        expiry = _as_utc("expiry", expiry)
        tau = (expiry - self.now).total_seconds() / SECONDS_PER_YEAR
        if tau <= 0:
            raise ChainCoverageError(f"expiry {expiry:%Y-%m-%d %H:%M} UTC is not after now")
        later = self._usable_from(expiry)
        if later is None:
            raise ChainCoverageError(
                f"expiry {expiry:%Y-%m-%d %H:%M} UTC is beyond the last usable option expiry"
            )
        earlier = self._usable_up_to(expiry)
        weight_next = 0.0
        if earlier is None:  # before the first expiry: variance grows linearly from zero at now
            parts = ((tau / later.tau, later),)
            forward = later.forward
        elif earlier is later:
            parts = ((1.0, later),)
            forward = later.forward
        else:
            weight_next = (tau - earlier.tau) / (later.tau - earlier.tau)
            parts = ((1.0 - weight_next, earlier), (weight_next, later))
            forward = math.exp(
                (1.0 - weight_next) * math.log(earlier.forward)
                + weight_next * math.log(later.forward)
            )
        return _Blend(
            tau=tau,
            forward=forward,
            parts=parts,
            k_min=max(part.k_min for _, part in parts),
            k_max=min(part.k_max for _, part in parts),
            weight_next=weight_next,
        )

    def digital(self, expiry: datetime) -> _Digital:
        expiry = _as_utc("expiry", expiry)
        if expiry not in self._digitals:
            self._digitals[expiry] = _build_digital(self.blend(expiry))
        return self._digitals[expiry]


def _surface(chain: Sequence[OptionQuote], now: datetime | None) -> VolSurface:
    if now is None:
        if not chain:
            raise ChainCoverageError("empty option chain")
        now = max(quote.timestamp for quote in chain)
    return VolSurface(chain, now)


def _check_strike(strike: float) -> None:
    if not (math.isfinite(strike) and strike > 0):
        raise ValueError(f"strike must be positive and finite, got {strike}")


# --- public API ------------------------------------------------------------------


def implied_vol(
    chain: list[OptionQuote], strike: float, expiry: datetime, now: datetime | None = None
) -> float:
    """Annualised implied vol (fraction) at `strike` for `expiry`, from the interpolated surface.

    Interpolates the smile across strikes and total variance across time. `now` defaults to the
    chain's snapshot time (its latest quote timestamp). Raises ChainCoverageError if the chain
    cannot reach `expiry`. Strikes outside the quoted range get the continued wing.
    """
    _check_strike(strike)
    blend = _surface(chain, now).blend(expiry)
    return blend.iv(math.log(strike / blend.forward))


def prob_above(chain: list[OptionQuote], strike: float, expiry: datetime, now: datetime) -> float:
    """Risk-neutral P(S_T > strike), including the smile correction to N(d2).

    Non-increasing in `strike` and within [0, 1] (up to float rounding). `chain` must hold one
    asset's quotes from one snapshot. Raises ChainCoverageError if `expiry` is not after `now`
    or lies beyond the last usable option expiry. Strikes outside the quoted range are priced on
    the continued wings: use `fair_probability`, which refuses them, for trading decisions.
    """
    _check_strike(strike)
    digital = _surface(chain, now).digital(expiry)
    return digital.p_above(math.log(strike / digital.forward))


def fair_probability(
    market: CryptoMarket,
    chain: list[OptionQuote],
    now: datetime,
    *,
    resolution_basis: float = 0.0,
) -> FairProbability | None:
    """None if the chain cannot cover the market (expiry beyond last option, strike far outside).

    Also None when the market expires within `MIN_TTE`, or a strike is not positive.

    `resolution_basis` is the premium of the resolution source over the Deribit index
    (Binance / Deribit - 1, see `DeribitClient.fetch_resolution_basis`). It is assumed to
    persist until expiry: strikes are mapped into index terms as K / (1 + basis). Default 0
    means "no correction"; the value used is stored in `meta["resolution_basis"]`.
    """
    now = _as_utc("now", now)
    if not -0.5 < resolution_basis < 0.5:
        raise ValueError(f"resolution_basis must be a small fraction, got {resolution_basis}")
    if market.expiry - now < MIN_TTE:
        return None

    quotes = [q for q in chain if q.asset is market.asset]
    surface = VolSurface(quotes, now)
    try:
        digital = surface.digital(market.expiry)
    except ChainCoverageError:
        return None

    match market.kind:
        case MarketKind.ABOVE:
            strikes = {"low": market.strike_low}
        case MarketKind.BELOW:
            strikes = {"high": market.strike_high}
        case MarketKind.RANGE:
            strikes = {"low": market.strike_low, "high": market.strike_high}

    shift = 1.0 + resolution_basis
    moneyness: dict[str, float] = {}
    for name, strike in strikes.items():
        if strike is None or not (math.isfinite(strike) and strike > 0):
            return None
        k = math.log(strike / shift / digital.forward)
        if not digital.covers(k):
            return None
        moneyness[name] = k

    above = {name: digital.p_above(k) for name, k in moneyness.items()}
    flat = {name: digital.p_above_flat_iv(k) for name, k in moneyness.items()}
    ivs = {name: digital.blend.iv(k) for name, k in moneyness.items()}
    match market.kind:
        case MarketKind.ABOVE:
            p_yes, p_flat = above["low"], flat["low"]
        case MarketKind.BELOW:
            p_yes, p_flat = 1.0 - above["high"], 1.0 - flat["high"]
        case MarketKind.RANGE:
            p_yes = above["low"] - above["high"]
            p_flat = flat["low"] - flat["high"]
    p_yes = min(max(p_yes, 0.0), 1.0)
    if not math.isfinite(p_yes):
        return None

    meta: dict[str, float] = {
        "tte_years": digital.blend.tau,
        "iv": sum(ivs.values()) / len(ivs),
        "forward": digital.forward,
        "resolution_basis": resolution_basis,
        "expiry_weight": digital.blend.weight_next,
        "chain_age_seconds": (now - max(q.timestamp for q in quotes)).total_seconds(),
        "p_flat_iv": min(max(p_flat, 0.0), 1.0),
        "smile_adjustment": p_yes - min(max(p_flat, 0.0), 1.0),
    }
    for name in strikes:
        meta[f"moneyness_{name}"] = moneyness[name]
        meta[f"p_above_{name}"] = above[name]
        if len(strikes) > 1:
            meta[f"iv_{name}"] = ivs[name]

    nearest = surface.nearest_forward()
    spot = (nearest if nearest is not None else digital.forward) * shift
    return FairProbability(
        market_id=market.market_id,
        p_yes=p_yes,
        computed_at=now,
        model=MODEL_NAME,
        spot=spot,
        meta=meta,
    )
