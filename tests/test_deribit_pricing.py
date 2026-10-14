import itertools
import json
import math
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest
from scipy.optimize import brentq

from polybot.deribit.client import parse_instrument_name, parse_option_chain
from polybot.deribit.pricing import (
    MIN_TTE,
    MODEL_NAME,
    ChainCoverageError,
    VolSurface,
    fair_probability,
    implied_vol,
    prob_above,
)
from polybot.models import Asset, CryptoMarket, MarketKind, OptionQuote

FIXTURES = Path(__file__).parent / "fixtures" / "deribit"
NORM = NormalDist()
SECONDS_PER_YEAR = 365 * 86400
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
FORWARD = 2430.0
STRIKES = [float(k) for k in range(2000, 3001, 50)]
EXPIRIES = [NOW + timedelta(hours=20), NOW + timedelta(hours=44), NOW + timedelta(days=16)]


# --- helpers ------------------------------------------------------------------------


def years(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / SECONDS_PER_YEAR


def black76(forward, strike, sigma, tau, *, call):
    """Undiscounted Black-76 price (Deribit's interest rate is 0)."""
    sd = sigma * math.sqrt(tau)
    d1 = (math.log(forward / strike) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if call:
        return forward * NORM.cdf(d1) - strike * NORM.cdf(d2)
    return strike * NORM.cdf(-d2) - forward * NORM.cdf(-d1)


def lognormal_digital(forward, strike, sigma, tau):
    """P(S_T > K) for a lognormal S_T with this forward and vol: plain N(d2)."""
    d2 = (math.log(forward / strike) - 0.5 * sigma * sigma * tau) / (sigma * math.sqrt(tau))
    return NORM.cdf(d2)


def make_chain(expiries, strikes, iv, *, forward=FORWARD, now=NOW, asset=Asset.ETH):
    """A call and a put at every strike. `iv(strike, expiry)` and `forward` may be callables."""
    quotes = []
    for expiry in expiries:
        fwd = forward(expiry) if callable(forward) else forward
        for strike in strikes:
            sigma = iv(strike, expiry) if callable(iv) else iv
            for is_call in (True, False):
                quotes.append(
                    OptionQuote(
                        instrument=f"{asset.value}-{expiry:%d%b%y}-{strike:g}-{'C' if is_call else 'P'}",
                        asset=asset,
                        strike=strike,
                        expiry=expiry,
                        is_call=is_call,
                        mark_iv=sigma,
                        bid_iv=None,
                        ask_iv=None,
                        underlying_price=fwd,
                        timestamp=now,
                    )
                )
    return quotes


def make_market(kind, low, high, expiry, *, asset=Asset.ETH, market_id="0xabc"):
    return CryptoMarket(
        market_id=market_id,
        event_id="evt",
        question="Will ETH ...?",
        asset=asset,
        kind=kind,
        strike_low=low,
        strike_high=high,
        expiry=expiry,
        yes_token_id="yes",
        no_token_id="no",
        resolution_source="Binance ETHUSDT 1m close",
        neg_risk=False,
        volume_usd=10_000.0,
    )


def load_real(asset):
    rows = json.loads((FIXTURES / f"book_summary_{asset.value.lower()}.json").read_text())["result"]
    chain = parse_option_chain(rows, asset)
    return chain, max(q.timestamp for q in chain), rows


@pytest.fixture(scope="module")
def eth():
    return load_real(Asset.ETH)


@pytest.fixture(scope="module")
def btc():
    return load_real(Asset.BTC)


REAL_TARGETS = [
    datetime(2026, 10, 15, 2, tzinfo=UTC),  # before the first Deribit expiry
    datetime(2026, 10, 15, 8, tzinfo=UTC),  # exactly a Deribit expiry
    datetime(2026, 10, 17, 16, tzinfo=UTC),  # between two dailies
    datetime(2026, 10, 26, 16, tzinfo=UTC),  # between a weekly and the monthly
    datetime(2026, 11, 20, 16, tzinfo=UTC),  # between the monthly and the next
]


# --- Black-Scholes on a flat smile ---------------------------------------------------


@pytest.mark.parametrize("sigma", [0.3, 0.6, 1.2])
@pytest.mark.parametrize(
    "target",
    [
        NOW + timedelta(hours=8),  # before the first expiry
        NOW + timedelta(hours=20),  # exactly an expiry
        NOW + timedelta(hours=30),  # between the first two
        NOW + timedelta(hours=44),
        NOW + timedelta(days=5),  # between the second and the third
    ],
)
def test_flat_smile_matches_black_scholes(sigma, target):
    chain = make_chain(EXPIRIES, STRIKES, sigma)
    tau = years(NOW, target)
    # Strikes beyond the quoted range too: a flat smile continues flat.
    for strike in np.linspace(1700.0, 3300.0, 33):
        expected = lognormal_digital(FORWARD, strike, sigma, tau)
        assert prob_above(chain, float(strike), target, NOW) == pytest.approx(expected, abs=1e-6)


def test_variance_is_interpolated_linearly_in_time_not_vol():
    first, second = EXPIRIES[:2]
    sigma1, sigma2 = 0.40, 0.70
    chain = make_chain([first, second], STRIKES, lambda k, e: sigma1 if e == first else sigma2)
    target = NOW + timedelta(hours=30)
    tau1, tau2, tau = years(NOW, first), years(NOW, second), years(NOW, target)
    weight = (tau - tau1) / (tau2 - tau1)
    expected = math.sqrt(((1 - weight) * sigma1**2 * tau1 + weight * sigma2**2 * tau2) / tau)

    assert implied_vol(chain, FORWARD, target) == pytest.approx(expected, rel=1e-9)
    assert abs(expected - ((1 - weight) * sigma1 + weight * sigma2)) > 0.05  # test can tell
    assert prob_above(chain, 2500.0, target, NOW) == pytest.approx(
        lognormal_digital(FORWARD, 2500.0, expected, tau), abs=1e-6
    )


def test_before_first_expiry_the_first_iv_is_held():
    first, second = EXPIRIES[:2]
    chain = make_chain([first, second], STRIKES, lambda k, e: 0.45 if e == first else 0.8)
    assert implied_vol(chain, FORWARD, NOW + timedelta(hours=6)) == pytest.approx(0.45)
    assert implied_vol(chain, FORWARD, NOW + timedelta(minutes=30)) == pytest.approx(0.45)


def test_forward_is_interpolated_log_linearly_in_time():
    first, second = EXPIRIES[:2]
    forwards = {first: 2430.0, second: 2470.0}
    sigma = 0.5
    chain = make_chain([first, second], STRIKES, sigma, forward=lambda e: forwards[e])
    target = NOW + timedelta(hours=26)
    tau1, tau2, tau = years(NOW, first), years(NOW, second), years(NOW, target)
    weight = (tau - tau1) / (tau2 - tau1)
    forward = math.exp((1 - weight) * math.log(2430.0) + weight * math.log(2470.0))
    for strike in (2350.0, 2450.0, 2550.0):
        assert prob_above(chain, strike, target, NOW) == pytest.approx(
            lognormal_digital(forward, strike, sigma, tau), abs=1e-6
        )


# --- the smile correction ------------------------------------------------------------

# (weight, forward / FORWARD, vol); weight-averaged forward is FORWARD, so mixture options are
# priced consistently with a single forward and have a genuine smile.
MIXTURES = {
    "vol_regimes": [(0.5, 1.0, 0.35), (0.5, 1.0, 0.75)],  # symmetric smile
    "skewed": [(0.8, 1.0125, 0.45), (0.2, 0.95, 0.90)],  # crash-risk skew
}


@pytest.mark.parametrize("name", MIXTURES)
@pytest.mark.parametrize("days", [2, 20])
def test_slope_correction_recovers_true_digital_of_a_mixture(name, days):
    """For a mixture of lognormals the true P(S > K) is known exactly.

    Plain N(d2) at the smile's IV is visibly wrong; N(d2) minus the vega * slope term recovers it.
    """
    components = [(w, f * FORWARD, s) for w, f, s in MIXTURES[name]]
    assert sum(w * f for w, f, _ in components) == pytest.approx(FORWARD)
    expiry = NOW + timedelta(days=days)
    tau = years(NOW, expiry)

    def mixture_iv(strike):
        is_call = strike >= FORWARD  # invert the out-of-the-money option: well conditioned
        price = sum(w * black76(f, strike, s, tau, call=is_call) for w, f, s in components)
        return brentq(
            lambda s: black76(FORWARD, strike, s, tau, call=is_call) - price, 1e-3, 5.0, xtol=1e-14
        )

    strikes = [round(FORWARD * math.exp(x), 2) for x in np.arange(-0.20, 0.2001, 0.01)]
    ivs = {k: mixture_iv(k) for k in strikes}
    chain = make_chain([expiry], strikes, lambda k, e: ivs[k])

    worst_inside = worst_edge = worst_plain = 0.0
    for x in np.linspace(-0.20, 0.20, 161):  # the whole quoted range, edge strikes included
        strike = FORWARD * math.exp(x)
        truth = sum(w * lognormal_digital(f, strike, s, tau) for w, f, s in components)
        error = abs(prob_above(chain, strike, expiry, NOW) - truth)
        if abs(x) <= 0.13:
            worst_inside = max(worst_inside, error)
            plain = lognormal_digital(FORWARD, strike, implied_vol(chain, strike, expiry), tau)
            worst_plain = max(worst_plain, abs(plain - truth))
        else:
            worst_edge = max(worst_edge, error)

    assert worst_inside < 2e-4
    assert worst_plain > 0.01  # without the slope term the error is two orders larger
    # Near the outermost strikes the wing must continue with the smile's slope: dropping the
    # slope there (flat continuation) leaves a 1-2 point kink that the monotone fit then spreads
    # into the quoted range.
    assert worst_edge < 1e-3


def test_real_chain_agrees_with_call_spread_from_marks(eth):
    """Independent check on real data: P(S > K) ~ -(C(K+h) - C(K-h)) / 2h from the mark prices."""
    chain, now, rows = eth
    expiry = datetime(2026, 10, 17, 8, tzinfo=UTC)  # a Deribit expiry: no time interpolation
    calls = {}
    for row in rows:
        parsed = parse_instrument_name(row["instrument_name"])
        if parsed and parsed[2] == expiry and parsed[3]:
            calls[parsed[1]] = row["mark_price"] * row["underlying_price"]  # coin -> USD

    plain_errors = []
    for strike in (2380.0, 2400.0, 2420.0, 2440.0, 2460.0, 2500.0, 2540.0):
        lower, upper = strike - 20.0, strike + 20.0  # Deribit lists ETH every $20 near the money
        spread = (calls[lower] - calls[upper]) / (upper - lower)
        fair = fair_probability(make_market(MarketKind.ABOVE, strike, None, expiry), chain, now)
        assert fair.p_yes == pytest.approx(spread, abs=0.008)
        plain_errors.append(abs(fair.meta["p_flat_iv"] - spread))
    assert max(plain_errors) > 0.03  # plain N(d2) is off by several points on this chain


# --- monotonicity and bounds -----------------------------------------------------------


@pytest.mark.parametrize("target", REAL_TARGETS)
@pytest.mark.parametrize("asset", [Asset.ETH, Asset.BTC])
def test_prob_above_is_non_increasing_in_strike_on_real_chains(asset, target, eth, btc):
    chain, now, _ = eth if asset is Asset.ETH else btc
    forward = chain[0].underlying_price
    strikes = np.linspace(0.8 * forward, 1.25 * forward, 300)
    probs = np.array([prob_above(chain, float(k), target, now) for k in strikes])
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.all(np.diff(probs) <= 1e-12)
    assert probs[0] > 0.9 and probs[-1] < 0.1  # and it is a real distribution, not a constant


@pytest.mark.parametrize("spike", [0.05, 0.15, 0.3])
def test_monotone_even_when_marks_contain_butterfly_arbitrage(spike):
    """An IV spike on one strike makes the raw -dC/dK increase somewhere; the output must not."""

    def iv(strike, expiry):
        return (
            0.5 + 2e-7 * (strike - FORWARD) ** 2 + spike * math.exp(-(((strike - 2450) / 15) ** 2))
        )

    chain = make_chain(EXPIRIES[:2], [float(k) for k in range(2000, 2901, 25)], iv)
    target = NOW + timedelta(hours=30)
    strikes = np.linspace(2000.0, 2900.0, 600)
    probs = np.array([prob_above(chain, float(k), target, NOW) for k in strikes])
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.all(np.diff(probs) <= 1e-12)


@pytest.mark.parametrize("target", REAL_TARGETS)
def test_probability_tends_to_one_and_zero_at_extreme_strikes(target, eth):
    chain, now, _ = eth
    assert prob_above(chain, 1e-6, target, now) > 1 - 1e-5
    assert prob_above(chain, 1.0, target, now) > 1 - 1e-5
    assert prob_above(chain, 1e9, target, now) < 1e-5
    assert prob_above(chain, 1e12, target, now) < 1e-5
    strikes = np.geomspace(100.0, 1e5, 400)
    probs = np.array([prob_above(chain, float(k), target, now) for k in strikes])
    assert np.all(np.diff(probs) <= 1e-12)


def test_surface_can_be_reused_across_strikes(eth):
    chain, now, _ = eth
    target = datetime(2026, 10, 17, 16, tzinfo=UTC)
    digital = VolSurface(chain, now).digital(target)
    for strike in (2300.0, 2430.0, 2600.0):
        k = math.log(strike / digital.forward)
        assert digital.p_above(k) == prob_above(chain, strike, target, now)


# --- implied_vol -----------------------------------------------------------------------


@pytest.mark.parametrize("strike", [2100.0, 2400.0, 2440.0, 2500.0, 2900.0])
def test_implied_vol_returns_the_mark_at_quoted_nodes(strike, eth):
    chain, _, _ = eth
    expiry = datetime(2026, 10, 17, 8, tzinfo=UTC)
    mark = next(q.mark_iv for q in chain if q.expiry == expiry and q.strike == strike)
    assert implied_vol(chain, strike, expiry) == pytest.approx(mark, abs=1e-9)


def test_implied_vol_between_strikes_and_expiries_stays_between_neighbours(eth):
    chain, now, _ = eth
    first = datetime(2026, 10, 17, 8, tzinfo=UTC)
    second = datetime(2026, 10, 18, 8, tzinfo=UTC)
    vol = {e: {q.strike: q.mark_iv for q in chain if q.expiry == e} for e in (first, second)}
    strike_iv = implied_vol(chain, 2410.0, first, now)
    assert min(vol[first][2400.0], vol[first][2420.0]) - 0.002 < strike_iv
    assert strike_iv < max(vol[first][2400.0], vol[first][2420.0]) + 0.002
    middle = implied_vol(chain, 2400.0, datetime(2026, 10, 17, 20, tzinfo=UTC), now)
    assert (
        min(vol[first][2400.0], vol[second][2400.0])
        < middle
        < max(vol[first][2400.0], vol[second][2400.0])
    )


# --- errors and unusable data ----------------------------------------------------------


def test_prob_above_rejects_expiry_not_after_now_or_beyond_the_chain():
    chain = make_chain(EXPIRIES, STRIKES, 0.5)
    with pytest.raises(ChainCoverageError):
        prob_above(chain, 2400.0, NOW, NOW)
    with pytest.raises(ChainCoverageError):
        prob_above(chain, 2400.0, NOW - timedelta(hours=1), NOW)
    with pytest.raises(ChainCoverageError):
        prob_above(chain, 2400.0, EXPIRIES[-1] + timedelta(hours=1), NOW)
    with pytest.raises(ChainCoverageError):
        implied_vol([], 2400.0, EXPIRIES[0])
    assert issubclass(ChainCoverageError, ValueError)


@pytest.mark.parametrize("strike", [0.0, -5.0, float("nan"), float("inf")])
def test_prob_above_rejects_bad_strikes(strike):
    chain = make_chain(EXPIRIES, STRIKES, 0.5)
    with pytest.raises(ValueError):
        prob_above(chain, strike, EXPIRIES[0], NOW)
    with pytest.raises(ValueError):
        implied_vol(chain, strike, EXPIRIES[0])


def test_naive_datetimes_are_rejected():
    chain = make_chain(EXPIRIES, STRIKES, 0.5)
    with pytest.raises(ValueError):
        prob_above(chain, 2400.0, EXPIRIES[0], NOW.replace(tzinfo=None))
    with pytest.raises(ValueError):
        prob_above(chain, 2400.0, EXPIRIES[0].replace(tzinfo=None), NOW)


def test_expiries_with_too_few_strikes_are_ignored():
    first, second = EXPIRIES[:2]
    sparse = make_chain([first], [2400.0, 2430.0, 2460.0], 2.0)  # absurd IV, but unusable
    proper = make_chain([second, EXPIRIES[2]], STRIKES, 0.5)
    chain = sparse + proper
    # Between the sparse expiry and the next, the next one's IV is scaled back to now.
    assert implied_vol(chain, FORWARD, NOW + timedelta(hours=30)) == pytest.approx(0.5)


def test_expiries_whose_strikes_do_not_straddle_the_forward_are_ignored():
    first, second = EXPIRIES[:2]
    one_sided = make_chain([first], [float(k) for k in range(2600, 3100, 50)], 2.0)
    proper = make_chain([second, EXPIRIES[2]], STRIKES, 0.5)
    assert implied_vol(one_sided + proper, FORWARD, NOW + timedelta(hours=30)) == pytest.approx(0.5)


def test_expired_quotes_are_ignored():
    chain = make_chain(EXPIRIES, STRIKES, 0.5)
    expired = make_chain([NOW - timedelta(hours=1)], STRIKES, 5.0)
    target = NOW + timedelta(hours=30)
    assert prob_above(expired + chain, 2500.0, target, NOW) == prob_above(
        chain, 2500.0, target, NOW
    )


# --- fair_probability --------------------------------------------------------------------

EXPIRY = datetime(2026, 10, 17, 16, tzinfo=UTC)


def test_fair_probability_fields_and_meta(eth):
    chain, now, _ = eth
    market = make_market(MarketKind.RANGE, 2400.0, 2450.0, EXPIRY, market_id="0xfeed")

    fair = fair_probability(market, chain, now)

    assert fair is not None
    assert (fair.market_id, fair.model, fair.computed_at) == ("0xfeed", MODEL_NAME, now)
    assert 0.0 < fair.p_yes < 1.0
    assert fair.spot == pytest.approx(
        2427.8, abs=2.0
    )  # the Deribit index is 2427.81 in the fixture
    meta = fair.meta
    assert all(isinstance(value, float) for value in meta.values())
    assert meta["tte_years"] == pytest.approx(years(now, EXPIRY))
    assert 0.3 < meta["iv"] < 0.7
    assert meta["iv_low"] != meta["iv_high"]
    assert meta["forward"] == pytest.approx(2427.9, abs=1.0)
    assert meta["resolution_basis"] == 0.0
    assert meta["chain_age_seconds"] == pytest.approx(0.0)
    assert 0.0 < meta["expiry_weight"] < 1.0
    assert meta["moneyness_low"] < meta["moneyness_high"]
    assert meta["p_above_low"] > meta["p_above_high"]
    assert fair.p_yes == pytest.approx(meta["p_above_low"] - meta["p_above_high"])
    assert meta["smile_adjustment"] == pytest.approx(fair.p_yes - meta["p_flat_iv"])
    assert abs(meta["smile_adjustment"]) > 0.005  # the slope term matters on this chain


def test_chain_age_is_recorded(eth):
    chain, now, _ = eth
    market = make_market(MarketKind.ABOVE, 2450.0, None, EXPIRY)
    fair = fair_probability(market, chain, now + timedelta(seconds=90))
    assert fair.meta["chain_age_seconds"] == pytest.approx(90.0)


def test_range_is_a_difference_and_below_is_the_complement(eth):
    chain, now, _ = eth
    low, high = 2400.0, 2500.0
    above_low = prob_above(chain, low, EXPIRY, now)
    above_high = prob_above(chain, high, EXPIRY, now)

    above = fair_probability(make_market(MarketKind.ABOVE, low, None, EXPIRY), chain, now)
    below = fair_probability(make_market(MarketKind.BELOW, None, high, EXPIRY), chain, now)
    inside = fair_probability(make_market(MarketKind.RANGE, low, high, EXPIRY), chain, now)

    assert above.p_yes == pytest.approx(above_low, abs=1e-12)
    assert below.p_yes == pytest.approx(1.0 - above_high, abs=1e-12)
    assert inside.p_yes == pytest.approx(above_low - above_high, abs=1e-12)
    assert inside.p_yes >= 0.0


def test_complete_ladder_of_markets_sums_to_one(eth):
    chain, now, _ = eth
    edges = [2150.0 + 50.0 * i for i in range(15)]  # 2150 ... 2850
    markets = [make_market(MarketKind.BELOW, None, edges[0], EXPIRY, market_id="below")]
    markets += [
        make_market(MarketKind.RANGE, low, high, EXPIRY, market_id=f"{low:g}-{high:g}")
        for low, high in itertools.pairwise(edges)
    ]
    markets.append(make_market(MarketKind.ABOVE, edges[-1], None, EXPIRY, market_id="above"))

    fairs = [fair_probability(m, chain, now) for m in markets]

    assert all(f is not None for f in fairs)
    assert all(f.p_yes >= 0.0 for f in fairs)
    assert sum(f.p_yes for f in fairs) == pytest.approx(1.0, abs=1e-9)
    # ...and the ranges alone add up to exactly what lies between the outer strikes
    ranges = sum(f.p_yes for f in fairs[1:-1])
    assert ranges == pytest.approx(
        prob_above(chain, edges[0], EXPIRY, now) - prob_above(chain, edges[-1], EXPIRY, now)
    )


def test_complete_ladder_sums_to_one_on_flat_smile_with_basis():
    chain = make_chain(EXPIRIES, STRIKES, 0.55)
    expiry = NOW + timedelta(hours=30)
    edges = [2200.0 + 40.0 * i for i in range(12)]
    markets = [make_market(MarketKind.BELOW, None, edges[0], expiry)]
    markets += [make_market(MarketKind.RANGE, a, b, expiry) for a, b in itertools.pairwise(edges)]
    markets.append(make_market(MarketKind.ABOVE, edges[-1], None, expiry))
    total = sum(fair_probability(m, chain, NOW, resolution_basis=0.0013).p_yes for m in markets)
    assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    ("kind", "low", "high"),
    [
        (MarketKind.ABOVE, 2450.0, None),
        (MarketKind.BELOW, None, 2450.0),
        (MarketKind.RANGE, 2400.0, 2450.0),
    ],
)
def test_fair_probability_agrees_with_black_scholes_on_a_flat_smile(kind, low, high):
    sigma = 0.6
    chain = make_chain(EXPIRIES, STRIKES, sigma)
    expiry = NOW + timedelta(hours=30)
    tau = years(NOW, expiry)

    def above(strike):
        return lognormal_digital(FORWARD, strike, sigma, tau)

    expected = {
        MarketKind.ABOVE: above(2450.0),
        MarketKind.BELOW: 1.0 - above(2450.0),
        MarketKind.RANGE: above(2400.0) - above(2450.0),
    }[kind]

    fair = fair_probability(make_market(kind, low, high, expiry), chain, NOW)

    assert fair.p_yes == pytest.approx(expected, abs=1e-6)
    assert fair.meta["smile_adjustment"] == pytest.approx(0.0, abs=1e-6)
    assert fair.meta["iv"] == pytest.approx(sigma)


def test_resolution_basis_shifts_strikes_into_index_terms(eth):
    chain, now, _ = eth
    basis = 0.002  # Binance trades 0.2% above Deribit's index
    above_binance = fair_probability(
        make_market(MarketKind.ABOVE, 2450.0, None, EXPIRY), chain, now, resolution_basis=basis
    )
    plain = fair_probability(make_market(MarketKind.ABOVE, 2450.0, None, EXPIRY), chain, now)
    equivalent = fair_probability(
        make_market(MarketKind.ABOVE, 2450.0 / (1 + basis), None, EXPIRY), chain, now
    )
    below_binance = fair_probability(
        make_market(MarketKind.BELOW, None, 2450.0, EXPIRY), chain, now, resolution_basis=basis
    )

    assert above_binance.p_yes == pytest.approx(equivalent.p_yes, abs=1e-12)
    assert above_binance.p_yes > plain.p_yes + 0.01  # 0.2% of spot is several points here
    assert below_binance.p_yes < 1.0 - plain.p_yes
    assert above_binance.meta["resolution_basis"] == basis
    assert plain.meta["resolution_basis"] == 0.0
    assert above_binance.spot == pytest.approx(plain.spot * (1 + basis))


@pytest.mark.parametrize("basis", [0.5, -0.5, 0.9, float("nan")])
def test_absurd_resolution_basis_is_rejected(basis, eth):
    chain, now, _ = eth
    with pytest.raises(ValueError):
        fair_probability(
            make_market(MarketKind.ABOVE, 2450.0, None, EXPIRY), chain, now, resolution_basis=basis
        )


def test_fair_probability_accepts_any_utc_offset(eth):
    chain, now, _ = eth
    market = make_market(MarketKind.ABOVE, 2450.0, None, EXPIRY)
    moscow = now.astimezone(timezone(timedelta(hours=3)))
    assert (
        fair_probability(market, chain, moscow).p_yes == fair_probability(market, chain, now).p_yes
    )


def test_other_assets_in_the_chain_are_ignored(eth, btc):
    eth_chain, now, _ = eth
    btc_chain, _, _ = btc
    market = make_market(MarketKind.ABOVE, 2450.0, None, EXPIRY)
    alone = fair_probability(market, eth_chain, now)
    mixed = fair_probability(market, btc_chain + eth_chain, now)
    assert mixed.p_yes == alone.p_yes
    assert fair_probability(market, btc_chain, now) is None


def test_btc_market_on_the_btc_chain(btc):
    chain, now, _ = btc
    market = make_market(MarketKind.ABOVE, 82000.0, None, EXPIRY, asset=Asset.BTC)
    fair = fair_probability(market, chain, now)
    assert 0.45 < fair.p_yes < 0.6  # near the forward (about 81.95k)
    assert fair.spot == pytest.approx(81940.0, abs=60.0)


def test_market_expiring_just_after_the_minimum_horizon_is_priced(eth):
    chain, now, _ = eth
    expiry = now + MIN_TTE + timedelta(minutes=1)
    fair = fair_probability(make_market(MarketKind.ABOVE, 2430.0, None, expiry), chain, now)
    assert fair is not None
    # Minutes out, the strike 0.1% above the forward is ~0.6 standard deviations away.
    plain = lognormal_digital(fair.meta["forward"], 2430.0, fair.meta["iv"], fair.meta["tte_years"])
    assert 0.2 < fair.p_yes < 0.4
    assert fair.p_yes == pytest.approx(plain, abs=0.02)


@pytest.mark.parametrize(
    "case",
    [
        "expiry beyond the last Deribit expiry",
        "expiry in the past",
        "expiry now",
        "expiry within the minimum horizon",
        "above: strike far above the chain",
        "below: strike far below the chain",
        "range: one strike far outside",
        "above: just beyond the highest quoted strike",
        "strike is zero",
        "empty chain",
    ],
)
def test_fair_probability_is_none_when_the_chain_cannot_cover_the_market(case, eth):
    chain, now, _ = eth
    market = make_market(MarketKind.ABOVE, 2450.0, None, EXPIRY)
    match case:
        case "expiry beyond the last Deribit expiry":
            market = make_market(MarketKind.ABOVE, 2450.0, None, datetime(2028, 1, 1, tzinfo=UTC))
        case "expiry in the past":
            market = make_market(MarketKind.ABOVE, 2450.0, None, now - timedelta(hours=2))
        case "expiry now":
            market = make_market(MarketKind.ABOVE, 2450.0, None, now)
        case "expiry within the minimum horizon":
            market = make_market(MarketKind.ABOVE, 2450.0, None, now + MIN_TTE / 2)
        case "above: strike far above the chain":
            market = make_market(MarketKind.ABOVE, 9000.0, None, EXPIRY)
        case "below: strike far below the chain":
            market = make_market(MarketKind.BELOW, None, 500.0, EXPIRY)
        case "range: one strike far outside":
            market = make_market(MarketKind.RANGE, 2400.0, 6000.0, EXPIRY)
        case "above: just beyond the highest quoted strike":
            market = make_market(MarketKind.ABOVE, 3000.0, None, EXPIRY)  # chain stops at 2900
        case "strike is zero":
            market = make_market(MarketKind.ABOVE, 0.0, None, EXPIRY)
        case "empty chain":
            chain = []
    assert fair_probability(market, chain, now) is None


def test_fair_probability_is_none_for_chains_without_usable_expiries():
    market = make_market(MarketKind.ABOVE, 2450.0, None, NOW + timedelta(hours=30))
    too_few = make_chain(EXPIRIES, [2400.0, 2430.0, 2460.0], 0.5)
    one_sided = make_chain(EXPIRIES, [float(k) for k in range(2600, 3100, 50)], 0.5)
    assert fair_probability(market, too_few, NOW) is None
    assert fair_probability(market, one_sided, NOW) is None


def test_strike_at_the_edge_of_the_quoted_range_is_still_priced(eth):
    chain, now, _ = eth
    expiry = datetime(2026, 10, 17, 8, tzinfo=UTC)
    assert fair_probability(make_market(MarketKind.ABOVE, 2100.0, None, expiry), chain, now)
    assert fair_probability(make_market(MarketKind.BELOW, None, 2900.0, expiry), chain, now)
