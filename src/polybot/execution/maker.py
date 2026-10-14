"""Task H. Market-making simulator: resting limit orders around the fair price.

Paper results here are optimistic by construction: queue position is unknown.
Assume we are last in the queue at our price level and only fill from trades
that print through it, never from the book merely touching our price.

Polymarket's NO book mirrors the YES book: a YES bid at p is a NO ask at 1 - p,
and a taker buying NO at q can fill a YES bid above 1 - q (the exchange mints the
pair). Post-only checks and fills below rely on that.
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import NormalDist

from polybot.models import (
    BookSide,
    CryptoMarket,
    FairProbability,
    Fill,
    MarketOutcome,
    Merge,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Outcome,
    Position,
    WalletTrade,
)

_EPS = 1e-9
_NORMAL = NormalDist()
# Trades reported later than this are ignored: they can no longer be deduplicated
# or matched against orders cancelled since.
_LATE_TRADES = timedelta(minutes=10)


@dataclass(frozen=True, slots=True)
class MakerParams:
    half_spread: float = 0.01  # min distance of each bid below fair; a merged pair earns twice this
    size: float = 50.0  # shares per side
    max_inventory: float = 250.0  # |YES - NO| shares at which the side adding to it stops
    skew_at_max: float = 0.02  # fair-price shift against inventory at max_inventory
    tick: float = 0.01
    min_fair: float = 0.05  # no quotes on near-certain outcomes: tail risk dwarfs the spread
    max_fair: float = 0.95
    min_time_to_expiry: timedelta = timedelta(hours=1)
    quote_lifetime: timedelta = timedelta(seconds=30)  # time until the next requote
    vol_mult: float = 1.0  # half spread >= vol_mult * expected fair move over quote_lifetime
    max_spot_move: float = 0.002  # pull quotes once spot is this far from fair.spot
    latency: timedelta = timedelta(seconds=1)  # orders go live and die this long after we act

    def __post_init__(self) -> None:
        if self.half_spread <= 0 or self.tick <= 0 or self.size < 0 or self.max_inventory <= 0:
            raise ValueError(f"invalid maker params: {self}")


DEFAULT_PARAMS = MakerParams()


@dataclass(frozen=True, slots=True)
class Quote:
    yes_bid: float | None
    no_bid: float | None
    size: float

    def __post_init__(self) -> None:
        for bid in (self.yes_bid, self.no_bid):
            if bid is not None and not 0.0 < bid < 1.0:
                raise ValueError(f"bid must be in (0, 1), got {bid}")
        if self.yes_bid is not None and self.no_bid is not None and self.yes_bid + self.no_bid >= 1:
            raise ValueError("yes_bid + no_bid must be < 1, otherwise our own quotes cross")
        if self.size < 0:
            raise ValueError(f"size must be >= 0, got {self.size}")


def fair_move_std(p_yes: float, horizon: timedelta, time_to_expiry: timedelta) -> float:
    """Std of the fair price change over `horizon` for a binary on a lognormal spot.

    p = N(d2) and d2 moves by dW / sqrt(T - t), so the move is about
    phi(d2) * sqrt(horizon / (T - t)) whatever the volatility level. For RANGE
    markets it is an upper bound. It blows up near expiry at the money.
    """
    if time_to_expiry <= horizon:
        return math.sqrt(p_yes * (1.0 - p_yes))
    d2 = _NORMAL.inv_cdf(min(max(p_yes, 1e-12), 1.0 - 1e-12))
    return _NORMAL.pdf(d2) * math.sqrt(horizon / time_to_expiry)


def make_quote(
    market: CryptoMarket,
    fair: FairProbability,
    book_yes: OrderBook,
    inventory: float,
    *,
    params: MakerParams = DEFAULT_PARAMS,
) -> Quote:
    """Bids on YES and NO such that yes_bid + no_bid < 1, skewed against inventory.

    `inventory` is net YES exposure in shares: YES held minus NO held (pairs merge
    away). The half spread is at least the expected fair move before the next
    requote, so it widens near expiry. Bids are post-only, at most one tick above
    the best bid, and floored to the tick grid.
    """
    if book_yes.token_id != market.yes_token_id or fair.market_id != market.market_id:
        raise ValueError("fair price and book must belong to the quoted market's YES token")
    no_quote = Quote(yes_bid=None, no_bid=None, size=0.0)
    time_left = market.expiry - fair.computed_at
    if time_left < params.min_time_to_expiry or not params.min_fair <= fair.p_yes <= params.max_fair:
        return no_quote

    move = fair_move_std(fair.p_yes, params.quote_lifetime, time_left)
    half = max(params.half_spread, params.vol_mult * move)
    skew = params.skew_at_max * max(-1.0, min(1.0, inventory / params.max_inventory))
    center = fair.p_yes - skew
    yes_bid = _post_only_bid(center - half, book_yes.best_bid, book_yes.best_ask, params.tick)
    no_bid = _post_only_bid(
        1.0 - center - half, _mirror(book_yes.best_ask), _mirror(book_yes.best_bid), params.tick
    )
    if inventory >= params.max_inventory:
        yes_bid = None
    if inventory <= -params.max_inventory:
        no_bid = None
    if yes_bid is None and no_bid is None:
        return no_quote
    return Quote(yes_bid=yes_bid, no_bid=no_bid, size=params.size)


def spot_moved_sharply(
    fair: FairProbability, spot: float, params: MakerParams = DEFAULT_PARAMS
) -> bool:
    """True once spot has run away from the spot the fair price was computed at."""
    if fair.spot <= 0:
        return True
    return abs(spot / fair.spot - 1.0) > params.max_spot_move


def _mirror(price: float | None) -> float | None:
    return None if price is None else 1.0 - price


def _post_only_bid(
    target: float, best_bid: float | None, best_ask: float | None, tick: float
) -> float | None:
    price = min(target, 1.0 - tick)
    if best_ask is not None:
        price = min(price, best_ask - tick)
    if best_bid is not None:
        price = min(price, best_bid + tick)
    price = round(math.floor(price / tick + _EPS) * tick, 10)
    return price if price >= tick else None


def _prints_through(trade: WalletTrade, bid: Order) -> bool:
    """True if a trade printed strictly past our bid, so it would have reached us first."""
    if trade.outcome is bid.outcome:
        return trade.side is BookSide.SELL and trade.price < bid.limit_price - _EPS
    return trade.side is BookSide.BUY and 1.0 - trade.price < bid.limit_price - _EPS


class MakerSimulator:
    """Paper maker: resting BUY limits on YES and NO, pairs merged into $1."""

    def __init__(self, fee_rate: float, params: MakerParams = DEFAULT_PARAMS) -> None:
        self.fee_rate = fee_rate
        self.params = params
        self.open_orders: dict[str, Order] = {}
        self.orders: dict[str, Order] = {}  # every order ever placed
        self.fills: list[Fill] = []
        self.merges: list[Merge] = []
        self.pulls: list[tuple[datetime, str, str]] = []  # (when, market_id, reason)
        self.positions: dict[tuple[str, Outcome], Position] = {}
        self.cash_usd = 0.0
        self.peak_capital_usd = 0.0  # most cash tied up at once
        self._live_from: dict[str, datetime] = {}
        self._dying: defaultdict[str, list[tuple[Order, datetime]]] = defaultdict(list)
        self._seen: dict[tuple[object, ...], datetime] = {}
        self._clock: datetime | None = None
        self._next_id = 1

    def requote(self, market: CryptoMarket, quote: Quote, now: datetime) -> None:
        """Make our resting bids match `quote`. Feed trades up to `now` to on_trades first.

        An order at an unchanged price is left alone; any other change is a cancel
        plus a new order, and both take `params.latency` to reach the book.
        """
        self._advance(now)
        horizon = now - _LATE_TRADES
        for market_id, dying in self._dying.items():
            self._dying[market_id] = [(o, until) for o, until in dying if until >= horizon]
        self._seen = {key: at for key, at in self._seen.items() if at >= horizon}

        for outcome, price in ((Outcome.YES, quote.yes_bid), (Outcome.NO, quote.no_bid)):
            current = self._open_order(market.market_id, outcome)
            if current is not None:
                if price is not None and abs(current.limit_price - price) < _EPS:
                    continue
                self._cancel(current, now)
            if price is not None and quote.size > 0:
                self._place(market, outcome, price, quote.size, now)

    def on_trades(self, trades: list[WalletTrade]) -> list[Fill]:
        """Fill our bids from public trades that printed through their price.

        A print exactly at our price fills nothing: the whole queue at that level is
        assumed ahead of us. Fills are at our limit price and each trade's size is used
        once. Maker-side records are fine too: one only reaches our bids if its resting
        order would have crossed them. A taker record may carry the average price of a
        sweep across levels, which blurs how deep it went.
        """
        fills: list[Fill] = []
        for trade in sorted(trades, key=lambda t: t.timestamp):
            if self._clock is not None and trade.timestamp < self._clock - _LATE_TRADES:
                continue
            key = (
                trade.tx_hash,
                trade.wallet,
                trade.market_id,
                trade.outcome,
                trade.side,
                trade.price,
                trade.size,
                trade.timestamp,
            )
            if key in self._seen:
                continue
            self._seen[key] = trade.timestamp
            self._advance(trade.timestamp)
            left = trade.size
            for order in self._reached_by(trade):
                size = min(order.remaining, left)
                fills.append(self._fill(order, size, trade.timestamp))
                left -= size
                if left <= _EPS:
                    break
        return fills

    def cancel_all(self, market_id: str, reason: str, now: datetime | None = None) -> None:
        """Pull quotes, e.g. when spot moves sharply faster than the fair price refreshes.

        The cancel lands `params.latency` after `now` (default: the latest time seen),
        so trades in between still fill the stale orders.
        """
        if now is not None:
            self._advance(now)
        when = now or self._clock
        orders = [o for o in self.open_orders.values() if o.market_id == market_id]
        if when is None or not orders:
            return
        for order in orders:
            self._cancel(order, when)
        self.pulls.append((when, market_id, reason))

    def inventory(self, market_id: str) -> float:
        """Net YES exposure in shares: YES held minus NO held."""
        return self._held(market_id, Outcome.YES) - self._held(market_id, Outcome.NO)

    def merge_pairs(self, market_id: str, now: datetime) -> Merge | None:
        """Turn every YES+NO pair held in one market into $1."""
        size = min(self._held(market_id, Outcome.YES), self._held(market_id, Outcome.NO))
        if size <= _EPS:
            return None
        yes = self.positions[(market_id, Outcome.YES)]
        no = self.positions[(market_id, Outcome.NO)]
        merge = Merge(
            market_ids=(market_id,),
            size=size,
            cost_basis_usd=size * (yes.avg_price + no.avg_price),
            proceeds_usd=size,
            timestamp=now,
        )
        for position in (yes, no):
            position.size = max(0.0, position.size - size)
        self.cash_usd += size
        self.merges.append(merge)
        return merge

    def settle(self, outcome: MarketOutcome) -> float:
        """Pay winning shares at $1, zero the losers. Returns PnL on the shares held."""
        self.cancel_all(outcome.market_id, "resolved", outcome.resolved_at)
        pnl = 0.0
        for side in Outcome:
            position = self.positions.pop((outcome.market_id, side), None)
            if position is None:
                continue
            payout = position.size if (side is Outcome.YES) == outcome.resolved_yes else 0.0
            self.cash_usd += payout
            pnl += payout - position.size * position.avg_price
        return pnl

    def equity(self, fair_yes: Mapping[str, float]) -> float:
        """Cash plus shares held, marked at fair. Needs a fair p_yes for every held market."""
        value = self.cash_usd
        for (market_id, outcome), position in self.positions.items():
            if position.size > _EPS:
                p_yes = fair_yes[market_id]
                value += position.size * (p_yes if outcome is Outcome.YES else 1.0 - p_yes)
        return value

    def own_trades(self, wallet: str = "paper-maker") -> list[WalletTrade]:
        """Our fills in the shape of public trades, to profile them like any wallet."""
        trades = []
        for fill in self.fills:
            order = self.orders[fill.order_id]
            trades.append(
                WalletTrade(
                    wallet=wallet,
                    market_id=order.market_id,
                    outcome=order.outcome,
                    side=order.side,
                    price=fill.price,
                    size=fill.size,
                    timestamp=fill.timestamp,
                    tx_hash=fill.order_id,
                )
            )
        return trades

    def _open_order(self, market_id: str, outcome: Outcome) -> Order | None:
        orders = self.open_orders.values()
        return next((o for o in orders if o.market_id == market_id and o.outcome is outcome), None)

    def _held(self, market_id: str, outcome: Outcome) -> float:
        position = self.positions.get((market_id, outcome))
        return 0.0 if position is None else position.size

    def _advance(self, now: datetime) -> None:
        self._clock = now if self._clock is None else max(self._clock, now)

    def _place(
        self, market: CryptoMarket, outcome: Outcome, price: float, size: float, now: datetime
    ) -> None:
        order = Order(
            order_id=f"mm-{self._next_id}",
            market_id=market.market_id,
            token_id=market.token_id(outcome),
            outcome=outcome,
            side=BookSide.BUY,
            order_type=OrderType.LIMIT,
            size=size,
            limit_price=price,
            created_at=now,
            strategy="maker",
        )
        self._next_id += 1
        self.open_orders[order.order_id] = order
        self.orders[order.order_id] = order
        self._live_from[order.order_id] = now + self.params.latency

    def _cancel(self, order: Order, now: datetime) -> None:
        del self.open_orders[order.order_id]
        order.status = OrderStatus.CANCELLED
        self._dying[order.market_id].append((order, now + self.params.latency))

    def _reached_by(self, trade: WalletTrade) -> list[Order]:
        """Our live bids this trade would have filled before printing where it did."""
        orders = [o for o in self.open_orders.values() if o.market_id == trade.market_id]
        orders += [o for o, until in self._dying.get(trade.market_id, ()) if trade.timestamp <= until]
        return [
            o
            for o in orders
            if o.remaining > _EPS
            and self._live_from[o.order_id] < trade.timestamp
            and _prints_through(trade, o)
        ]

    def _fill(self, order: Order, size: float, at: datetime) -> Fill:
        price = order.limit_price
        fee = self.fee_rate * price * size
        order.filled_size += size
        order.avg_fill_price = price
        if order.remaining <= _EPS:
            order.status = OrderStatus.FILLED
            self.open_orders.pop(order.order_id, None)
        elif order.status is OrderStatus.OPEN:
            order.status = OrderStatus.PARTIALLY_FILLED

        position = self.positions.setdefault(
            (order.market_id, order.outcome), Position(order.market_id, order.outcome)
        )
        cost = price * size + fee
        position.avg_price = (position.avg_price * position.size + cost) / (position.size + size)
        position.size += size
        self.cash_usd -= cost
        self.peak_capital_usd = max(self.peak_capital_usd, -self.cash_usd)

        fill = Fill(order.order_id, price, size, fee, is_maker=True, timestamp=at)
        self.fills.append(fill)
        return fill


# --- Replay -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tick:
    """One poll of one market: what the bot knew at `time`."""

    time: datetime
    fair: FairProbability
    book_yes: OrderBook
    spot: float | None = None  # latest spot; may be fresher than fair.spot


@dataclass(frozen=True, slots=True)
class SimulationResult:
    n_fills: int
    bought_yes: float  # shares
    bought_no: float
    merged: float  # shares turned into $1 each
    merge_pnl_usd: float
    inventory: float  # YES - NO left unpaired at the end
    pnl_usd: float  # settled if the outcome is known, else marked at the last fair price
    peak_capital_usd: float
    edge_at_fill: float | None  # fair - price per share, by the fair known at fill time
    edge_after: float | None  # same by the fair `markout` later; the drop is adverse selection
    n_pulls: int


def replay(
    market: CryptoMarket,
    ticks: Sequence[Tick],
    trades: Sequence[WalletTrade],
    sim: MakerSimulator,
    *,
    outcome: MarketOutcome | None = None,
    markout: timedelta = timedelta(minutes=5),
) -> SimulationResult:
    """Run the maker over one market's recorded ticks and public trades. Pass a fresh `sim`.

    On every tick: fill from trades since the previous tick, then pull quotes if
    spot ran away from the fair price, else requote and merge pairs. After the last
    tick quotes are cancelled; later trades can only hit them within the latency.
    """
    if not ticks:
        raise ValueError("replay needs at least one tick")
    ticks = sorted(ticks, key=lambda t: t.time)
    tape = sorted((t for t in trades if t.market_id == market.market_id), key=lambda t: t.timestamp)
    tape_times = [t.timestamp for t in tape]
    market_id = market.market_id

    start = 0
    for tick in ticks:
        end = bisect_right(tape_times, tick.time)
        sim.on_trades(tape[start:end])
        start = end
        if tick.spot is not None and spot_moved_sharply(tick.fair, tick.spot, sim.params):
            sim.cancel_all(market_id, "spot_move", tick.time)
            continue
        quote = make_quote(
            market, tick.fair, tick.book_yes, sim.inventory(market_id), params=sim.params
        )
        sim.requote(market, quote, tick.time)
        sim.merge_pairs(market_id, tick.time)
    last = ticks[-1]
    sim.cancel_all(market_id, "end_of_replay", last.time)
    sim.on_trades(tape[start:])
    sim.merge_pairs(market_id, last.time + sim.params.latency)

    fills = [(sim.orders[f.order_id], f) for f in sim.fills]
    fills = [(order, fill) for order, fill in fills if order.market_id == market_id]
    tick_times = [t.time for t in ticks]

    def fair_at(when: datetime) -> FairProbability:
        return ticks[max(0, bisect_right(tick_times, when) - 1)].fair

    shares = sum(fill.size for _, fill in fills)
    edge_at_fill = edge_after = None
    if shares > 0:
        edge_at_fill = sum(
            fill.size * (fair_at(fill.timestamp).p(order.outcome) - fill.price)
            for order, fill in fills
        ) / shares
        edge_after = sum(
            fill.size * (fair_at(fill.timestamp + markout).p(order.outcome) - fill.price)
            for order, fill in fills
        ) / shares

    merges = [m for m in sim.merges if m.market_ids == (market_id,)]
    inventory = sim.inventory(market_id)
    if outcome is not None:
        sim.settle(outcome)
        pnl = sim.cash_usd
    else:
        pnl = sim.equity({market_id: last.fair.p_yes})
    return SimulationResult(
        n_fills=len(fills),
        bought_yes=sum(f.size for o, f in fills if o.outcome is Outcome.YES),
        bought_no=sum(f.size for o, f in fills if o.outcome is Outcome.NO),
        merged=sum(m.size for m in merges),
        merge_pnl_usd=sum(m.pnl_usd for m in merges),
        inventory=inventory,
        pnl_usd=pnl,
        peak_capital_usd=sim.peak_capital_usd,
        edge_at_fill=edge_at_fill,
        edge_after=edge_after,
        n_pulls=sum(1 for _, m, _ in sim.pulls if m == market_id),
    )


# --- Comparison with real wallets (task G) -----------------------------------------


@dataclass(frozen=True, slots=True)
class TradingProfile:
    """How a wallet, or our simulator, trades, in terms that map onto MakerParams."""

    n_trades: int
    n_markets: int
    median_size: float  # shares per trade: compare with size
    share_buys: float  # of volume; a merging maker almost only buys
    share_paired: float  # of bought volume matched by buys of the other outcome
    pair_cost: float | None  # avg YES price + avg NO price per pair: 1 - 2 * half_spread for us
    typical_unpaired: float  # median over markets of the largest |YES - NO|: max_inventory
    median_price: float  # where in [0, 1] it trades: min_fair / max_fair


def profile_trades(trades: Iterable[WalletTrade]) -> TradingProfile:
    rows = sorted(trades, key=lambda t: t.timestamp)
    if not rows:
        raise ValueError("no trades to profile")
    bought: defaultdict[tuple[str, Outcome], list[float]] = defaultdict(lambda: [0.0, 0.0])
    imbalance: defaultdict[str, float] = defaultdict(float)
    peak_unpaired: defaultdict[str, float] = defaultdict(float)
    for t in rows:
        if t.side is BookSide.BUY:
            shares_usd = bought[(t.market_id, t.outcome)]
            shares_usd[0] += t.size
            shares_usd[1] += t.size * t.price
        adds_yes = (t.outcome is Outcome.YES) == (t.side is BookSide.BUY)
        imbalance[t.market_id] += t.size if adds_yes else -t.size
        peak_unpaired[t.market_id] = max(peak_unpaired[t.market_id], abs(imbalance[t.market_id]))

    paired = paired_cost = 0.0
    for market_id in imbalance:
        yes_shares, yes_usd = bought.get((market_id, Outcome.YES), (0.0, 0.0))
        no_shares, no_usd = bought.get((market_id, Outcome.NO), (0.0, 0.0))
        if yes_shares > 0 and no_shares > 0:
            n = min(yes_shares, no_shares)
            paired += n
            paired_cost += n * (yes_usd / yes_shares + no_usd / no_shares)
    volume = sum(t.size for t in rows)
    buy_volume = sum(t.size for t in rows if t.side is BookSide.BUY)
    return TradingProfile(
        n_trades=len(rows),
        n_markets=len(imbalance),
        median_size=statistics.median(t.size for t in rows),
        share_buys=buy_volume / volume if volume else 0.0,
        share_paired=2 * paired / buy_volume if buy_volume else 0.0,
        pair_cost=paired_cost / paired if paired else None,
        typical_unpaired=statistics.median(peak_unpaired.values()),
        median_price=statistics.median(t.price for t in rows),
    )


_PROFILE_ROWS = (
    ("trades", "n_trades", "{:.0f}"),
    ("markets", "n_markets", "{:.0f}"),
    ("median size, shares", "median_size", "{:.1f}"),
    ("buys, share of volume", "share_buys", "{:.2f}"),
    ("paired YES+NO buys", "share_paired", "{:.2f}"),
    ("pair cost YES+NO", "pair_cost", "{:.4f}"),
    ("typical max |YES-NO|", "typical_unpaired", "{:.0f}"),
    ("median price", "median_price", "{:.2f}"),
)


def format_comparison(profiles: Mapping[str, TradingProfile]) -> str:
    """Side-by-side table, e.g. {"we": profile_trades(sim.own_trades()), "Lennie": ...}."""
    width = max(len(label) for label, _, _ in _PROFILE_ROWS)
    lines = [" " * width + "".join(f"{name:>12}" for name in profiles)]
    for label, attr, fmt in _PROFILE_ROWS:
        cells = []
        for profile in profiles.values():
            value = getattr(profile, attr)
            cells.append(f"{'-' if value is None else fmt.format(value):>12}")
        lines.append(f"{label:<{width}}" + "".join(cells))
    return "\n".join(lines)
