"""Task F. Does the model have an edge? Built only from Journal data.

The main score is the Brier score of the model's P(YES) against the Brier score of
the market's P(YES) on the same resolved snapshots. Lower is better. If the model
is not strictly better than the market, there is no edge.

Market P(YES) is the YES mid recorded next to the model (``meta["market_yes_bid"]`` and
``meta["market_yes_ask"]``, written by the main loop). A snapshot whose spread is wider than
``MAX_SCORED_SPREAD`` is not scored: the mid of an empty or one-sided book is ~0.5, which
would hand the model a fake win. Without those keys the YES-token mid from the book snapshot
known at that moment is used (``markets()`` and ``book_snapshots()``); when that is missing
too, a signal recorded at the same timestamp supplies the price. With no fair-probability
rows at all, each signal is scored on its own fair price and entry price.

Paper PnL is the hold-to-resolution result of directional trades. When ``orders()``
and ``fills()`` are available they are replayed (they are the trades that actually
happened) and the matching signals are not added again. Complete-set profit is
merge PnL plus whatever complete-set inventory was left to settle, or the locked
profit of a complete-set opportunity when that trade has no execution record.
"""

from __future__ import annotations

import inspect
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from polybot.models import (
    BookSide,
    CompleteSetOpportunity,
    CryptoMarket,
    FairProbability,
    Fill,
    MarketOutcome,
    Merge,
    Order,
    OrderBook,
    Outcome,
    Signal,
)
from polybot.storage.journal import Journal

MAX_SCORED_SPREAD = 0.05

_DUST = 1e-9
_EDGE = 1e-12
_BUCKETS = 10
_QUOTE_KEYS = ("market_yes_bid", "market_yes_ask")
# Same-tick clock skew only. The next poll is much later, so this does not use a future book.
_BOOK_SKEW = timedelta(seconds=5)


@dataclass(frozen=True, slots=True)
class CalibrationBucket:
    p_low: float
    p_high: float
    n: int
    model_mean_p: float
    market_mean_p: float
    realized_freq: float


@dataclass(frozen=True, slots=True)
class Report:
    n_signals: int
    n_resolved: int  # forecast snapshots in the Brier score
    paper_pnl_usd: float
    brier_model: float  # lower is better
    brier_market: float  # the model must beat this to have an edge
    calibration: tuple[CalibrationBucket, ...]
    complete_set_profit_usd: float


@dataclass(frozen=True, slots=True)
class _Forecast:
    market_id: str
    at: datetime
    model_p: float
    market_p: float
    resolved_yes: bool


@dataclass(slots=True)
class _Pool:
    """One outcome token. Buys blend into a single average, matching PaperBroker."""

    size: float = 0.0
    cost_usd: float = 0.0
    strategy_size: dict[str, float] = field(default_factory=dict)
    realized: dict[str, float] = field(default_factory=dict)


def build_report(journal: Journal) -> Report:
    outcomes = _index_outcomes(_load_list(journal, "outcomes"))
    signals = _load_list(journal, "signals")
    fairs = _load_list(journal, "fair_probabilities")
    fills = _load_list(journal, "fills")
    orders = _load_optional(journal, "orders")
    merges = _load_optional(journal, "merges")
    opportunities = _load_optional(journal, "complete_sets", "complete_set_opportunities")
    markets = _load_optional(journal, "markets")
    # A week of snapshots does not fit in memory: only load books if some fair lacks its quote.
    books = (
        _normalize_books(_load_raw(journal, "book_snapshots", "books", "order_books"))
        if any(isinstance(f, FairProbability) and not _has_quote(f) for f in fairs)
        else []
    )

    forecasts = _forecasts(fairs, signals, outcomes, markets, orders, books)
    paper_pnl, complete_pnl = _profits(
        signals, fills, orders, merges, opportunities, outcomes
    )
    brier_model, brier_market, calibration = _scores(forecasts)
    return Report(
        n_signals=len(signals),
        n_resolved=len(forecasts),
        paper_pnl_usd=paper_pnl,
        brier_model=brier_model,
        brier_market=brier_market,
        calibration=calibration,
        complete_set_profit_usd=complete_pnl,
    )


def format_report(report: Report) -> str:
    gap = report.brier_market - report.brier_model
    lines = [
        "Polybot edge report",
        f"Signals: {report.n_signals}",
        f"Scored forecasts: {report.n_resolved}",
        f"Paper PnL (USD): {_fmt_usd(report.paper_pnl_usd)}",
        f"Complete-set profit (USD): {_fmt_usd(report.complete_set_profit_usd)}",
        f"Brier model: {_fmt_num(report.brier_model)}",
        f"Brier market: {_fmt_num(report.brier_market)}",
        f"Brier gap (market - model): {_fmt_num(gap)}",
        f"Verdict: {_verdict(report)}",
        "Calibration by model P(YES):",
    ]
    if report.calibration:
        for bucket in report.calibration:
            lines.append(
                f"  {_bucket_label(bucket)} n={bucket.n} "
                f"model={bucket.model_mean_p:.4f} market={bucket.market_mean_p:.4f} "
                f"realized={bucket.realized_freq:.4f}"
            )
    else:
        lines.append("  (none)")
    return "\n".join(lines) + "\n"


def _verdict(report: Report) -> str:
    if report.n_resolved == 0 or not _finite(report.brier_model) or not _finite(report.brier_market):
        return "No resolved forecasts yet, so there is no evidence of an edge."
    gap = report.brier_market - report.brier_model
    if gap > _EDGE:
        return "Model beats the market (lower Brier). There may be an edge."
    if gap < -_EDGE:
        return "Model does not beat the market. No edge."
    return "Model matches the market. No edge."


def _forecasts(
    fairs: list[Any],
    signals: list[Any],
    outcomes: dict[str, MarketOutcome],
    markets: list[Any],
    orders: list[Any],
    books: list[tuple[str | None, OrderBook]],
) -> list[_Forecast]:
    fair_rows = [row for row in fairs if isinstance(row, FairProbability)]
    signal_rows = [row for row in signals if isinstance(row, Signal)]
    model = _latest_model(fair_rows)
    roles = _token_roles(markets, orders)
    book_index = _index_books(books, roles)
    signals_at: dict[tuple[str, datetime], list[Signal]] = defaultdict(list)
    for signal in signal_rows:
        signals_at[(signal.market_id, signal.created_at)].append(signal)

    forecasts: list[_Forecast] = []
    if model is not None:
        for fair in fair_rows:
            if fair.model != model:
                continue
            outcome = outcomes.get(fair.market_id)
            if outcome is None or fair.computed_at > outcome.resolved_at:
                continue
            if _has_quote(fair):
                market_p = _quoted_yes_price(fair)
            else:
                market_p = _market_yes_price(
                    fair.market_id,
                    fair.computed_at,
                    book_index,
                    roles,
                    signals_at.get((fair.market_id, fair.computed_at), []),
                )
            model_p = _as_prob(fair.p_yes)
            if market_p is None or model_p is None:
                continue
            forecasts.append(
                _Forecast(fair.market_id, fair.computed_at, model_p, market_p, outcome.resolved_yes)
            )
    else:
        grouped: dict[tuple[str, datetime], list[Signal]] = defaultdict(list)
        for signal in signal_rows:
            grouped[(signal.market_id, signal.created_at)].append(signal)
        for (market_id, at), grouped_signals in grouped.items():
            outcome = outcomes.get(market_id)
            if outcome is None or at > outcome.resolved_at:
                continue
            model_p = _model_yes_from_signals(grouped_signals)
            market_p = _yes_price_from_signals(grouped_signals)
            if model_p is None or market_p is None:
                continue
            forecasts.append(_Forecast(market_id, at, model_p, market_p, outcome.resolved_yes))

    forecasts.sort(key=lambda row: (row.at, row.market_id, row.model_p))
    return forecasts


def _scores(
    forecasts: list[_Forecast],
) -> tuple[float, float, tuple[CalibrationBucket, ...]]:
    if not forecasts:
        return float("nan"), float("nan"), ()
    model_errors = [(row.model_p - _yes(row)) ** 2 for row in forecasts]
    market_errors = [(row.market_p - _yes(row)) ** 2 for row in forecasts]
    return (
        _mean(model_errors),
        _mean(market_errors),
        _calibration(forecasts),
    )


def _calibration(forecasts: list[_Forecast]) -> tuple[CalibrationBucket, ...]:
    groups: dict[int, list[_Forecast]] = defaultdict(list)
    for row in forecasts:
        groups[_bucket_id(row.model_p)].append(row)
    buckets: list[CalibrationBucket] = []
    for idx in sorted(groups):
        rows = groups[idx]
        buckets.append(
            CalibrationBucket(
                p_low=idx / _BUCKETS,
                p_high=(idx + 1) / _BUCKETS,
                n=len(rows),
                model_mean_p=_mean([row.model_p for row in rows]),
                market_mean_p=_mean([row.market_p for row in rows]),
                realized_freq=_mean([_yes(row) for row in rows]),
            )
        )
    return tuple(buckets)


def _profits(
    signals: list[Any],
    fills: list[Any],
    orders: list[Any],
    merges: list[Any],
    opportunities: list[Any],
    outcomes: dict[str, MarketOutcome],
) -> tuple[float, float]:
    order_rows = [row for row in orders if isinstance(row, Order)]
    fill_rows = [row for row in fills if isinstance(row, Fill)]
    merge_rows = [row for row in merges if isinstance(row, Merge)]
    opportunity_rows = [row for row in opportunities if isinstance(row, CompleteSetOpportunity)]
    signal_rows = [row for row in signals if isinstance(row, Signal)]

    realized = _replay(order_rows, fill_rows, merge_rows, outcomes)
    paper = sum(pnl for (strategy, pnl) in realized.items() if strategy != "complete_set")
    complete = sum(pnl for (strategy, pnl) in realized.items() if strategy == "complete_set")
    complete += sum(merge.pnl_usd for merge in merge_rows)

    directional_orders = {
        (order.market_id, order.outcome)
        for order in order_rows
        if order.strategy != "complete_set"
    }
    complete_orders = {
        (order.market_id, order.outcome)
        for order in order_rows
        if order.strategy == "complete_set"
    }
    merged_markets = {market_id for merge in merge_rows for market_id in merge.market_ids}
    opportunity_markets = {
        market_id for opp in opportunity_rows for market_id in opp.market_ids
    }

    for signal in signal_rows:
        outcome = outcomes.get(signal.market_id)
        if outcome is None:
            continue
        pnl = _signal_hold_pnl(signal, outcome.resolved_yes)
        if signal.reason == "complete_set":
            covered = (signal.market_id, signal.outcome) in complete_orders
            covered = covered or signal.market_id in merged_markets
            covered = covered or signal.market_id in opportunity_markets
            if not covered:
                complete += pnl
            continue
        if (signal.market_id, signal.outcome) in directional_orders:
            continue
        paper += pnl

    for opp in opportunity_rows:
        if not _opportunity_executed(opp, merge_rows, order_rows):
            complete += opp.profit_usd
    return paper, complete


def _replay(
    orders: list[Order],
    fills: list[Fill],
    merges: list[Merge],
    outcomes: dict[str, MarketOutcome],
) -> dict[str, float]:
    fills_by_order: dict[str, list[Fill]] = defaultdict(list)
    for fill in fills:
        fills_by_order[fill.order_id].append(fill)

    events: list[tuple[datetime, int, int, str, Order | None, Fill | None, Merge | None]] = []
    seq = 0
    for order in orders:
        for fill in fills_by_order.get(order.order_id, []):
            priority = 0 if order.side is BookSide.BUY else 1
            events.append((fill.timestamp, priority, seq, "fill", order, fill, None))
            seq += 1
    for merge in merges:
        events.append((merge.timestamp, 2, seq, "merge", None, None, merge))
        seq += 1
    events.sort(key=lambda event: (event[0], event[1], event[2]))

    pools: dict[tuple[str, Outcome], _Pool] = defaultdict(_Pool)
    for _at, _priority, _seq, kind, order, fill, merge in events:
        if kind == "fill" and order is not None and fill is not None:
            _apply_fill(pools[(order.market_id, order.outcome)], order, fill)
        elif merge is not None:
            _apply_merge(pools, merge)

    for (market_id, outcome), pool in pools.items():
        resolved = outcomes.get(market_id)
        if resolved is None or pool.size <= _DUST:
            continue
        won = _leg_won(outcome, resolved.resolved_yes)
        _settle(pool, won)

    totals: dict[str, float] = defaultdict(float)
    for pool in pools.values():
        for strategy, pnl in pool.realized.items():
            totals[strategy] += pnl
    return totals


def _apply_fill(pool: _Pool, order: Order, fill: Fill) -> None:
    if fill.size <= _DUST:
        return
    if order.side is BookSide.BUY:
        pool.size += fill.size
        pool.cost_usd += fill.price * fill.size + fill.fee_usd
        pool.strategy_size[order.strategy] = pool.strategy_size.get(order.strategy, 0.0) + fill.size
        return

    held = pool.strategy_size.get(order.strategy, 0.0)
    take = min(held, fill.size, pool.size)
    if take <= _DUST or pool.size <= _DUST:
        return
    average = pool.cost_usd / pool.size
    removed_cost = average * take
    fee = fill.fee_usd * (take / fill.size)
    proceeds = fill.price * take - fee
    pool.realized[order.strategy] = pool.realized.get(order.strategy, 0.0) + proceeds - removed_cost
    pool.size -= take
    pool.cost_usd -= removed_cost
    pool.strategy_size[order.strategy] = held - take
    _squash(pool)


def _apply_merge(pools: dict[tuple[str, Outcome], _Pool], merge: Merge) -> None:
    if len(merge.market_ids) == 1:
        market_id = merge.market_ids[0]
        _consume(pools[(market_id, Outcome.YES)], merge.size)
        _consume(pools[(market_id, Outcome.NO)], merge.size)
        return
    for market_id in merge.market_ids:
        yes_pool = pools[(market_id, Outcome.YES)]
        no_pool = pools[(market_id, Outcome.NO)]
        outcome = Outcome.YES if yes_pool.size >= no_pool.size else Outcome.NO
        _consume(pools[(market_id, outcome)], merge.size)


def _consume(pool: _Pool, size: float) -> None:
    """Drop merged shares. Their PnL is Merge.pnl_usd, so this does not realize it."""
    if size <= _DUST or pool.size <= _DUST:
        return
    take = min(pool.size, size)
    average = pool.cost_usd / pool.size
    pool.cost_usd -= average * take
    pool.size -= take
    left = take
    for strategy in _strategy_order(pool):
        if left <= _DUST:
            break
        held = pool.strategy_size.get(strategy, 0.0)
        cut = min(held, left)
        pool.strategy_size[strategy] = held - cut
        left -= cut
    _squash(pool)


def _settle(pool: _Pool, won: bool) -> None:
    if pool.size <= _DUST:
        return
    payout = 1.0 if won else 0.0
    average = pool.cost_usd / pool.size
    for strategy, held in list(pool.strategy_size.items()):
        if held <= _DUST:
            continue
        pnl = held * (payout - average)
        pool.realized[strategy] = pool.realized.get(strategy, 0.0) + pnl
    pool.size = 0.0
    pool.cost_usd = 0.0
    pool.strategy_size.clear()


def _squash(pool: _Pool) -> None:
    if pool.size <= _DUST:
        pool.size = 0.0
        pool.cost_usd = 0.0
        pool.strategy_size.clear()
        return
    for strategy, held in list(pool.strategy_size.items()):
        if held <= _DUST:
            del pool.strategy_size[strategy]


def _strategy_order(pool: _Pool) -> list[str]:
    return sorted(pool.strategy_size, key=lambda name: (name != "complete_set", name))


def _signal_hold_pnl(signal: Signal, resolved_yes: bool) -> float:
    if signal.size <= 0.0:
        return 0.0
    payoff = 1.0 if _leg_won(signal.outcome, resolved_yes) else 0.0
    fee_per_share = signal.fair_price - signal.entry_price - signal.edge_after_costs
    return signal.size * (payoff - signal.entry_price - fee_per_share)


def _opportunity_executed(
    opp: CompleteSetOpportunity, merges: list[Merge], orders: list[Order]
) -> bool:
    for merge in merges:
        same_time = merge.timestamp == opp.found_at
        if same_time and _same_markets(merge.market_ids, opp.market_ids):
            return True
    return any(
        order.strategy == "complete_set"
        and order.created_at == opp.found_at
        and order.market_id in opp.market_ids
        for order in orders
    )


def _has_quote(fair: FairProbability) -> bool:
    return all(key in fair.meta for key in _QUOTE_KEYS)


def _quoted_yes_price(fair: FairProbability) -> float | None:
    bid, ask = (fair.meta[key] for key in _QUOTE_KEYS)
    if not (_finite(bid) and _finite(ask)) or ask - bid > MAX_SCORED_SPREAD + _DUST:
        return None
    return _as_prob((bid + ask) / 2.0)


def _market_yes_price(
    market_id: str,
    when: datetime,
    book_index: dict[tuple[str, str], list[OrderBook]],
    roles: dict[str, dict[Outcome, str]],
    signals: list[Signal],
) -> float | None:
    rolemap = roles.get(market_id, {})
    yes_token = rolemap.get(Outcome.YES)
    no_token = rolemap.get(Outcome.NO)
    if yes_token is not None:
        book = _select_book(book_index.get((market_id, yes_token), []), when)
        if book is not None:
            price = _book_prob(book)
            if price is not None:
                return price
    if no_token is not None:
        book = _select_book(book_index.get((market_id, no_token), []), when)
        if book is not None:
            price = _book_prob(book)
            if price is not None:
                return _as_prob(1.0 - price)
    return _yes_price_from_signals(signals)


def _yes_price_from_signals(signals: list[Signal]) -> float | None:
    yes_prices = _prices(signal.entry_price for signal in signals if signal.outcome is Outcome.YES)
    no_prices = _prices(signal.entry_price for signal in signals if signal.outcome is Outcome.NO)
    if yes_prices and no_prices:
        yes_ask = _mean(yes_prices)
        no_ask = _mean(no_prices)
        return _as_prob((yes_ask + (1.0 - no_ask)) / 2.0)
    if yes_prices:
        return _mean(yes_prices)
    if no_prices:
        return _as_prob(1.0 - _mean(no_prices))
    return None


def _model_yes_from_signals(signals: list[Signal]) -> float | None:
    probabilities: list[float] = []
    for signal in signals:
        raw = signal.fair_price if signal.outcome is Outcome.YES else 1.0 - signal.fair_price
        probability = _as_prob(raw)
        if probability is not None:
            probabilities.append(probability)
    if not probabilities:
        return None
    return _mean(probabilities)


def _select_book(books: list[OrderBook], when: datetime) -> OrderBook | None:
    if not books:
        return None
    earlier = [book for book in books if book.timestamp <= when]
    if earlier:
        return max(earlier, key=lambda book: book.timestamp)
    nearest = min(books, key=lambda book: abs(book.timestamp - when))
    if abs(nearest.timestamp - when) <= _BOOK_SKEW:
        return nearest
    return None


def _book_prob(book: OrderBook) -> float | None:
    if book.mid is not None:
        return _as_prob(book.mid)
    if book.best_bid is not None:
        return _as_prob(book.best_bid)
    if book.best_ask is not None:
        return _as_prob(book.best_ask)
    return None


def _index_books(
    books: list[tuple[str | None, OrderBook]],
    roles: dict[str, dict[Outcome, str]],
) -> dict[tuple[str, str], list[OrderBook]]:
    token_market: dict[str, str] = {}
    for market_id, rolemap in roles.items():
        for token_id in rolemap.values():
            token_market.setdefault(token_id, market_id)
    index: dict[tuple[str, str], list[OrderBook]] = defaultdict(list)
    for market_id, book in books:
        resolved_market = market_id or token_market.get(book.token_id)
        if resolved_market is None:
            continue
        index[(resolved_market, book.token_id)].append(book)
    return index


def _token_roles(markets: list[Any], orders: list[Any]) -> dict[str, dict[Outcome, str]]:
    roles: dict[str, dict[Outcome, str]] = defaultdict(dict)
    for market in markets:
        if not isinstance(market, CryptoMarket):
            continue
        roles[market.market_id][Outcome.YES] = market.yes_token_id
        roles[market.market_id][Outcome.NO] = market.no_token_id
    for order in orders:
        if isinstance(order, Order):
            roles[order.market_id].setdefault(order.outcome, order.token_id)
    return roles


def _normalize_books(raw: Any) -> list[tuple[str | None, OrderBook]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        rows: list[tuple[str | None, OrderBook]] = []
        for market_id, value in raw.items():
            if isinstance(value, OrderBook):
                rows.append((str(market_id), value))
                continue
            for book in value:
                if isinstance(book, OrderBook):
                    rows.append((str(market_id), book))
        return rows
    rows = []
    for entry in raw:
        if isinstance(entry, OrderBook):
            rows.append((None, entry))
        elif isinstance(entry, tuple) and len(entry) == 2:
            left, right = entry
            if isinstance(right, OrderBook):
                rows.append((None if left is None else str(left), right))
            elif isinstance(left, OrderBook):
                rows.append((None if right is None else str(right), left))
    return rows


def _index_outcomes(rows: list[Any]) -> dict[str, MarketOutcome]:
    chosen: dict[str, MarketOutcome] = {}
    for row in rows:
        if not isinstance(row, MarketOutcome):
            continue
        current = chosen.get(row.market_id)
        if current is None or row.resolved_at >= current.resolved_at:
            chosen[row.market_id] = row
    return chosen


def _latest_model(fairs: list[FairProbability]) -> str | None:
    if not fairs:
        return None
    latest = max(fairs, key=lambda fair: (fair.computed_at, fair.model))
    return latest.model


def _load_list(journal: Journal, name: str) -> list[Any]:
    return _as_list(getattr(journal, name)())


def _load_optional(journal: Journal, *names: str) -> list[Any]:
    raw = _load_raw(journal, *names)
    if raw is None:
        return []
    return _as_list(raw)


def _load_raw(journal: Journal, *names: str) -> Any:
    for name in names:
        fn = getattr(journal, name, None)
        if not callable(fn) or _needs_arguments(fn):
            continue
        return fn()
    return None


def _needs_arguments(fn: Any) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty:
            return True
    return False


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def _bucket_id(probability: float) -> int:
    if probability >= 1.0:
        return _BUCKETS - 1
    if probability <= 0.0:
        return 0
    return min(_BUCKETS - 1, math.floor(probability * _BUCKETS + 1e-12))


def _bucket_label(bucket: CalibrationBucket) -> str:
    right = "]" if bucket.p_high >= 1.0 else ")"
    return f"[{bucket.p_low:.1f}, {bucket.p_high:.1f}{right}"


def _as_prob(value: float) -> float | None:
    if not _finite(value):
        return None
    if -1e-9 <= value <= 1.0 + 1e-9:
        return min(1.0, max(0.0, value))
    return None


def _prices(values: Any) -> list[float]:
    parsed = [_as_prob(value) for value in values]
    return [price for price in parsed if price is not None]


def _leg_won(outcome: Outcome, resolved_yes: bool) -> bool:
    return resolved_yes if outcome is Outcome.YES else not resolved_yes


def _yes(row: _Forecast) -> float:
    return 1.0 if row.resolved_yes else 0.0


def _same_markets(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return len(left) == len(right) and set(left) == set(right)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _fmt_usd(value: float) -> str:
    if not _finite(value):
        return "n/a"
    if abs(value) < 5e-5:
        value = 0.0
    sign = "+" if value > 0.0 else ""
    return f"{sign}{value:.2f}"


def _fmt_num(value: float) -> str:
    if not _finite(value):
        return "n/a"
    return f"{value:.4f}"
