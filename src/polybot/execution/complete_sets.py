"""Task C. Riskless complete sets: YES+NO of one market, or YES of every exclusive range.

A set is only complete if its legs are truly mutually exclusive and exhaustive
under the markets' resolution rules; same event_id, neg_risk, matching expiry
and contiguous ranges covering everything must all be checked.
"""

from __future__ import annotations

import math
from datetime import datetime

from polybot.execution.costs import estimate_fill
from polybot.models import (
    BookSide,
    CompleteSetOpportunity,
    CryptoMarket,
    MarketKind,
    OrderBook,
    PriceLevel,
)

_EPS = 1e-9


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware UTC, got naive {value!r}")


def _build_opportunity(
    market_ids: tuple[str, ...],
    token_ids: tuple[str, ...],
    books: dict[str, OrderBook],
    fee_rate: float,
    now: datetime,
) -> CompleteSetOpportunity | None:
    """Walk ask books of all legs in lockstep while marginal cost per unit set < 1 USDC."""
    if fee_rate < 0.0:
        raise ValueError(f"fee_rate must be >= 0, got {fee_rate}")

    leg_books: list[OrderBook] = []
    for tid in token_ids:
        book = books.get(tid)
        if book is None or not book.asks:
            return None
        leg_books.append(book)

    # Filter out any zero-sized levels.
    leg_levels: list[list[PriceLevel]] = [
        [lvl for lvl in book.asks if lvl.size > 0.0] for book in leg_books
    ]
    if any(not lvls for lvls in leg_levels):
        return None

    indices = [0] * len(leg_levels)
    rem_sizes = [lvls[0].size for lvls in leg_levels]
    total_size = 0.0

    while True:
        marginal_cost = sum(
            leg_levels[i][indices[i]].price * (1.0 + fee_rate) for i in range(len(leg_levels))
        )
        if marginal_cost >= 1.0 - _EPS:
            break

        step = min(rem_sizes)
        total_size += step

        exhausted = False
        for i in range(len(leg_levels)):
            rem_sizes[i] -= step
            if rem_sizes[i] <= _EPS:
                indices[i] += 1
                if indices[i] >= len(leg_levels[i]):
                    exhausted = True
                else:
                    rem_sizes[i] = leg_levels[i][indices[i]].size
        if exhausted:
            break

    if total_size <= 0.0:
        return None

    fills = [estimate_fill(book, BookSide.BUY, total_size, fee_rate) for book in leg_books]
    if not all(f.fully_filled for f in fills):
        return None

    total_cost_usd = sum(f.cost_usd for f in fills)
    payout_usd = total_size
    if total_cost_usd >= payout_usd - _EPS:
        return None

    return CompleteSetOpportunity(
        market_ids=market_ids,
        token_ids=token_ids,
        size=total_size,
        total_cost_usd=total_cost_usd,
        payout_usd=payout_usd,
        found_at=now,
    )


def find_binary_sets(
    markets: list[CryptoMarket], books: dict[str, OrderBook], fee_rate: float, now: datetime
) -> list[CompleteSetOpportunity]:
    _require_utc("now", now)
    opportunities: list[CompleteSetOpportunity] = []

    for market in markets:
        if market.expiry <= now:
            continue
        opp = _build_opportunity(
            market_ids=(market.market_id,),
            token_ids=(market.yes_token_id, market.no_token_id),
            books=books,
            fee_rate=fee_rate,
            now=now,
        )
        if opp is not None:
            opportunities.append(opp)

    return opportunities


def _ordered_exhaustive_event(group: list[CryptoMarket]) -> list[CryptoMarket] | None:
    """Return markets ordered from lowest to highest strike if they form a valid complete set."""
    if len(group) < 2:
        return None

    first = group[0]
    if not first.event_id:
        return None

    # All legs must have neg_risk=True, same expiry, same asset, same resolution_source,
    # and distinct market_ids / yes_token_ids.
    if not all(
        m.neg_risk
        and m.expiry == first.expiry
        and m.asset == first.asset
        and m.resolution_source == first.resolution_source
        for m in group
    ):
        return None

    if len({m.market_id for m in group}) != len(group):
        return None
    if len({m.yes_token_id for m in group}) != len(group):
        return None

    below = [m for m in group if m.kind is MarketKind.BELOW]
    above = [m for m in group if m.kind is MarketKind.ABOVE]
    ranges = [m for m in group if m.kind is MarketKind.RANGE]

    # Must have exactly one left tail (< K_1) and one right tail (> K_m),
    # and every market in the group must be accounted for.
    if len(below) != 1 or len(above) != 1:
        return None
    if len(below) + len(ranges) + len(above) != len(group):
        return None

    below_mkt = below[0]
    above_mkt = above[0]
    if below_mkt.strike_low is not None and below_mkt.strike_low > 0.0:
        return None
    if above_mkt.strike_high is not None:
        return None

    ranges_sorted = sorted(ranges, key=lambda m: (m.strike_low or 0.0, m.strike_high or 0.0))

    cursor = below_mkt.strike_high
    assert cursor is not None
    for r_mkt in ranges_sorted:
        assert r_mkt.strike_low is not None and r_mkt.strike_high is not None
        if not math.isclose(r_mkt.strike_low, cursor, rel_tol=0.0, abs_tol=_EPS):
            return None
        cursor = r_mkt.strike_high

    assert above_mkt.strike_low is not None
    if not math.isclose(above_mkt.strike_low, cursor, rel_tol=0.0, abs_tol=_EPS):
        return None

    return [below_mkt, *ranges_sorted, above_mkt]


def find_range_sets(
    markets: list[CryptoMarket], books: dict[str, OrderBook], fee_rate: float, now: datetime
) -> list[CompleteSetOpportunity]:
    _require_utc("now", now)
    by_event: dict[str, list[CryptoMarket]] = {}
    for market in markets:
        by_event.setdefault(market.event_id, []).append(market)

    opportunities: list[CompleteSetOpportunity] = []
    for group in by_event.values():
        ordered = _ordered_exhaustive_event(group)
        if ordered is None:
            continue
        if ordered[0].expiry <= now:
            continue

        opp = _build_opportunity(
            market_ids=tuple(m.market_id for m in ordered),
            token_ids=tuple(m.yes_token_id for m in ordered),
            books=books,
            fee_rate=fee_rate,
            now=now,
        )
        if opp is not None:
            opportunities.append(opp)

    return opportunities
