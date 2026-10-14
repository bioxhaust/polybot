"""Task C. Taker execution costs from a real order book."""

from __future__ import annotations

import math

from polybot.models import BookSide, FillEstimate, OrderBook

_EPS = 1e-12


def estimate_fill(book: OrderBook, side: BookSide, size: float, fee_rate: float) -> FillEstimate:
    """Walk the book level by level. Partial fill if the book is too thin."""
    if size < 0.0:
        raise ValueError(f"size must be >= 0, got {size}")
    if fee_rate < 0.0:
        raise ValueError(f"fee_rate must be >= 0, got {fee_rate}")

    levels = book.asks if side is BookSide.BUY else book.bids
    if size == 0.0 or not levels:
        return FillEstimate(
            token_id=book.token_id,
            side=side,
            size_requested=size,
            size_filled=0.0,
            avg_price=0.0,
            worst_price=0.0,
            slippage=0.0,
            fee_usd=0.0,
        )

    remaining = size
    size_filled = 0.0
    notional = 0.0
    best_price: float | None = None
    worst_price = 0.0
    tol = _EPS * max(1.0, size)

    for level in levels:
        if level.size <= 0.0:
            continue
        if best_price is None:
            best_price = level.price

        if level.size >= remaining - tol:
            take = remaining
            size_filled = size
            notional += take * level.price
            worst_price = level.price
            remaining = 0.0
            break

        take = level.size
        size_filled += take
        notional += take * level.price
        worst_price = level.price
        remaining = size - size_filled

    if size_filled == 0.0 or best_price is None:
        return FillEstimate(
            token_id=book.token_id,
            side=side,
            size_requested=size,
            size_filled=0.0,
            avg_price=0.0,
            worst_price=0.0,
            slippage=0.0,
            fee_usd=0.0,
        )

    avg_price = notional / size_filled
    raw_slippage = avg_price - best_price if side is BookSide.BUY else best_price - avg_price
    slippage = max(0.0, raw_slippage)
    fee_usd = notional * fee_rate

    return FillEstimate(
        token_id=book.token_id,
        side=side,
        size_requested=size,
        size_filled=size_filled,
        avg_price=avg_price,
        worst_price=worst_price,
        slippage=slippage,
        fee_usd=fee_usd,
    )


def edge_after_costs(fair_price: float, fill: FillEstimate) -> float:
    """Per-share edge of a BUY after slippage and fees. Negative means no trade."""
    if fill.side is not BookSide.BUY:
        raise ValueError(f"edge_after_costs requires a BUY fill, got {fill.side}")
    if fill.size_filled <= 0.0:
        return -math.inf
    fee_per_share = fill.fee_usd / fill.size_filled
    return fair_price - fill.avg_price - fee_per_share


def max_size_for_edge(
    book: OrderBook, fair_price: float, min_edge: float, fee_rate: float
) -> float:
    """Largest BUY size whose edge after costs is still >= min_edge."""
    if fee_rate < 0.0:
        raise ValueError(f"fee_rate must be >= 0, got {fee_rate}")
    if not book.asks:
        return 0.0

    max_avg_cost = fair_price - min_edge
    if max_avg_cost <= 0.0:
        return 0.0

    cum_size = 0.0
    cum_cost = 0.0

    for level in book.asks:
        if level.size <= 0.0:
            continue
        unit_cost = level.price * (1.0 + fee_rate)
        level_cost = level.size * unit_cost

        if cum_size == 0.0 and unit_cost > max_avg_cost + _EPS:
            return 0.0

        new_size = cum_size + level.size
        new_cost = cum_cost + level_cost
        if new_cost / new_size <= max_avg_cost + _EPS:
            cum_size = new_size
            cum_cost = new_cost
        else:
            if unit_cost > max_avg_cost:
                extra = (cum_size * max_avg_cost - cum_cost) / (unit_cost - max_avg_cost)
                cum_size += max(0.0, min(level.size, extra))
            break

    # Guard against tiny IEEE-754 rounding pushing edge 1 ULP below min_edge.
    for _ in range(8):
        if cum_size <= 0.0:
            return 0.0
        fill = estimate_fill(book, BookSide.BUY, cum_size, fee_rate)
        if edge_after_costs(fair_price, fill) >= min_edge:
            break
        cum_size = math.nextafter(cum_size, 0.0)

    return cum_size
