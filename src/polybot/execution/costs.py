"""Task C. Taker execution costs from a real order book."""

from __future__ import annotations

from polybot.models import BookSide, FillEstimate, OrderBook


def estimate_fill(book: OrderBook, side: BookSide, size: float, fee_rate: float) -> FillEstimate:
    """Walk the book level by level. Partial fill if the book is too thin."""
    raise NotImplementedError


def edge_after_costs(fair_price: float, fill: FillEstimate) -> float:
    """Per-share edge of a BUY after slippage and fees. Negative means no trade."""
    raise NotImplementedError


def max_size_for_edge(
    book: OrderBook, fair_price: float, min_edge: float, fee_rate: float
) -> float:
    """Largest BUY size whose edge after costs is still >= min_edge."""
    raise NotImplementedError
