from datetime import UTC, datetime

import pytest

from polybot.execution.costs import edge_after_costs, estimate_fill, max_size_for_edge
from polybot.models import BookSide, OrderBook, PriceLevel

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def make_book(
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
    token_id: str = "yes_tok",
) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        bids=tuple(PriceLevel(p, s) for p, s in (bids or [])),
        asks=tuple(PriceLevel(p, s) for p, s in (asks or [])),
        timestamp=NOW,
    )


def test_estimate_fill_single_level_buy():
    book = make_book(asks=[(0.40, 100.0), (0.45, 200.0)])
    fill = estimate_fill(book, BookSide.BUY, size=60.0, fee_rate=0.01)

    assert fill.token_id == "yes_tok"
    assert fill.side is BookSide.BUY
    assert fill.size_requested == 60.0
    assert fill.size_filled == pytest.approx(60.0)
    assert fill.fully_filled is True
    assert fill.avg_price == pytest.approx(0.40)
    assert fill.worst_price == pytest.approx(0.40)
    assert fill.slippage == pytest.approx(0.0)
    assert fill.fee_usd == pytest.approx(60.0 * 0.40 * 0.01)
    assert fill.cost_usd == pytest.approx(60.0 * 0.40 * 1.01)


def test_estimate_fill_multi_level_buy():
    book = make_book(asks=[(0.40, 50.0), (0.44, 50.0), (0.50, 100.0)])
    fill = estimate_fill(book, BookSide.BUY, size=150.0, fee_rate=0.02)

    expected_notional = 50.0 * 0.40 + 50.0 * 0.44 + 50.0 * 0.50  # 67.0
    expected_avg = expected_notional / 150.0

    assert fill.fully_filled is True
    assert fill.size_filled == pytest.approx(150.0)
    assert fill.avg_price == pytest.approx(expected_avg)
    assert fill.worst_price == pytest.approx(0.50)
    assert fill.slippage == pytest.approx(expected_avg - 0.40)
    assert fill.fee_usd == pytest.approx(expected_notional * 0.02)
    assert fill.cost_usd == pytest.approx(expected_notional * 1.02)


def test_estimate_fill_multi_level_sell():
    book = make_book(bids=[(0.60, 40.0), (0.55, 60.0), (0.50, 100.0)])
    fill = estimate_fill(book, BookSide.SELL, size=100.0, fee_rate=0.01)

    assert fill.fully_filled is True
    assert fill.size_filled == pytest.approx(100.0)
    assert fill.avg_price == pytest.approx(0.57)
    assert fill.worst_price == pytest.approx(0.55)
    assert fill.slippage == pytest.approx(0.60 - 0.57)
    assert fill.fee_usd == pytest.approx(0.57)
    assert fill.cost_usd == pytest.approx(57.0 - 0.57)


def test_estimate_fill_partial_when_book_thin():
    book = make_book(asks=[(0.40, 30.0), (0.45, 20.0)])
    fill = estimate_fill(book, BookSide.BUY, size=80.0, fee_rate=0.0)

    assert fill.fully_filled is False
    assert fill.size_requested == 80.0
    assert fill.size_filled == pytest.approx(50.0)
    assert fill.avg_price == pytest.approx((30.0 * 0.40 + 20.0 * 0.45) / 50.0)
    assert fill.worst_price == pytest.approx(0.45)


def test_estimate_fill_empty_book_and_zero_size():
    empty = make_book()
    fill_empty = estimate_fill(empty, BookSide.BUY, size=50.0, fee_rate=0.01)
    assert fill_empty.fully_filled is False
    assert fill_empty.size_filled == 0.0
    assert fill_empty.avg_price == 0.0
    assert fill_empty.worst_price == 0.0
    assert fill_empty.slippage == 0.0
    assert fill_empty.fee_usd == 0.0

    book = make_book(asks=[(0.40, 100.0)])
    fill_zero = estimate_fill(book, BookSide.BUY, size=0.0, fee_rate=0.01)
    assert fill_zero.fully_filled is True
    assert fill_zero.size_filled == 0.0
    assert fill_zero.cost_usd == 0.0


def test_estimate_fill_float_accumulation_exact():
    # 0.3 + 0.6 + 0.1 in float64 is 0.9999999999999999 < 1.0 without tolerance handling.
    book = make_book(asks=[(0.40, 0.3), (0.42, 0.6), (0.45, 0.1)])
    fill = estimate_fill(book, BookSide.BUY, size=1.0, fee_rate=0.0)
    assert fill.fully_filled is True
    assert fill.size_filled == 1.0


def test_edge_after_costs():
    book = make_book(asks=[(0.40, 50.0), (0.50, 50.0)])
    fill = estimate_fill(book, BookSide.BUY, size=100.0, fee_rate=0.02)
    # avg_price = 0.45, fee per share = 0.45 * 0.02 = 0.009, total cost/share = 0.459
    edge = edge_after_costs(fair_price=0.50, fill=fill)
    assert edge == pytest.approx(0.50 - 0.459)

    assert edge_after_costs(fair_price=0.45, fill=fill) < 0.0

    empty_fill = estimate_fill(make_book(), BookSide.BUY, size=10.0, fee_rate=0.0)
    assert edge_after_costs(fair_price=0.90, fill=empty_fill) < 0.0


def test_edge_after_costs_rejects_sell():
    book = make_book(bids=[(0.50, 10.0)])
    sell_fill = estimate_fill(book, BookSide.SELL, size=10.0, fee_rate=0.0)
    with pytest.raises(ValueError):
        edge_after_costs(0.40, sell_fill)


def test_max_size_for_edge_partial_level():
    book = make_book(asks=[(0.40, 100.0), (0.50, 200.0), (0.60, 100.0)])
    # fair = 0.48, min_edge = 0.03 -> max avg cost = 0.45.
    # 100 @ 0.40 + 100 @ 0.50 gives avg cost = 0.45 -> total size = 200.
    size = max_size_for_edge(book, fair_price=0.48, min_edge=0.03, fee_rate=0.0)
    assert size == pytest.approx(200.0)

    fill_at_max = estimate_fill(book, BookSide.BUY, size=size, fee_rate=0.0)
    assert edge_after_costs(0.48, fill_at_max) >= 0.03
    assert edge_after_costs(0.48, fill_at_max) == pytest.approx(0.03)

    fill_above_max = estimate_fill(book, BookSide.BUY, size=size + 1.0, fee_rate=0.0)
    assert edge_after_costs(0.48, fill_above_max) < 0.03


def test_max_size_for_edge_with_fees_and_boundaries():
    book = make_book(asks=[(0.40, 100.0), (0.45, 100.0)])

    # Even best ask (0.40 * 1.05 = 0.42) leaves edge 0.02 < min_edge 0.03.
    assert max_size_for_edge(book, fair_price=0.44, min_edge=0.03, fee_rate=0.05) == 0.0

    # Entire book has edge >= min_edge.
    assert max_size_for_edge(book, fair_price=0.60, min_edge=0.05, fee_rate=0.01) == pytest.approx(
        200.0
    )

    # Empty book returns 0.
    assert max_size_for_edge(make_book(), fair_price=0.60, min_edge=0.01, fee_rate=0.0) == 0.0


def test_costs_invalid_inputs():
    book = make_book(asks=[(0.40, 10.0)])
    with pytest.raises(ValueError):
        estimate_fill(book, BookSide.BUY, size=-1.0, fee_rate=0.0)
    with pytest.raises(ValueError):
        estimate_fill(book, BookSide.BUY, size=1.0, fee_rate=-0.01)
    with pytest.raises(ValueError):
        max_size_for_edge(book, fair_price=0.5, min_edge=0.01, fee_rate=-0.01)
