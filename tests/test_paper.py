from datetime import UTC, datetime

import pytest

from polybot.execution.complete_sets import find_binary_sets
from polybot.execution.paper import PaperBroker
from polybot.models import (
    Asset,
    BookSide,
    CryptoMarket,
    MarketKind,
    MarketOutcome,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Outcome,
    PriceLevel,
)

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
EXPIRY = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)


def make_book(
    token_id: str = "yes_tok",
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        bids=tuple(PriceLevel(p, s) for p, s in (bids or [])),
        asks=tuple(PriceLevel(p, s) for p, s in (asks or [])),
        timestamp=NOW,
    )


def make_market_order(
    order_id: str = "ord_1",
    market_id: str = "mkt_1",
    token_id: str = "yes_tok",
    outcome: Outcome = Outcome.YES,
    side: BookSide = BookSide.BUY,
    size: float = 100.0,
) -> Order:
    return Order(
        order_id=order_id,
        market_id=market_id,
        token_id=token_id,
        outcome=outcome,
        side=side,
        order_type=OrderType.MARKET,
        size=size,
        limit_price=None,
        created_at=NOW,
    )


def test_submit_market_multi_level_buy():
    broker = PaperBroker(fee_rate=0.01)
    book = make_book("yes_tok", asks=[(0.40, 60.0), (0.45, 80.0)])
    order = make_market_order(size=100.0)

    fills = broker.submit_market(order, book)
    assert len(fills) == 2
    assert fills[0].price == pytest.approx(0.40)
    assert fills[0].size == pytest.approx(60.0)
    assert fills[0].fee_usd == pytest.approx(60.0 * 0.40 * 0.01)
    assert fills[0].is_maker is False
    assert fills[1].price == pytest.approx(0.45)
    assert fills[1].size == pytest.approx(40.0)

    expected_notional = 60.0 * 0.40 + 40.0 * 0.45  # 42.0
    assert order.status is OrderStatus.FILLED
    assert order.filled_size == pytest.approx(100.0)
    assert order.remaining == pytest.approx(0.0)
    assert order.avg_fill_price == pytest.approx(0.42)

    pos = broker.positions[("mkt_1", Outcome.YES)]
    assert pos.size == pytest.approx(100.0)
    assert pos.avg_price == pytest.approx(0.42)
    assert broker.cash_usd == pytest.approx(-expected_notional * 1.01)


def test_submit_market_partial_fill_then_complete():
    broker = PaperBroker(fee_rate=0.0)
    order = make_market_order(size=100.0)

    fills_1 = broker.submit_market(order, make_book("yes_tok", asks=[(0.40, 40.0)]))
    assert len(fills_1) == 1
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.filled_size == pytest.approx(40.0)
    assert order.remaining == pytest.approx(60.0)
    assert order.avg_fill_price == pytest.approx(0.40)

    fills_2 = broker.submit_market(order, make_book("yes_tok", asks=[(0.50, 100.0)]))
    assert len(fills_2) == 1
    assert fills_2[0].size == pytest.approx(60.0)
    assert order.status is OrderStatus.FILLED
    assert order.filled_size == pytest.approx(100.0)
    assert order.avg_fill_price == pytest.approx(0.46)


def test_submit_market_empty_book():
    broker = PaperBroker(fee_rate=0.0)
    order = make_market_order(size=50.0)
    fills = broker.submit_market(order, make_book("yes_tok"))

    assert fills == []
    assert order.status is OrderStatus.OPEN
    assert order.filled_size == 0.0
    assert order.avg_fill_price is None
    assert broker.cash_usd == 0.0
    assert broker.positions == {}


def test_submit_market_buy_and_sell_position():
    broker = PaperBroker(fee_rate=0.01)
    buy_order = make_market_order(order_id="b1", size=100.0)
    broker.submit_market(buy_order, make_book("yes_tok", asks=[(0.40, 100.0)]))

    sell_half = make_market_order(order_id="s1", side=BookSide.SELL, size=40.0)
    broker.submit_market(sell_half, make_book("yes_tok", bids=[(0.55, 100.0)]))

    pos = broker.positions[("mkt_1", Outcome.YES)]
    assert pos.size == pytest.approx(60.0)
    assert pos.avg_price == pytest.approx(0.40)

    sell_rest = make_market_order(order_id="s2", side=BookSide.SELL, size=60.0)
    broker.submit_market(sell_rest, make_book("yes_tok", bids=[(0.50, 100.0)]))

    assert pos.size == 0.0
    assert pos.avg_price == 0.0
    assert pos.realized_pnl_usd == pytest.approx(broker.cash_usd)


def test_merge_and_complete_set_roundtrip():
    market = CryptoMarket(
        market_id="mkt_1",
        event_id="evt_1",
        question="Will ETH be above $4,400?",
        asset=Asset.ETH,
        kind=MarketKind.ABOVE,
        strike_low=4400.0,
        strike_high=None,
        expiry=EXPIRY,
        yes_token_id="yes_tok",
        no_token_id="no_tok",
        resolution_source="Binance ETHUSDT",
        neg_risk=False,
        volume_usd=50_000.0,
    )
    books = {
        "yes_tok": make_book("yes_tok", asks=[(0.44, 100.0)]),
        "no_tok": make_book("no_tok", asks=[(0.50, 80.0)]),
    }
    fee_rate = 0.01
    opps = find_binary_sets([market], books, fee_rate=fee_rate, now=NOW)
    assert len(opps) == 1
    opp = opps[0]

    broker = PaperBroker(fee_rate=fee_rate)
    broker.submit_market(
        make_market_order("o_yes", "mkt_1", "yes_tok", Outcome.YES, BookSide.BUY, opp.size),
        books["yes_tok"],
    )
    broker.submit_market(
        make_market_order("o_no", "mkt_1", "no_tok", Outcome.NO, BookSide.BUY, opp.size),
        books["no_tok"],
    )

    merge = broker.merge("mkt_1", opp.size, NOW)
    assert merge.market_ids == ("mkt_1",)
    assert merge.size == pytest.approx(80.0)
    assert merge.cost_basis_usd == pytest.approx(opp.total_cost_usd)
    assert merge.proceeds_usd == pytest.approx(opp.payout_usd)
    assert merge.pnl_usd == pytest.approx(opp.profit_usd)
    assert broker.cash_usd == pytest.approx(opp.profit_usd)
    assert broker.positions[("mkt_1", Outcome.YES)].size == 0.0
    assert broker.positions[("mkt_1", Outcome.NO)].size == 0.0


def test_settle_winner_and_loser():
    broker = PaperBroker(fee_rate=0.01)
    broker.submit_market(
        make_market_order("o_yes", "mkt_1", "yes_tok", Outcome.YES, BookSide.BUY, 100.0),
        make_book("yes_tok", asks=[(0.40, 100.0)]),
    )
    broker.submit_market(
        make_market_order("o_no", "mkt_1", "no_tok", Outcome.NO, BookSide.BUY, 50.0),
        make_book("no_tok", asks=[(0.50, 50.0)]),
    )

    pnl = broker.settle(MarketOutcome(market_id="mkt_1", resolved_yes=True, resolved_at=NOW))
    expected_yes_pnl = 100.0 - 100.0 * 0.40 * 1.01
    expected_no_pnl = 0.0 - 50.0 * 0.50 * 1.01
    assert pnl == pytest.approx(expected_yes_pnl + expected_no_pnl)
    assert broker.cash_usd == pytest.approx(expected_yes_pnl + expected_no_pnl)
    assert broker.positions[("mkt_1", Outcome.YES)].size == 0.0
    assert broker.positions[("mkt_1", Outcome.NO)].size == 0.0

    # Settling again when flat returns 0.0.
    assert broker.settle(MarketOutcome(market_id="mkt_1", resolved_yes=True, resolved_at=NOW)) == 0.0


def test_paper_broker_validation_errors():
    broker = PaperBroker(fee_rate=0.0)
    book = make_book("yes_tok", asks=[(0.40, 100.0)], bids=[(0.38, 100.0)])

    # Selling without holding position:
    with pytest.raises(ValueError):
        broker.submit_market(make_market_order(side=BookSide.SELL, size=10.0), book)

    # Wrong token_id:
    with pytest.raises(ValueError):
        broker.submit_market(make_market_order(token_id="other_tok", size=10.0), book)

    # Limit order passed to submit_market:
    limit_order = Order(
        order_id="lim_1",
        market_id="mkt_1",
        token_id="yes_tok",
        outcome=Outcome.YES,
        side=BookSide.BUY,
        order_type=OrderType.LIMIT,
        size=10.0,
        limit_price=0.40,
        created_at=NOW,
    )
    with pytest.raises(ValueError):
        broker.submit_market(limit_order, book)

    # Merging without shares:
    with pytest.raises(ValueError):
        broker.merge("mkt_1", 10.0, NOW)
