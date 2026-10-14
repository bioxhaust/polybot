from datetime import UTC, datetime

import pytest

from polybot.models import (
    Asset,
    BookSide,
    CryptoMarket,
    FairProbability,
    MarketKind,
    Merge,
    Order,
    OrderBook,
    OrderType,
    Outcome,
    PriceLevel,
)

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def make_market(**overrides) -> CryptoMarket:
    fields = {
        "market_id": "0xabc",
        "event_id": "evt1",
        "question": "Will ETH be above $4,400 on October 17?",
        "asset": Asset.ETH,
        "kind": MarketKind.ABOVE,
        "strike_low": 4400.0,
        "strike_high": None,
        "expiry": datetime(2026, 10, 17, 16, 0, tzinfo=UTC),
        "yes_token_id": "yes",
        "no_token_id": "no",
        "resolution_source": "Binance ETHUSDT 1m close",
        "neg_risk": False,
        "volume_usd": 50_000.0,
    }
    fields.update(overrides)
    return CryptoMarket(**fields)


def test_market_valid():
    m = make_market()
    assert m.token_id(Outcome.NO) == "no"


@pytest.mark.parametrize(
    "overrides",
    [
        {"strike_low": None},
        {"kind": MarketKind.BELOW, "strike_low": None},
        {"kind": MarketKind.RANGE, "strike_high": None},
        {"kind": MarketKind.RANGE, "strike_low": 4500.0, "strike_high": 4400.0},
        {"expiry": datetime(2026, 10, 17, 16, 0)},
    ],
)
def test_market_invalid(overrides):
    with pytest.raises(ValueError):
        make_market(**overrides)


def test_order_book_properties():
    book = OrderBook(
        token_id="yes",
        bids=(PriceLevel(0.40, 100), PriceLevel(0.39, 200)),
        asks=(PriceLevel(0.42, 50), PriceLevel(0.45, 300)),
        timestamp=NOW,
    )
    assert book.best_bid == 0.40
    assert book.best_ask == 0.42
    assert book.spread == pytest.approx(0.02)
    assert book.mid == pytest.approx(0.41)


def test_order_book_rejects_unsorted():
    with pytest.raises(ValueError):
        OrderBook("yes", bids=(PriceLevel(0.39, 1), PriceLevel(0.40, 1)), asks=(), timestamp=NOW)


def test_empty_book():
    book = OrderBook("yes", bids=(), asks=(), timestamp=NOW)
    assert book.mid is None and book.spread is None


def test_price_bounds():
    with pytest.raises(ValueError):
        PriceLevel(1.2, 1)
    with pytest.raises(ValueError):
        FairProbability("m", p_yes=-0.1, computed_at=NOW, model="x", spot=1.0)


def test_fair_probability_outcomes():
    fair = FairProbability("m", p_yes=0.7, computed_at=NOW, model="x", spot=4500.0)
    assert fair.p(Outcome.NO) == pytest.approx(0.3)
    assert Outcome.YES.opposite is Outcome.NO


def test_limit_order_needs_price():
    with pytest.raises(ValueError):
        Order("o1", "m", "yes", Outcome.YES, BookSide.BUY, OrderType.LIMIT, 10, None, NOW)


def test_merge_pnl():
    merge = Merge(("m",), size=100, cost_basis_usd=98.0, proceeds_usd=100.0, timestamp=NOW)
    assert merge.pnl_usd == pytest.approx(2.0)
