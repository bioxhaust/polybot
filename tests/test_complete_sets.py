from datetime import UTC, datetime

import pytest

from polybot.execution.complete_sets import find_binary_sets, find_range_sets
from polybot.models import Asset, CryptoMarket, MarketKind, OrderBook, PriceLevel

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
EXPIRY = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)


def make_market(**overrides) -> CryptoMarket:
    fields = {
        "market_id": "mkt_1",
        "event_id": "evt_1",
        "question": "Will BTC be above $60,000?",
        "asset": Asset.BTC,
        "kind": MarketKind.ABOVE,
        "strike_low": 60_000.0,
        "strike_high": None,
        "expiry": EXPIRY,
        "yes_token_id": "yes_1",
        "no_token_id": "no_1",
        "resolution_source": "Binance BTCUSDT 1m close",
        "neg_risk": False,
        "volume_usd": 100_000.0,
    }
    fields.update(overrides)
    return CryptoMarket(**fields)


def make_book(token_id: str, asks: list[tuple[float, float]]) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        bids=(),
        asks=tuple(PriceLevel(p, s) for p, s in asks),
        timestamp=NOW,
    )


def test_find_binary_sets_single_and_multi_level():
    m = make_market()
    books = {
        "yes_1": make_book("yes_1", [(0.45, 50.0), (0.48, 50.0), (0.55, 100.0)]),
        "no_1": make_book("no_1", [(0.48, 80.0), (0.50, 100.0)]),
    }
    # Slice 1: 50 @ (0.45 + 0.48 = 0.93)
    # Slice 2: 30 @ (0.48 + 0.48 = 0.96)
    # Slice 3: 20 @ (0.48 + 0.50 = 0.98)
    # Next level has 0.55 + 0.50 = 1.05 >= 1.0 -> stop at size = 100.
    opps = find_binary_sets([m], books, fee_rate=0.0, now=NOW)
    assert len(opps) == 1
    opp = opps[0]
    assert opp.market_ids == ("mkt_1",)
    assert opp.token_ids == ("yes_1", "no_1")
    assert opp.size == pytest.approx(100.0)
    expected_cost = (50.0 * 0.45 + 50.0 * 0.48) + (80.0 * 0.48 + 20.0 * 0.50)
    assert opp.total_cost_usd == pytest.approx(expected_cost)
    assert opp.payout_usd == pytest.approx(100.0)
    assert opp.profit_usd == pytest.approx(100.0 - expected_cost)
    assert opp.found_at == NOW


def test_find_binary_sets_unprofitable_or_killed_by_fees():
    m = make_market()
    books = {
        "yes_1": make_book("yes_1", [(0.49, 100.0)]),
        "no_1": make_book("no_1", [(0.50, 100.0)]),
    }
    # Profitable without fees (0.99 < 1.0):
    assert len(find_binary_sets([m], books, fee_rate=0.0, now=NOW)) == 1
    # Unprofitable with 2% taker fee (0.99 * 1.02 = 1.0098 > 1.0):
    assert find_binary_sets([m], books, fee_rate=0.02, now=NOW) == []


def test_find_binary_sets_missing_book_or_expired():
    m = make_market()
    books = {"yes_1": make_book("yes_1", [(0.40, 100.0)])}
    assert find_binary_sets([m], books, fee_rate=0.0, now=NOW) == []

    expired = make_market(expiry=NOW)
    both_books = {
        "yes_1": make_book("yes_1", [(0.40, 100.0)]),
        "no_1": make_book("no_1", [(0.40, 100.0)]),
    }
    assert find_binary_sets([expired], both_books, fee_rate=0.0, now=NOW) == []


def make_range_event() -> tuple[list[CryptoMarket], dict[str, OrderBook]]:
    m_below = make_market(
        market_id="m_below",
        event_id="evt_btc",
        kind=MarketKind.BELOW,
        strike_low=None,
        strike_high=60_000.0,
        yes_token_id="t_below",
        no_token_id="t_below_no",
        neg_risk=True,
    )
    m_r1 = make_market(
        market_id="m_r1",
        event_id="evt_btc",
        kind=MarketKind.RANGE,
        strike_low=60_000.0,
        strike_high=65_000.0,
        yes_token_id="t_r1",
        no_token_id="t_r1_no",
        neg_risk=True,
    )
    m_r2 = make_market(
        market_id="m_r2",
        event_id="evt_btc",
        kind=MarketKind.RANGE,
        strike_low=65_000.0,
        strike_high=70_000.0,
        yes_token_id="t_r2",
        no_token_id="t_r2_no",
        neg_risk=True,
    )
    m_above = make_market(
        market_id="m_above",
        event_id="evt_btc",
        kind=MarketKind.ABOVE,
        strike_low=70_000.0,
        strike_high=None,
        yes_token_id="t_above",
        no_token_id="t_above_no",
        neg_risk=True,
    )
    books = {
        "t_below": make_book("t_below", [(0.15, 100.0)]),
        "t_r1": make_book("t_r1", [(0.30, 80.0)]),
        "t_r2": make_book("t_r2", [(0.30, 120.0)]),
        "t_above": make_book("t_above", [(0.20, 90.0)]),
    }
    # Pass shuffled to ensure strike sorting is deterministic.
    return [m_r2, m_above, m_below, m_r1], books


def test_find_range_sets_valid_event():
    markets, books = make_range_event()
    opps = find_range_sets(markets, books, fee_rate=0.0, now=NOW)

    assert len(opps) == 1
    opp = opps[0]
    assert opp.market_ids == ("m_below", "m_r1", "m_r2", "m_above")
    assert opp.token_ids == ("t_below", "t_r1", "t_r2", "t_above")
    assert opp.size == pytest.approx(80.0)
    assert opp.total_cost_usd == pytest.approx(80.0 * (0.15 + 0.30 + 0.30 + 0.20))
    assert opp.payout_usd == pytest.approx(80.0)
    assert opp.profit_usd == pytest.approx(80.0 * 0.05)


def test_find_range_sets_rejects_non_neg_risk():
    markets, books = make_range_event()
    markets[0] = make_market(
        market_id=markets[0].market_id,
        event_id=markets[0].event_id,
        kind=markets[0].kind,
        strike_low=markets[0].strike_low,
        strike_high=markets[0].strike_high,
        yes_token_id=markets[0].yes_token_id,
        no_token_id=markets[0].no_token_id,
        neg_risk=False,
    )
    assert find_range_sets(markets, books, fee_rate=0.0, now=NOW) == []


def test_find_range_sets_rejects_missing_tail_or_gap_or_overlap():
    markets, books = make_range_event()

    # Missing BELOW tail:
    no_below = [m for m in markets if m.kind is not MarketKind.BELOW]
    assert find_range_sets(no_below, books, fee_rate=0.0, now=NOW) == []

    # Missing ABOVE tail:
    no_above = [m for m in markets if m.kind is not MarketKind.ABOVE]
    assert find_range_sets(no_above, books, fee_rate=0.0, now=NOW) == []

    # Gap between 64,000 and 65,000:
    gapped = [
        m
        if m.market_id != "m_r1"
        else make_market(
            market_id="m_r1",
            event_id="evt_btc",
            kind=MarketKind.RANGE,
            strike_low=60_000.0,
            strike_high=64_000.0,
            yes_token_id="t_r1",
            no_token_id="t_r1_no",
            neg_risk=True,
        )
        for m in markets
    ]
    assert find_range_sets(gapped, books, fee_rate=0.0, now=NOW) == []

    # Overlap between [60,000, 66,000) and [65,000, 70,000):
    overlapped = [
        m
        if m.market_id != "m_r1"
        else make_market(
            market_id="m_r1",
            event_id="evt_btc",
            kind=MarketKind.RANGE,
            strike_low=60_000.0,
            strike_high=66_000.0,
            yes_token_id="t_r1",
            no_token_id="t_r1_no",
            neg_risk=True,
        )
        for m in markets
    ]
    assert find_range_sets(overlapped, books, fee_rate=0.0, now=NOW) == []


def test_find_range_sets_rejects_mismatched_expiry_or_source():
    markets, books = make_range_event()
    wrong_expiry = [
        m
        if m.market_id != "m_above"
        else make_market(
            market_id="m_above",
            event_id="evt_btc",
            kind=MarketKind.ABOVE,
            strike_low=70_000.0,
            strike_high=None,
            expiry=datetime(2026, 10, 18, 16, 0, tzinfo=UTC),
            yes_token_id="t_above",
            no_token_id="t_above_no",
            neg_risk=True,
        )
        for m in markets
    ]
    assert find_range_sets(wrong_expiry, books, fee_rate=0.0, now=NOW) == []

    wrong_source = [
        m
        if m.market_id != "m_above"
        else make_market(
            market_id="m_above",
            event_id="evt_btc",
            kind=MarketKind.ABOVE,
            strike_low=70_000.0,
            strike_high=None,
            resolution_source="Coinbase BTCUSD",
            yes_token_id="t_above",
            no_token_id="t_above_no",
            neg_risk=True,
        )
        for m in markets
    ]
    assert find_range_sets(wrong_source, books, fee_rate=0.0, now=NOW) == []
