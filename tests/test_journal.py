from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from polybot.models import (
    Asset,
    BookSide,
    CompleteSetOpportunity,
    CryptoMarket,
    FairProbability,
    Fill,
    FillEstimate,
    MarketKind,
    MarketOutcome,
    Merge,
    OptionQuote,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Outcome,
    Position,
    PriceLevel,
    Signal,
    WalletTrade,
)
from polybot.storage.journal import SCHEMA_VERSION, Journal

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
LATER = datetime(2026, 10, 14, 12, 0, 30, tzinfo=UTC)
EXPIRY = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)
EXPIRY_PLUS3 = EXPIRY.astimezone(timezone(timedelta(hours=3)))
DERIBIT_EXPIRY = datetime(2026, 10, 17, 8, 0, tzinfo=UTC)

_MODEL_TABLES = {
    "markets",
    "book_snapshots",
    "book_levels",
    "fair_probabilities",
    "signals",
    "orders",
    "fills",
    "merges",
    "complete_sets",
    "outcomes",
    "wallet_trades",
    "option_quotes",
    "fill_estimates",
    "positions",
}


@pytest.fixture
def journal(tmp_path: Path) -> Iterator[Journal]:
    db = Journal(tmp_path / "journal.sqlite")
    db.migrate()
    try:
        yield db
    finally:
        db.close()


def _market(
    market_id: str,
    *,
    kind: MarketKind = MarketKind.ABOVE,
    strike_low: float | None = 4400.0,
    strike_high: float | None = None,
    neg_risk: bool = False,
    asset: Asset = Asset.ETH,
    expiry: datetime = EXPIRY,
    volume_usd: float = 50_000.0,
    question: str | None = None,
) -> CryptoMarket:
    return CryptoMarket(
        market_id=market_id,
        event_id=f"evt-{market_id}",
        question=question or f"Will {asset.value} finish beyond the strike? ({market_id})",
        asset=asset,
        kind=kind,
        strike_low=strike_low,
        strike_high=strike_high,
        expiry=expiry,
        yes_token_id=f"{market_id}-yes",
        no_token_id=f"{market_id}-no",
        resolution_source="Binance ETHUSDT 1m close",
        neg_risk=neg_risk,
        volume_usd=volume_usd,
    )


def _book(
    token_id: str,
    timestamp: datetime,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        bids=tuple(PriceLevel(price, size) for price, size in bids),
        asks=tuple(PriceLevel(price, size) for price, size in asks),
        timestamp=timestamp,
    )


def _signal(
    *,
    created_at: datetime = NOW,
    reason: str = "model_edge",
    market_id: str = "0xabove",
    outcome: Outcome = Outcome.YES,
    stop_price: float | None = None,
    take_profit_price: float | None = 0.62,
) -> Signal:
    return Signal(
        market_id=market_id,
        outcome=outcome,
        fair_price=0.62,
        entry_price=0.55,
        edge_after_costs=0.04,
        size=25.0,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        created_at=created_at,
        reason=reason,
    )


def _trade(
    wallet: str,
    tx_hash: str,
    *,
    price: float = 0.42,
    size: float = 15.0,
    timestamp: datetime = NOW,
    market_id: str = "0xabove",
) -> WalletTrade:
    return WalletTrade(
        wallet=wallet,
        market_id=market_id,
        outcome=Outcome.NO,
        side=BookSide.SELL,
        price=price,
        size=size,
        timestamp=timestamp,
        tx_hash=tx_hash,
    )


def _order(
    order_id: str,
    order_type: OrderType,
    *,
    status: OrderStatus,
    filled_size: float,
    avg_fill_price: float | None,
    strategy: str,
) -> Order:
    limit_price = None if order_type is OrderType.MARKET else 0.41
    return Order(
        order_id=order_id,
        market_id="0xabove",
        token_id="0xabove-yes",
        outcome=Outcome.YES,
        side=BookSide.BUY,
        order_type=order_type,
        size=10.0,
        limit_price=limit_price,
        created_at=NOW,
        status=status,
        filled_size=filled_size,
        avg_fill_price=avg_fill_price,
        strategy=strategy,
    )


@dataclass
class _Payload:
    markets: list[CryptoMarket]
    books: list[tuple[str, OrderBook]]
    fairs: list[FairProbability]
    signals: list[Signal]
    orders: list[Order]
    fills: list[Fill]
    merges: list[Merge]
    complete_sets: list[CompleteSetOpportunity]
    outcomes: list[MarketOutcome]
    trades: list[WalletTrade]
    quotes: list[OptionQuote]
    estimates: list[FillEstimate]
    positions: list[Position]


def _payload() -> _Payload:
    assert EXPIRY_PLUS3.utcoffset() == timedelta(hours=3)
    markets = [
        _market("0xabove"),
        _market(
            "0xbelow",
            kind=MarketKind.BELOW,
            strike_low=None,
            strike_high=4300.0,
            asset=Asset.BTC,
            expiry=EXPIRY_PLUS3,
            volume_usd=0.0,
        ),
        _market(
            "0xrange",
            kind=MarketKind.RANGE,
            strike_low=4200.0,
            strike_high=4600.0,
            neg_risk=True,
            question="ETH price ≥ 4200 and < 4600?",
        ),
    ]
    books = [
        (
            "0xabove",
            _book(
                "0xabove-yes",
                NOW,
                bids=[(0.40, 5.0), (0.33, 9.0), (0.0, 1.0)],
                asks=[(0.42, 4.0), (0.48, 8.0), (1.0, 2.5)],
            ),
        ),
        ("0xabove", _book("0xabove-no", NOW, bids=[], asks=[])),
    ]
    fairs = [
        FairProbability(
            market_id="0xabove",
            p_yes=0.0,
            computed_at=NOW,
            model="bsm_smile_v1",
            spot=4412.5,
            meta={},
        ),
        FairProbability(
            market_id="0xabove",
            p_yes=0.62,
            computed_at=LATER,
            model="bsm_smile_v1",
            spot=4412.5,
            meta={"basis_usd": 1.25, "iv": 0.1, "tte_years": 0.008},
        ),
        FairProbability(
            market_id="0xbelow",
            p_yes=1.0,
            computed_at=LATER,
            model="bsm_smile_v1",
            spot=60000.0,
            meta={"iv": 0.5},
        ),
    ]
    signals = [
        _signal(),
        _signal(
            created_at=LATER,
            reason="complete_set",
            market_id="0xrange",
            outcome=Outcome.NO,
            stop_price=0.3,
            take_profit_price=0.7,
        ),
    ]
    orders = [
        _order(
            "ord-market",
            OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_size=10.0,
            avg_fill_price=0.44,
            strategy="signal",
        ),
        _order(
            "ord-partial",
            OrderType.LIMIT,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_size=4.0,
            avg_fill_price=0.41,
            strategy="maker",
        ),
        _order(
            "ord-cancelled",
            OrderType.LIMIT,
            status=OrderStatus.CANCELLED,
            filled_size=0.0,
            avg_fill_price=None,
            strategy="maker",
        ),
    ]
    fills = [
        Fill(
            order_id="ord-market",
            price=0.44,
            size=10.0,
            fee_usd=0.0,
            is_maker=False,
            timestamp=NOW,
        ),
        Fill(
            order_id="ord-partial",
            price=0.41,
            size=4.0,
            fee_usd=0.02,
            is_maker=True,
            timestamp=LATER,
        ),
    ]
    merges = [
        Merge(
            market_ids=("0xabove",),
            size=5.0,
            cost_basis_usd=4.8,
            proceeds_usd=5.0,
            timestamp=NOW,
        ),
        Merge(
            market_ids=("0xrange-a", "0xrange-b"),
            size=2.0,
            cost_basis_usd=1.9,
            proceeds_usd=2.0,
            timestamp=LATER,
        ),
    ]
    complete_sets = [
        CompleteSetOpportunity(
            market_ids=("0xabove",),
            token_ids=("0xabove-yes", "0xabove-no"),
            size=5.0,
            total_cost_usd=4.85,
            payout_usd=5.0,
            found_at=NOW,
        ),
        CompleteSetOpportunity(
            market_ids=("0xrange-a", "0xrange-b"),
            token_ids=("0xrange-a-yes", "0xrange-b-yes"),
            size=2.0,
            total_cost_usd=1.92,
            payout_usd=2.0,
            found_at=LATER,
        ),
    ]
    outcomes = [
        MarketOutcome("0xabove", True, LATER),
        MarketOutcome("0xbelow", False, NOW),
    ]
    trades = [
        _trade("0xwallet", "0xtx1", price=0.01),
        _trade("0xother", "0xtx2", price=0.99, size=1.5, timestamp=LATER),
    ]
    quotes = [
        OptionQuote(
            instrument="ETH-17OCT26-4400-C",
            asset=Asset.ETH,
            strike=4400.0,
            expiry=DERIBIT_EXPIRY,
            is_call=True,
            mark_iv=0.55,
            bid_iv=None,
            ask_iv=None,
            underlying_price=4410.0,
            timestamp=NOW,
        ),
        OptionQuote(
            instrument="ETH-17OCT26-4400-C",
            asset=Asset.ETH,
            strike=4400.0,
            expiry=DERIBIT_EXPIRY,
            is_call=True,
            mark_iv=0.57,
            bid_iv=0.56,
            ask_iv=0.58,
            underlying_price=4420.0,
            timestamp=LATER,
        ),
        OptionQuote(
            instrument="BTC-17OCT26-60000-P",
            asset=Asset.BTC,
            strike=60000.0,
            expiry=DERIBIT_EXPIRY,
            is_call=False,
            mark_iv=0.48,
            bid_iv=0.46,
            ask_iv=0.50,
            underlying_price=60500.0,
            timestamp=LATER,
        ),
    ]
    estimates = [
        FillEstimate(
            token_id="0xabove-yes",
            side=BookSide.BUY,
            size_requested=10.0,
            size_filled=10.0,
            avg_price=0.44,
            worst_price=0.44,
            slippage=0.0,
            fee_usd=0.0,
        ),
        FillEstimate(
            token_id="0xabove-no",
            side=BookSide.SELL,
            size_requested=8.0,
            size_filled=5.0,
            avg_price=0.6,
            worst_price=0.55,
            slippage=0.02,
            fee_usd=0.01,
        ),
    ]
    positions = [
        Position("0xabove", Outcome.YES, size=10.0, avg_price=0.44, realized_pnl_usd=0.0),
        Position("0xabove", Outcome.NO, size=0.0, avg_price=0.0, realized_pnl_usd=1.25),
    ]
    return _Payload(
        markets=markets,
        books=books,
        fairs=fairs,
        signals=signals,
        orders=orders,
        fills=fills,
        merges=merges,
        complete_sets=complete_sets,
        outcomes=outcomes,
        trades=trades,
        quotes=quotes,
        estimates=estimates,
        positions=positions,
    )


def _write(journal: Journal, payload: _Payload) -> None:
    for market in payload.markets:
        journal.save_market(market)
    for market_id, book in payload.books:
        journal.save_book(market_id, book)
    for fair in payload.fairs:
        journal.save_fair_probability(fair)
    for signal in payload.signals:
        journal.save_signal(signal)
    for order in payload.orders:
        journal.save_order(order)
    for fill in payload.fills:
        journal.save_fill(fill)
    for merge in payload.merges:
        journal.save_merge(merge)
    for opp in payload.complete_sets:
        journal.save_complete_set(opp)
    for outcome in payload.outcomes:
        journal.save_outcome(outcome)
    journal.save_wallet_trades(payload.trades)
    for quote in payload.quotes:
        journal.save_option_quote(quote)
    for estimate in payload.estimates:
        journal.save_fill_estimate(estimate)
    for position in payload.positions:
        journal.save_position(position)


def _assert_read_equals(journal: Journal, payload: _Payload) -> None:
    assert journal.markets() == payload.markets
    assert journal.book_snapshots() == payload.books
    assert journal.fair_probabilities() == payload.fairs
    assert journal.signals() == payload.signals
    assert journal.orders() == payload.orders
    assert journal.fills() == payload.fills
    assert journal.merges() == payload.merges
    assert journal.complete_sets() == payload.complete_sets
    assert journal.outcomes() == payload.outcomes
    assert journal.wallet_trades() == payload.trades
    assert journal.option_quotes() == payload.quotes
    assert journal.fill_estimates() == payload.estimates
    assert journal.positions() == payload.positions
    for market in journal.markets():
        assert market.expiry.utcoffset() == timedelta(0)
    for _, book in journal.book_snapshots():
        assert book.timestamp.utcoffset() == timedelta(0)
    for fair in journal.fair_probabilities():
        assert fair.computed_at.utcoffset() == timedelta(0)


def test_migrate_is_idempotent_and_versioned(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "polybot.sqlite"
    journal = Journal(path)
    try:
        assert journal.schema_version() == 0
        journal.migrate()
        assert path.is_file()
        assert journal.schema_version() == SCHEMA_VERSION
        journal.migrate()
        assert journal.schema_version() == SCHEMA_VERSION
    finally:
        journal.close()

    with sqlite3.connect(path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()
        assert version is not None
        assert version[0] == SCHEMA_VERSION


def test_migration_upgrades_previous_version(tmp_path: Path) -> None:
    path = tmp_path / "journal.sqlite"
    market = _market("0xkeep")
    journal = Journal(path)
    journal.migrate()
    journal.save_market(market)
    journal.close()

    with sqlite3.connect(path) as conn:
        conn.execute("DROP INDEX IF EXISTS idx_book_snapshots_market_ts")
        conn.execute("PRAGMA user_version = 1")

    upgraded = Journal(path)
    try:
        assert upgraded.schema_version() == 1
        upgraded.migrate()
        assert upgraded.schema_version() == SCHEMA_VERSION
        assert upgraded.markets() == [market]
    finally:
        upgraded.close()

    with sqlite3.connect(path) as conn:
        found = conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'index' AND name = 'idx_book_snapshots_market_ts'
            """
        ).fetchone()
        assert found is not None
        version = conn.execute("PRAGMA user_version").fetchone()
        assert version is not None
        assert version[0] == SCHEMA_VERSION


def test_schema_has_a_table_for_every_model(tmp_path: Path) -> None:
    path = tmp_path / "journal.sqlite"
    with Journal(path) as journal:
        journal.migrate()

    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    names = {row[0] for row in rows}
    assert _MODEL_TABLES <= names


def test_roundtrip_equals_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "data" / "polybot.sqlite"
    payload = _payload()
    with Journal(path) as journal:
        _write(journal, payload)
        assert journal.schema_version() == SCHEMA_VERSION
        _assert_read_equals(journal, payload)

    with Journal(path) as journal:
        _assert_read_equals(journal, payload)


def test_book_snapshots_are_kept_for_every_tick(journal: Journal) -> None:
    first = _book(
        "yes",
        NOW,
        bids=[(0.40, 5.0), (0.33, 9.0)],
        asks=[(0.42, 4.0), (0.48, 8.0)],
    )
    same_again = first
    later = _book("yes", LATER, bids=[(0.37, 6.0)], asks=[(0.39, 6.0)])
    empty = _book("no", NOW, bids=[], asks=[])

    journal.save_book("m1", first)
    journal.save_book("m1", same_again)
    journal.save_book("m1", empty)
    journal.save_book("m2", later)

    expected = [("m1", first), ("m1", same_again), ("m1", empty), ("m2", later)]
    assert journal.book_snapshots() == expected
    assert journal.book_snapshots(market_id="m1") == expected[:3]
    assert journal.book_snapshots(since=LATER) == [("m2", later)]
    assert journal.book_snapshots(market_id="m2", since=LATER) == [("m2", later)]
    assert journal.book_snapshots(market_id="m1", since=LATER) == []


def test_histories_are_not_overwritten(journal: Journal) -> None:
    early = FairProbability(
        market_id="0xabove",
        p_yes=0.4,
        computed_at=NOW,
        model="bsm_smile_v1",
        spot=100.0,
        meta={"iv": 0.2},
    )
    late = FairProbability(
        market_id="0xabove",
        p_yes=0.6,
        computed_at=LATER,
        model="bsm_smile_v1",
        spot=101.0,
        meta={"iv": 0.3},
    )
    journal.save_fair_probability(early)
    journal.save_fair_probability(late)
    assert journal.fair_probabilities() == [early, late]

    fill = Fill("ord-1", price=0.5, size=1.0, fee_usd=0.0, is_maker=False, timestamp=NOW)
    journal.save_fill(fill)
    journal.save_fill(fill)
    assert journal.fills() == [fill, fill]


def test_mutable_rows_upsert(journal: Journal) -> None:
    market = _market("0xabc")
    journal.save_market(market)
    revised = replace(market, volume_usd=123.0, question="revised question")
    journal.save_market(revised)
    assert journal.markets() == [revised]

    order = _order(
        "ord-1",
        OrderType.LIMIT,
        status=OrderStatus.OPEN,
        filled_size=0.0,
        avg_fill_price=None,
        strategy="maker",
    )
    journal.save_order(order)
    order.status = OrderStatus.FILLED
    order.filled_size = order.size
    order.avg_fill_price = 0.41
    journal.save_order(order)
    assert journal.orders() == [order]

    position = Position("0xabc", Outcome.YES, size=1.0, avg_price=0.4, realized_pnl_usd=0.0)
    journal.save_position(position)
    position.size = 2.0
    position.realized_pnl_usd = 1.5
    journal.save_position(position)
    assert journal.positions() == [position]


def test_unresolved_market_ids_drop_out_once_resolved(journal: Journal) -> None:
    journal.save_market(_market("0x1"))
    journal.save_market(_market("0x2"))
    assert journal.unresolved_market_ids() == ["0x1", "0x2"]

    journal.save_outcome(MarketOutcome("0x1", True, NOW))
    assert journal.unresolved_market_ids() == ["0x2"]
    assert journal.outcomes() == [MarketOutcome("0x1", True, NOW)]

    journal.save_outcome(MarketOutcome("0x2", False, LATER))
    journal.save_outcome(MarketOutcome("0x2", True, LATER))
    assert journal.unresolved_market_ids() == []
    assert journal.outcomes() == [
        MarketOutcome("0x1", True, NOW),
        MarketOutcome("0x2", True, LATER),
    ]


def test_since_filters_are_inclusive(journal: Journal) -> None:
    early_signal = _signal(created_at=NOW)
    late_signal = _signal(created_at=LATER, reason="late")
    journal.save_signal(early_signal)
    journal.save_signal(late_signal)

    same_instant = datetime(2026, 10, 14, 15, 0, tzinfo=timezone(timedelta(hours=3)))
    assert journal.signals() == [early_signal, late_signal]
    assert journal.signals(since=same_instant) == [early_signal, late_signal]
    assert journal.signals(since=LATER) == [late_signal]
    assert journal.signals(since=LATER + timedelta(microseconds=1)) == []

    early_fair = FairProbability("0xabove", 0.2, NOW, "bsm_smile_v1", 1.0, {"iv": 0.2})
    late_fair = FairProbability("0xabove", 0.3, LATER, "bsm_smile_v1", 1.0, {})
    journal.save_fair_probability(early_fair)
    journal.save_fair_probability(late_fair)
    assert journal.fair_probabilities(since=LATER) == [late_fair]
    assert journal.fair_probabilities(since=NOW) == [early_fair, late_fair]

    early_fill = Fill("ord-1", 0.4, 1.0, 0.0, False, NOW)
    late_fill = Fill("ord-1", 0.5, 2.0, 0.0, True, LATER)
    journal.save_fill(early_fill)
    journal.save_fill(late_fill)
    assert journal.fills(since=LATER) == [late_fill]
    assert journal.fills(since=same_instant) == [early_fill, late_fill]


def test_naive_since_is_rejected(journal: Journal) -> None:
    naive = datetime(2026, 10, 14, 12, 0)
    with pytest.raises(ValueError):
        journal.signals(since=naive)
    with pytest.raises(ValueError):
        journal.fills(since=naive)
    with pytest.raises(ValueError):
        journal.fair_probabilities(since=naive)
    with pytest.raises(ValueError):
        journal.book_snapshots(since=naive)


def test_wallet_trades_dedup_and_filter(journal: Journal) -> None:
    first = _trade("0xwallet", "0xtx1")
    different_size = replace(first, size=3.0)
    second = _trade("0xwallet", "0xtx2", timestamp=LATER)
    other = _trade("0xother", "0xtx3")
    same_instant_other_zone = replace(
        first,
        timestamp=NOW.astimezone(timezone(timedelta(hours=3))),
    )

    journal.save_wallet_trades([first, different_size])
    journal.save_wallet_trades([same_instant_other_zone, second, other])
    journal.save_wallet_trades([])

    assert journal.wallet_trades() == [first, different_size, second, other]
    assert journal.wallet_trades(wallet="0xwallet") == [first, different_size, second]
    assert journal.wallet_trades(wallet="0xmissing") == []


def test_empty_reads(journal: Journal) -> None:
    assert journal.markets() == []
    assert journal.book_snapshots() == []
    assert journal.fair_probabilities() == []
    assert journal.signals() == []
    assert journal.orders() == []
    assert journal.fills() == []
    assert journal.merges() == []
    assert journal.complete_sets() == []
    assert journal.outcomes() == []
    assert journal.wallet_trades() == []
    assert journal.option_quotes() == []
    assert journal.fill_estimates() == []
    assert journal.positions() == []
    assert journal.unresolved_market_ids() == []
