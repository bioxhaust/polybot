from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

import polybot.engine.loop as loop_mod
from polybot.config import Settings
from polybot.engine.risk import RiskManager
from polybot.models import (
    Asset,
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
    Outcome,
    Position,
    PriceLevel,
    Signal,
)

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
EXPIRY = datetime(2026, 10, 17, 8, 0, tzinfo=UTC)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "polybot.sqlite",
        assets=(Asset.ETH,),
        poll_seconds=5.0,
        bankroll_usd=10_000.0,
        risk_unit_pct=0.005,
        daily_loss_limit_r=3.0,
        weekly_loss_limit_r=7.0,
        theme_limit_r=3.0,
        max_share_of_volume=0.01,
        min_edge=0.03,
        taker_fee_rate=0.0,
    )


MARKET = CryptoMarket(
    market_id="m1",
    event_id="e1",
    question="Will ETH be above $4,400 on October 17?",
    asset=Asset.ETH,
    kind=MarketKind.ABOVE,
    strike_low=4400.0,
    strike_high=None,
    expiry=EXPIRY,
    yes_token_id="yes_m1",
    no_token_id="no_m1",
    resolution_source="Binance ETHUSDT 1m close",
    neg_risk=False,
    volume_usd=100_000.0,
)

EXPIRED_MARKET = CryptoMarket(
    market_id="m0",
    event_id="e0",
    question="Will ETH be above $4,300 on October 13?",
    asset=Asset.ETH,
    kind=MarketKind.ABOVE,
    strike_low=4300.0,
    strike_high=None,
    expiry=NOW - timedelta(days=1),
    yes_token_id="yes_m0",
    no_token_id="no_m0",
    resolution_source="Binance ETHUSDT 1m close",
    neg_risk=False,
    volume_usd=100_000.0,
)

BOOK_YES = OrderBook(
    token_id="yes_m1",
    bids=(PriceLevel(0.48, 200.0),),
    asks=(PriceLevel(0.50, 500.0),),
    timestamp=NOW,
)

BOOK_NO = OrderBook(
    token_id="no_m1",
    bids=(PriceLevel(0.48, 200.0),),
    asks=(PriceLevel(0.52, 500.0),),
    timestamp=NOW,
)

QUOTE = OptionQuote(
    instrument="ETH-17OCT26-4400-C",
    asset=Asset.ETH,
    strike=4400.0,
    expiry=EXPIRY,
    is_call=True,
    mark_iv=0.55,
    bid_iv=0.54,
    ask_iv=0.56,
    underlying_price=4500.0,
    timestamp=NOW,
)


class FakePolymarket:
    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self.outcome: MarketOutcome | None = None
        self.outcome_requests: list[str] = []

    def fetch_crypto_markets(self, assets: tuple[Asset, ...]) -> list[dict[str, Any]]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise httpx.ConnectError("transient network error")
        return [{"id": "m1"}]

    def fetch_order_books(self, token_ids: list[str]) -> dict[str, OrderBook]:
        return {"yes_m1": BOOK_YES, "no_m1": BOOK_NO}

    def fetch_outcome(self, market_id: str) -> MarketOutcome | None:
        self.outcome_requests.append(market_id)
        return self.outcome


class FakeDeribit:
    def __init__(self, basis: float = 0.0, chain_error: Exception | None = None) -> None:
        self.basis = basis
        self.chain_error = chain_error

    def fetch_option_chain(self, asset: Asset) -> list[OptionQuote]:
        if self.chain_error is not None:
            raise self.chain_error
        return [QUOTE]

    def fetch_resolution_basis(self, asset: Asset) -> float:
        return self.basis


class FakeBroker:
    def __init__(self) -> None:
        self.positions: dict[tuple[str, Outcome], Position] = {}
        self.submitted_orders: list[Order] = []
        self.merges: list[Merge] = []
        self.settled: list[MarketOutcome] = []

    def submit_market(self, order: Order, book: OrderBook) -> list[Fill]:
        self.submitted_orders.append(order)
        pos = self.positions.setdefault(
            (order.market_id, order.outcome),
            Position(market_id=order.market_id, outcome=order.outcome),
        )
        pos.size += order.size
        pos.avg_price = book.best_ask or 0.5
        return [
            Fill(
                order_id=order.order_id,
                price=book.best_ask or 0.5,
                size=order.size,
                fee_usd=0.0,
                is_maker=False,
                timestamp=NOW,
            )
        ]

    def merge(self, market_id: str, size: float, now: datetime) -> Merge:
        m = Merge(
            market_ids=(market_id,),
            size=size,
            cost_basis_usd=size * 0.96,
            proceeds_usd=size,
            timestamp=now,
        )
        self.merges.append(m)
        return m

    def settle(self, outcome: MarketOutcome) -> float:
        self.settled.append(outcome)
        return -50.0


class FakeJournal:
    def __init__(self) -> None:
        self.migrated = False
        self.saved_markets: list[CryptoMarket] = []
        self.saved_books: list[tuple[str, OrderBook]] = []
        self.saved_fairs: list[FairProbability] = []
        self.saved_signals: list[Signal] = []
        self.saved_orders: list[Order] = []
        self.saved_fills: list[Fill] = []
        self.saved_merges: list[Merge] = []
        self.saved_complete_sets: list[CompleteSetOpportunity] = []
        self.saved_outcomes: list[MarketOutcome] = []
        self.unresolved: list[str] = []

    def migrate(self) -> None:
        self.migrated = True

    def save_market(self, market: CryptoMarket) -> None:
        self.saved_markets.append(market)

    def save_book(self, market_id: str, book: OrderBook) -> None:
        self.saved_books.append((market_id, book))

    def save_fair_probability(self, fair: FairProbability) -> None:
        self.saved_fairs.append(fair)

    def save_signal(self, signal: Signal) -> None:
        self.saved_signals.append(signal)

    def save_order(self, order: Order) -> None:
        self.saved_orders.append(order)

    def save_fill(self, fill: Fill) -> None:
        self.saved_fills.append(fill)

    def save_merge(self, merge: Merge) -> None:
        self.saved_merges.append(merge)

    def save_complete_set(self, opp: CompleteSetOpportunity) -> None:
        self.saved_complete_sets.append(opp)

    def save_outcome(self, outcome: MarketOutcome) -> None:
        self.saved_outcomes.append(outcome)

    def unresolved_market_ids(self) -> list[str]:
        return list(self.unresolved)

    def markets(self) -> list[CryptoMarket]:
        return list(self.saved_markets)


def _fake_fair(market: CryptoMarket, chain: list[OptionQuote], now: datetime, **_: Any):
    return FairProbability(
        market_id=market.market_id,
        p_yes=0.60,
        computed_at=now,
        model="test_v1",
        spot=4500.0,
    )


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, *, include_complete_set: bool = False) -> None:
    monkeypatch.setattr(loop_mod, "parse_market", lambda raw: MARKET)
    monkeypatch.setattr(loop_mod, "fair_probability", _fake_fair)
    monkeypatch.setattr(
        loop_mod,
        "max_size_for_edge",
        lambda book, fair_p, min_edge, fee_rate: 300.0 if fair_p - (book.best_ask or 1.0) >= min_edge else 0.0,
    )
    monkeypatch.setattr(
        loop_mod,
        "estimate_fill",
        lambda book, side, size, fee_rate: FillEstimate(
            token_id=book.token_id,
            side=side,
            size_requested=size,
            size_filled=size,
            avg_price=book.best_ask or 0.5,
            worst_price=book.best_ask or 0.5,
            slippage=0.0,
            fee_usd=0.0,
        ),
    )
    monkeypatch.setattr(
        loop_mod,
        "edge_after_costs",
        lambda fair_p, fill: fair_p - fill.avg_price,
    )
    if include_complete_set:
        monkeypatch.setattr(
            loop_mod,
            "find_binary_sets",
            lambda markets, books, fee_rate, now: [
                CompleteSetOpportunity(
                    market_ids=("m1",),
                    token_ids=("yes_m1", "no_m1"),
                    size=25.0,
                    total_cost_usd=24.0,
                    payout_usd=25.0,
                    found_at=now,
                )
            ],
        )
    else:
        monkeypatch.setattr(loop_mod, "find_binary_sets", lambda markets, books, fee_rate, now: [])
    monkeypatch.setattr(loop_mod, "find_range_sets", lambda markets, books, fee_rate, now: [])


def test_run_once_full_pipeline_and_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_pipeline(monkeypatch, include_complete_set=True)
    settings = make_settings(tmp_path)
    pm = FakePolymarket(fail_times=2)
    pm.outcome = MarketOutcome(market_id="m0", resolved_yes=False, resolved_at=NOW)
    db = FakeDeribit()
    journal = FakeJournal()
    journal.unresolved = ["m0", "m1"]
    broker = FakeBroker()
    risk = RiskManager(settings)
    sleeps: list[float] = []

    loop_mod.run_once(
        settings,
        polymarket=pm,  # type: ignore[arg-type]
        deribit=db,  # type: ignore[arg-type]
        journal=journal,  # type: ignore[arg-type]
        broker=broker,  # type: ignore[arg-type]
        risk=risk,
        now=NOW,
        sleep_fn=sleeps.append,
        known_markets={"m0": EXPIRED_MARKET},
    )

    assert journal.migrated
    assert len(sleeps) == 2  # 2 retries before 3rd attempt succeeded
    assert journal.saved_markets == [MARKET]
    assert len(journal.saved_books) == 2
    assert len(journal.saved_fairs) == 1
    assert len(journal.saved_complete_sets) == 1
    assert len(journal.saved_merges) == 1
    assert len(journal.saved_signals) == 1
    assert journal.saved_signals[0].outcome is Outcome.YES
    assert journal.saved_signals[0].size == pytest.approx(100.0)
    assert len(journal.saved_orders) == 3  # 2 complete set legs + 1 directional signal
    assert len(journal.saved_fills) == 3
    assert len(journal.saved_outcomes) == 1
    assert pm.outcome_requests == ["m0"]  # m1 has not expired, so it cannot have resolved
    assert risk.daily_pnl_usd(NOW) == pytest.approx(-50.0)


def test_run_survives_errors_and_exits_cleanly_on_ctrl_c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_pipeline(monkeypatch)
    settings = make_settings(tmp_path)
    pm = FakePolymarket()
    db = FakeDeribit()
    journal = FakeJournal()
    broker = FakeBroker()
    risk = RiskManager(settings)

    tick_count = 0

    def flaky_run_once(*args: Any, **kwargs: Any) -> None:
        nonlocal tick_count
        tick_count += 1
        if tick_count == 1:
            raise RuntimeError("transient API outage")

    monkeypatch.setattr(loop_mod, "run_once", flaky_run_once)

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise KeyboardInterrupt

    loop_mod.run(
        settings,
        polymarket=pm,  # type: ignore[arg-type]
        deribit=db,  # type: ignore[arg-type]
        journal=journal,  # type: ignore[arg-type]
        broker=broker,  # type: ignore[arg-type]
        risk=risk,
        sleep_fn=fake_sleep,
    )

    assert tick_count == 2
    assert sleeps[0] == pytest.approx(loop_mod.DEFAULT_RETRY_BASE_DELAY)
    assert sleeps[1] == pytest.approx(settings.poll_seconds)


def test_run_once_with_real_paper_broker_and_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from polybot.execution.paper import PaperBroker

    settings = make_settings(tmp_path)
    pm = FakePolymarket()
    db = FakeDeribit()
    journal = FakeJournal()
    broker = PaperBroker(settings.taker_fee_rate)
    risk = RiskManager(settings)

    monkeypatch.setattr(loop_mod, "parse_market", lambda raw: MARKET)
    monkeypatch.setattr(loop_mod, "fair_probability", _fake_fair)

    loop_mod.run_once(
        settings,
        polymarket=pm,  # type: ignore[arg-type]
        deribit=db,  # type: ignore[arg-type]
        journal=journal,  # type: ignore[arg-type]
        broker=broker,
        risk=risk,
        now=NOW,
    )

    assert len(journal.saved_signals) == 1
    sig = journal.saved_signals[0]
    assert sig.outcome is Outcome.YES
    assert sig.size == pytest.approx(100.0)
    assert sig.edge_after_costs == pytest.approx(0.10)
    assert broker.positions[("m1", Outcome.YES)].size == pytest.approx(100.0)
    assert broker.cash_usd == pytest.approx(-50.0)


def _run_tick(
    tmp_path: Path, *, deribit: FakeDeribit | None = None, **kwargs: Any
) -> tuple[FakePolymarket, FakeJournal]:
    pm = FakePolymarket()
    pm.outcome = MarketOutcome(market_id="m0", resolved_yes=True, resolved_at=NOW)
    journal = FakeJournal()
    journal.unresolved = ["m0"]
    loop_mod.run_once(
        kwargs.pop("settings", make_settings(tmp_path)),
        polymarket=pm,  # type: ignore[arg-type]
        deribit=deribit or FakeDeribit(),  # type: ignore[arg-type]
        journal=journal,  # type: ignore[arg-type]
        broker=FakeBroker(),  # type: ignore[arg-type]
        now=NOW,
        known_markets={"m0": EXPIRED_MARKET},
        **kwargs,
    )
    return pm, journal


def test_resolution_basis_reaches_the_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_pipeline(monkeypatch)
    seen: list[float] = []

    def fair(market, chain, now, *, resolution_basis: float = 0.0):
        seen.append(resolution_basis)
        return _fake_fair(market, chain, now)

    monkeypatch.setattr(loop_mod, "fair_probability", fair)
    _run_tick(tmp_path, deribit=FakeDeribit(basis=0.0012))
    assert seen == [pytest.approx(0.0012)]


def test_fair_probability_carries_the_market_quote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_pipeline(monkeypatch)
    _, journal = _run_tick(tmp_path)
    meta = journal.saved_fairs[0].meta
    assert meta["market_yes_bid"] == pytest.approx(0.48)
    assert meta["market_yes_ask"] == pytest.approx(0.50)


def test_deribit_outage_skips_pricing_but_not_the_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_pipeline(monkeypatch)
    _, journal = _run_tick(
        tmp_path,
        deribit=FakeDeribit(chain_error=httpx.ReadTimeout("deribit is slow")),
        sleep_fn=lambda _: None,
    )
    assert len(journal.saved_books) == 2
    assert journal.saved_fairs == []
    assert journal.saved_signals == []
    assert len(journal.saved_outcomes) == 1


def test_tick_without_snapshot_trades_but_does_not_persist_or_settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_pipeline(monkeypatch)
    pm, journal = _run_tick(tmp_path, save_snapshots=False)
    assert journal.saved_books == []
    assert journal.saved_fairs == []
    assert len(journal.saved_signals) == 1
    assert pm.outcome_requests == []


def test_no_directional_entries_close_to_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_pipeline(monkeypatch)
    soon = dataclasses.replace(MARKET, expiry=NOW + timedelta(minutes=30))
    monkeypatch.setattr(loop_mod, "parse_market", lambda raw: soon)
    _, journal = _run_tick(tmp_path)
    assert len(journal.saved_fairs) == 1
    assert journal.saved_signals == []

