"""Task F. Edge report: Brier, calibration, paper PnL, complete-set profit."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from polybot.models import (
    Asset,
    BookSide,
    CompleteSetOpportunity,
    CryptoMarket,
    FairProbability,
    Fill,
    MarketKind,
    MarketOutcome,
    Merge,
    Order,
    OrderBook,
    OrderType,
    Outcome,
    PriceLevel,
    Signal,
)
from polybot.report import build_report, format_report
from polybot.storage.journal import Journal

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
LATER = NOW + timedelta(hours=1)


class MemoryJournal:
    def __init__(self) -> None:
        self.signals_list: list[Signal] = []
        self.fairs: list[FairProbability] = []
        self.fills_list: list[Fill] = []
        self.outcomes_list: list[MarketOutcome] = []
        self.orders_list: list[Order] = []
        self.merges_list: list[Merge] = []
        self.complete_sets_list: list[CompleteSetOpportunity] = []
        self.markets_list: list[CryptoMarket] = []
        self.books_list: list[tuple[str, OrderBook]] = []

    def signals(self, since: datetime | None = None) -> list[Signal]:
        return list(self.signals_list)

    def fair_probabilities(self, since: datetime | None = None) -> list[FairProbability]:
        return list(self.fairs)

    def fills(self, since: datetime | None = None) -> list[Fill]:
        return list(self.fills_list)

    def outcomes(self) -> list[MarketOutcome]:
        return list(self.outcomes_list)

    def orders(self) -> list[Order]:
        return list(self.orders_list)

    def merges(self) -> list[Merge]:
        return list(self.merges_list)

    def complete_sets(self) -> list[CompleteSetOpportunity]:
        return list(self.complete_sets_list)

    def markets(self) -> list[CryptoMarket]:
        return list(self.markets_list)

    def books(self) -> list[tuple[str, OrderBook]]:
        return list(self.books_list)


def make_market(market_id: str = "m1", yes: str = "yes", no: str = "no") -> CryptoMarket:
    return CryptoMarket(
        market_id=market_id,
        event_id="evt",
        question="Will ETH be above 4400?",
        asset=Asset.ETH,
        kind=MarketKind.ABOVE,
        strike_low=4400.0,
        strike_high=None,
        expiry=datetime(2026, 10, 17, 16, 0, tzinfo=UTC),
        yes_token_id=yes,
        no_token_id=no,
        resolution_source="Binance ETHUSDT 1m close",
        neg_risk=False,
        volume_usd=1_000.0,
    )


def make_fair(
    market_id: str = "m1",
    p_yes: float = 0.6,
    at: datetime = NOW,
    model: str = "deribit_iv_v1",
) -> FairProbability:
    return FairProbability(
        market_id=market_id,
        p_yes=p_yes,
        computed_at=at,
        model=model,
        spot=4500.0,
    )


def make_signal(
    market_id: str = "m1",
    outcome: Outcome = Outcome.YES,
    fair: float = 0.6,
    entry: float = 0.4,
    edge: float = 0.2,
    size: float = 10.0,
    at: datetime = NOW,
    reason: str = "model_edge",
) -> Signal:
    return Signal(
        market_id=market_id,
        outcome=outcome,
        fair_price=fair,
        entry_price=entry,
        edge_after_costs=edge,
        size=size,
        stop_price=None,
        take_profit_price=None,
        created_at=at,
        reason=reason,
    )


def make_book(
    token: str,
    bid: float | None,
    ask: float | None,
    at: datetime = NOW,
) -> OrderBook:
    bids = () if bid is None else (PriceLevel(bid, 100.0),)
    asks = () if ask is None else (PriceLevel(ask, 100.0),)
    return OrderBook(token_id=token, bids=bids, asks=asks, timestamp=at)


def make_order(
    order_id: str,
    market_id: str,
    token_id: str,
    outcome: Outcome,
    side: BookSide = BookSide.BUY,
    size: float = 10.0,
    at: datetime = NOW,
    strategy: str = "signal",
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
        created_at=at,
        strategy=strategy,
    )


def make_fill(
    order_id: str,
    price: float,
    size: float,
    fee: float = 0.0,
    at: datetime = NOW,
) -> Fill:
    return Fill(
        order_id=order_id,
        price=price,
        size=size,
        fee_usd=fee,
        is_maker=False,
        timestamp=at,
    )


def resolved(market_id: str, yes: bool, at: datetime = LATER) -> MarketOutcome:
    return MarketOutcome(market_id=market_id, resolved_yes=yes, resolved_at=at)


def test_empty_journal_has_no_score():
    report = build_report(MemoryJournal())

    assert report.n_signals == 0
    assert report.n_resolved == 0
    assert report.paper_pnl_usd == 0.0
    assert report.complete_set_profit_usd == 0.0
    assert report.brier_model != report.brier_model  # NaN
    assert report.brier_market != report.brier_market
    assert report.calibration == ()

    text = format_report(report)
    assert "Scored forecasts: 0" in text
    assert "Brier model: n/a" in text
    assert "no evidence of an edge" in text
    assert text.endswith("\n")


def test_empty_sqlite_journal(tmp_path: Path):
    with Journal(tmp_path / "polybot.sqlite") as journal:
        report = build_report(journal)
    assert report.n_signals == 0
    assert report.n_resolved == 0
    assert report.paper_pnl_usd == 0.0


def test_sqlite_journal_scores_the_book_mid_and_the_fill(tmp_path: Path):
    with Journal(tmp_path / "polybot.sqlite") as journal:
        journal.save_market(make_market())
        journal.save_book("m1", make_book("yes", bid=0.40, ask=0.60))
        journal.save_fair_probability(make_fair(p_yes=0.80))
        journal.save_signal(make_signal(fair=0.80, entry=0.55, edge=0.25, size=10))
        journal.save_order(make_order("o1", "m1", "yes", Outcome.YES, size=10))
        journal.save_fill(make_fill("o1", price=0.55, size=10))
        journal.save_merge(
            Merge(market_ids=("other",), size=5, cost_basis_usd=4, proceeds_usd=5, timestamp=NOW)
        )
        journal.save_outcome(resolved("m1", yes=True))
        report = build_report(journal)

    assert report.n_signals == 1
    assert report.n_resolved == 1
    # Mid 0.50, not the 0.55 fill. The fill is the trade; the mid is the market forecast.
    assert report.brier_model == pytest.approx((0.80 - 1.0) ** 2)
    assert report.brier_market == pytest.approx((0.50 - 1.0) ** 2)
    assert report.paper_pnl_usd == pytest.approx(10 * (1.0 - 0.55))
    assert report.complete_set_profit_usd == pytest.approx(1.0)


def test_brier_uses_yes_mid_on_the_same_snapshots():
    journal = MemoryJournal()
    journal.markets_list.append(make_market())
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.fairs.extend(
        [
            make_fair(p_yes=0.80, at=NOW),
            make_fair(p_yes=0.60, at=NOW + timedelta(minutes=1)),
        ]
    )
    journal.books_list.extend(
        [
            ("m1", make_book("yes", bid=0.30, ask=0.50, at=NOW)),  # mid 0.40
            ("m1", make_book("yes", bid=0.60, ask=0.80, at=NOW + timedelta(minutes=1))),  # 0.70
        ]
    )

    report = build_report(journal)

    # (0.80-1)^2 = 0.04, (0.60-1)^2 = 0.16 -> 0.10
    # (0.40-1)^2 = 0.36, (0.70-1)^2 = 0.09 -> 0.225
    assert report.n_resolved == 2
    assert report.brier_model == pytest.approx(0.10)
    assert report.brier_market == pytest.approx(0.225)
    assert report.brier_model < report.brier_market

    text = format_report(report)
    gap = report.brier_market - report.brier_model
    assert "There may be an edge" in text
    assert f"Brier gap (market - model): {gap:.4f}" in text


def test_book_mid_is_preferred_over_the_signal_ask():
    journal = MemoryJournal()
    journal.markets_list.append(make_market())
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.fairs.append(make_fair(p_yes=0.62))
    journal.books_list.append(("m1", make_book("yes", bid=0.40, ask=0.50)))  # mid 0.45
    journal.signals_list.append(make_signal(fair=0.99, entry=0.50, edge=0.49))

    report = build_report(journal)

    assert report.n_resolved == 1
    assert report.brier_model == pytest.approx((0.62 - 1.0) ** 2)
    assert report.brier_market == pytest.approx((0.45 - 1.0) ** 2)


def test_asof_book_ignores_a_later_snapshot():
    journal = MemoryJournal()
    journal.markets_list.append(make_market())
    journal.outcomes_list.append(resolved("m1", yes=True, at=LATER))
    fair_at = NOW + timedelta(seconds=10)
    journal.fairs.append(make_fair(p_yes=1.0, at=fair_at))
    journal.books_list.extend(
        [
            ("m1", make_book("yes", bid=0.40, ask=0.60, at=NOW)),  # mid 0.50
            ("m1", make_book("yes", bid=0.70, ask=0.90, at=NOW + timedelta(seconds=30))),
        ]
    )

    report = build_report(journal)
    assert report.brier_market == pytest.approx((0.50 - 1.0) ** 2)


def test_book_a_few_seconds_ahead_covers_clock_skew_only():
    journal = MemoryJournal()
    journal.markets_list.append(make_market())
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.fairs.append(make_fair(p_yes=1.0))
    journal.books_list.append(("m1", make_book("yes", bid=0.30, ask=0.50, at=NOW + timedelta(seconds=1))))

    scored = build_report(journal)
    assert scored.n_resolved == 1
    assert scored.brier_market == pytest.approx((0.40 - 1.0) ** 2)

    later = MemoryJournal()
    later.markets_list.append(make_market())
    later.outcomes_list.append(resolved("m1", yes=True))
    later.fairs.append(make_fair(p_yes=1.0))
    later.books_list.append(("m1", make_book("yes", bid=0.30, ask=0.50, at=NOW + timedelta(seconds=30))))
    assert build_report(later).n_resolved == 0


def test_yes_token_can_be_identified_from_an_order():
    journal = MemoryJournal()
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.fairs.append(make_fair(p_yes=0.70))
    journal.orders_list.append(make_order("o1", "m1", "yes_tok", Outcome.YES))
    journal.books_list.append(("m1", make_book("yes_tok", bid=0.40, ask=0.60)))

    report = build_report(journal)

    assert report.n_resolved == 1
    assert report.brier_market == pytest.approx((0.50 - 1.0) ** 2)
    assert report.paper_pnl_usd == pytest.approx(0.0)


def test_no_book_mid_is_converted_to_yes_probability():
    journal = MemoryJournal()
    journal.markets_list.append(make_market())
    journal.outcomes_list.append(resolved("m1", yes=False))
    journal.fairs.append(make_fair(p_yes=0.25))
    journal.books_list.append(("m1", make_book("no", bid=0.70, ask=0.90)))  # NO mid 0.80 -> YES 0.20

    report = build_report(journal)
    assert report.brier_model == pytest.approx(0.25**2)
    assert report.brier_market == pytest.approx(0.20**2)
    assert "does not beat the market" in format_report(report)


def test_one_sided_bid_is_the_market_price():
    journal = MemoryJournal()
    journal.markets_list.append(make_market())
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.fairs.append(make_fair(p_yes=0.42))
    journal.books_list.append(("m1", make_book("yes", bid=0.42, ask=None)))

    report = build_report(journal)
    assert report.brier_model == pytest.approx((0.42 - 1.0) ** 2)
    assert report.brier_market == pytest.approx(report.brier_model)
    assert "matches the market" in format_report(report)


def test_same_timestamp_signals_price_a_fair_without_a_book():
    journal = MemoryJournal()
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.fairs.append(make_fair(p_yes=0.60))
    journal.signals_list.extend(
        [
            make_signal(outcome=Outcome.YES, fair=0.60, entry=0.55, edge=0.05, size=10),
            make_signal(outcome=Outcome.NO, fair=0.40, entry=0.50, edge=-0.10, size=10),
        ]
    )

    report = build_report(journal)

    market_p = (0.55 + (1.0 - 0.50)) / 2.0
    assert report.n_signals == 2
    assert report.n_resolved == 1
    assert report.brier_model == pytest.approx((0.60 - 1.0) ** 2)
    assert report.brier_market == pytest.approx((market_p - 1.0) ** 2)
    # YES wins: 10*(1-0.55) + 10*(0-0.50)
    assert report.paper_pnl_usd == pytest.approx(4.5 - 5.0)


def test_signal_only_journal_scores_each_trade_in_yes_space():
    journal = MemoryJournal()
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.signals_list.append(
        make_signal(outcome=Outcome.NO, fair=0.80, entry=0.30, edge=0.50, size=4)
    )

    report = build_report(journal)

    assert report.n_resolved == 1
    assert report.brier_model == pytest.approx((0.20 - 1.0) ** 2)
    assert report.brier_market == pytest.approx((0.70 - 1.0) ** 2)
    assert report.paper_pnl_usd == pytest.approx(4 * (0.0 - 0.30))
    assert "Paper PnL (USD): -1.20" in format_report(report)


def test_forecast_after_resolution_is_excluded_and_latest_outcome_wins():
    journal = MemoryJournal()
    journal.markets_list.append(make_market())
    journal.outcomes_list.extend(
        [
            resolved("m1", yes=False, at=NOW),
            resolved("m1", yes=True, at=NOW + timedelta(minutes=5)),
        ]
    )
    journal.fairs.extend(
        [
            make_fair(p_yes=1.0, at=NOW + timedelta(minutes=5)),
            make_fair(p_yes=1.0, at=NOW + timedelta(minutes=6)),
        ]
    )
    journal.books_list.append(("m1", make_book("yes", bid=0.0, ask=0.0, at=NOW)))

    report = build_report(journal)
    assert report.n_resolved == 1
    assert report.brier_model == pytest.approx(0.0)
    assert report.brier_market == pytest.approx(1.0)


def test_only_the_latest_model_is_scored():
    journal = MemoryJournal()
    journal.markets_list.append(make_market())
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.books_list.append(("m1", make_book("yes", bid=0.40, ask=0.60)))
    journal.fairs.extend(
        [
            make_fair(p_yes=0.0, at=NOW, model="m1"),
            make_fair(p_yes=1.0, at=NOW, model="m2"),
            make_fair(p_yes=0.0, at=NOW - timedelta(minutes=1), model="m1"),
        ]
    )

    report = build_report(journal)
    assert report.n_resolved == 1
    assert report.brier_model == pytest.approx(0.0)


def test_unresolved_signal_is_counted_but_not_scored_or_paid():
    journal = MemoryJournal()
    journal.markets_list.append(make_market())
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.fairs.append(make_fair(p_yes=0.7))
    journal.books_list.append(("m1", make_book("yes", bid=0.40, ask=0.40)))
    journal.signals_list.extend(
        [
            make_signal(market_id="m1", fair=0.7, entry=0.4, edge=0.3, size=10),
            make_signal(market_id="open", fair=0.7, entry=0.4, edge=0.3, size=100),
        ]
    )

    report = build_report(journal)
    assert report.n_signals == 2
    assert report.n_resolved == 1
    assert report.paper_pnl_usd == pytest.approx(10 * (1.0 - 0.4))


def test_signal_fee_comes_out_of_hold_to_resolution_pnl():
    journal = MemoryJournal()
    journal.signals_list.append(make_signal(fair=0.70, entry=0.40, edge=0.27, size=100))
    journal.outcomes_list.append(resolved("m1", yes=True))

    report = build_report(journal)
    # fee per share = 0.70 - 0.40 - 0.27 = 0.03
    assert report.paper_pnl_usd == pytest.approx(100 * (1.0 - 0.40 - 0.03))
    assert report.brier_model == pytest.approx((0.70 - 1.0) ** 2)
    assert "Paper PnL (USD): +57.00" in format_report(report)


def test_invalid_signal_probability_still_has_pnl_but_no_brier():
    journal = MemoryJournal()
    journal.signals_list.append(make_signal(fair=1.5, entry=0.40, edge=1.0, size=10))
    journal.outcomes_list.append(resolved("m1", yes=True))

    report = build_report(journal)
    assert report.n_resolved == 0
    assert report.paper_pnl_usd == pytest.approx(10 * (1.0 - 0.40 - 0.1))


def test_order_fills_are_the_trade_and_are_not_added_twice():
    journal = MemoryJournal()
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.signals_list.append(make_signal(fair=0.70, entry=0.40, edge=0.27, size=100))
    journal.orders_list.append(make_order("o1", "m1", "yes", Outcome.YES, size=100))
    journal.fills_list.append(make_fill("o1", price=0.40, size=40, fee=1.2))

    report = build_report(journal)
    # Partial fill: payout 40, cost 16 + 1.2. The signal's 100 shares are not added.
    assert report.paper_pnl_usd == pytest.approx(40.0 - 16.0 - 1.2)
    assert report.complete_set_profit_usd == pytest.approx(0.0)


def test_signal_on_another_outcome_is_kept_when_the_order_is_a_different_leg():
    journal = MemoryJournal()
    journal.outcomes_list.append(resolved("m1", yes=False))
    journal.orders_list.append(make_order("o1", "m1", "yes", Outcome.YES, size=10))
    journal.fills_list.append(make_fill("o1", price=0.50, size=10))
    journal.signals_list.append(
        make_signal(outcome=Outcome.NO, fair=0.80, entry=0.20, edge=0.60, size=10)
    )

    report = build_report(journal)
    # YES lost: -5. NO won: 10*(1-0.20) = 8.
    assert report.paper_pnl_usd == pytest.approx(-5.0 + 8.0)


def test_sell_then_resolution_realizes_both_legs():
    journal = MemoryJournal()
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.orders_list.extend(
        [
            make_order("buy", "m1", "yes", Outcome.YES, BookSide.BUY, size=10, at=NOW),
            make_order(
                "sell",
                "m1",
                "yes",
                Outcome.YES,
                BookSide.SELL,
                size=4,
                at=NOW + timedelta(minutes=1),
            ),
        ]
    )
    journal.fills_list.extend(
        [
            make_fill("buy", price=0.40, size=10, at=NOW),
            make_fill("sell", price=0.60, size=4, at=NOW + timedelta(minutes=1)),
        ]
    )

    report = build_report(journal)
    # Sell 4: proceeds 2.4, cost 1.6, pnl 0.8. Remainder 6 pays 6 - 2.4 = 3.6.
    assert report.paper_pnl_usd == pytest.approx(0.8 + 3.6)


def test_merge_does_not_also_settle_the_merged_shares():
    journal = MemoryJournal()
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.orders_list.extend(
        [
            make_order("y", "m1", "yes", Outcome.YES, size=10, strategy="complete_set"),
            make_order("n", "m1", "no", Outcome.NO, size=10, strategy="complete_set"),
        ]
    )
    journal.fills_list.extend(
        [
            make_fill("y", price=0.40, size=10),
            make_fill("n", price=0.40, size=10),
        ]
    )
    journal.merges_list.append(
        Merge(
            market_ids=("m1",),
            size=10,
            cost_basis_usd=8,
            proceeds_usd=10,
            timestamp=NOW,
        )
    )

    report = build_report(journal)
    assert report.paper_pnl_usd == pytest.approx(0.0)
    assert report.complete_set_profit_usd == pytest.approx(2.0)


def test_merge_leaves_directional_shares_at_the_blended_average():
    journal = MemoryJournal()
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.orders_list.extend(
        [
            make_order("sig", "m1", "yes", Outcome.YES, size=10, strategy="signal"),
            make_order("cy", "m1", "yes", Outcome.YES, size=10, strategy="complete_set"),
            make_order("cn", "m1", "no", Outcome.NO, size=10, strategy="complete_set"),
        ]
    )
    journal.fills_list.extend(
        [
            make_fill("sig", price=0.30, size=10),
            make_fill("cy", price=0.50, size=10),
            make_fill("cn", price=0.40, size=10),
        ]
    )
    journal.merges_list.append(
        Merge(market_ids=("m1",), size=10, cost_basis_usd=8, proceeds_usd=10, timestamp=NOW)
    )

    report = build_report(journal)
    # Blended YES average is 0.40. Ten signal shares remain and pay 10 - 4.
    assert report.paper_pnl_usd == pytest.approx(6.0)
    assert report.complete_set_profit_usd == pytest.approx(2.0)


def test_range_merge_consumes_each_leg():
    journal = MemoryJournal()
    journal.outcomes_list.extend([resolved("a", yes=True), resolved("b", yes=True)])
    journal.orders_list.extend(
        [
            make_order("a", "a", "ya", Outcome.YES, size=10, strategy="complete_set"),
            make_order("b", "b", "yb", Outcome.YES, size=10, strategy="complete_set"),
        ]
    )
    journal.fills_list.extend(
        [
            make_fill("a", price=0.40, size=10),
            make_fill("b", price=0.40, size=10),
        ]
    )
    journal.merges_list.append(
        Merge(market_ids=("a", "b"), size=10, cost_basis_usd=8, proceeds_usd=10, timestamp=NOW)
    )

    report = build_report(journal)
    assert report.complete_set_profit_usd == pytest.approx(2.0)
    assert report.paper_pnl_usd == pytest.approx(0.0)


def test_unmerged_complete_set_orders_settle_into_complete_set_profit():
    journal = MemoryJournal()
    journal.outcomes_list.extend([resolved("a", yes=True), resolved("b", yes=False)])
    journal.orders_list.extend(
        [
            make_order("a", "a", "ya", Outcome.YES, size=10, strategy="complete_set"),
            make_order("b", "b", "yb", Outcome.YES, size=10, strategy="complete_set"),
        ]
    )
    journal.fills_list.extend(
        [
            make_fill("a", price=0.40, size=10),
            make_fill("b", price=0.55, size=10),
        ]
    )

    report = build_report(journal)
    assert report.paper_pnl_usd == pytest.approx(0.0)
    assert report.complete_set_profit_usd == pytest.approx(6.0 - 5.5)


def test_opportunity_profit_is_used_when_the_set_was_not_executed():
    journal = MemoryJournal()
    journal.complete_sets_list.append(
        CompleteSetOpportunity(
            market_ids=("a", "b"),
            token_ids=("ya", "yb"),
            size=10,
            total_cost_usd=9.2,
            payout_usd=10,
            found_at=NOW,
        )
    )

    report = build_report(journal)
    assert report.complete_set_profit_usd == pytest.approx(0.8)
    assert report.paper_pnl_usd == pytest.approx(0.0)


def test_executed_opportunity_is_not_added_on_top_of_the_merge():
    journal = MemoryJournal()
    journal.complete_sets_list.append(
        CompleteSetOpportunity(
            market_ids=("m1",),
            token_ids=("yes", "no"),
            size=10,
            total_cost_usd=8.5,
            payout_usd=10,
            found_at=NOW,
        )
    )
    journal.merges_list.append(
        Merge(market_ids=("m1",), size=10, cost_basis_usd=8.8, proceeds_usd=10, timestamp=NOW)
    )

    report = build_report(journal)
    assert report.complete_set_profit_usd == pytest.approx(1.2)


def test_unfilled_complete_set_order_does_not_book_the_opportunity():
    journal = MemoryJournal()
    journal.orders_list.append(
        make_order("o", "m1", "yes", Outcome.YES, strategy="complete_set")
    )
    journal.complete_sets_list.append(
        CompleteSetOpportunity(
            market_ids=("m1",),
            token_ids=("yes", "no"),
            size=10,
            total_cost_usd=8.5,
            payout_usd=10,
            found_at=NOW,
        )
    )

    report = build_report(journal)
    assert report.complete_set_profit_usd == pytest.approx(0.0)


def test_complete_set_signal_is_not_paper_pnl():
    journal = MemoryJournal()
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.signals_list.append(
        make_signal(fair=0.40, entry=0.30, edge=0.10, size=10, reason="complete_set")
    )

    report = build_report(journal)
    assert report.paper_pnl_usd == pytest.approx(0.0)
    assert report.complete_set_profit_usd == pytest.approx(10 * (1.0 - 0.30))


def test_calibration_buckets_and_boundaries():
    journal = MemoryJournal()
    specs = [
        (0.00, 0.10, False, NOW),
        (0.22, 0.40, False, NOW + timedelta(minutes=1)),
        (0.25, 0.50, True, NOW + timedelta(minutes=2)),
        (0.30, 0.30, True, NOW + timedelta(minutes=3)),
        (1.00, 0.80, True, NOW + timedelta(minutes=4)),
    ]
    for index, (p_model, p_market, yes, at) in enumerate(specs):
        market_id = f"m{index}"
        journal.markets_list.append(make_market(market_id))
        journal.outcomes_list.append(resolved(market_id, yes=yes, at=LATER))
        journal.fairs.append(make_fair(market_id, p_yes=p_model, at=at))
        if p_market == 0.0:
            bid, ask = 0.0, 0.0
        else:
            bid, ask = p_market - 0.01, p_market + 0.01
        journal.books_list.append((market_id, make_book("yes", bid=bid, ask=ask, at=at)))

    report = build_report(journal)
    by_low = {bucket.p_low: bucket for bucket in report.calibration}

    assert by_low[0.0].p_high == pytest.approx(0.1)
    assert by_low[0.0].n == 1
    assert by_low[0.0].realized_freq == pytest.approx(0.0)

    low = by_low[0.2]
    assert low.n == 2
    assert low.model_mean_p == pytest.approx(0.235)
    assert low.market_mean_p == pytest.approx(0.45)
    assert low.realized_freq == pytest.approx(0.5)

    assert by_low[0.3].p_low == pytest.approx(0.3)
    assert by_low[0.3].n == 1

    top = by_low[0.9]
    assert top.p_high == pytest.approx(1.0)
    assert top.n == 1
    assert top.realized_freq == pytest.approx(1.0)

    text = format_report(report)
    assert "[0.0, 0.1)" in text
    assert "[0.2, 0.3)" in text
    assert "[0.3, 0.4)" in text
    assert "[0.9, 1.0]" in text
    assert "n=2" in text


class QuotedOnlyJournal(MemoryJournal):
    def books(self) -> list[tuple[str, OrderBook]]:
        raise AssertionError("books must not be loaded when every fair carries its quote")


def quoted_fair(p_yes: float, bid: float, ask: float, market_id: str = "m1") -> FairProbability:
    return replace(
        make_fair(market_id, p_yes=p_yes), meta={"market_yes_bid": bid, "market_yes_ask": ask}
    )


def test_quote_recorded_with_the_fair_is_the_market_price():
    journal = QuotedOnlyJournal()
    journal.markets_list.append(make_market())
    journal.outcomes_list.append(resolved("m1", yes=True))
    journal.fairs.append(quoted_fair(p_yes=0.70, bid=0.58, ask=0.62))

    report = build_report(journal)
    assert report.n_resolved == 1
    assert report.brier_model == pytest.approx((0.70 - 1.0) ** 2)
    assert report.brier_market == pytest.approx((0.60 - 1.0) ** 2)


def test_wide_or_empty_quote_is_not_scored():
    journal = QuotedOnlyJournal()
    for market_id, (bid, ask) in {"m1": (0.0, 1.0), "m2": (0.40, 0.60)}.items():
        journal.markets_list.append(make_market(market_id))
        journal.outcomes_list.append(resolved(market_id, yes=False))
        journal.fairs.append(quoted_fair(p_yes=0.02, bid=bid, ask=ask, market_id=market_id))

    report = build_report(journal)
    assert report.n_resolved == 0
    assert "No resolved forecasts" in format_report(report)
