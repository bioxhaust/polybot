import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from polybot.execution.maker import (
    DEFAULT_PARAMS,
    MakerParams,
    MakerSimulator,
    Quote,
    SimulationResult,
    Tick,
    fair_move_std,
    format_comparison,
    make_quote,
    profile_trades,
    replay,
    spot_moved_sharply,
)
from polybot.models import (
    Asset,
    BookSide,
    CryptoMarket,
    FairProbability,
    MarketKind,
    MarketOutcome,
    OrderBook,
    OrderStatus,
    Outcome,
    PriceLevel,
    WalletTrade,
)

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
SEC = timedelta(seconds=1)
FIXTURES = Path(__file__).parent / "fixtures"
YES, NO = Outcome.YES, Outcome.NO
BUY, SELL = BookSide.BUY, BookSide.SELL

MARKET = CryptoMarket(
    market_id="m1",
    event_id="e1",
    question="Will the price of Bitcoin be above $100,000 on October 17?",
    asset=Asset.BTC,
    kind=MarketKind.ABOVE,
    strike_low=100_000.0,
    strike_high=None,
    expiry=NOW + timedelta(days=3),
    yes_token_id="yes",
    no_token_id="no",
    resolution_source="Binance BTCUSDT 1m close",
    neg_risk=False,
    volume_usd=100_000.0,
)


def fair(p: float, at: datetime = NOW, spot: float = 100_000.0) -> FairProbability:
    return FairProbability("m1", p_yes=p, computed_at=at, model="test", spot=spot)


def book(bid: float | None = None, ask: float | None = None, at: datetime = NOW) -> OrderBook:
    return OrderBook(
        "yes",
        bids=() if bid is None else (PriceLevel(bid, 1_000.0),),
        asks=() if ask is None else (PriceLevel(ask, 1_000.0),),
        timestamp=at,
    )


def trade(outcome, side, price, size=10.0, at=NOW + 10 * SEC, tx="0xt") -> WalletTrade:
    return WalletTrade("0xtaker", "m1", outcome, side, price, size, at, tx)


def quoted(yes=0.49, no=0.49, size=50.0, fee_rate=0.0) -> MakerSimulator:
    sim = MakerSimulator(fee_rate=fee_rate)
    sim.requote(MARKET, Quote(yes, no, size), NOW)
    return sim


def bids(quote: Quote) -> tuple[float | None, float | None]:
    return quote.yes_bid, quote.no_bid


# --- Quotes -----------------------------------------------------------------------


def test_quote_brackets_fair_price():
    quote = make_quote(MARKET, fair(0.5), book(), inventory=0)
    assert bids(quote) == (0.49, 0.49)
    assert quote.size == DEFAULT_PARAMS.size


@pytest.mark.parametrize("p", [0.06, 0.2, 0.37, 0.5, 0.63, 0.8, 0.94])
@pytest.mark.parametrize("inventory", [-400.0, -120.0, 0.0, 120.0, 400.0])
def test_quotes_never_cross_and_sit_on_tick_grid(p, inventory):
    quote = make_quote(MARKET, fair(p), book(), inventory)
    assert quote.yes_bid is not None or quote.no_bid is not None
    if quote.yes_bid is not None and quote.no_bid is not None:
        assert quote.yes_bid + quote.no_bid <= 1 - 2 * DEFAULT_PARAMS.half_spread + 1e-9
    for bid in bids(quote):
        if bid is not None:
            assert bid * 100 == pytest.approx(round(bid * 100))
    if inventory == 0:
        assert quote.yes_bid <= p - DEFAULT_PARAMS.half_spread + 1e-9
        assert quote.no_bid <= 1 - p - DEFAULT_PARAMS.half_spread + 1e-9


def test_skew_moves_both_bids_against_inventory():
    params = MakerParams(skew_at_max=0.04)
    flat = make_quote(MARKET, fair(0.5), book(), 0, params=params)
    long_yes = make_quote(MARKET, fair(0.5), book(), 125, params=params)
    short_yes = make_quote(MARKET, fair(0.5), book(), -125, params=params)
    assert bids(flat) == (0.49, 0.49)
    assert bids(long_yes) == (0.47, 0.51)
    assert bids(short_yes) == (0.51, 0.47)


def test_inventory_limit_stops_the_side_that_adds_risk():
    long_yes = make_quote(MARKET, fair(0.5), book(), DEFAULT_PARAMS.max_inventory)
    short_yes = make_quote(MARKET, fair(0.5), book(), -DEFAULT_PARAMS.max_inventory)
    assert long_yes.yes_bid is None and long_yes.no_bid is not None
    assert short_yes.no_bid is None and short_yes.yes_bid is not None


@pytest.mark.parametrize(
    ("bid", "ask", "expected"),
    [
        (0.40, 0.45, (0.41, 0.49)),  # YES capped one tick above the best bid
        (0.55, 0.60, (0.49, 0.41)),  # NO book is the mirror: NO bid 0.40, ask 0.45
        (0.44, 0.47, (0.45, 0.49)),  # never at or above the best ask
        (0.48, 0.50, (0.49, 0.49)),  # tight book: fair-based bids already fit
    ],
)
def test_bids_are_post_only_and_do_not_overpay_the_queue(bid, ask, expected):
    quote = make_quote(MARKET, fair(0.5), book(bid, ask), 0)
    assert bids(quote) == expected
    assert quote.yes_bid < ask
    assert quote.no_bid < 1 - bid


def test_spread_widens_as_expiry_nears():
    soon = replace(MARKET, expiry=NOW + timedelta(hours=2))
    assert bids(make_quote(MARKET, fair(0.5), book(), 0)) == (0.49, 0.49)
    assert bids(make_quote(soon, fair(0.5), book(), 0)) == (0.47, 0.47)


@pytest.mark.parametrize(
    ("market", "p"),
    [
        (replace(MARKET, expiry=NOW + timedelta(minutes=30)), 0.5),
        (MARKET, 0.97),
        (MARKET, 0.03),
    ],
)
def test_no_quotes_close_to_expiry_or_on_near_certain_outcomes(market, p):
    assert make_quote(market, fair(p), book(), 0) == Quote(None, None, 0.0)


def test_quote_rejects_foreign_book():
    with pytest.raises(ValueError):
        make_quote(MARKET, fair(0.5), OrderBook("no", (), (), NOW), 0)


def test_quote_rejects_crossed_bids():
    with pytest.raises(ValueError):
        Quote(yes_bid=0.5, no_bid=0.5, size=10)


@pytest.mark.parametrize("sigma", [0.3, 1.2])
def test_fair_move_std_matches_black_scholes_simulation(sigma):
    rng = np.random.default_rng(7)
    spot, strike = 100.0, 103.0
    tau, horizon = 2 / 365, 2 / (365 * 24 * 60)

    def p_above(s, t):
        return norm.cdf((np.log(s / strike) - sigma**2 * t / 2) / (sigma * np.sqrt(t)))

    p0 = float(p_above(spot, tau))
    shocks = sigma * np.sqrt(horizon) * rng.standard_normal(400_000)
    later = spot * np.exp(shocks - sigma**2 * horizon / 2)
    simulated = float(np.std(p_above(later, tau - horizon) - p0))
    assert fair_move_std(p0, timedelta(minutes=2), timedelta(days=2)) == pytest.approx(
        simulated, rel=0.02
    )


def test_spot_move_detection():
    assert not spot_moved_sharply(fair(0.5, spot=100_000), 100_150)
    assert spot_moved_sharply(fair(0.5, spot=100_000), 100_250)
    assert spot_moved_sharply(fair(0.5, spot=100_000), 99_700)


# --- Fills ------------------------------------------------------------------------


def test_touching_our_price_never_fills_but_trading_through_does():
    sim = quoted()
    assert sim.on_trades([trade(YES, SELL, 0.49, tx="touch")]) == []
    fills = sim.on_trades([trade(YES, SELL, 0.48, size=30, tx="through")])
    assert [(f.price, f.size, f.is_maker) for f in fills] == [(0.49, 30.0, True)]
    assert sim.inventory("m1") == 30


def test_takers_of_the_other_outcome_reach_our_bid_through_the_mirror():
    sim = quoted()
    # buying NO at 0.52 means YES bids up to 0.48 were hit: our 0.49 YES bid was better
    fills = sim.on_trades([trade(NO, BUY, 0.52, tx="a")])
    assert sim.orders[fills[0].order_id].outcome is YES
    assert sim.on_trades([trade(NO, BUY, 0.51, tx="b")]) == []
    fills = sim.on_trades([trade(YES, BUY, 0.53, tx="c")])
    assert sim.orders[fills[0].order_id].outcome is NO


def test_trades_that_cannot_reach_our_bids_do_not_fill():
    sim = quoted()
    trades = [
        trade(YES, BUY, 0.30, tx="a"),  # a buyer cannot fill a bid on the same token
        trade(NO, BUY, 0.30, tx="b"),  # mirrored YES price 0.70 is above our YES bid
        trade(YES, SELL, 0.60, tx="c"),  # higher bids were ahead of ours
        replace(trade(YES, SELL, 0.10, tx="d"), market_id="other"),
    ]
    assert sim.on_trades(trades) == []


@pytest.mark.parametrize(
    ("taker", "maker"),
    [
        ((YES, SELL, 0.45), (YES, BUY, 0.45)),  # plain match
        ((NO, BUY, 0.55), (YES, BUY, 0.45)),  # mint: both bought, YES 0.45 + NO 0.55
        ((YES, SELL, 0.45), (NO, SELL, 0.55)),  # merge: both sold
    ],
)
def test_match_seen_from_both_sides_fills_us_once(taker, maker):
    sim = quoted()
    both = [trade(*taker, tx="0xm"), replace(trade(*maker, tx="0xm"), wallet="0xmaker")]
    fills = sim.on_trades(both)
    assert len(fills) == 1 and sim.orders[fills[0].order_id].outcome is YES


def test_fill_is_capped_by_order_and_trade_size():
    sim = quoted(size=50)
    fills = sim.on_trades(
        [trade(YES, SELL, 0.45, size=30, tx="a"), trade(YES, SELL, 0.45, size=40, tx="b")]
    )
    assert [f.size for f in fills] == [30, 20]
    order = sim.orders[fills[0].order_id]
    assert order.status is OrderStatus.FILLED and order.order_id not in sim.open_orders
    assert sim.on_trades([trade(YES, SELL, 0.40, tx="c")]) == []


def test_the_same_trade_is_counted_once():
    sim = quoted()
    t = trade(YES, SELL, 0.45, size=10, tx="a")
    assert len(sim.on_trades([t, t])) == 1
    assert sim.on_trades([t]) == []


def test_orders_fill_only_while_live_in_the_book():
    sim = quoted()
    assert sim.on_trades([trade(YES, SELL, 0.40, at=NOW - 5 * SEC, tx="before")]) == []
    assert sim.on_trades([trade(YES, SELL, 0.40, at=NOW + SEC, tx="in_flight")]) == []
    sim.cancel_all("m1", "spot_move", NOW + 20 * SEC)
    # the cancel takes `latency` to land: stale quotes get picked off meanwhile
    assert len(sim.on_trades([trade(YES, SELL, 0.40, at=NOW + 21 * SEC, tx="stale")])) == 1
    assert sim.on_trades([trade(NO, SELL, 0.40, at=NOW + 22 * SEC, tx="gone")]) == []
    assert sim.pulls == [(NOW + 20 * SEC, "m1", "spot_move")]


def test_requote_keeps_unchanged_orders_and_replaces_moved_ones():
    sim = quoted()
    before = {o.outcome: o for o in sim.open_orders.values()}
    sim.requote(MARKET, Quote(0.49, 0.47, 50), NOW + 30 * SEC)
    after = {o.outcome: o for o in sim.open_orders.values()}
    assert after[YES] is before[YES]
    assert after[NO].limit_price == 0.47 and before[NO].status is OrderStatus.CANCELLED
    sim.requote(MARKET, Quote(None, None, 0), NOW + 60 * SEC)
    assert sim.open_orders == {}


# --- Accounting -------------------------------------------------------------------


def test_merge_turns_pairs_into_a_dollar():
    sim = quoted()
    sim.on_trades([trade(YES, SELL, 0.45, size=50, tx="a"), trade(NO, SELL, 0.45, size=30, tx="b")])
    merge = sim.merge_pairs("m1", NOW + 30 * SEC)
    assert merge.size == 30 and merge.pnl_usd == pytest.approx(30 * 0.02)
    assert sim.inventory("m1") == pytest.approx(20)
    assert sim.cash_usd == pytest.approx(-80 * 0.49 + 30)
    assert sim.peak_capital_usd == pytest.approx(80 * 0.49)
    assert sim.merge_pairs("m1", NOW + 31 * SEC) is None


def test_fees_are_part_of_the_cost_basis():
    sim = quoted(fee_rate=0.01)
    sim.on_trades([trade(YES, SELL, 0.45, size=50, tx="a"), trade(NO, SELL, 0.45, size=50, tx="b")])
    assert sim.merge_pairs("m1", NOW + 30 * SEC).pnl_usd == pytest.approx(50 - 50 * 0.98 * 1.01)


def test_settle_pays_winners_and_pulls_quotes():
    sim = quoted()
    sim.on_trades([trade(YES, SELL, 0.45, size=50, tx="a")])
    assert sim.equity({"m1": 0.6}) == pytest.approx(-24.5 + 30)
    resolved = MarketOutcome("m1", resolved_yes=False, resolved_at=NOW + timedelta(days=3))
    assert sim.settle(resolved) == pytest.approx(-24.5)
    assert sim.open_orders == {} and sim.inventory("m1") == 0
    assert sim.cash_usd == pytest.approx(-24.5)


# --- Replay -----------------------------------------------------------------------


def ticks(path: list[float], spots: list[float] | None = None) -> list[Tick]:
    out = []
    for i, p in enumerate(path):
        at = NOW + 30 * i * SEC
        spot = None if spots is None else spots[i]
        out.append(Tick(at, fair(p, at), book(at=at), spot))
    return out


def two_sided_replay() -> tuple[MakerSimulator, SimulationResult]:
    sim = MakerSimulator(fee_rate=0.0, params=MakerParams(size=20))
    tape = [
        trade(YES, SELL, 0.47, 20, NOW + (30 * i + 15) * SEC, f"s{i}")
        if i % 2 == 0
        else trade(YES, BUY, 0.53, 20, NOW + (30 * i + 15) * SEC, f"b{i}")
        for i in range(118)
    ]
    return sim, replay(MARKET, ticks([0.5] * 120), tape, sim)


def test_replay_of_balanced_flow_earns_the_spread_through_merges():
    _, result = two_sided_replay()
    assert result.bought_yes == result.bought_no == result.merged == pytest.approx(59 * 20)
    assert result.merge_pnl_usd == pytest.approx(59 * 20 * 0.02)
    assert result.pnl_usd == pytest.approx(result.merge_pnl_usd)
    assert result.inventory == pytest.approx(0)
    assert result.edge_at_fill == pytest.approx(0.01)
    assert result.edge_after == pytest.approx(0.01)
    assert result.peak_capital_usd == pytest.approx(2 * 20 * 0.49)


def test_replay_of_informed_flow_loses_and_shows_adverse_selection():
    path = [0.60 - 0.0025 * i for i in range(120)]
    tape = [
        trade(YES, SELL, p - 0.05, 50, NOW + (30 * i + 15) * SEC, f"s{i}")
        for i, p in enumerate(path)
    ]
    resolved = MarketOutcome("m1", resolved_yes=False, resolved_at=MARKET.expiry)
    result = replay(MARKET, ticks(path), tape, MakerSimulator(0.0), outcome=resolved)
    assert result.bought_no == 0 and result.merged == 0
    assert result.inventory == DEFAULT_PARAMS.max_inventory
    assert result.edge_at_fill > 0
    assert result.edge_after < result.edge_at_fill - 0.02
    assert result.pnl_usd < -100


def test_replay_pulls_quotes_when_spot_runs_away_from_a_stale_fair_price():
    path = [0.5, 0.5, 0.5]
    tape = [trade(YES, SELL, 0.40, 50, NOW + 40 * SEC, "after_pull")]
    sim = MakerSimulator(0.0)
    result = replay(MARKET, ticks(path, spots=[100_000, 101_000, 101_000]), tape, sim)
    assert result.n_fills == 0
    assert result.n_pulls == 1 and sim.pulls[0][2] == "spot_move"


# --- Comparison with wallets ------------------------------------------------------


def load_wallet_trades(name: str) -> list[WalletTrade]:
    rows = json.loads((FIXTURES / name).read_text())
    return [
        WalletTrade(
            wallet=r["proxyWallet"],
            market_id=r["conditionId"],
            outcome=YES if r["outcomeIndex"] == 0 else NO,
            side=BookSide(r["side"]),
            price=r["price"],
            size=r["size"],
            timestamp=datetime.fromtimestamp(r["timestamp"], UTC),
            tx_hash=r["transactionHash"],
        )
        for r in rows
    ]


def test_profile_of_our_simulated_quotes():
    sim, _ = two_sided_replay()
    ours = profile_trades(sim.own_trades())
    assert ours.n_markets == 1 and ours.median_size == 20
    assert ours.share_buys == 1 and ours.share_paired == pytest.approx(1)
    assert ours.pair_cost == pytest.approx(0.98)
    assert ours.typical_unpaired == 20


def test_lennie_buys_both_sides_but_carries_large_unpaired_inventory():
    """Real trades of Lennie in four BTC/ETH range markets, July 2025 (maker and taker)."""
    lennie = profile_trades(load_wallet_trades("maker_lennie_trades.json"))
    assert (lennie.n_trades, lennie.n_markets) == (399, 4)
    assert lennie.share_buys == 1 and lennie.share_paired == pytest.approx(0.94, abs=0.01)
    # pairs assembled over a market's life cost more than $1 here: the spread did not
    # cover the drift between buying one side and the other
    assert lennie.pair_cost == pytest.approx(1.063, abs=0.001)
    assert lennie.typical_unpaired > 20 * DEFAULT_PARAMS.max_inventory

    sim, _ = two_sided_replay()
    table = format_comparison({"we": profile_trades(sim.own_trades()), "Lennie": lennie})
    assert "pair cost YES+NO" in table and "Lennie" in table
