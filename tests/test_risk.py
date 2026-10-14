from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from polybot.config import Settings
from polybot.engine.risk import RiskManager
from polybot.models import Asset, CryptoMarket, MarketKind, Outcome, Signal

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def make_settings(**overrides: object) -> Settings:
    fields: dict[str, object] = {
        "db_path": Path("data/test.sqlite"),
        "assets": (Asset.BTC, Asset.ETH),
        "poll_seconds": 30.0,
        "bankroll_usd": 10_000.0,
        "risk_unit_pct": 0.005,  # 1R = $50
        "daily_loss_limit_r": 3.0,  # -$150
        "weekly_loss_limit_r": 7.0,  # -$350
        "theme_limit_r": 3.0,  # 3R per asset
        "max_share_of_volume": 0.01,  # 1% of market volume_usd
        "min_edge": 0.03,
        "taker_fee_rate": 0.0,
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


def make_market(**overrides: object) -> CryptoMarket:
    fields: dict[str, object] = {
        "market_id": "mkt_eth_1",
        "event_id": "evt_1",
        "question": "Will ETH be above $4,400 on October 17?",
        "asset": Asset.ETH,
        "kind": MarketKind.ABOVE,
        "strike_low": 4400.0,
        "strike_high": None,
        "expiry": datetime(2026, 10, 17, 16, 0, tzinfo=UTC),
        "yes_token_id": "yes_1",
        "no_token_id": "no_1",
        "resolution_source": "Binance ETHUSDT 1m close",
        "neg_risk": False,
        "volume_usd": 100_000.0,
    }
    fields.update(overrides)
    return CryptoMarket(**fields)  # type: ignore[arg-type]


def make_signal(**overrides: object) -> Signal:
    fields: dict[str, object] = {
        "market_id": "mkt_eth_1",
        "outcome": Outcome.YES,
        "fair_price": 0.60,
        "entry_price": 0.50,
        "edge_after_costs": 0.10,
        "size": 0.0,
        "stop_price": None,
        "take_profit_price": 0.60,
        "created_at": NOW,
        "reason": "model_edge",
    }
    fields.update(overrides)
    return Signal(**fields)  # type: ignore[arg-type]


def test_sizing_from_r_without_stop():
    # 1R = $50, entry = 0.50, no stop -> worst case is full entry loss ($0.50/share) -> 100 shares
    rm = RiskManager(make_settings())
    market = make_market()
    sig = make_signal(entry_price=0.50, stop_price=None)

    shares = rm.size(sig, market, NOW)
    assert shares == pytest.approx(100.0)
    assert rm.state.open_risk_by_theme["ETH"] == pytest.approx(1.0)


def test_sizing_accounts_for_stop_gap():
    # entry = 0.50, stop = 0.30.
    # Naive stop distance is 0.20 (250 shares), full loss is 0.50 (100 shares).
    # With stop slippage (default 0.5), effective stop = 0.15, risk_per_share = 0.35.
    rm = RiskManager(make_settings())
    market = make_market()
    sig = make_signal(entry_price=0.50, stop_price=0.30)

    shares = rm.size(sig, market, NOW)
    naive_shares = 50.0 / (0.50 - 0.30)
    full_loss_shares = 50.0 / 0.50

    assert full_loss_shares < shares < naive_shares
    assert shares == pytest.approx(50.0 / 0.35)


def test_daily_loss_limit_blocks_after_3r():
    rm = RiskManager(make_settings())
    market = make_market()

    # Lose 3R ($150) today
    rm.record_pnl(market, -150.0, NOW)
    assert rm.size(make_signal(), market, NOW) == 0.0

    # Next UTC day resets daily limit (weekly loss is only -3R < 7R)
    tomorrow = NOW + timedelta(days=1)
    assert rm.size(make_signal(), market, tomorrow) == pytest.approx(100.0)


def test_weekly_loss_limit_halves_size_after_7r():
    rm = RiskManager(make_settings())
    market = make_market()

    # Spread -7R (-$350) across previous days in the same 7-day window so daily limit isn't hit today
    rm.record_pnl(market, -100.0, NOW - timedelta(days=3))  # -2R
    rm.record_pnl(market, -150.0, NOW - timedelta(days=2))  # -3R
    rm.record_pnl(market, -100.0, NOW - timedelta(days=1))  # -2R

    shares = rm.size(make_signal(entry_price=0.50, stop_price=None), market, NOW)
    assert shares == pytest.approx(50.0)
    assert rm.state.open_risk_by_theme["ETH"] == pytest.approx(0.5)


def test_theme_limit_caps_at_3r_per_asset():
    rm = RiskManager(make_settings())

    for i in range(3):
        m = make_market(market_id=f"eth_{i}")
        s = make_signal(market_id=f"eth_{i}")
        assert rm.size(s, m, NOW) == pytest.approx(100.0)

    assert rm.state.open_risk_by_theme["ETH"] == pytest.approx(3.0)

    # 4th ETH market is blocked by 3R theme limit
    m4 = make_market(market_id="eth_4")
    s4 = make_signal(market_id="eth_4")
    assert rm.size(s4, m4, NOW) == 0.0

    # BTC is a separate theme and is still allowed
    m_btc = make_market(market_id="btc_1", asset=Asset.BTC)
    s_btc = make_signal(market_id="btc_1")
    assert rm.size(s_btc, m_btc, NOW) == pytest.approx(100.0)

    # Settling one ETH market releases 1R of ETH theme risk
    rm.record_pnl(make_market(market_id="eth_0"), 25.0, NOW)
    assert rm.state.open_risk_by_theme["ETH"] == pytest.approx(2.0)
    assert rm.size(s4, m4, NOW) == pytest.approx(100.0)


def test_max_two_attempts_per_setup():
    rm = RiskManager(make_settings())
    market = make_market(market_id="eth_setup")
    sig = make_signal(market_id="eth_setup", outcome=Outcome.YES)

    assert rm.size(sig, market, NOW) == pytest.approx(100.0)
    rm.record_pnl(market, -10.0, NOW)  # release theme risk

    assert rm.size(sig, market, NOW) == pytest.approx(100.0)
    rm.record_pnl(market, -10.0, NOW)

    # 3rd attempt on the same setup is blocked
    assert rm.size(sig, market, NOW) == 0.0


def test_volume_cap_1_percent_of_market_volume():
    rm = RiskManager(make_settings())
    # volume_usd = $1,000 -> 1% max notional = $10 -> at entry 0.50, max 20 shares (< 100 from 1R)
    thin_market = make_market(volume_usd=1_000.0)
    sig = make_signal(entry_price=0.50, stop_price=None)

    shares = rm.size(sig, thin_market, NOW)
    assert shares == pytest.approx(20.0)
    assert rm.state.open_risk_by_theme["ETH"] == pytest.approx(0.2)


def test_rejects_naive_datetime_and_invalid_inputs():
    rm = RiskManager(make_settings())
    market = make_market()
    sig = make_signal()

    with pytest.raises(ValueError, match="timezone-aware UTC"):
        rm.size(sig, market, datetime(2026, 10, 14, 12, 0))

    with pytest.raises(ValueError, match="timezone-aware UTC"):
        rm.record_pnl(market, 10.0, datetime(2026, 10, 14, 12, 0))

    assert rm.size(make_signal(edge_after_costs=0.0), market, NOW) == 0.0
    assert rm.size(make_signal(entry_price=0.40, stop_price=0.45), market, NOW) == 0.0


def test_settlement_releases_every_entry_of_the_market():
    rm = RiskManager(make_settings())
    market = make_market(market_id="eth_both")
    assert rm.size(make_signal(market_id="eth_both", outcome=Outcome.YES), market, NOW) > 0
    assert rm.size(make_signal(market_id="eth_both", outcome=Outcome.YES), market, NOW) > 0
    assert rm.size(make_signal(market_id="eth_both", outcome=Outcome.NO), market, NOW) > 0
    assert rm.state.open_risk_by_theme["ETH"] == pytest.approx(3.0)

    rm.record_pnl(market, 0.0, NOW)

    assert rm.state.open_risk_by_theme.get("ETH", 0.0) == pytest.approx(0.0)


def test_settling_an_untraded_market_keeps_open_risk():
    rm = RiskManager(make_settings())
    traded = make_market(market_id="eth_traded")
    assert rm.size(make_signal(market_id="eth_traded"), traded, NOW) > 0

    rm.record_pnl(make_market(market_id="eth_never_traded"), 0.0, NOW)

    assert rm.state.open_risk_by_theme["ETH"] == pytest.approx(1.0)
