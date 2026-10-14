"""Task E. Risk rules from part 2 of the strategy articles, measured in R (risk units).

Prediction-market prices gap through stops on news, so sizing must assume a
stop can fill far below stop_price; worst case for a long is losing the full entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from polybot.config import Settings
from polybot.models import CryptoMarket, Signal, _require_utc

DEFAULT_STOP_SLIPPAGE = 0.5
DEFAULT_MAX_ATTEMPTS_PER_SETUP = 2
_EPS = 1e-9


@dataclass(slots=True)
class RiskState:
    realized_pnl_by_day: dict[str, float] = field(default_factory=dict)  # "YYYY-MM-DD" -> USD
    open_risk_by_theme: dict[str, float] = field(default_factory=dict)  # theme -> R at risk
    attempts_by_setup: dict[str, int] = field(default_factory=dict)


class RiskManager:
    def __init__(
        self,
        settings: Settings,
        *,
        stop_slippage: float = DEFAULT_STOP_SLIPPAGE,
        max_attempts_per_setup: int = DEFAULT_MAX_ATTEMPTS_PER_SETUP,
    ) -> None:
        if not 0.0 <= stop_slippage <= 1.0:
            raise ValueError(f"stop_slippage must be in [0, 1], got {stop_slippage}")
        if max_attempts_per_setup < 1:
            raise ValueError(f"max_attempts_per_setup must be >= 1, got {max_attempts_per_setup}")
        self.settings = settings
        self.state = RiskState()
        self.stop_slippage = stop_slippage
        self.max_attempts_per_setup = max_attempts_per_setup
        self._open_risk_by_market: dict[str, list[float]] = {}

    def theme(self, market: CryptoMarket) -> str:
        """Correlated markets share a theme; v1: one theme per asset."""
        return market.asset.value

    def setup_key(self, signal: Signal, market: CryptoMarket) -> str:
        """Identifier for a trading setup (market + outcome)."""
        return f"{market.market_id}:{signal.outcome.value}"

    def daily_pnl_usd(self, now: datetime) -> float:
        _require_utc("now", now)
        return self.state.realized_pnl_by_day.get(now.strftime("%Y-%m-%d"), 0.0)

    def weekly_pnl_usd(self, now: datetime) -> float:
        _require_utc("now", now)
        today = now.date()
        return sum(
            self.state.realized_pnl_by_day.get((today - timedelta(days=i)).isoformat(), 0.0)
            for i in range(7)
        )

    def _risk_per_share(self, signal: Signal) -> float | None:
        """Per-share loss assumed for sizing, accounting for stop gap-through risk."""
        entry = signal.entry_price
        if entry <= 0.0 or entry >= 1.0:
            return None
        if signal.stop_price is None:
            return entry
        if signal.stop_price < 0.0 or signal.stop_price >= entry:
            return None
        effective_stop = signal.stop_price * (1.0 - self.stop_slippage)
        return entry - effective_stop

    def size(self, signal: Signal, market: CryptoMarket, now: datetime) -> float:
        """Shares allowed for this signal, 0 if any limit blocks it."""
        _require_utc("now", now)

        r_usd = self.settings.risk_unit_usd
        if r_usd <= 0.0 or signal.edge_after_costs <= 0.0 or market.volume_usd <= 0.0:
            return 0.0

        risk_per_share = self._risk_per_share(signal)
        if risk_per_share is None or risk_per_share <= 0.0:
            return 0.0

        # 1. Daily loss limit: -3R stops trading for the rest of the UTC day.
        if self.daily_pnl_usd(now) <= -self.settings.daily_loss_limit_r * r_usd + _EPS:
            return 0.0

        # 2. Setup attempts limit: max 2 attempts per setup.
        setup_id = self.setup_key(signal, market)
        attempts = max(
            self.state.attempts_by_setup.get(setup_id, 0),
            self.state.attempts_by_setup.get(market.market_id, 0),
        )
        if attempts >= self.max_attempts_per_setup:
            return 0.0

        # 3. Theme open-risk limit: max 3R concurrently at risk per theme (asset).
        theme = self.theme(market)
        open_theme_r = self.state.open_risk_by_theme.get(theme, 0.0)
        remaining_theme_r = self.settings.theme_limit_r - open_theme_r
        if remaining_theme_r <= _EPS:
            return 0.0

        # 4. Weekly loss limit: after -7R over the rolling 7-day window, halve position size.
        base_r = 1.0
        if self.weekly_pnl_usd(now) <= -self.settings.weekly_loss_limit_r * r_usd + _EPS:
            base_r = 0.5

        allowed_r = min(base_r, remaining_theme_r)
        shares_from_r = (allowed_r * r_usd) / risk_per_share

        # 5. Liquidity cap: position notional cannot exceed 1% of market volume.
        max_notional_usd = self.settings.max_share_of_volume * market.volume_usd
        max_shares_by_volume = max_notional_usd / signal.entry_price

        shares = min(shares_from_r, max_shares_by_volume)
        if shares <= 0.0:
            return 0.0

        actual_r = (shares * risk_per_share) / r_usd
        self.state.open_risk_by_theme[theme] = open_theme_r + actual_r
        self._open_risk_by_market.setdefault(market.market_id, []).append(actual_r)

        if (
            market.market_id in self.state.attempts_by_setup
            and setup_id not in self.state.attempts_by_setup
        ):
            self.state.attempts_by_setup[market.market_id] += 1
        else:
            self.state.attempts_by_setup[setup_id] = (
                self.state.attempts_by_setup.get(setup_id, 0) + 1
            )

        return shares

    def record_pnl(self, market: CryptoMarket, pnl_usd: float, now: datetime) -> None:
        _require_utc("now", now)
        day_key = now.strftime("%Y-%m-%d")
        self.state.realized_pnl_by_day[day_key] = (
            self.state.realized_pnl_by_day.get(day_key, 0.0) + pnl_usd
        )

        theme = self.theme(market)
        released_r = sum(self._open_risk_by_market.pop(market.market_id, []))
        if released_r <= 0.0:
            return

        current_r = self.state.open_risk_by_theme.get(theme, 0.0)
        new_r = max(0.0, current_r - released_r)
        if new_r > _EPS:
            self.state.open_risk_by_theme[theme] = new_r
        else:
            self.state.open_risk_by_theme.pop(theme, None)

