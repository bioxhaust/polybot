"""Task E. Risk rules from part 2 of the strategy articles, measured in R (risk units).

Prediction-market prices gap through stops on news, so sizing must assume a
stop can fill far below stop_price; worst case for a long is losing the full entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from polybot.config import Settings
from polybot.models import CryptoMarket, Signal


@dataclass(slots=True)
class RiskState:
    realized_pnl_by_day: dict[str, float] = field(default_factory=dict)  # "YYYY-MM-DD" -> USD
    open_risk_by_theme: dict[str, float] = field(default_factory=dict)  # theme -> R at risk
    attempts_by_setup: dict[str, int] = field(default_factory=dict)


class RiskManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = RiskState()

    def theme(self, market: CryptoMarket) -> str:
        """Correlated markets share a theme; v1: one theme per asset."""
        return market.asset.value

    def size(self, signal: Signal, market: CryptoMarket, now: datetime) -> float:
        """Shares allowed for this signal, 0 if any limit blocks it."""
        raise NotImplementedError

    def record_pnl(self, market: CryptoMarket, pnl_usd: float, now: datetime) -> None:
        raise NotImplementedError
