"""Task G. Public wallet trades and per-wallet performance stats."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import httpx

from polybot.models import WalletTrade

DATA_API_URL = "https://data-api.polymarket.com"


@dataclass(frozen=True, slots=True)
class WalletStats:
    wallet: str
    n_trades: int
    n_markets: int
    realized_pnl_usd: float
    max_drawdown_usd: float
    active_days: int
    share_crypto_markets: float  # fraction of volume in crypto price markets
    share_maker: float | None  # fraction of fills as maker, if derivable
    n_merges: int


class WalletClient:
    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(timeout=10.0)

    def fetch_trades(self, wallet: str, since: datetime | None = None) -> list[WalletTrade]:
        raise NotImplementedError

    def fetch_market_trades(self, market_id: str, since: datetime | None = None) -> list[WalletTrade]:
        """All public trades in one market, used to find active wallets."""
        raise NotImplementedError


def compute_stats(wallet: str, trades: list[WalletTrade]) -> WalletStats:
    raise NotImplementedError
