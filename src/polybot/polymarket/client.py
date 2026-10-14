"""Task A. Read-only Polymarket client (Gamma API for markets, CLOB API for books)."""

from __future__ import annotations

from typing import Any

import httpx

from polybot.models import Asset, MarketOutcome, OrderBook

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


class PolymarketClient:
    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(timeout=10.0)

    def fetch_crypto_markets(self, assets: tuple[Asset, ...]) -> list[dict[str, Any]]:
        """Active, not yet resolved price markets for the given assets, as raw Gamma JSON."""
        raise NotImplementedError

    def fetch_order_book(self, token_id: str) -> OrderBook:
        raise NotImplementedError

    def fetch_order_books(self, token_ids: list[str]) -> dict[str, OrderBook]:
        """Batch variant; must respect CLOB rate limits."""
        raise NotImplementedError

    def fetch_outcome(self, market_id: str) -> MarketOutcome | None:
        """None while the market is unresolved."""
        raise NotImplementedError
