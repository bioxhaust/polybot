"""Task B. Read-only Deribit public API client."""

from __future__ import annotations

import httpx

from polybot.models import Asset, OptionQuote

DERIBIT_URL = "https://www.deribit.com/api/v2"


class DeribitClient:
    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(timeout=10.0)

    def fetch_option_chain(self, asset: Asset) -> list[OptionQuote]:
        """All live options for the asset (get_book_summary_by_currency, kind=option)."""
        raise NotImplementedError

    def fetch_index_price(self, asset: Asset) -> float:
        raise NotImplementedError
