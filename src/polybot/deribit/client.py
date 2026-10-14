"""Task B. Read-only Deribit public API client.

Only public, unauthenticated endpoints are used. Nothing here can place an order.

The client also fetches the Binance spot price: Polymarket crypto markets resolve on a
Binance candle, while Deribit options settle on Deribit's own composite index. The two differ
by roughly 0.1% (USDT basis, exchange-specific noise), which is worth ~2 points of probability
for a daily at-the-money market, so the engine needs both numbers to correct for it.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import httpx

from polybot.models import Asset, OptionQuote

DERIBIT_URL = "https://www.deribit.com/api/v2"
# Binance's market-data-only host first (reachable from more regions), the main API as fallback.
BINANCE_URLS = ("https://data-api.binance.vision", "https://api.binance.com")

log = logging.getLogger(__name__)

_MONTHS = {
    name: number
    for number, name in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"),
        start=1,
    )
}
# "ETH-17OCT26-4400-C", "BTC-5NOV26-80000-P"; Deribit writes a decimal point in strikes as "d".
_INSTRUMENT_RE = re.compile(
    r"^(?P<base>[A-Z0-9]+)-(?P<day>\d{1,2})(?P<month>[A-Z]{3})(?P<year>\d{2})"
    r"-(?P<strike>\d+(?:d\d+)?)-(?P<kind>[CP])$"
)
_SETTLEMENT_HOUR_UTC = 8


class DeribitError(RuntimeError):
    """Deribit (or the Binance price source) answered with an error or unusable data."""


def parse_instrument_name(name: str) -> tuple[str, float, datetime, bool] | None:
    """`ETH-17OCT26-4400-C` -> ("ETH", 4400.0, 2026-10-17 08:00 UTC, True); None if not an option."""
    match = _INSTRUMENT_RE.match(name)
    if match is None:
        return None
    month = _MONTHS.get(match["month"])
    if month is None:
        return None
    try:
        expiry = datetime(
            2000 + int(match["year"]),
            month,
            int(match["day"]),
            _SETTLEMENT_HOUR_UTC,
            tzinfo=UTC,
        )
    except ValueError:  # e.g. 31FEB26
        return None
    strike = float(match["strike"].replace("d", "."))
    return match["base"], strike, expiry, match["kind"] == "C"


def parse_option_chain(
    summaries: Iterable[dict[str, Any]], asset: Asset, fallback_timestamp: datetime | None = None
) -> list[OptionQuote]:
    """Convert `get_book_summary_by_currency` rows into quotes, sorted by expiry and strike.

    Rows that cannot be used are dropped (never guessed): not an option on `asset`, no mark IV
    (Deribit reports it as null/0 for instruments without a fitted surface), no underlying price.
    Deribit reports IV in percent; `OptionQuote.mark_iv` is a fraction. The summary endpoint has
    no bid/ask IV, so those stay None. Call and put marks at one strike coincide on Deribit.
    """
    quotes: list[OptionQuote] = []
    dropped = 0
    for row in summaries:
        quote = _parse_summary(row, asset, fallback_timestamp)
        if quote is None:
            dropped += 1
        else:
            quotes.append(quote)
    if dropped:
        log.debug("dropped %d unusable %s option summaries", dropped, asset.value)
    quotes.sort(key=lambda q: (q.expiry, q.strike, not q.is_call))
    return quotes


def _parse_summary(
    row: dict[str, Any], asset: Asset, fallback_timestamp: datetime | None
) -> OptionQuote | None:
    name = row.get("instrument_name")
    parsed = parse_instrument_name(name) if isinstance(name, str) else None
    if parsed is None:
        return None
    base, strike, expiry, is_call = parsed
    if base != asset.value:
        return None

    mark_iv = row.get("mark_iv")
    underlying = row.get("underlying_price")
    if not _positive(mark_iv) or not _positive(underlying):
        return None

    created_ms = row.get("creation_timestamp")
    if _positive(created_ms):
        timestamp = datetime.fromtimestamp(created_ms / 1000.0, tz=UTC)
    elif fallback_timestamp is not None:
        timestamp = fallback_timestamp
    else:
        timestamp = datetime.now(UTC)

    return OptionQuote(
        instrument=name,
        asset=asset,
        strike=strike,
        expiry=expiry,
        is_call=is_call,
        mark_iv=mark_iv / 100.0,
        bid_iv=None,
        ask_iv=None,
        underlying_price=float(underlying),
        timestamp=timestamp,
    )


def _positive(value: Any) -> bool:
    is_number = isinstance(value, int | float) and not isinstance(value, bool)
    return is_number and math.isfinite(value) and value > 0


class DeribitClient:
    def __init__(self, http: httpx.Client | None = None) -> None:
        # A full option chain is ~1 MB: on a slow link it needs more than the usual 10 s.
        self._http = http or httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))

    def close(self) -> None:
        self._http.close()

    def fetch_option_chain(self, asset: Asset) -> list[OptionQuote]:
        """All live options for the asset (get_book_summary_by_currency, kind=option)."""
        payload = self._get_json(
            "get_book_summary_by_currency", {"currency": asset.value, "kind": "option"}
        )
        result = payload.get("result")
        if not isinstance(result, list):
            raise DeribitError("get_book_summary_by_currency: 'result' is not a list")
        server_time = payload.get("usOut")
        fallback = (
            datetime.fromtimestamp(server_time / 1e6, tz=UTC) if _positive(server_time) else None
        )
        return parse_option_chain(result, asset, fallback)

    def fetch_index_price(self, asset: Asset) -> float:
        """Deribit's composite USD index for the asset (what its options settle against)."""
        payload = self._get_json("get_index_price", {"index_name": f"{asset.value.lower()}_usd"})
        result = payload.get("result")
        price = result.get("index_price") if isinstance(result, dict) else None
        if not _positive(price):
            raise DeribitError(f"get_index_price: no usable index_price in {payload!r:.200}")
        return float(price)

    def fetch_binance_spot(self, asset: Asset) -> float:
        """Last Binance <ASSET>USDT trade price: the source most Polymarket crypto markets use."""
        symbol = f"{asset.value}USDT"
        problems: list[str] = []
        for base_url in BINANCE_URLS:
            try:
                response = self._http.get(
                    f"{base_url}/api/v3/ticker/price", params={"symbol": symbol}
                )
                response.raise_for_status()
                price = float(response.json()["price"])
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                problems.append(f"{base_url}: {exc!r}")
                continue
            if math.isfinite(price) and price > 0:
                return price
            problems.append(f"{base_url}: unusable price {price!r}")
        raise DeribitError(f"Binance {symbol} price unavailable ({'; '.join(problems)})")

    def fetch_resolution_basis(self, asset: Asset) -> float:
        """Binance spot / Deribit index - 1, e.g. 0.0011 when Binance trades 0.11% above Deribit.

        Pass it to `pricing.fair_probability(..., resolution_basis=...)`.
        """
        return self.fetch_binance_spot(asset) / self.fetch_index_price(asset) - 1.0

    def _get_json(self, method: str, params: dict[str, str]) -> dict[str, Any]:
        response = self._http.get(f"{DERIBIT_URL}/public/{method}", params=params)
        try:
            payload = response.json()
        except ValueError:
            response.raise_for_status()
            raise DeribitError(f"{method}: response is not JSON") from None
        # Deribit reports JSON-RPC errors in the body, usually with HTTP 400: keep its message.
        if isinstance(payload, dict) and payload.get("error"):
            error = payload["error"]
            raise DeribitError(f"{method}: {error!r}")
        response.raise_for_status()
        if not isinstance(payload, dict):
            raise DeribitError(f"{method}: unexpected payload {payload!r:.200}")
        return payload
