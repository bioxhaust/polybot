"""Task A. Read-only Polymarket client (Gamma API for markets, CLOB API for books).

Facts about the live APIs this module is built around (checked against them, not assumed):

* Gamma `/events` can exclude tags (`exclude_tag_id`), `/markets` silently ignores that filter.
  So markets are discovered through events and every market gets its event attached as
  `raw["events"] == [event]`, exactly the shape `/markets` returns and `parse_market` reads.
* Gamma pages hold at most 100 rows and refuse offsets above ~2100.
* Events marked open can still contain closed markets (e.g. a "reach" market that already hit).
* `/markets?condition_ids=...` only returns *closed* markets when asked with `closed=true`.
* CLOB books arrive worst-first: bids ascending, asks descending. `OrderBook` wants best-first.
* `POST /books` silently omits tokens that have no book; `GET /book` answers 404 for them.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, Self

import httpx

from polybot.models import Asset, MarketOutcome, OrderBook, PriceLevel
from polybot.polymarket.parser import parse_json_list, parse_timestamp

log = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

# Gamma tag ids (stable database ids, see GET /tags/slug/<slug>).
ASSET_TAG_IDS: dict[Asset, int] = {Asset.BTC: 235, Asset.ETH: 39}  # "bitcoin", "ethereum"
# "up-or-down" (relative move, no strike) and "hit-price" (path-dependent) can never be parsed.
# Asking the API to drop them shrinks a Bitcoin poll from ~450 events to a dozen.
EXCLUDED_TAG_IDS: tuple[int, ...] = (102127, 102134)

_ASSET_WORDS: dict[Asset, re.Pattern[str]] = {
    Asset.BTC: re.compile(r"\b(?:bitcoin|btc)\b", re.IGNORECASE),
    Asset.ETH: re.compile(r"\b(?:ethereum|eth)\b", re.IGNORECASE),
}

EVENTS_PAGE_SIZE = 100  # Gamma's hard cap per page
MAX_EVENT_PAGES = 20  # Gamma refuses offsets above ~2100 anyway
BOOKS_BATCH_SIZE = 100  # tokens per POST /books (the API accepts more; this keeps replies small)
DEFAULT_HORIZON_DAYS = 30.0
_MAX_RETRY_AFTER_S = 60.0


class PolymarketError(RuntimeError):
    """The API kept failing, answered with a client error, or sent something unexpected."""


class OrderBookNotFound(PolymarketError):
    """CLOB has no book for this token (market not open yet, already settled, or a bad id)."""


# --- pure converters (exposed so they can be tested without HTTP) ----------------------


def _levels(raw_levels: Any) -> list[PriceLevel]:
    levels = [PriceLevel(float(item["price"]), float(item["size"])) for item in raw_levels or []]
    return [level for level in levels if level.size > 0]


def _book_time(raw_ms: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(int(raw_ms) / 1000, tz=UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def parse_order_book(
    raw: dict[str, Any], *, token_id: str | None = None, received_at: datetime | None = None
) -> OrderBook:
    """CLOB book JSON -> OrderBook with bids best-first (descending) and asks ascending.

    The book's own `timestamp` (epoch ms) is used; `received_at` (default: now) only fills in
    when the API omits it. Raises KeyError/ValueError/TypeError on a malformed payload.
    """
    book_token = str(raw["asset_id"]) if raw.get("asset_id") is not None else token_id
    if book_token is None:
        raise KeyError("asset_id")
    if token_id is not None and book_token != token_id:
        raise ValueError(f"asked for token {token_id}, got the book of {book_token}")
    bids = sorted(_levels(raw.get("bids")), key=lambda level: level.price, reverse=True)
    asks = sorted(_levels(raw.get("asks")), key=lambda level: level.price)
    timestamp = _book_time(raw.get("timestamp")) or received_at or datetime.now(UTC)
    return OrderBook(book_token, tuple(bids), tuple(asks), timestamp)


def outcome_from_market(raw: dict[str, Any]) -> MarketOutcome | None:
    """Gamma market JSON -> MarketOutcome, or None unless it is closed with a clean YES/NO result.

    A market voided 50/50 (or anything else that is not a plain win for one side) is also None:
    `MarketOutcome.resolved_yes` cannot express it. `resolved_at` is the time Gamma closed the
    market; if that is missing we fall back to the UMA end time, then to the market's endDate.
    """
    market_id = raw.get("conditionId")
    if not isinstance(market_id, str) or not market_id or raw.get("closed") is not True:
        return None
    status = raw.get("umaResolutionStatus")
    if status is not None and status != "resolved":
        return None

    outcomes = parse_json_list(raw.get("outcomes"))
    prices = parse_json_list(raw.get("outcomePrices"))
    if outcomes is None or prices is None or len(outcomes) != len(prices):
        return None
    labels = [str(label).strip().lower() for label in outcomes]
    if sorted(labels) != ["no", "yes"]:
        return None
    try:
        yes_price = float(prices[labels.index("yes")])
        no_price = float(prices[labels.index("no")])
    except (TypeError, ValueError):
        return None
    if yes_price == 1.0 and no_price == 0.0:
        resolved_yes = True
    elif yes_price == 0.0 and no_price == 1.0:
        resolved_yes = False
    else:
        return None

    resolved_at = (
        parse_timestamp(raw.get("closedTime"))
        or parse_timestamp(raw.get("umaEndDate"))
        or parse_timestamp(raw.get("endDate"))
    )
    if resolved_at is None:
        return None
    return MarketOutcome(market_id, resolved_yes, resolved_at)


def _is_tradable(market: dict[str, Any], now: datetime) -> bool:
    """Open, accepting orders on the CLOB, and not past its expiry."""
    if market.get("closed") or market.get("archived"):
        return False
    if market.get("active") is False or market.get("enableOrderBook") is False:
        return False
    if market.get("acceptingOrders") is False:
        return False
    end_date = parse_timestamp(market.get("endDate"))
    return end_date is None or end_date > now


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _retry_after(response: httpx.Response) -> float | None:
    try:
        seconds = float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return None
    return max(0.0, min(seconds, _MAX_RETRY_AFTER_S))


# --- client ---------------------------------------------------------------------------


class PolymarketClient:
    """Thin read-only client. No keys, no order placement.

    Every request is spaced `min_interval_s` apart from the previous one and retried with
    exponential backoff on network errors, HTTP 429 (honouring `Retry-After`) and 5xx. Other
    4xx answers are not retried. After `max_retries` retries a `PolymarketError` is raised, so
    the caller (the main loop) decides how long to back off further.
    """

    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        min_interval_s: float = 0.05,
        books_batch_size: int = BOOKS_BATCH_SIZE,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if books_batch_size < 1:
            raise ValueError("books_batch_size must be >= 1")
        self._owns_http = http is None
        self._http = http or httpx.Client(timeout=10.0)
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._min_interval_s = min_interval_s
        self._books_batch_size = books_batch_size
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_at: float | None = None

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- transport --

    def _wait_turn(self) -> None:
        if self._last_request_at is not None and self._min_interval_s > 0:
            wait = self._last_request_at + self._min_interval_s - self._monotonic()
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._monotonic()

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """One logical request. Returns any non-retryable response (2xx and 4xx) as is."""
        attempt = 0
        while True:
            self._wait_turn()
            retry_after: float | None = None
            try:
                response = self._http.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                reason = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code != 429 and response.status_code < 500:
                    return response
                reason = f"HTTP {response.status_code}"
                retry_after = _retry_after(response)
            if attempt >= self._max_retries:
                raise PolymarketError(f"{method} {url} failed after {attempt + 1} tries: {reason}")
            delay = retry_after if retry_after is not None else self._backoff_s * 2**attempt
            log.warning("%s %s: %s; retry %d in %.2fs", method, url, reason, attempt + 1, delay)
            self._sleep(delay)
            attempt += 1

    def _json(self, method: str, url: str, **kwargs: Any) -> Any:
        response = self._send(method, url, **kwargs)
        if response.status_code >= 400:
            raise PolymarketError(
                f"{method} {url}: HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise PolymarketError(f"{method} {url}: reply is not JSON") from exc

    def _json_list(self, method: str, url: str, **kwargs: Any) -> list[Any]:
        payload = self._json(method, url, **kwargs)
        if not isinstance(payload, list):
            raise PolymarketError(
                f"{method} {url}: expected a JSON list, got {type(payload).__name__}"
            )
        return payload

    # -- markets (Gamma) --

    def _iter_events(self, tag_id: int, now: datetime, until: datetime) -> Iterator[dict[str, Any]]:
        for page in range(MAX_EVENT_PAGES):
            params: dict[str, Any] = {
                "tag_id": tag_id,
                "exclude_tag_id": list(EXCLUDED_TAG_IDS),
                "active": "true",
                "closed": "false",
                "end_date_min": _iso(now),
                "end_date_max": _iso(until),
                "order": "endDate",
                "ascending": "true",
                "limit": EVENTS_PAGE_SIZE,
                "offset": page * EVENTS_PAGE_SIZE,
            }
            events = self._json_list("GET", f"{GAMMA_URL}/events", params=params)
            yield from (event for event in events if isinstance(event, dict))
            if len(events) < EVENTS_PAGE_SIZE:
                return

    def fetch_crypto_markets(
        self,
        assets: tuple[Asset, ...],
        *,
        max_days_ahead: float = DEFAULT_HORIZON_DAYS,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Active, not yet resolved price markets for the given assets, as raw Gamma JSON.

        Only markets that can be traded right now are returned: open, accepting orders, with an
        order book, expiring after `now` and within `max_days_ahead`. "Up or Down" and
        "hit price" events are excluded by the API; everything else tagged with the asset comes
        back and `parse_market` decides what it is. Each item is a market dict with its event
        attached as `events == [event_without_markets]`.

        The list changes slowly (daily and hourly series appear a day or a week ahead); refresh
        it every few minutes, not on every tick.
        """
        now = now or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        until = now + timedelta(days=max_days_ahead)

        found: dict[str, dict[str, Any]] = {}
        for asset in assets:
            if asset not in ASSET_TAG_IDS:
                raise ValueError(f"no Polymarket tag configured for {asset!r}")
            mentions_asset = _ASSET_WORDS[asset]
            for event in self._iter_events(ASSET_TAG_IDS[asset], now, until):
                shell = {key: value for key, value in event.items() if key != "markets"}
                for market in event.get("markets") or []:
                    if not isinstance(market, dict):
                        continue
                    condition_id = market.get("conditionId")
                    if not condition_id or condition_id in found:
                        continue
                    if not mentions_asset.search(str(market.get("question") or "")):
                        continue
                    if _is_tradable(market, now):
                        found[condition_id] = {**market, "events": [shell]}
        return list(found.values())

    def fetch_outcome(self, market_id: str) -> MarketOutcome | None:
        """None while the market is unresolved (or if it was voided rather than won)."""
        rows = self._json_list(
            "GET", f"{GAMMA_URL}/markets", params={"condition_ids": market_id, "closed": "true"}
        )
        for row in rows:
            if (
                isinstance(row, dict)
                and str(row.get("conditionId", "")).lower() == market_id.lower()
            ):
                return outcome_from_market(row)
        return None  # Gamma lists a market under closed=true only once it is closed

    # -- order books (CLOB) --

    def fetch_order_book(self, token_id: str) -> OrderBook:
        response = self._send("GET", f"{CLOB_URL}/book", params={"token_id": token_id})
        if response.status_code == 404:
            raise OrderBookNotFound(f"no order book for token {token_id}")
        if response.status_code >= 400:
            raise PolymarketError(f"GET /book: HTTP {response.status_code}: {response.text[:200]}")
        try:
            return parse_order_book(response.json(), token_id=token_id)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise PolymarketError(f"malformed order book for token {token_id}: {exc!r}") from exc

    def fetch_order_books(self, token_ids: list[str]) -> dict[str, OrderBook]:
        """Batch variant; must respect CLOB rate limits.

        Tokens are de-duplicated and sent `books_batch_size` at a time. Tokens without a book
        (or with a malformed one) are simply missing from the result.
        """
        unique = list(dict.fromkeys(token_ids))
        books: dict[str, OrderBook] = {}
        for start in range(0, len(unique), self._books_batch_size):
            chunk = unique[start : start + self._books_batch_size]
            wanted = set(chunk)
            payload = self._json_list(
                "POST", f"{CLOB_URL}/books", json=[{"token_id": token} for token in chunk]
            )
            received_at = datetime.now(UTC)
            for raw in payload:
                try:
                    book = parse_order_book(raw, received_at=received_at)
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    log.warning("skipping malformed order book: %r", exc)
                    continue
                if book.token_id in wanted:
                    books[book.token_id] = book
        return books
