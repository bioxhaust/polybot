"""Task A: PolymarketClient against recorded real responses. No network: HTTP is mocked by respx.

Fixtures (captured from the live APIs on 2026-10-14, ~15:30Z):
* polymarket_gamma_events_btc.json / _eth.json - `GET /events` pages as the client asks for them.
  BTC holds the daily "above" event, the daily neg-risk "price" event, and an open "reach/dip"
  event that already contains two closed markets (trimmed to four markets).
* polymarket_clob_book.json / _books.json - `GET /book` and `POST /books` for the YES/NO tokens of
  "Will the price of Bitcoin be above $82,000 on October 14?". Levels arrive worst-first.
* polymarket_gamma_resolved.json - `GET /markets?condition_ids=..&closed=true` for two resolved
  markets: Bitcoin above $76,000 (YES) and above $84,000 (NO) on October 13.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

import polybot.polymarket.client as client_module
from polybot.models import Asset, MarketOutcome
from polybot.polymarket.client import (
    CLOB_URL,
    GAMMA_URL,
    OrderBookNotFound,
    PolymarketClient,
    PolymarketError,
    outcome_from_market,
    parse_order_book,
)
from polybot.polymarket.parser import parse_market

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str):
    return json.loads((FIXTURES / name).read_text())


BTC_EVENTS = load("polymarket_gamma_events_btc.json")
ETH_EVENTS = load("polymarket_gamma_events_eth.json")
BOOK = load("polymarket_clob_book.json")
BOOKS = load("polymarket_clob_books.json")
RESOLVED = load("polymarket_gamma_resolved.json")

NOW = datetime(2026, 10, 14, 15, 30, tzinfo=UTC)
YES_TOKEN, NO_TOKEN = BOOKS[0]["asset_id"], BOOKS[1]["asset_id"]
EVENTS_URL = f"{GAMMA_URL}/events"
MARKETS_URL = f"{GAMMA_URL}/markets"
BOOK_URL = f"{CLOB_URL}/book"
BOOKS_URL = f"{CLOB_URL}/books"


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def client(respx_mock, sleeps):
    with httpx.Client() as http:
        yield PolymarketClient(http, min_interval_s=0.0, sleep=sleeps.append)


def book_for(token_id: str) -> dict:
    return {**BOOK, "asset_id": token_id}


def serve_events(respx_mock, by_tag: dict[str, list]):
    """Answer GET /events from `by_tag`; returns the list collecting every request's params."""
    seen: list[httpx.QueryParams] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params)
        return httpx.Response(200, json=by_tag[request.url.params["tag_id"]])

    respx_mock.get(EVENTS_URL).mock(side_effect=handler)
    return seen


# --- fetch_crypto_markets -------------------------------------------------------------------


def test_each_asset_gets_one_filtered_query(respx_mock, client):
    seen = serve_events(respx_mock, {"235": BTC_EVENTS, "39": ETH_EVENTS})
    client.fetch_crypto_markets((Asset.BTC, Asset.ETH), now=NOW)

    assert [params["tag_id"] for params in seen] == ["235", "39"]  # bitcoin, ethereum
    for params in seen:
        assert params.get_list("exclude_tag_id") == ["102127", "102134"]  # up-or-down, hit-price
        assert (params["active"], params["closed"]) == ("true", "false")
        assert params["end_date_min"] == "2026-10-14T15:30:00Z"
        assert params["end_date_max"] == "2026-11-13T15:30:00Z"  # default horizon: 30 days
        assert (params["order"], params["ascending"]) == ("endDate", "true")
        assert (params["limit"], params["offset"]) == ("100", "0")


def test_horizon_is_configurable(respx_mock, client):
    seen = serve_events(respx_mock, {"235": []})
    assert client.fetch_crypto_markets((Asset.BTC,), now=NOW, max_days_ahead=2) == []
    assert seen[0]["end_date_max"] == "2026-10-16T15:30:00Z"


def test_markets_come_back_raw_with_their_event_attached(respx_mock, client):
    serve_events(respx_mock, {"235": BTC_EVENTS})
    raws = client.fetch_crypto_markets((Asset.BTC,), now=NOW)

    # 11 (above) + 11 (neg-risk price) + the 2 still-open reach markets; the 2 closed are gone
    assert len(raws) == 24
    questions = {raw["question"] for raw in raws}
    assert "Will Bitcoin reach $84,000 on October 14?" not in questions
    assert "Will Bitcoin dip to $83,000 on October 14?" not in questions
    assert {
        "Will Bitcoin reach $91,000 on October 14?",
        "Will Bitcoin reach $90,000 on October 14?",
    } <= questions
    assert len({raw["conditionId"] for raw in raws}) == 24

    for raw in raws:
        (event,) = raw["events"]
        assert "markets" not in event
        assert event["id"] and event["slug"]
        assert raw["clobTokenIds"] and raw["conditionId"]  # untouched Gamma fields


def test_each_market_points_at_its_own_event(respx_mock, client):
    serve_events(respx_mock, {"235": BTC_EVENTS})
    by_question = {
        raw["question"]: raw for raw in client.fetch_crypto_markets((Asset.BTC,), now=NOW)
    }
    assert (
        by_question["Will the price of Bitcoin be above $74,000 on October 14?"]["events"][0]["id"]
        == "1148918"
    )
    assert (
        by_question["Will the price of Bitcoin be less than $74,000 on October 14?"]["events"][0][
            "id"
        ]
        == "1148913"
    )


def test_the_events_payload_is_not_mutated(respx_mock, client):
    pristine = copy.deepcopy(BTC_EVENTS)
    serve_events(respx_mock, {"235": BTC_EVENTS})
    client.fetch_crypto_markets((Asset.BTC,), now=NOW)
    assert BTC_EVENTS == pristine


def test_untradable_markets_are_dropped(respx_mock, client):
    events = copy.deepcopy(BTC_EVENTS[:1])  # the daily "above" event, 11 markets
    markets = events[0]["markets"]
    markets[0]["closed"] = True
    markets[1]["active"] = False
    markets[2]["acceptingOrders"] = False
    markets[3]["enableOrderBook"] = False
    markets[4]["endDate"] = "2026-10-14T15:00:00Z"  # expired before NOW
    markets[5]["archived"] = True
    markets[6]["endDate"] = "2026-10-14T15:30:00Z"  # expires exactly now
    serve_events(respx_mock, {"235": events})

    kept = client.fetch_crypto_markets((Asset.BTC,), now=NOW)
    assert [raw["question"] for raw in kept] == [m["question"] for m in markets[7:]]


def test_unknown_flags_are_not_treated_as_untradable(respx_mock, client):
    events = copy.deepcopy(BTC_EVENTS[:1])
    for market in events[0]["markets"]:
        for flag in ("closed", "archived", "active", "acceptingOrders", "enableOrderBook"):
            market.pop(flag, None)
    serve_events(respx_mock, {"235": events})
    assert len(client.fetch_crypto_markets((Asset.BTC,), now=NOW)) == 11


def test_only_markets_about_the_requested_asset_are_returned(respx_mock, client):
    mixed = BTC_EVENTS + ETH_EVENTS  # an Ethereum event wrongly tagged "bitcoin"
    serve_events(respx_mock, {"235": mixed, "39": mixed})

    btc = client.fetch_crypto_markets((Asset.BTC,), now=NOW)
    assert len(btc) == 24 and all("Bitcoin" in raw["question"] for raw in btc)
    eth = client.fetch_crypto_markets((Asset.ETH,), now=NOW)
    assert len(eth) == 11 and all("Ethereum" in raw["question"] for raw in eth)


def test_junk_inside_an_event_is_skipped(respx_mock, client):
    events = copy.deepcopy(BTC_EVENTS[:1])
    events[0]["markets"] = ["junk", None, {"question": "no condition id"}, *events[0]["markets"]]
    serve_events(respx_mock, {"235": [*events, "not an event"]})
    assert len(client.fetch_crypto_markets((Asset.BTC,), now=NOW)) == 11


def test_a_market_seen_twice_is_returned_once(respx_mock, client):
    both = copy.deepcopy(BTC_EVENTS[:1])
    both[0]["markets"][0]["question"] = "Will ETH/BTC be above 0.03 on October 14?"  # mentions both
    serve_events(respx_mock, {"235": both, "39": both})
    raws = client.fetch_crypto_markets((Asset.BTC, Asset.ETH), now=NOW)
    questions = [raw["question"] for raw in raws]
    assert questions.count("Will ETH/BTC be above 0.03 on October 14?") == 1


def test_pages_are_followed_until_a_short_page(respx_mock, client, monkeypatch):
    monkeypatch.setattr(client_module, "EVENTS_PAGE_SIZE", 2)
    first, second, third = BTC_EVENTS
    offsets: list[str] = []
    pages = {"0": [first, second], "2": [second, third], "4": []}  # page 2 overlaps page 1

    def handler(request: httpx.Request) -> httpx.Response:
        offsets.append(request.url.params["offset"])
        assert request.url.params["limit"] == "2"
        return httpx.Response(200, json=pages[request.url.params["offset"]])

    respx_mock.get(EVENTS_URL).mock(side_effect=handler)
    raws = client.fetch_crypto_markets((Asset.BTC,), now=NOW)
    assert offsets == ["0", "2", "4"]
    assert len(raws) == 24 and len({raw["conditionId"] for raw in raws}) == 24


def test_a_short_first_page_is_the_last_page(respx_mock, client):
    seen = serve_events(respx_mock, {"235": BTC_EVENTS})
    client.fetch_crypto_markets((Asset.BTC,), now=NOW)
    assert len(seen) == 1


def test_fetched_markets_parse_end_to_end(respx_mock, client):
    serve_events(respx_mock, {"235": BTC_EVENTS, "39": ETH_EVENTS})
    parsed = [
        parse_market(raw) for raw in client.fetch_crypto_markets((Asset.BTC, Asset.ETH), now=NOW)
    ]
    markets = [market for market in parsed if market is not None]

    # 11 + 11 Bitcoin and 11 Ethereum point-in-time markets; the two reach markets are rejected
    assert len(markets) == 33 and parsed.count(None) == 2
    assert {market.asset for market in markets} == {Asset.BTC, Asset.ETH}
    assert {market.event_id for market in markets} == {"1148918", "1148913", "1148917"}


def test_asset_without_a_tag_is_an_error(respx_mock, client, monkeypatch):
    monkeypatch.delitem(client_module.ASSET_TAG_IDS, Asset.ETH)
    with pytest.raises(ValueError, match="no Polymarket tag"):
        client.fetch_crypto_markets((Asset.ETH,), now=NOW)


def test_naive_now_is_rejected(client):
    with pytest.raises(ValueError, match="timezone-aware"):
        client.fetch_crypto_markets((Asset.BTC,), now=datetime(2026, 10, 14, 15, 30))


def test_gamma_validation_error_is_raised_without_retries(respx_mock, client, sleeps):
    body = {"type": "validation error", "error": "offset too large"}
    route = respx_mock.get(EVENTS_URL).mock(return_value=httpx.Response(422, json=body))
    with pytest.raises(PolymarketError, match="422"):
        client.fetch_crypto_markets((Asset.BTC,), now=NOW)
    assert route.call_count == 1 and sleeps == []


@pytest.mark.parametrize("body", ['{"events": []}', '"oops"', "null", "42"])
def test_unexpected_gamma_payload_is_an_error(respx_mock, client, body):
    respx_mock.get(EVENTS_URL).mock(return_value=httpx.Response(200, content=body))
    with pytest.raises(PolymarketError, match="expected a JSON list"):
        client.fetch_crypto_markets((Asset.BTC,), now=NOW)


def test_non_json_reply_is_an_error(respx_mock, client):
    respx_mock.get(EVENTS_URL).mock(return_value=httpx.Response(200, text="<html>nope</html>"))
    with pytest.raises(PolymarketError, match="not JSON"):
        client.fetch_crypto_markets((Asset.BTC,), now=NOW)


# --- order books ----------------------------------------------------------------------------------


def test_real_clob_levels_really_arrive_worst_first():
    # the premise of everything below: the API does not send best-first
    bids = [float(level["price"]) for level in BOOK["bids"]]
    asks = [float(level["price"]) for level in BOOK["asks"]]
    assert bids == sorted(bids) and asks == sorted(asks, reverse=True)


def test_single_book_is_sorted_best_first(respx_mock, client):
    route = respx_mock.get(BOOK_URL).mock(return_value=httpx.Response(200, json=BOOK))
    book = client.fetch_order_book(BOOK["asset_id"])

    assert route.calls.last.request.url.params["token_id"] == BOOK["asset_id"]
    assert book.token_id == BOOK["asset_id"]
    assert (len(book.bids), len(book.asks)) == (36, 57)
    assert (book.best_bid, book.best_ask) == (0.176, 0.201)
    assert book.mid == pytest.approx(0.1885) and book.spread == pytest.approx(0.025)
    assert [level.price for level in book.bids] == sorted(
        (level.price for level in book.bids), reverse=True
    )
    assert [level.price for level in book.asks] == sorted(level.price for level in book.asks)
    assert (
        book.bids[0].size == 87.52 and book.asks[0].size == 62.0
    )  # sizes travel with their prices
    assert book.timestamp == datetime(2026, 10, 14, 15, 12, 35, 94000, tzinfo=UTC)


def test_book_content_is_preserved(respx_mock, client):
    respx_mock.get(BOOK_URL).mock(return_value=httpx.Response(200, json=BOOK))
    book = client.fetch_order_book(BOOK["asset_id"])
    sent_bids = sorted((float(x["price"]), float(x["size"])) for x in BOOK["bids"])
    sent_asks = sorted((float(x["price"]), float(x["size"])) for x in BOOK["asks"])
    assert sorted((level.price, level.size) for level in book.bids) == sent_bids
    assert sorted((level.price, level.size) for level in book.asks) == sent_asks


def test_missing_book_is_a_specific_error(respx_mock, client, sleeps):
    body = {"error": "No orderbook exists for the requested token id"}
    route = respx_mock.get(BOOK_URL).mock(return_value=httpx.Response(404, json=body))
    with pytest.raises(OrderBookNotFound):
        client.fetch_order_book("123")
    assert route.call_count == 1 and sleeps == []
    assert issubclass(OrderBookNotFound, PolymarketError)


def test_book_of_another_token_is_an_error(respx_mock, client):
    respx_mock.get(BOOK_URL).mock(return_value=httpx.Response(200, json=book_for("someone-else")))
    with pytest.raises(PolymarketError, match="malformed order book"):
        client.fetch_order_book("123")


@pytest.mark.parametrize(
    "payload", [[], {"asset_id": "123", "bids": [{"price": "2", "size": "1"}]}]
)
def test_malformed_single_book_is_an_error(respx_mock, client, payload):
    respx_mock.get(BOOK_URL).mock(return_value=httpx.Response(200, json=payload))
    with pytest.raises(PolymarketError, match="malformed order book"):
        client.fetch_order_book("123")


def test_batch_is_chunked_deduplicated_and_skips_missing(respx_mock, sleeps):
    requested: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tokens = [item["token_id"] for item in json.loads(request.content)]
        requested.append(tokens)
        return httpx.Response(200, json=[book_for(t) for t in tokens if t != "t4"])  # t4: no book

    respx_mock.post(BOOKS_URL).mock(side_effect=handler)
    with httpx.Client() as http:
        client = PolymarketClient(http, books_batch_size=2, min_interval_s=0.0, sleep=sleeps.append)
        books = client.fetch_order_books(["t1", "t2", "t2", "t3", "t4", "t5", "t1"])

    assert requested == [["t1", "t2"], ["t3", "t4"], ["t5"]]
    assert set(books) == {"t1", "t2", "t3", "t5"}
    assert all(book.best_bid == 0.176 and book.best_ask == 0.201 for book in books.values())


def test_batch_of_real_books(respx_mock, client):
    route = respx_mock.post(BOOKS_URL).mock(return_value=httpx.Response(200, json=BOOKS))
    books = client.fetch_order_books([YES_TOKEN, NO_TOKEN])
    assert json.loads(route.calls.last.request.content) == [
        {"token_id": YES_TOKEN},
        {"token_id": NO_TOKEN},
    ]
    assert set(books) == {YES_TOKEN, NO_TOKEN}
    for book in books.values():
        assert book.bids == tuple(sorted(book.bids, key=lambda level: -level.price))
        assert book.asks == tuple(sorted(book.asks, key=lambda level: level.price))
        assert book.best_bid < book.best_ask


def test_empty_batch_makes_no_request(client):
    assert client.fetch_order_books([]) == {}  # respx would fail the test on any request


def test_batch_ignores_malformed_and_unrequested_books(respx_mock, client):
    payload = [
        book_for("t1"),
        {**book_for("t2"), "bids": [{"price": "1.5", "size": "10"}]},  # price outside [0, 1]
        {"asset_id": "t3", "bids": "oops"},
        {"bids": []},  # no asset_id
        book_for("not-asked-for"),
        "garbage",
    ]
    respx_mock.post(BOOKS_URL).mock(return_value=httpx.Response(200, json=payload))
    assert set(client.fetch_order_books(["t1", "t2", "t3"])) == {"t1"}


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        PolymarketClient(books_batch_size=0)


# --- parse_order_book ---------------------------------------------------------------------------


def test_zero_size_levels_are_dropped():
    raw = {
        "asset_id": "t",
        "bids": [{"price": "0.40", "size": "0"}, {"price": "0.39", "size": "5"}],
        "asks": [{"price": "0.42", "size": "3"}, {"price": "0.43", "size": "0.0"}],
        "timestamp": "1791990755094",
    }
    book = parse_order_book(raw)
    assert [(level.price, level.size) for level in book.bids] == [(0.39, 5.0)]
    assert [(level.price, level.size) for level in book.asks] == [(0.42, 3.0)]


def test_empty_sides_give_an_empty_book():
    book = parse_order_book(
        {"asset_id": "t", "bids": [], "asks": None, "timestamp": "1791990755094"}
    )
    assert book.bids == () and book.asks == () and book.mid is None


def test_missing_timestamp_falls_back_to_receive_time():
    received = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    for stamp in ({}, {"timestamp": None}, {"timestamp": "soon"}):
        book = parse_order_book(
            {"asset_id": "t", "bids": [], "asks": [], **stamp}, received_at=received
        )
        assert book.timestamp == received


def test_asset_id_falls_back_to_the_requested_token_and_must_match():
    assert parse_order_book({"bids": [], "asks": []}, token_id="t").token_id == "t"
    with pytest.raises(KeyError):
        parse_order_book({"bids": [], "asks": []})
    with pytest.raises(ValueError, match="got the book of other"):
        parse_order_book({"asset_id": "other"}, token_id="t")


# --- fetch_outcome -----------------------------------------------------------------------------------


def serve_resolved(respx_mock, rows):
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        assert params["closed"] == "true"  # Gamma hides closed markets unless asked
        wanted = params["condition_ids"].lower()
        return httpx.Response(200, json=[r for r in rows if r["conditionId"].lower() == wanted])

    return respx_mock.get(MARKETS_URL).mock(side_effect=handler)


def test_resolved_yes_and_no_markets(respx_mock, client):
    serve_resolved(respx_mock, RESOLVED)
    resolved_at = datetime(2026, 10, 13, 16, 12, 15, tzinfo=UTC)

    yes_market, no_market = RESOLVED
    assert yes_market["question"].endswith("$76,000 on October 13?")
    assert client.fetch_outcome(yes_market["conditionId"]) == MarketOutcome(
        yes_market["conditionId"], True, resolved_at
    )
    assert client.fetch_outcome(no_market["conditionId"]) == MarketOutcome(
        no_market["conditionId"], False, resolved_at
    )


def test_unresolved_or_unknown_market_is_none(respx_mock, client):
    serve_resolved(respx_mock, [])
    assert client.fetch_outcome("0x" + "ab" * 32) is None


def test_condition_id_is_matched_case_insensitively(respx_mock, client):
    serve_resolved(respx_mock, RESOLVED)
    outcome = client.fetch_outcome(RESOLVED[0]["conditionId"].upper().replace("0X", "0x"))
    assert outcome is not None and outcome.resolved_yes is True


def test_a_different_market_in_the_reply_is_ignored(respx_mock, client):
    respx_mock.get(MARKETS_URL).mock(return_value=httpx.Response(200, json=RESOLVED))
    assert client.fetch_outcome("0x" + "cd" * 32) is None


def resolved(**changes) -> dict:
    return {**copy.deepcopy(RESOLVED[0]), **changes}


def test_outcome_from_market_requires_a_closed_resolved_market():
    assert outcome_from_market(resolved()).resolved_yes is True
    assert outcome_from_market(resolved(outcomePrices='["0", "1"]')).resolved_yes is False
    assert outcome_from_market(resolved(closed=False)) is None
    assert outcome_from_market(resolved(closed=None)) is None
    assert outcome_from_market(resolved(umaResolutionStatus="proposed")) is None
    assert outcome_from_market(resolved(umaResolutionStatus="disputed")) is None
    assert outcome_from_market(resolved(umaResolutionStatus=None)).resolved_yes is True


@pytest.mark.parametrize(
    "prices",
    [
        '["0.5", "0.5"]',
        '["0.6", "0.4"]',
        '["0.9995", "0.0005"]',
        '["1", "1"]',
        '["0", "0"]',
        '["1"]',
        "nope",
    ],
)
def test_voided_or_not_final_prices_are_none(prices):
    assert outcome_from_market(resolved(outcomePrices=prices)) is None


def test_outcome_is_matched_by_label_not_position():
    flipped = resolved(outcomes='["No", "Yes"]', outcomePrices='["1", "0"]')
    assert outcome_from_market(flipped).resolved_yes is False
    assert outcome_from_market(resolved(outcomes='["Up", "Down"]')) is None


def test_resolved_at_falls_back_to_uma_time_then_end_date():
    both_gone = {"closedTime": None, "umaEndDate": None}
    assert outcome_from_market(
        resolved(closedTime=None, umaEndDate="2026-10-13T16:30:00Z")
    ).resolved_at == (datetime(2026, 10, 13, 16, 30, tzinfo=UTC))
    assert outcome_from_market(resolved(**both_gone)).resolved_at == datetime(
        2026, 10, 13, 16, 0, tzinfo=UTC
    )
    assert outcome_from_market(resolved(**both_gone, endDate=None)) is None


@pytest.mark.parametrize("bad_id", [None, "", 5])
def test_outcome_needs_a_condition_id(bad_id):
    assert outcome_from_market(resolved(conditionId=bad_id)) is None


# --- retries, back-off, rate limits ------------------------------------------------------------


def test_429_waits_for_retry_after_then_succeeds(respx_mock, client, sleeps):
    route = respx_mock.get(BOOK_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(200, json=BOOK),
        ]
    )
    assert client.fetch_order_book(BOOK["asset_id"]).best_bid == 0.176
    assert route.call_count == 2 and sleeps == [2.0]


@pytest.mark.parametrize(
    ("header", "expected_wait"),
    [("3600", 60.0), ("-5", 0.0), ("soon", 0.5), (None, 0.5)],
    ids=["capped", "negative", "not-a-number", "absent"],
)
def test_retry_after_is_sanitised(respx_mock, client, sleeps, header, expected_wait):
    headers = {} if header is None else {"Retry-After": header}
    respx_mock.get(BOOK_URL).mock(
        side_effect=[httpx.Response(429, headers=headers), httpx.Response(200, json=BOOK)]
    )
    client.fetch_order_book(BOOK["asset_id"])
    assert sleeps == [expected_wait]


def test_server_errors_back_off_exponentially(respx_mock, client, sleeps):
    route = respx_mock.get(BOOK_URL).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(502),
            httpx.Response(503),
            httpx.Response(200, json=BOOK),
        ]
    )
    client.fetch_order_book(BOOK["asset_id"])
    assert route.call_count == 4 and sleeps == [0.5, 1.0, 2.0]


def test_gives_up_after_max_retries(respx_mock, client, sleeps):
    route = respx_mock.get(BOOK_URL).mock(return_value=httpx.Response(503))
    with pytest.raises(PolymarketError, match="failed after 4 tries: HTTP 503"):
        client.fetch_order_book("123")
    assert route.call_count == 4 and sleeps == [0.5, 1.0, 2.0]


def test_retries_can_be_turned_off(respx_mock, sleeps):
    route = respx_mock.get(BOOK_URL).mock(return_value=httpx.Response(500))
    with httpx.Client() as http:
        client = PolymarketClient(http, max_retries=0, min_interval_s=0.0, sleep=sleeps.append)
        with pytest.raises(PolymarketError):
            client.fetch_order_book("123")
    assert route.call_count == 1 and sleeps == []


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_client_errors_are_not_retried(respx_mock, client, sleeps, status):
    route = respx_mock.get(BOOK_URL).mock(
        return_value=httpx.Response(status, json={"error": "nope"})
    )
    with pytest.raises(PolymarketError, match=str(status)):
        client.fetch_order_book("123")
    assert route.call_count == 1 and sleeps == []


@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), httpx.RemoteProtocolError("cut")],
)
def test_network_errors_are_retried(respx_mock, client, sleeps, failure):
    route = respx_mock.get(BOOK_URL).mock(side_effect=[failure, httpx.Response(200, json=BOOK)])
    assert client.fetch_order_book(BOOK["asset_id"]).best_ask == 0.201
    assert route.call_count == 2 and sleeps == [0.5]


def test_persistent_network_errors_surface_as_polymarket_error(respx_mock, client):
    respx_mock.get(BOOK_URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(PolymarketError, match="ConnectError"):
        client.fetch_order_book("123")


def test_post_body_is_resent_on_retry(respx_mock, client):
    route = respx_mock.post(BOOKS_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json=BOOKS),
        ]
    )
    assert len(client.fetch_order_books([YES_TOKEN, NO_TOKEN])) == 2
    bodies = [json.loads(call.request.content) for call in route.calls]
    assert bodies[0] == bodies[1] == [{"token_id": YES_TOKEN}, {"token_id": NO_TOKEN}]


def test_requests_are_spaced_by_min_interval(respx_mock):
    respx_mock.get(BOOK_URL).mock(return_value=httpx.Response(200, json=BOOK))
    clock = [100.0]
    waited: list[float] = []

    def sleep(seconds: float) -> None:
        waited.append(seconds)
        clock[0] += seconds

    with httpx.Client() as http:
        client = PolymarketClient(http, min_interval_s=1.0, sleep=sleep, monotonic=lambda: clock[0])
        client.fetch_order_book(BOOK["asset_id"])  # first request goes out at once
        assert waited == []
        client.fetch_order_book(BOOK["asset_id"])  # immediately again: must wait a full interval
        assert waited == [1.0]
        clock[0] += 5.0  # long pause: no extra waiting
        client.fetch_order_book(BOOK["asset_id"])
        assert waited == [1.0]
        clock[0] += 0.25
        client.fetch_order_book(BOOK["asset_id"])
        assert waited == [1.0, 0.75]


def test_batches_are_spaced_too(respx_mock):
    respx_mock.post(BOOKS_URL).mock(side_effect=lambda request: httpx.Response(200, json=[]))
    clock = [0.0]
    waited: list[float] = []
    with httpx.Client() as http:
        client = PolymarketClient(
            http,
            books_batch_size=1,
            min_interval_s=0.5,
            sleep=lambda s: (waited.append(s), clock.__setitem__(0, clock[0] + s)),
            monotonic=lambda: clock[0],
        )
        client.fetch_order_books(["a", "b", "c"])
    assert waited == [0.5, 0.5]


def test_owned_http_client_is_closed_but_a_borrowed_one_is_not():
    with PolymarketClient() as owned:
        assert not owned._http.is_closed
    assert owned._http.is_closed

    with httpx.Client() as borrowed_http:
        PolymarketClient(borrowed_http).close()
        assert not borrowed_http.is_closed
