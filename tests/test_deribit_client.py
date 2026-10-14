import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from polybot.deribit.client import (
    BINANCE_URLS,
    DERIBIT_URL,
    DeribitClient,
    DeribitError,
    parse_instrument_name,
    parse_option_chain,
)
from polybot.models import Asset

FIXTURES = Path(__file__).parent / "fixtures" / "deribit"
SUMMARY_URL = f"{DERIBIT_URL}/public/get_book_summary_by_currency"
INDEX_URL = f"{DERIBIT_URL}/public/get_index_price"


def fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


def summary_row(**overrides):
    row = {
        "instrument_name": "ETH-17OCT26-2400-C",
        "base_currency": "ETH",
        "mark_iv": 48.49,
        "underlying_price": 2427.87,
        "creation_timestamp": 1791988167622,
        "mark_price": 0.0412,
        "bid_price": 0.04,
        "ask_price": 0.0425,
    }
    row.update(overrides)
    return row


# --- instrument names ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ETH-17OCT26-4400-C", ("ETH", 4400.0, datetime(2026, 10, 17, 8, tzinfo=UTC), True)),
        ("BTC-5NOV26-80000-P", ("BTC", 80000.0, datetime(2026, 11, 5, 8, tzinfo=UTC), False)),
        ("BTC-25DEC26-250000-C", ("BTC", 250000.0, datetime(2026, 12, 25, 8, tzinfo=UTC), True)),
        ("XRP-30JAN27-0d5-P", ("XRP", 0.5, datetime(2027, 1, 30, 8, tzinfo=UTC), False)),
    ],
)
def test_parse_instrument_name(name, expected):
    assert parse_instrument_name(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "",
        "ETH-PERPETUAL",
        "ETH-17OCT26",  # a future, not an option
        "ETH-17OCT26-4400",
        "ETH-17OCT26-4400-X",
        "ETH-17XXX26-4400-C",
        "ETH-31FEB26-4400-C",
        "ETH_USDC-17OCT26-4400-C",  # linear option: different contract, not what we price
        "eth-17oct26-4400-c",
    ],
)
def test_parse_instrument_name_rejects_non_options(name):
    assert parse_instrument_name(name) is None


# --- option chain -------------------------------------------------------------------


@pytest.mark.parametrize("asset", [Asset.ETH, Asset.BTC])
def test_fetch_option_chain_parses_real_response(respx_mock, asset):
    payload = fixture(f"book_summary_{asset.value.lower()}.json")
    route = respx_mock.get(SUMMARY_URL, params={"currency": asset.value, "kind": "option"}).mock(
        return_value=httpx.Response(200, json=payload)
    )

    chain = DeribitClient().fetch_option_chain(asset)

    assert route.call_count == 1
    assert len(chain) == len(payload["result"]) > 400
    assert [q.instrument for q in chain] == [
        q.instrument for q in sorted(chain, key=lambda q: (q.expiry, q.strike, not q.is_call))
    ]
    assert len({q.instrument for q in chain}) == len(chain)
    for quote in chain:
        assert quote.asset is asset
        assert quote.expiry.utcoffset().total_seconds() == 0
        assert (quote.expiry.hour, quote.expiry.minute) == (8, 0)  # Deribit settles at 08:00 UTC
        assert quote.timestamp.utcoffset().total_seconds() == 0
        assert quote.strike > 0 and quote.underlying_price > 0
        assert 0.05 < quote.mark_iv < 5.0  # a fraction, not percent
        assert quote.bid_iv is None and quote.ask_iv is None
    assert {q.is_call for q in chain} == {True, False}


def test_fetch_option_chain_converts_every_field(respx_mock):
    payload = fixture("book_summary_eth.json")
    respx_mock.get(SUMMARY_URL).mock(return_value=httpx.Response(200, json=payload))

    chain = {q.instrument: q for q in DeribitClient().fetch_option_chain(Asset.ETH)}

    for row in payload["result"]:
        quote = chain[row["instrument_name"]]
        assert quote.mark_iv == pytest.approx(row["mark_iv"] / 100.0)
        assert quote.underlying_price == row["underlying_price"]
        assert quote.timestamp == datetime.fromtimestamp(row["creation_timestamp"] / 1000, UTC)
    put = chain["ETH-18OCT26-2420-P"]
    assert (put.strike, put.is_call) == (2420.0, False)
    assert put.expiry == datetime(2026, 10, 18, 8, tzinfo=UTC)
    assert put.mark_iv == pytest.approx(0.4358)


def test_fetch_option_chain_drops_unusable_rows(respx_mock):
    rows = [
        summary_row(),
        summary_row(instrument_name="ETH-17OCT26-2500-C", mark_iv=None),
        summary_row(instrument_name="ETH-17OCT26-2600-C", mark_iv=0),
        summary_row(instrument_name="ETH-17OCT26-2700-C", underlying_price=None),
        summary_row(instrument_name="BTC-17OCT26-80000-C"),  # other asset
        summary_row(instrument_name="ETH-PERPETUAL"),
        summary_row(instrument_name=None),
        {"instrument_name": "ETH-17OCT26-2800-P"},  # nothing else
    ]
    respx_mock.get(SUMMARY_URL).mock(
        return_value=httpx.Response(200, json={"jsonrpc": "2.0", "result": rows, "usOut": 1e15})
    )

    chain = DeribitClient().fetch_option_chain(Asset.ETH)

    assert [q.instrument for q in chain] == ["ETH-17OCT26-2400-C"]


def test_missing_creation_timestamp_falls_back_to_server_time():
    quotes = parse_option_chain(
        [summary_row(creation_timestamp=None)],
        Asset.ETH,
        fallback_timestamp=datetime(2026, 10, 14, 12, tzinfo=UTC),
    )
    assert quotes[0].timestamp == datetime(2026, 10, 14, 12, tzinfo=UTC)


def test_fetch_option_chain_rejects_malformed_payload(respx_mock):
    respx_mock.get(SUMMARY_URL).mock(return_value=httpx.Response(200, json={"result": {"a": 1}}))
    with pytest.raises(DeribitError):
        DeribitClient().fetch_option_chain(Asset.ETH)


# --- errors -------------------------------------------------------------------------


def test_jsonrpc_error_keeps_deribit_message_even_on_http_400(respx_mock):
    respx_mock.get(INDEX_URL).mock(
        return_value=httpx.Response(400, json=fixture("index_price_error.json"))
    )
    with pytest.raises(DeribitError, match="Invalid params"):
        DeribitClient().fetch_index_price(Asset.ETH)


def test_jsonrpc_error_with_http_200(respx_mock):
    body = {"jsonrpc": "2.0", "error": {"code": 13009, "message": "too_many_requests"}}
    respx_mock.get(SUMMARY_URL).mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(DeribitError, match="too_many_requests"):
        DeribitClient().fetch_option_chain(Asset.BTC)


def test_http_error_without_json_body_propagates(respx_mock):
    respx_mock.get(SUMMARY_URL).mock(return_value=httpx.Response(503, text="upstream down"))
    with pytest.raises(httpx.HTTPStatusError):
        DeribitClient().fetch_option_chain(Asset.BTC)


def test_transport_errors_propagate(respx_mock):
    respx_mock.get(SUMMARY_URL).mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(httpx.ConnectTimeout):
        DeribitClient().fetch_option_chain(Asset.BTC)


def test_non_json_200_is_an_error(respx_mock):
    respx_mock.get(INDEX_URL).mock(
        return_value=httpx.Response(200, text="<html>maintenance</html>")
    )
    with pytest.raises(DeribitError):
        DeribitClient().fetch_index_price(Asset.BTC)


# --- index price --------------------------------------------------------------------


@pytest.mark.parametrize("asset", [Asset.BTC, Asset.ETH])
def test_fetch_index_price(respx_mock, asset):
    payload = fixture(f"index_price_{asset.value.lower()}.json")
    route = respx_mock.get(INDEX_URL, params={"index_name": f"{asset.value.lower()}_usd"}).mock(
        return_value=httpx.Response(200, json=payload)
    )
    price = DeribitClient().fetch_index_price(asset)
    assert route.call_count == 1
    assert price == payload["result"]["index_price"] > 0


@pytest.mark.parametrize("result", [{}, {"index_price": 0}, {"index_price": None}, None])
def test_fetch_index_price_rejects_unusable_result(respx_mock, result):
    respx_mock.get(INDEX_URL).mock(
        return_value=httpx.Response(200, json={"jsonrpc": "2.0", "result": result})
    )
    with pytest.raises(DeribitError):
        DeribitClient().fetch_index_price(Asset.ETH)


# --- Binance spot and the resolution basis -------------------------------------------


def binance_url(base: str) -> str:
    return f"{base}/api/v3/ticker/price"


def test_fetch_binance_spot_from_real_response(respx_mock):
    payload = fixture("binance_ticker_ethusdt.json")
    route = respx_mock.get(binance_url(BINANCE_URLS[0]), params={"symbol": "ETHUSDT"}).mock(
        return_value=httpx.Response(200, json=payload)
    )
    assert DeribitClient().fetch_binance_spot(Asset.ETH) == float(payload["price"])
    assert route.call_count == 1


def test_fetch_binance_spot_falls_back_to_second_host(respx_mock):
    respx_mock.get(binance_url(BINANCE_URLS[0])).mock(return_value=httpx.Response(451))
    respx_mock.get(binance_url(BINANCE_URLS[1]), params={"symbol": "BTCUSDT"}).mock(
        return_value=httpx.Response(200, json={"symbol": "BTCUSDT", "price": "81895.36000000"})
    )
    assert DeribitClient().fetch_binance_spot(Asset.BTC) == pytest.approx(81895.36)


def test_fetch_binance_spot_fails_loudly_when_every_host_fails(respx_mock):
    respx_mock.get(binance_url(BINANCE_URLS[0])).mock(side_effect=httpx.ConnectError("down"))
    respx_mock.get(binance_url(BINANCE_URLS[1])).mock(
        return_value=httpx.Response(200, json={"price": "0"})
    )
    with pytest.raises(DeribitError, match="BTCUSDT"):
        DeribitClient().fetch_binance_spot(Asset.BTC)


def test_fetch_resolution_basis(respx_mock):
    respx_mock.get(INDEX_URL, params={"index_name": "eth_usd"}).mock(
        return_value=httpx.Response(200, json=fixture("index_price_eth.json"))
    )
    respx_mock.get(binance_url(BINANCE_URLS[0]), params={"symbol": "ETHUSDT"}).mock(
        return_value=httpx.Response(200, json=fixture("binance_ticker_ethusdt.json"))
    )
    index = fixture("index_price_eth.json")["result"]["index_price"]
    binance = float(fixture("binance_ticker_ethusdt.json")["price"])

    basis = DeribitClient().fetch_resolution_basis(Asset.ETH)

    assert basis == pytest.approx(binance / index - 1.0)
    assert 0.0 < basis < 0.01  # Binance trades ~0.1% above Deribit's index in the snapshot


def test_client_accepts_injected_http_client(respx_mock):
    respx_mock.get(INDEX_URL).mock(
        return_value=httpx.Response(200, json=fixture("index_price_btc.json"))
    )
    with httpx.Client() as http:
        assert DeribitClient(http).fetch_index_price(Asset.BTC) > 0
