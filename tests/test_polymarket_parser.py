"""Task A: parse_market on real Gamma markets.

`fixtures/polymarket_gamma_markets.json` holds 45 markets captured from the live Gamma API on
2026-10-14, each with its event attached as `events == [event]` (the shape
`PolymarketClient.fetch_crypto_markets` returns). Expected values below are written out by hand,
including expiries in UTC (noon ET on 14 Oct 2026 is EDT, UTC-4, hence 16:00Z).
"""

from __future__ import annotations

import copy
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from polybot.models import Asset, MarketKind
from polybot.polymarket.parser import parse_json_list, parse_market, parse_timestamp

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str):
    return json.loads((FIXTURES / name).read_text())


RAW = {market["question"]: market for market in load("polymarket_gamma_markets.json")}


def raw(question: str) -> dict:
    return copy.deepcopy(RAW[question])


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text)


DAILY_ABOVE = "Will the price of Bitcoin be above $74,000 on October 14?"
HOURLY_ABOVE = "Bitcoin above 85,000 on October 14, 11AM ET?"

B, E = Asset.BTC, Asset.ETH
ABOVE, BELOW, RANGE = MarketKind.ABOVE, MarketKind.BELOW, MarketKind.RANGE
BTC_1M, ETH_1M = "Binance BTCUSDT 1m close", "Binance ETHUSDT 1m close"
BTC_1H, ETH_1H = "Binance BTCUSDT 1h close", "Binance ETHUSDT 1h close"

# question -> (asset, kind, strike_low, strike_high, expiry, neg_risk, event_id, resolution_source)
SUPPORTED = {
    # daily, 1-minute candle closing at 12:00 ET (16:00Z while New York is on EDT)
    DAILY_ABOVE: (B, ABOVE, 74_000, None, "2026-10-14T16:00:00Z", False, "1148918", BTC_1M),
    "Will the price of Ethereum be above $2,100 on October 14?": (
        E,
        ABOVE,
        2_100,
        None,
        "2026-10-14T16:00:00Z",
        False,
        "1148914",
        ETH_1M,
    ),
    "Will the price of Bitcoin be between $74,000 and $76,000 on October 14?": (
        B,
        RANGE,
        74_000,
        76_000,
        "2026-10-14T16:00:00Z",
        True,
        "1148913",
        BTC_1M,
    ),
    "Will the price of Ethereum be between $2,100 and $2,200 on October 14?": (
        E,
        RANGE,
        2_100,
        2_200,
        "2026-10-14T16:00:00Z",
        True,
        "1148917",
        ETH_1M,
    ),
    "Will the price of Bitcoin be less than $74,000 on October 14?": (
        B,
        BELOW,
        None,
        74_000,
        "2026-10-14T16:00:00Z",
        True,
        "1148913",
        BTC_1M,
    ),
    "Will the price of Bitcoin be greater than $92,000 on October 14?": (
        B,
        ABOVE,
        92_000,
        None,
        "2026-10-14T16:00:00Z",
        True,
        "1148913",
        BTC_1M,
    ),
    "Will the price of Ethereum be less than $2,100 on October 14?": (
        E,
        BELOW,
        None,
        2_100,
        "2026-10-14T16:00:00Z",
        True,
        "1148917",
        ETH_1M,
    ),
    "Will the price of Ethereum be greater than $3,000 on October 14?": (
        E,
        ABOVE,
        3_000,
        None,
        "2026-10-14T16:00:00Z",
        True,
        "1148917",
        ETH_1M,
    ),
    # hourly, 1-hour candle that ends at the time in the title
    HOURLY_ABOVE: (B, ABOVE, 85_000, None, "2026-10-14T15:00:00Z", False, "1188400", BTC_1H),
    "Ethereum above 2,560 on October 14, 11AM ET?": (
        E,
        ABOVE,
        2_560,
        None,
        "2026-10-14T15:00:00Z",
        False,
        "1188401",
        ETH_1H,
    ),
    "Bitcoin above 83,800 on October 14, 12PM ET?": (
        B,
        ABOVE,
        83_800,
        None,
        "2026-10-14T16:00:00Z",
        False,
        "1188620",
        BTC_1H,
    ),
    "Ethereum above 2,500 on October 14, 12PM ET?": (
        E,
        ABOVE,
        2_500,
        None,
        "2026-10-14T16:00:00Z",
        False,
        "1188621",
        ETH_1H,
    ),
    # 11PM ET on the 13th is already the 14th in UTC
    "Bitcoin above 85,200 on October 13, 11PM ET?": (
        B,
        ABOVE,
        85_200,
        None,
        "2026-10-14T03:00:00Z",
        False,
        "1185874",
        BTC_1H,
    ),
    "Ethereum above 2,580 on October 13, 11PM ET?": (
        E,
        ABOVE,
        2_580,
        None,
        "2026-10-14T03:00:00Z",
        False,
        "1185875",
        ETH_1H,
    ),
}

# Everything else in the fixture must come back None. Grouped by *why*, so a failure says what
# kind of market started to slip through.
REJECTED = {
    "path-dependent (touches the level at any time)": [
        "Will Bitcoin reach $91,000 on October 14?",
        "Will Bitcoin dip to $83,000 on October 14?",
        "Will Ethereum reach $2,850 on October 14?",
        "Will Bitcoin reach $98,000 October 12-18?",
        "Will Ethereum dip to $2,400 October 12-18?",
        "Will Ethereum reach $4,500 in October?",
        "Will Bitcoin dip to $82,500 in October?",
        "Will Bitcoin reach $95,000 by December 31, 2026?",
        "Will Ethereum hit $3k by September 30, 2026?",
        "Will Bitcoin hit $150k by March 31, 2026?",
        "Bitcoin all time high by March 31, 2026?",
        "Will Ethereum hit $1,000 or $3,000 first?",
    ],
    "relative move, no strike": [
        "Bitcoin Up or Down - October 14, 10AM ET",
        "Ethereum Up or Down - December 19, 11:30AM-11:35AM ET",
        "Bitcoin Up or Down on October 14?",
        "Bitcoin Up or Down - October 14, 8:00PM-12:00AM ET",
    ],
    "not a spot price": [
        "Will Bitcoin dominance be less than 58.00% on October 16?",
        "Will Bitcoin dominance be between 58.00% and 59.00% on October 16?",
        "Will Bitcoin dominance be greater than 62.00% on October 16?",
        "Will ETH/BTC reach 0.0335 October 12-18?",
        "Will the Bitcoin Volatility Index hit 70 by October 31?",
        "Will Bitcoin have 0 green days the week of October 12?",
        "Bitcoin ETF Flows on October 13?",
        "Will Bitcoin be the best performing coin the week of October 12?",
        "Will CryptoPunks floor price reach 50 ETH before 2027?",
        "Bitcoin below MicroStrategy average buy price in 2026?",
    ],
    "another asset": [
        "Will Ethena reach $0.28 by December 31, 2026?",
        "Will MegaETH hit $0.10 by September 30, 2026?",
        "Will the price of Solana be above $90 on October 14?",
        "Will the price of XRP be above $0.90 on October 14?",
    ],
    "rules do not say which candle (older wording, no 'closing at')": [
        "Will the price of Bitcoin be above $74,000 on October 6?",
    ],
}


def test_fixture_has_many_real_wordings_and_is_fully_classified():
    rejected = [q for questions in REJECTED.values() for q in questions]
    assert len(RAW) >= 20
    assert set(SUPPORTED) | set(rejected) == set(RAW)
    assert not set(SUPPORTED) & set(rejected)
    assert len(rejected) == len(set(rejected))


@pytest.mark.parametrize("question", list(SUPPORTED))
def test_supported_market_is_parsed_exactly(question):
    asset, kind, low, high, expiry, neg_risk, event_id, source = SUPPORTED[question]
    source_json = raw(question)
    market = parse_market(source_json)

    assert market is not None
    assert (market.asset, market.kind) == (asset, kind)
    assert (market.strike_low, market.strike_high) == (low, high)
    assert market.expiry == utc(expiry)
    assert market.expiry.utcoffset().total_seconds() == 0
    assert market.neg_risk is neg_risk
    assert market.event_id == event_id
    assert market.resolution_source == source
    assert market.question == question
    assert market.market_id == source_json["conditionId"]
    # Gamma leaves every volume field out for a market nobody has traded yet
    assert market.volume_usd == source_json.get("volumeNum", 0.0)
    # Gamma lists outcomes as ["Yes", "No"] with tokens in the same order
    assert json.loads(source_json["outcomes"]) == ["Yes", "No"]
    yes_token, no_token = json.loads(source_json["clobTokenIds"])
    assert (market.yes_token_id, market.no_token_id) == (yes_token, no_token)


@pytest.mark.parametrize(
    ("reason", "question"),
    [(reason, q) for reason, questions in REJECTED.items() for q in questions],
    ids=lambda value: value[:60],
)
def test_everything_else_is_none(reason, question):
    assert parse_market(raw(question)) is None, reason


# --- the expiry must be rebuilt from title + rules and agree with endDate -----------------


def test_end_date_that_disagrees_with_the_rules_is_ambiguous():
    for wrong in ("2026-10-14T17:00:00Z", "2026-10-14T16:01:00Z", "2026-10-15T16:00:00Z"):
        market = raw(DAILY_ABOVE)
        market["endDate"] = wrong
        assert parse_market(market) is None, wrong


@pytest.mark.parametrize("end_date", [None, "", "soon", "2026-10-14", 1_792_000_000])
def test_missing_or_unusable_end_date(end_date):
    market = raw(DAILY_ABOVE)
    market["endDate"] = end_date
    assert parse_market(market) is None


def test_daily_expiry_follows_new_york_into_winter_time():
    # New York leaves daylight saving on 1 Nov 2026: noon ET is then 17:00Z, not 16:00Z.
    market = raw(DAILY_ABOVE)
    market["question"] = "Will the price of Bitcoin be above $74,000 on November 2?"
    market["endDate"] = "2026-11-02T17:00:00Z"
    parsed = parse_market(market)
    assert parsed is not None and parsed.expiry == utc("2026-11-02T17:00:00Z")

    market["endDate"] = "2026-11-02T16:00:00Z"  # summer-time offset applied to a winter date
    assert parse_market(market) is None


def test_hourly_expiry_follows_new_york_into_winter_time():
    market = raw(HOURLY_ABOVE)
    market["question"] = "Bitcoin above 85,000 on November 1, 11AM ET?"
    market["endDate"] = "2026-11-01T16:00:00Z"  # 11:00 EST
    parsed = parse_market(market)
    assert parsed is not None and parsed.expiry == utc("2026-11-01T16:00:00Z")

    market["endDate"] = "2026-11-01T15:00:00Z"
    assert parse_market(market) is None


def test_midnight_and_noon_hours_of_the_twelve_hour_clock():
    midnight = raw(HOURLY_ABOVE)
    midnight["question"] = "Bitcoin above 85,000 on October 15, 12AM ET?"
    midnight["endDate"] = "2026-10-15T04:00:00Z"
    parsed = parse_market(midnight)
    assert parsed is not None and parsed.expiry == utc("2026-10-15T04:00:00Z")

    noon = raw(HOURLY_ABOVE)
    noon["question"] = "Bitcoin above 85,000 on October 15, 12PM ET?"
    noon["endDate"] = "2026-10-15T16:00:00Z"
    parsed = parse_market(noon)
    assert parsed is not None and parsed.expiry == utc("2026-10-15T16:00:00Z")


def test_year_is_taken_in_new_york_not_in_utc():
    # 11PM on New Year's Eve in New York is already next year in UTC.
    market = raw(HOURLY_ABOVE)
    market["question"] = "Bitcoin above 85,000 on December 31, 11PM ET?"
    market["endDate"] = "2027-01-01T04:00:00Z"
    parsed = parse_market(market)
    assert parsed is not None and parsed.expiry == utc("2027-01-01T04:00:00Z")


@pytest.mark.parametrize(
    "question",
    [
        "Bitcoin above 85,000 on February 30, 11AM ET?",
        "Bitcoin above 85,000 on October 14, 13PM ET?",
        "Bitcoin above 85,000 on October 14, 0AM ET?",
        "Bitcoin above 85,000 on October 14, 11:30AM ET?",
        "Bitcoin above 85,000 on October 14, 2026, 11AM ET?",
        "Bitcoin above 85,000 on Oct 14, 11AM ET?",
    ],
)
def test_impossible_or_unexpected_dates_are_none(question):
    market = raw(HOURLY_ABOVE)
    market["question"] = question
    assert parse_market(market) is None


# --- the rules text -----------------------------------------------------------------------


def test_rules_are_read_from_the_event_when_the_market_has_none():
    market = raw(DAILY_ABOVE)
    del market["description"]
    assert market["events"][0]["description"], "fixture event must carry the rules"
    assert parse_market(market) is not None

    del market["events"][0]["description"]
    assert parse_market(market) is None


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.replace("Binance", "Kucoin"),
        lambda d: d.replace('"Close"', '"Open"'),
        lambda d: d.replace("BTC/USDT", "ETH/USDT"),  # a Bitcoin question resolved on ETH
        lambda d: d.replace("BTC/USDT", "BTC/USDC"),
        lambda d: d.replace("1 minute candle", "5 minute candle"),
        lambda d: d.replace("12:00 in the ET timezone", "12:00 in the UTC timezone"),
        lambda d: d + " Prices are also cross-checked against Coinbase BTC-USD.",
        lambda d: (
            d + ' It also resolves "Yes" if any 1 minute candle has a "High" above the price.'
        ),
        lambda d: d + " It resolves Yes if the price is reached at any time before expiry.",
    ],
)
def test_rules_that_are_not_a_clean_binance_close_are_none(edit):
    market = raw(DAILY_ABOVE)
    market["description"] = edit(market["description"])
    assert parse_market(market) is None


def test_rules_naming_both_moments_or_the_wrong_one_are_none():
    daily, hourly = raw(DAILY_ABOVE), raw(HOURLY_ABOVE)
    # a daily question under hourly rules, and the other way round
    daily_question, hourly_question = copy.deepcopy(daily), copy.deepcopy(hourly)
    daily_question["description"] = hourly["description"]
    hourly_question["description"] = daily["description"]
    assert parse_market(daily_question) is None
    assert parse_market(hourly_question) is None
    # rules that describe both wordings at once
    both = copy.deepcopy(daily)
    both["description"] += " " + hourly["description"]
    assert parse_market(both) is None


def test_resolution_url_must_point_at_binance_if_present():
    market = raw(DAILY_ABOVE)
    market["resolutionSource"] = "https://www.coinbase.com/price/bitcoin"
    assert parse_market(market) is None
    for harmless in (None, ""):
        market["resolutionSource"] = harmless
        assert parse_market(market) is not None


# --- the other fields -------------------------------------------------------------------------


def test_group_title_must_repeat_the_strikes():
    between = "Will the price of Bitcoin be between $74,000 and $76,000 on October 14?"
    for question, wrong in [
        (DAILY_ABOVE, "75,000"),
        (between, "74,000-78,000"),
        (between, "74,000"),
        ("Will the price of Bitcoin be less than $74,000 on October 14?", "<76,000"),
    ]:
        market = raw(question)
        market["groupItemTitle"] = wrong
        assert parse_market(market) is None, (question, wrong)
    for harmless in (None, ""):  # nothing to cross-check against
        market = raw(between)
        market["groupItemTitle"] = harmless
        assert parse_market(market) is not None


@pytest.mark.parametrize("bad", [None, "false", 0, 1])
def test_neg_risk_must_be_a_real_boolean(bad):
    market = raw(DAILY_ABOVE)
    market["negRisk"] = bad
    assert parse_market(market) is None
    del market["negRisk"]
    assert parse_market(market) is None


@pytest.mark.parametrize("bad", [None, "", 0])
def test_market_id_is_required(bad):
    market = raw(DAILY_ABOVE)
    market["conditionId"] = bad
    assert parse_market(market) is None


def test_tokens_are_matched_by_outcome_label_not_position():
    market = raw(DAILY_ABOVE)
    yes_token, no_token = json.loads(market["clobTokenIds"])
    market["outcomes"] = json.dumps(["No", "Yes"])  # No first: tokens follow the same order
    market["clobTokenIds"] = json.dumps([no_token, yes_token])
    parsed = parse_market(market)
    assert parsed is not None
    assert (parsed.yes_token_id, parsed.no_token_id) == (yes_token, no_token)

    market["clobTokenIds"] = [yes_token, no_token]  # real lists work as well as JSON strings
    parsed = parse_market(market)
    assert parsed is not None
    assert (parsed.yes_token_id, parsed.no_token_id) == (no_token, yes_token)


@pytest.mark.parametrize(
    ("outcomes", "tokens"),
    [
        ('["Up", "Down"]', '["1", "2"]'),
        ('["Yes", "No", "Maybe"]', '["1", "2", "3"]'),
        ('["Yes", "No"]', '["1"]'),
        ('["Yes", "No"]', '["7", "7"]'),
        ('["Yes", "No"]', '["", "2"]'),
        ('["Yes", "Yes"]', '["1", "2"]'),
        ("not json", "not json"),
        (None, None),
    ],
)
def test_malformed_outcomes_or_tokens_are_none(outcomes, tokens):
    market = raw(DAILY_ABOVE)
    market["outcomes"], market["clobTokenIds"] = outcomes, tokens
    assert parse_market(market) is None


def test_event_must_be_exactly_one_with_an_id():
    market = raw(DAILY_ABOVE)
    event = market["events"][0]

    for bad in (None, [], "1148918", [None]):
        broken = copy.deepcopy(market)
        broken["events"] = bad
        assert parse_market(broken) is None, bad
    broken = copy.deepcopy(market)
    broken["events"] = [event, copy.deepcopy(event)]  # which one groups the set? don't guess
    assert parse_market(broken) is None
    del broken["events"]
    assert parse_market(broken) is None
    for no_id in (None, ""):
        broken = copy.deepcopy(market)
        broken["events"][0]["id"] = no_id
        assert parse_market(broken) is None
    del market["events"][0]["id"]
    assert parse_market(market) is None


def test_numeric_event_id_becomes_a_string():
    market = raw(DAILY_ABOVE)
    market["events"][0]["id"] = 1148918
    parsed = parse_market(market)
    assert parsed is not None and parsed.event_id == "1148918"


def test_untraded_real_market_has_no_volume_fields_and_zero_volume():
    market = raw("Bitcoin above 83,800 on October 14, 12PM ET?")
    assert not {"volume", "volumeNum", "volumeClob"} & set(market)
    assert parse_market(market).volume_usd == 0.0


def test_volume_falls_back_and_defaults_to_zero():
    market = raw(DAILY_ABOVE)
    market["volumeNum"] = None
    market["volume"] = "1234.5"
    assert parse_market(market).volume_usd == 1234.5

    market["volume"] = None
    market["volumeClob"] = 99
    assert parse_market(market).volume_usd == 99.0

    for junk in (None, "n/a", -5, float("nan")):
        market["volumeNum"] = market["volume"] = market["volumeClob"] = junk
        assert parse_market(market).volume_usd == 0.0


def test_question_whitespace_is_normalised():
    market = raw(DAILY_ABOVE)
    market["question"] = "  Will the price of  Bitcoin be above $74,000\u00a0on October 14?\n"
    parsed = parse_market(market)
    assert parsed is not None and parsed.question == DAILY_ABOVE


@pytest.mark.parametrize(
    "question",
    [
        "will the price of bitcoin be above $74,000 on October 14?",  # wording is exact
        "Will the price of BTC be above $74,000 on October 14?",
        "Will the price of Bitcoin be above $74k on October 14?",
        "Will the price of Bitcoin be above 74,000 on October 14?",  # daily titles carry the $
        "Will the price of Bitcoin be above $74,00 on October 14?",
        "Will the price of Bitcoin be below $74,000 on October 14?",
        "Will the price of Bitcoin be above $74,000 on October 14",  # no question mark
        "Will the price of Bitcoin be above $74,000 on October 14? (Binance)",
        "Will the price of Bitcoin Cash be above $74,000 on October 14?",
        "",
    ],
)
def test_wording_is_matched_strictly(question):
    market = raw(DAILY_ABOVE)
    market["question"] = question
    assert parse_market(market) is None


def test_range_needs_low_below_high():
    market = raw("Will the price of Bitcoin be between $74,000 and $76,000 on October 14?")
    market["groupItemTitle"] = None  # leave only the question to speak
    assert parse_market(market) is not None
    for low, high in (("76,000", "74,000"), ("74,000", "74,000")):
        market["question"] = (
            f"Will the price of Bitcoin be between ${low} and ${high} on October 14?"
        )
        assert parse_market(market) is None


@pytest.mark.parametrize(
    "junk", [{}, {"question": None}, {"question": 5}, {"conditionId": "0x1"}, None, [], "x"]
)
def test_garbage_never_raises(junk):
    assert parse_market(junk) is None


def test_a_closed_market_is_still_parsed():
    # parsing is about structure; whether a market is tradable is the client's concern
    market = raw(DAILY_ABOVE)
    market.update(closed=True, active=False, acceptingOrders=False)
    assert parse_market(market) is not None


# --- what downstream code relies on: a neg-risk event is a gapless set --------------------


def test_neg_risk_event_parses_into_a_gapless_complete_set():
    events = {event["slug"]: event for event in load("polymarket_gamma_events_btc.json")}
    event = events["bitcoin-price-on-october-14-2026"]
    markets = [
        parse_market({**market, "events": [{k: v for k, v in event.items() if k != "markets"}]})
        for market in event["markets"]
    ]
    assert len(markets) == 11 and all(market is not None for market in markets)

    below = [m for m in markets if m.kind is BELOW]
    above = [m for m in markets if m.kind is ABOVE]
    ranges = sorted((m for m in markets if m.kind is RANGE), key=lambda m: m.strike_low)
    assert (len(below), len(above), len(ranges)) == (1, 1, 9)
    assert below[0].strike_high == ranges[0].strike_low == 74_000
    assert all(a.strike_high == b.strike_low for a, b in itertools.pairwise(ranges))
    assert ranges[-1].strike_high == above[0].strike_low == 92_000
    assert {m.event_id for m in markets} == {"1148913"}
    assert {m.expiry for m in markets} == {utc("2026-10-14T16:00:00Z")}
    assert all(m.neg_risk for m in markets)


# --- shared helpers ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-10-14T16:00:00Z", utc("2026-10-14T16:00:00Z")),
        ("2026-10-13 16:12:15+00", utc("2026-10-13T16:12:15Z")),
        ("2026-10-14T14:52:59.229738Z", datetime(2026, 10, 14, 14, 52, 59, 229738, tzinfo=UTC)),
        ("2026-10-14T12:00:00-04:00", utc("2026-10-14T16:00:00Z")),
        ("2026-10-14", None),  # a bare date is not a moment
        ("2026-10-14T16:00:00", None),  # naive
        ("garbage", None),
        ("", None),
        (None, None),
        (1_792_000_000, None),
    ],
)
def test_parse_timestamp(value, expected):
    assert parse_timestamp(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('["Yes", "No"]', ["Yes", "No"]),
        (["Yes", "No"], ["Yes", "No"]),
        ("[]", []),
        ('{"a": 1}', None),
        ("nope", None),
        (None, None),
        (5, None),
    ],
)
def test_parse_json_list(value, expected):
    assert parse_json_list(value) == expected
