"""Task A. Turn raw Gamma market JSON into CryptoMarket.

Must return None for anything it cannot parse with certainty (unknown wording,
"reach/hit any time" path-dependent markets, ambiguous expiry). Never guess.

Supported families (the only "price at a fixed moment" markets Polymarket runs on BTC/ETH):

* daily   - "Will the price of Bitcoin be above $74,000 on October 14?" (also "greater than",
            "less than", "between $A and $B"). Rules: the Binance 1-minute candle closing at
            12:00 ET decides.
* hourly  - "Bitcoin above 85,000 on October 14, 11AM ET?". Rules: the Binance 1-hour candle
            that ends at the time in the title decides.

A market is accepted only if three independent sources agree: the wording of the question, the
rules text (which names the exact moment and the price source) and the market's own `endDate`.
The title carries no year, so the expiry is rebuilt from title + rules in US Eastern time and
must equal `endDate` to the second; otherwise the expiry is ambiguous and we return None.

Everything else - "reach/dip/hit" (path-dependent), "Up or Down" (no strike), dominance,
volatility indices, ETH/BTC ratios, other coins, ... - is rejected on purpose.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from polybot.models import Asset, CryptoMarket, MarketKind

ET = ZoneInfo("America/New_York")

_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_MONTH_NUMBER = {name: number for number, name in enumerate(_MONTHS, start=1)}

# 74,000 | 2,560 | 840 | 0.5  (thousands separators are required above 999)
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_ASSET = r"(?P<asset>Bitcoin|Ethereum)"
_DATE = r"(?P<month>" + "|".join(_MONTHS) + r") (?P<day>\d{1,2})"
_TIME = r"(?P<hour>\d{1,2})(?P<ampm>AM|PM) ET"

_DAILY_PREFIX = rf"^Will the price of {_ASSET} be "
_QUESTION_PATTERNS: tuple[tuple[re.Pattern[str], MarketKind], ...] = (
    (
        re.compile(_DAILY_PREFIX + rf"(?:above|greater than) \$(?P<low>{_NUM}) on {_DATE}\?$"),
        MarketKind.ABOVE,
    ),
    (
        re.compile(_DAILY_PREFIX + rf"less than \$(?P<high>{_NUM}) on {_DATE}\?$"),
        MarketKind.BELOW,
    ),
    (
        re.compile(
            _DAILY_PREFIX + rf"between \$(?P<low>{_NUM}) and \$(?P<high>{_NUM}) on {_DATE}\?$"
        ),
        MarketKind.RANGE,
    ),
    (
        re.compile(rf"^{_ASSET} above \$?(?P<low>{_NUM}) on {_DATE}, {_TIME}\?$"),
        MarketKind.ABOVE,
    ),
)

_ASSETS = {"Bitcoin": Asset.BTC, "Ethereum": Asset.ETH}

# Rules texts (whitespace-normalised, straight quotes). Both name the moment and the price used.
_DAILY_RULES = re.compile(
    r"Binance (?P<n>\d+) minute candle for (?P<base>[A-Z]+)/(?P<quote>[A-Z]+) closing at "
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2}) in the ET timezone"
)
_HOURLY_RULES = re.compile(
    r'"Close" price for the (?P<base>[A-Z]+)/(?P<quote>[A-Z]+) (?P<n>\d+) hour candle '
    r"that ends on the time and date specified in the title"
)
# Path-dependent wording ("any candle ... High price") must never slip through next to a
# point-in-time phrase, and neither may a price source other than Binance.
_PATH_DEPENDENT = re.compile(
    r'\bany\b[^.]{0,60}\bcandle\b|"(?:High|Low)"|\bat any (?:time|point)\b', re.IGNORECASE
)
_OTHER_SOURCES = re.compile(
    r"Coinbase|Kraken|Bitstamp|OKX|Bybit|Chainlink|CoinGecko|CoinMarketCap|TradingView|Pyth",
    re.IGNORECASE,
)
_BASE_BY_ASSET = {Asset.BTC: "BTC", Asset.ETH: "ETH"}


# --- small shared helpers (also used by the client) ---------------------------------


def parse_timestamp(value: Any) -> datetime | None:
    """Gamma timestamp ("2026-10-14T16:00:00Z", "2026-10-13 16:12:15+00") -> aware UTC, else None."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None  # a bare date is not a moment in time
    return parsed.astimezone(UTC)


def parse_json_list(value: Any) -> list[Any] | None:
    """Gamma ships lists either as real lists or as JSON strings ('["Yes", "No"]')."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return None
        return decoded if isinstance(decoded, list) else None
    return None


# --- question -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Question:
    asset: Asset
    kind: MarketKind
    strike_low: float | None
    strike_high: float | None
    month: int
    day: int
    hour: int | None  # 0-23 ET if the title names a time of day (hourly family)


def _clean(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    return re.sub(r"\s+", " ", text).strip()


def _number(text: str | None) -> float | None:
    return float(text.replace(",", "")) if text is not None else None


def _parse_question(question: str) -> _Question | None:
    for pattern, kind in _QUESTION_PATTERNS:
        match = pattern.match(question)
        if match is None:
            continue
        groups = match.groupdict()
        hour: int | None = None
        if groups.get("hour") is not None:
            clock = int(groups["hour"])
            if not 1 <= clock <= 12:
                return None
            hour = clock % 12 + (12 if groups["ampm"] == "PM" else 0)
        return _Question(
            asset=_ASSETS[groups["asset"]],
            kind=kind,
            strike_low=_number(groups.get("low")),
            strike_high=_number(groups.get("high")),
            month=_MONTH_NUMBER[groups["month"]],
            day=int(groups["day"]),
            hour=hour,
        )
    return None


# --- rules ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Rules:
    daily: bool  # True: 1m candle closing at `clock`; False: 1h candle ending at the title's hour
    clock: tuple[int, int] | None  # (hour, minute) ET named by the rules (daily only)
    source: str  # e.g. "Binance BTCUSDT 1m close"


def _rules_text(raw: dict[str, Any]) -> str:
    text = _clean(raw.get("description"))
    if text:
        return text
    events = raw.get("events")
    if isinstance(events, list) and len(events) == 1 and isinstance(events[0], dict):
        return _clean(events[0].get("description"))
    return ""


def _parse_rules(raw: dict[str, Any], asset: Asset) -> _Rules | None:
    text = _rules_text(raw)
    if not text or '"Close"' not in text or "Binance" not in text:
        return None
    if _PATH_DEPENDENT.search(text) or _OTHER_SOURCES.search(text):
        return None
    url = raw.get("resolutionSource")
    if isinstance(url, str) and url and "binance.com" not in url:
        return None

    daily = _DAILY_RULES.search(text)
    hourly = _HOURLY_RULES.search(text)
    if daily is not None and hourly is None:
        match, is_daily = daily, True
    elif hourly is not None and daily is None:
        match, is_daily = hourly, False
    else:  # neither wording, or both at once: the moment of observation is not clear
        return None
    if match["base"] != _BASE_BY_ASSET[asset] or match["quote"] != "USDT" or match["n"] != "1":
        return None

    if is_daily:
        hour, minute = int(match["hour"]), int(match["minute"])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        return _Rules(True, (hour, minute), f"Binance {match['base']}USDT 1m close")
    return _Rules(False, None, f"Binance {match['base']}USDT 1h close")


# --- the rest of the market -----------------------------------------------------------


def _token_ids(raw: dict[str, Any]) -> tuple[str, str] | None:
    """(yes_token_id, no_token_id), matched by outcome label rather than by position."""
    outcomes = parse_json_list(raw.get("outcomes"))
    tokens = parse_json_list(raw.get("clobTokenIds"))
    if outcomes is None or tokens is None or len(outcomes) != 2 or len(tokens) != 2:
        return None
    labels = [str(label).strip().lower() for label in outcomes]
    ids = [str(token).strip() for token in tokens]
    if sorted(labels) != ["no", "yes"] or not all(ids) or ids[0] == ids[1]:
        return None
    return ids[labels.index("yes")], ids[labels.index("no")]


def _event_id(raw: dict[str, Any]) -> str | None:
    events = raw.get("events")
    if not isinstance(events, list) or len(events) != 1 or not isinstance(events[0], dict):
        return None  # no event, or several: which one groups the complete set is a guess
    event_id = events[0].get("id")
    return str(event_id) if event_id not in (None, "") else None


def _expiry(question: _Question, rules: _Rules, end_date: datetime) -> datetime | None:
    """Moment the price is observed, rebuilt from title + rules and checked against endDate."""
    if rules.daily:
        if question.hour is not None or rules.clock is None:
            return None
        hour, minute = rules.clock
    else:
        if question.hour is None:
            return None
        hour, minute = question.hour, 0
    year = end_date.astimezone(ET).year  # titles carry no year
    expiry = datetime(year, question.month, question.day, hour, minute, tzinfo=ET).astimezone(UTC)
    return expiry if expiry == end_date else None


def _strikes_match_group_title(question: _Question, group_title: Any) -> bool:
    """`groupItemTitle` ("74,000", "74,000-76,000", "<74,000", ">92,000") must repeat the strikes."""
    if group_title in (None, ""):
        return True  # nothing to cross-check against
    numbers = [_number(n) for n in re.findall(_NUM, str(group_title))]
    expected = {
        MarketKind.ABOVE: [question.strike_low],
        MarketKind.BELOW: [question.strike_high],
        MarketKind.RANGE: [question.strike_low, question.strike_high],
    }[question.kind]
    return numbers == expected


def _volume(raw: dict[str, Any]) -> float:
    for key in ("volumeNum", "volume", "volumeClob"):
        try:
            value = float(raw[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            return value
    return 0.0


def _parse(raw: dict[str, Any]) -> CryptoMarket | None:
    question_text = _clean(raw.get("question"))
    question = _parse_question(question_text)
    if question is None:
        return None

    market_id = raw.get("conditionId")
    neg_risk = raw.get("negRisk")
    end_date = parse_timestamp(raw.get("endDate"))
    tokens = _token_ids(raw)
    event_id = _event_id(raw)
    if not isinstance(market_id, str) or not market_id or not isinstance(neg_risk, bool):
        return None
    if end_date is None or tokens is None or event_id is None:
        return None
    if not _strikes_match_group_title(question, raw.get("groupItemTitle")):
        return None

    rules = _parse_rules(raw, question.asset)
    if rules is None:
        return None
    expiry = _expiry(question, rules, end_date)
    if expiry is None:
        return None

    return CryptoMarket(
        market_id=market_id,
        event_id=event_id,
        question=question_text,
        asset=question.asset,
        kind=question.kind,
        strike_low=question.strike_low,
        strike_high=question.strike_high,
        expiry=expiry,
        yes_token_id=tokens[0],
        no_token_id=tokens[1],
        resolution_source=rules.source,
        neg_risk=neg_risk,
        volume_usd=_volume(raw),
    )


def parse_market(raw: dict[str, Any]) -> CryptoMarket | None:
    try:
        return _parse(raw)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        # Garbage payloads and impossible values (30 February, low >= high, ...) are "not sure".
        return None
