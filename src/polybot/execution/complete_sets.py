"""Task C. Riskless complete sets: YES+NO of one market, or YES of every exclusive range.

A set is only complete if its legs are truly mutually exclusive and exhaustive
under the markets' resolution rules; same event_id, neg_risk, matching expiry
and contiguous ranges covering everything must all be checked.
"""

from __future__ import annotations

from datetime import datetime

from polybot.models import CompleteSetOpportunity, CryptoMarket, OrderBook


def find_binary_sets(
    markets: list[CryptoMarket], books: dict[str, OrderBook], fee_rate: float, now: datetime
) -> list[CompleteSetOpportunity]:
    raise NotImplementedError


def find_range_sets(
    markets: list[CryptoMarket], books: dict[str, OrderBook], fee_rate: float, now: datetime
) -> list[CompleteSetOpportunity]:
    raise NotImplementedError
