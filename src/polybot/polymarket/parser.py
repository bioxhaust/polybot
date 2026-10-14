"""Task A. Turn raw Gamma market JSON into CryptoMarket.

Must return None for anything it cannot parse with certainty (unknown wording,
"reach/hit any time" path-dependent markets, ambiguous expiry). Never guess.
"""

from __future__ import annotations

from typing import Any

from polybot.models import CryptoMarket


def parse_market(raw: dict[str, Any]) -> CryptoMarket | None:
    raise NotImplementedError
