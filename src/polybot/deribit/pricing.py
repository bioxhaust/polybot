"""Task B. Options-implied probabilities for Polymarket price markets.

Polymarket expiries rarely match Deribit's 08:00 UTC settlement, so implied
variance must be interpolated in time between the two surrounding expiries,
and IV interpolated across strikes (the smile matters for far strikes).
"""

from __future__ import annotations

from datetime import datetime

from polybot.models import CryptoMarket, FairProbability, OptionQuote

MODEL_NAME = "deribit_iv_v1"


def implied_vol(chain: list[OptionQuote], strike: float, expiry: datetime) -> float:
    raise NotImplementedError


def prob_above(chain: list[OptionQuote], strike: float, expiry: datetime, now: datetime) -> float:
    """Risk-neutral P(S_T > strike), including the smile correction to N(d2)."""
    raise NotImplementedError


def fair_probability(
    market: CryptoMarket, chain: list[OptionQuote], now: datetime
) -> FairProbability | None:
    """None if the chain cannot cover the market (expiry beyond last option, strike far outside)."""
    raise NotImplementedError
