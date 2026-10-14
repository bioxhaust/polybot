"""Task H. Market-making simulator: resting limit orders around the fair price.

Paper results here are optimistic by construction: queue position is unknown.
Assume we are last in the queue at our price level and only fill from trades
that print through it, never from the book merely touching our price.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from polybot.models import CryptoMarket, FairProbability, Fill, Order, OrderBook, WalletTrade


@dataclass(frozen=True, slots=True)
class Quote:
    yes_bid: float | None
    no_bid: float | None
    size: float


def make_quote(
    market: CryptoMarket, fair: FairProbability, book_yes: OrderBook, inventory: float
) -> Quote:
    """Bids on YES and NO such that yes_bid + no_bid < 1, skewed against inventory."""
    raise NotImplementedError


class MakerSimulator:
    def __init__(self, fee_rate: float) -> None:
        self.fee_rate = fee_rate
        self.open_orders: dict[str, Order] = {}

    def requote(self, market: CryptoMarket, quote: Quote, now: datetime) -> None:
        raise NotImplementedError

    def on_trades(self, trades: list[WalletTrade]) -> list[Fill]:
        raise NotImplementedError

    def cancel_all(self, market_id: str, reason: str) -> None:
        """Pull quotes, e.g. when spot moves sharply faster than the fair price refreshes."""
        raise NotImplementedError
