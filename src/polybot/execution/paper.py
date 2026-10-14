"""Task C. Paper broker. Fills only against liquidity that was really in the book."""

from __future__ import annotations

from datetime import datetime

from polybot.models import Fill, MarketOutcome, Merge, Order, OrderBook, Outcome, Position


class PaperBroker:
    def __init__(self, fee_rate: float) -> None:
        self.fee_rate = fee_rate
        self.positions: dict[tuple[str, Outcome], Position] = {}
        self.cash_usd: float = 0.0

    def submit_market(self, order: Order, book: OrderBook) -> list[Fill]:
        raise NotImplementedError

    def merge(self, market_id: str, size: float, now: datetime) -> Merge:
        """Turn `size` YES + `size` NO shares of one market into `size` USDC."""
        raise NotImplementedError

    def settle(self, outcome: MarketOutcome) -> float:
        """Pay out winning shares at 1, zero the losers. Returns realized PnL."""
        raise NotImplementedError
