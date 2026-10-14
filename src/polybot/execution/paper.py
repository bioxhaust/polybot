"""Task C. Paper broker. Fills only against liquidity that was really in the book."""

from __future__ import annotations

from datetime import datetime

from polybot.models import (
    BookSide,
    Fill,
    MarketOutcome,
    Merge,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Outcome,
    Position,
)

_EPS = 1e-12


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware UTC, got naive {value!r}")


class PaperBroker:
    def __init__(self, fee_rate: float) -> None:
        if fee_rate < 0.0:
            raise ValueError(f"fee_rate must be >= 0, got {fee_rate}")
        self.fee_rate = fee_rate
        self.positions: dict[tuple[str, Outcome], Position] = {}
        self.cash_usd: float = 0.0

    def submit_market(self, order: Order, book: OrderBook) -> list[Fill]:
        if order.order_type is not OrderType.MARKET:
            raise ValueError(f"submit_market requires OrderType.MARKET, got {order.order_type}")
        if order.token_id != book.token_id:
            raise ValueError(
                f"order token_id {order.token_id!r} does not match book token_id {book.token_id!r}"
            )
        if order.size <= 0.0:
            raise ValueError(f"order size must be > 0, got {order.size}")
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            raise ValueError(f"cannot submit order with status {order.status}")

        remaining = order.remaining
        if remaining <= _EPS:
            order.status = OrderStatus.FILLED
            return []

        key = (order.market_id, order.outcome)
        pos = self.positions.get(key)
        if order.side is BookSide.SELL:
            held = pos.size if pos is not None else 0.0
            if remaining > held + _EPS:
                raise ValueError(
                    f"cannot sell {remaining} shares of {key}; only {held} held in position"
                )

        levels = book.asks if order.side is BookSide.BUY else book.bids
        fills: list[Fill] = []
        tol = _EPS * max(1.0, order.size)

        for level in levels:
            if level.size <= 0.0:
                continue
            if level.size >= remaining - tol:
                take = remaining
                remaining = 0.0
            else:
                take = level.size
                remaining -= take

            fee_usd = take * level.price * self.fee_rate
            fills.append(
                Fill(
                    order_id=order.order_id,
                    price=level.price,
                    size=take,
                    fee_usd=fee_usd,
                    is_maker=False,
                    timestamp=book.timestamp,
                )
            )
            if remaining <= tol:
                break

        if not fills:
            return []

        prev_filled = order.filled_size
        prev_notional = (order.avg_fill_price or 0.0) * prev_filled
        added_size = sum(f.size for f in fills)
        added_notional = sum(f.size * f.price for f in fills)
        added_fees = sum(f.fee_usd for f in fills)

        new_filled = prev_filled + added_size
        if abs(new_filled - order.size) <= tol:
            new_filled = order.size

        order.filled_size = new_filled
        order.avg_fill_price = (prev_notional + added_notional) / new_filled
        order.status = (
            OrderStatus.FILLED
            if order.filled_size >= order.size
            else OrderStatus.PARTIALLY_FILLED
        )

        if pos is None:
            pos = Position(market_id=order.market_id, outcome=order.outcome)
            self.positions[key] = pos

        if order.side is BookSide.BUY:
            new_pos_size = pos.size + added_size
            pos.avg_price = (pos.size * pos.avg_price + added_notional) / new_pos_size
            pos.size = new_pos_size
            self.cash_usd -= added_notional + added_fees
        else:
            entry_cost = added_size * pos.avg_price * (1.0 + self.fee_rate)
            net_proceeds = added_notional - added_fees
            pos.realized_pnl_usd += net_proceeds - entry_cost
            pos.size = max(0.0, pos.size - added_size)
            if pos.size <= tol:
                pos.size = 0.0
                pos.avg_price = 0.0
            self.cash_usd += net_proceeds

        return fills

    def merge(self, market_id: str, size: float, now: datetime) -> Merge:
        """Turn `size` YES + `size` NO shares of one market into `size` USDC."""
        _require_utc("now", now)
        if size <= 0.0:
            raise ValueError(f"merge size must be > 0, got {size}")

        yes_pos = self.positions.get((market_id, Outcome.YES))
        no_pos = self.positions.get((market_id, Outcome.NO))
        tol = _EPS * max(1.0, size)

        if yes_pos is None or yes_pos.size < size - tol:
            held = 0.0 if yes_pos is None else yes_pos.size
            raise ValueError(f"insufficient YES shares to merge: need {size}, have {held}")
        if no_pos is None or no_pos.size < size - tol:
            held = 0.0 if no_pos is None else no_pos.size
            raise ValueError(f"insufficient NO shares to merge: need {size}, have {held}")

        yes_cost = size * yes_pos.avg_price * (1.0 + self.fee_rate)
        no_cost = size * no_pos.avg_price * (1.0 + self.fee_rate)
        cost_basis_usd = yes_cost + no_cost
        proceeds_usd = size

        yes_pos.realized_pnl_usd += 0.5 * size - yes_cost
        no_pos.realized_pnl_usd += 0.5 * size - no_cost

        yes_pos.size = max(0.0, yes_pos.size - size)
        if yes_pos.size <= tol:
            yes_pos.size = 0.0
            yes_pos.avg_price = 0.0

        no_pos.size = max(0.0, no_pos.size - size)
        if no_pos.size <= tol:
            no_pos.size = 0.0
            no_pos.avg_price = 0.0

        self.cash_usd += proceeds_usd

        return Merge(
            market_ids=(market_id,),
            size=size,
            cost_basis_usd=cost_basis_usd,
            proceeds_usd=proceeds_usd,
            timestamp=now,
        )

    def settle(self, outcome: MarketOutcome) -> float:
        """Pay out winning shares at 1, zero the losers. Returns realized PnL."""
        winning_outcome = Outcome.YES if outcome.resolved_yes else Outcome.NO
        realized_pnl = 0.0

        for out in (Outcome.YES, Outcome.NO):
            pos = self.positions.get((outcome.market_id, out))
            if pos is None or pos.size <= 0.0:
                continue

            payout_price = 1.0 if out is winning_outcome else 0.0
            payout_usd = pos.size * payout_price
            cost_basis_usd = pos.size * pos.avg_price * (1.0 + self.fee_rate)
            pnl = payout_usd - cost_basis_usd

            self.cash_usd += payout_usd
            pos.realized_pnl_usd += pnl
            pos.size = 0.0
            pos.avg_price = 0.0
            realized_pnl += pnl

        return realized_pnl
