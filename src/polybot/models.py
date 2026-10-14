"""Shared data types. Every module talks to the others only through these.

Conventions:
- All datetimes are timezone-aware UTC.
- Prices of Polymarket outcome tokens are in USDC per share, in [0, 1].
- Probabilities refer to the YES outcome unless the name says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Asset(StrEnum):
    BTC = "BTC"
    ETH = "ETH"


class MarketKind(StrEnum):
    ABOVE = "above"  # YES if price > strike_low at expiry
    BELOW = "below"  # YES if price < strike_high at expiry
    RANGE = "range"  # YES if strike_low <= price < strike_high at expiry


class Outcome(StrEnum):
    YES = "YES"
    NO = "NO"

    @property
    def opposite(self) -> Outcome:
        return Outcome.NO if self is Outcome.YES else Outcome.YES


class BookSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "market"  # taker: walks the book immediately
    LIMIT = "limit"  # maker: rests in the book until filled or cancelled


class OrderStatus(StrEnum):
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware UTC, got naive {value!r}")


def _require_price(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


# --- Polymarket -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CryptoMarket:
    """A binary Polymarket market on a crypto price, parsed into structured form."""

    market_id: str  # Polymarket condition id
    event_id: str  # markets of one event share it; their ranges may form a complete set
    question: str
    asset: Asset
    kind: MarketKind
    strike_low: float | None
    strike_high: float | None
    expiry: datetime  # moment the resolution price is observed
    yes_token_id: str
    no_token_id: str
    resolution_source: str  # e.g. "Binance ETHUSDT 1m close"
    neg_risk: bool  # True if part of a mutually exclusive multi-outcome event
    volume_usd: float

    def __post_init__(self) -> None:
        _require_utc("expiry", self.expiry)
        match self.kind:
            case MarketKind.ABOVE if self.strike_low is None:
                raise ValueError("ABOVE market needs strike_low")
            case MarketKind.BELOW if self.strike_high is None:
                raise ValueError("BELOW market needs strike_high")
            case MarketKind.RANGE if self.strike_low is None or self.strike_high is None:
                raise ValueError("RANGE market needs strike_low and strike_high")
            case MarketKind.RANGE if self.strike_low >= self.strike_high:  # type: ignore[operator]
                raise ValueError("RANGE market needs strike_low < strike_high")
            case _:
                pass

    def token_id(self, outcome: Outcome) -> str:
        return self.yes_token_id if outcome is Outcome.YES else self.no_token_id


@dataclass(frozen=True, slots=True)
class PriceLevel:
    price: float
    size: float  # shares

    def __post_init__(self) -> None:
        _require_price("price", self.price)
        if self.size < 0:
            raise ValueError(f"size must be >= 0, got {self.size}")


@dataclass(frozen=True, slots=True)
class OrderBook:
    """Book of one outcome token. Bids sorted best (highest) first, asks best (lowest) first."""

    token_id: str
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]
    timestamp: datetime

    def __post_init__(self) -> None:
        _require_utc("timestamp", self.timestamp)
        if any(a.price < b.price for a, b in zip(self.bids, self.bids[1:])):
            raise ValueError("bids must be sorted descending")
        if any(a.price > b.price for a, b in zip(self.asks, self.asks[1:])):
            raise ValueError("asks must be sorted ascending")

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid


@dataclass(frozen=True, slots=True)
class MarketOutcome:
    market_id: str
    resolved_yes: bool
    resolved_at: datetime

    def __post_init__(self) -> None:
        _require_utc("resolved_at", self.resolved_at)


@dataclass(frozen=True, slots=True)
class WalletTrade:
    """A public trade by any wallet, from the Polymarket data API (task G)."""

    wallet: str
    market_id: str
    outcome: Outcome
    side: BookSide
    price: float
    size: float
    timestamp: datetime
    tx_hash: str

    def __post_init__(self) -> None:
        _require_utc("timestamp", self.timestamp)
        _require_price("price", self.price)


# --- Deribit --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OptionQuote:
    instrument: str  # e.g. "ETH-17OCT26-4400-C"
    asset: Asset
    strike: float
    expiry: datetime  # Deribit settles at 08:00 UTC
    is_call: bool
    mark_iv: float  # annualized, as a fraction (0.55 == 55%)
    bid_iv: float | None
    ask_iv: float | None
    underlying_price: float  # forward/underlying used by Deribit for this expiry
    timestamp: datetime

    def __post_init__(self) -> None:
        _require_utc("expiry", self.expiry)
        _require_utc("timestamp", self.timestamp)


@dataclass(frozen=True, slots=True)
class FairProbability:
    market_id: str
    p_yes: float
    computed_at: datetime
    model: str  # model name + version, so the journal can compare models
    spot: float
    meta: dict[str, float] = field(default_factory=dict)  # iv used, time to expiry, etc.

    def __post_init__(self) -> None:
        _require_utc("computed_at", self.computed_at)
        _require_price("p_yes", self.p_yes)

    def p(self, outcome: Outcome) -> float:
        return self.p_yes if outcome is Outcome.YES else 1.0 - self.p_yes


# --- Execution ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FillEstimate:
    """Result of walking the book for a taker order (task C)."""

    token_id: str
    side: BookSide
    size_requested: float
    size_filled: float
    avg_price: float
    worst_price: float
    slippage: float  # avg_price vs best price, in price units, always >= 0
    fee_usd: float

    @property
    def fully_filled(self) -> bool:
        return self.size_filled >= self.size_requested

    @property
    def cost_usd(self) -> float:
        notional = self.avg_price * self.size_filled
        return notional + self.fee_usd if self.side is BookSide.BUY else notional - self.fee_usd


@dataclass(frozen=True, slots=True)
class Signal:
    market_id: str
    outcome: Outcome
    fair_price: float
    entry_price: float  # expected avg fill price incl. slippage
    edge_after_costs: float  # fair_price - entry_price - fee per share
    size: float  # shares, after risk sizing
    stop_price: float | None
    take_profit_price: float | None
    created_at: datetime
    reason: str  # "model_edge", "complete_set", ...

    def __post_init__(self) -> None:
        _require_utc("created_at", self.created_at)


@dataclass(slots=True)
class Order:
    """A paper order. Mutable: status and filled size change over time."""

    order_id: str
    market_id: str
    token_id: str
    outcome: Outcome
    side: BookSide
    order_type: OrderType
    size: float
    limit_price: float | None  # None for MARKET
    created_at: datetime
    status: OrderStatus = OrderStatus.OPEN
    filled_size: float = 0.0
    avg_fill_price: float | None = None
    strategy: str = ""  # "signal", "maker", "complete_set"

    def __post_init__(self) -> None:
        _require_utc("created_at", self.created_at)
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT order needs limit_price")

    @property
    def remaining(self) -> float:
        return self.size - self.filled_size


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    price: float
    size: float
    fee_usd: float
    is_maker: bool
    timestamp: datetime

    def __post_init__(self) -> None:
        _require_utc("timestamp", self.timestamp)


@dataclass(slots=True)
class Position:
    """Net shares held in one outcome token."""

    market_id: str
    outcome: Outcome
    size: float = 0.0
    avg_price: float = 0.0
    realized_pnl_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class Merge:
    """Exchange of N YES + N NO shares (or a full set of exclusive outcomes) for N USDC."""

    market_ids: tuple[str, ...]
    size: float
    cost_basis_usd: float  # what the merged shares cost us
    proceeds_usd: float  # == size
    timestamp: datetime

    def __post_init__(self) -> None:
        _require_utc("timestamp", self.timestamp)

    @property
    def pnl_usd(self) -> float:
        return self.proceeds_usd - self.cost_basis_usd


@dataclass(frozen=True, slots=True)
class CompleteSetOpportunity:
    """Buying every leg of a complete set for less than its guaranteed payout (task C)."""

    market_ids: tuple[str, ...]
    token_ids: tuple[str, ...]
    size: float
    total_cost_usd: float  # incl. slippage and fees
    payout_usd: float  # == size
    found_at: datetime

    @property
    def profit_usd(self) -> float:
        return self.payout_usd - self.total_cost_usd
