"""SQLite journal. Everything the bot sees and decides goes here.

Book snapshots are stored for every scanned market on every tick, not only when
a signal fires: they are the raw material for backtests.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from polybot.models import (
    Asset,
    BookSide,
    CompleteSetOpportunity,
    CryptoMarket,
    FairProbability,
    Fill,
    FillEstimate,
    MarketKind,
    MarketOutcome,
    Merge,
    OptionQuote,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Outcome,
    Position,
    PriceLevel,
    Signal,
    WalletTrade,
)

# v1 is the tables. Later entries are incremental upgrades; each runs in one
# transaction and then bumps PRAGMA user_version.
_MIGRATION_V1: tuple[str, ...] = (
    """
    CREATE TABLE markets (
        market_id TEXT PRIMARY KEY,
        event_id TEXT NOT NULL,
        question TEXT NOT NULL,
        asset TEXT NOT NULL CHECK (asset IN ('BTC', 'ETH')),
        kind TEXT NOT NULL CHECK (kind IN ('above', 'below', 'range')),
        strike_low REAL,
        strike_high REAL,
        expiry TEXT NOT NULL,
        yes_token_id TEXT NOT NULL,
        no_token_id TEXT NOT NULL,
        resolution_source TEXT NOT NULL,
        neg_risk INTEGER NOT NULL CHECK (neg_risk IN (0, 1)),
        volume_usd REAL NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE book_snapshots (
        id INTEGER PRIMARY KEY,
        market_id TEXT NOT NULL,
        token_id TEXT NOT NULL,
        timestamp TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE book_levels (
        snapshot_id INTEGER NOT NULL REFERENCES book_snapshots(id) ON DELETE CASCADE,
        side TEXT NOT NULL CHECK (side IN ('bid', 'ask')),
        level_index INTEGER NOT NULL CHECK (level_index >= 0),
        price REAL NOT NULL CHECK (price >= 0.0 AND price <= 1.0),
        size REAL NOT NULL CHECK (size >= 0.0),
        PRIMARY KEY (snapshot_id, side, level_index)
    ) STRICT
    """,
    """
    CREATE TABLE fair_probabilities (
        id INTEGER PRIMARY KEY,
        market_id TEXT NOT NULL,
        p_yes REAL NOT NULL CHECK (p_yes >= 0.0 AND p_yes <= 1.0),
        computed_at TEXT NOT NULL,
        model TEXT NOT NULL,
        spot REAL NOT NULL,
        meta TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE signals (
        id INTEGER PRIMARY KEY,
        market_id TEXT NOT NULL,
        outcome TEXT NOT NULL CHECK (outcome IN ('YES', 'NO')),
        fair_price REAL NOT NULL,
        entry_price REAL NOT NULL,
        edge_after_costs REAL NOT NULL,
        size REAL NOT NULL,
        stop_price REAL,
        take_profit_price REAL,
        created_at TEXT NOT NULL,
        reason TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE orders (
        order_id TEXT PRIMARY KEY,
        market_id TEXT NOT NULL,
        token_id TEXT NOT NULL,
        outcome TEXT NOT NULL CHECK (outcome IN ('YES', 'NO')),
        side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
        order_type TEXT NOT NULL CHECK (order_type IN ('market', 'limit')),
        size REAL NOT NULL,
        limit_price REAL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('open', 'partially_filled', 'filled', 'cancelled')
        ),
        filled_size REAL NOT NULL,
        avg_fill_price REAL,
        strategy TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE fills (
        id INTEGER PRIMARY KEY,
        order_id TEXT NOT NULL,
        price REAL NOT NULL,
        size REAL NOT NULL,
        fee_usd REAL NOT NULL,
        is_maker INTEGER NOT NULL CHECK (is_maker IN (0, 1)),
        timestamp TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE merges (
        id INTEGER PRIMARY KEY,
        market_ids TEXT NOT NULL,
        size REAL NOT NULL,
        cost_basis_usd REAL NOT NULL,
        proceeds_usd REAL NOT NULL,
        timestamp TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE complete_sets (
        id INTEGER PRIMARY KEY,
        market_ids TEXT NOT NULL,
        token_ids TEXT NOT NULL,
        size REAL NOT NULL,
        total_cost_usd REAL NOT NULL,
        payout_usd REAL NOT NULL,
        found_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE outcomes (
        market_id TEXT PRIMARY KEY,
        resolved_yes INTEGER NOT NULL CHECK (resolved_yes IN (0, 1)),
        resolved_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE wallet_trades (
        id INTEGER PRIMARY KEY,
        wallet TEXT NOT NULL,
        market_id TEXT NOT NULL,
        outcome TEXT NOT NULL CHECK (outcome IN ('YES', 'NO')),
        side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
        price REAL NOT NULL CHECK (price >= 0.0 AND price <= 1.0),
        size REAL NOT NULL,
        timestamp TEXT NOT NULL,
        tx_hash TEXT NOT NULL,
        UNIQUE (wallet, tx_hash, market_id, outcome, side, price, size, timestamp)
    ) STRICT
    """,
    """
    CREATE TABLE option_quotes (
        id INTEGER PRIMARY KEY,
        instrument TEXT NOT NULL,
        asset TEXT NOT NULL CHECK (asset IN ('BTC', 'ETH')),
        strike REAL NOT NULL,
        expiry TEXT NOT NULL,
        is_call INTEGER NOT NULL CHECK (is_call IN (0, 1)),
        mark_iv REAL NOT NULL,
        bid_iv REAL,
        ask_iv REAL,
        underlying_price REAL NOT NULL,
        timestamp TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE fill_estimates (
        id INTEGER PRIMARY KEY,
        token_id TEXT NOT NULL,
        side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
        size_requested REAL NOT NULL,
        size_filled REAL NOT NULL,
        avg_price REAL NOT NULL,
        worst_price REAL NOT NULL,
        slippage REAL NOT NULL,
        fee_usd REAL NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE positions (
        market_id TEXT NOT NULL,
        outcome TEXT NOT NULL CHECK (outcome IN ('YES', 'NO')),
        size REAL NOT NULL,
        avg_price REAL NOT NULL,
        realized_pnl_usd REAL NOT NULL,
        PRIMARY KEY (market_id, outcome)
    ) STRICT
    """,
)

_MIGRATION_V2: tuple[str, ...] = (
    """
    CREATE INDEX IF NOT EXISTS idx_book_snapshots_market_ts
    ON book_snapshots(market_id, timestamp)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_book_snapshots_token_ts
    ON book_snapshots(token_id, timestamp)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fair_probabilities_market
    ON fair_probabilities(market_id, computed_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_signals_market
    ON signals(market_id, created_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fills_order
    ON fills(order_id, timestamp)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_orders_market
    ON orders(market_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_merges_timestamp
    ON merges(timestamp)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_complete_sets_found_at
    ON complete_sets(found_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_wallet_trades_wallet_ts
    ON wallet_trades(wallet, timestamp)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_option_quotes_instrument_ts
    ON option_quotes(instrument, timestamp)
    """,
)

MIGRATIONS: tuple[tuple[str, ...], ...] = (_MIGRATION_V1, _MIGRATION_V2)
SCHEMA_VERSION = len(MIGRATIONS)

_BOOK_SELECT = """
SELECT s.id, s.market_id, s.token_id, s.timestamp,
       l.side, l.level_index, l.price, l.size
FROM book_snapshots AS s
LEFT JOIN book_levels AS l ON l.snapshot_id = s.id
"""


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware UTC, got naive {value!r}")


def _dump_dt(name: str, value: datetime) -> str:
    """Fixed-width UTC text so lexicographic comparison matches time order."""
    _require_utc(name, value)
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _load_dt(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"expected a datetime string, got {value!r}")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"stored datetime must be timezone-aware, got {value!r}")
    return parsed.astimezone(UTC)


def _dump_strings(values: tuple[str, ...]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def _load_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise TypeError(f"expected JSON text, got {value!r}")
    parsed = json.loads(value)
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise TypeError(f"expected a JSON list of strings, got {value!r}")
    return tuple(parsed)


def _dump_meta(meta: dict[str, float]) -> str:
    clean = {str(key): float(item) for key, item in meta.items()}
    return json.dumps(clean, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _load_meta(value: object) -> dict[str, float]:
    if not isinstance(value, str):
        raise TypeError(f"expected JSON text, got {value!r}")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError(f"expected a JSON object, got {value!r}")
    meta: dict[str, float] = {}
    for key, item in parsed.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"meta[{key!r}] must be a number, got {item!r}")
        meta[str(key)] = float(item)
    return meta


def _opt_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def _since_clause(column: str, since: datetime | None) -> tuple[str, tuple[object, ...]]:
    if since is None:
        return "", ()
    return f" WHERE {column} >= ?", (_dump_dt("since", since),)


def _insert_sql(table: str, columns: tuple[str, ...], conflict: tuple[str, ...] = ()) -> str:
    names = ", ".join(columns)
    marks = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO {table} ({names}) VALUES ({marks})"
    if conflict:
        assignments = ", ".join(
            f"{column} = excluded.{column}" for column in columns if column not in conflict
        )
        keys = ", ".join(conflict)
        sql += f" ON CONFLICT({keys}) DO UPDATE SET {assignments}"
    return sql


def _insert_ignore_sql(table: str, columns: tuple[str, ...]) -> str:
    names = ", ".join(columns)
    marks = ", ".join("?" for _ in columns)
    return f"INSERT OR IGNORE INTO {table} ({names}) VALUES ({marks})"


_MARKET_COLUMNS = (
    "market_id",
    "event_id",
    "question",
    "asset",
    "kind",
    "strike_low",
    "strike_high",
    "expiry",
    "yes_token_id",
    "no_token_id",
    "resolution_source",
    "neg_risk",
    "volume_usd",
)
_INSERT_MARKET = _insert_sql("markets", _MARKET_COLUMNS, ("market_id",))

_FAIR_COLUMNS = ("market_id", "p_yes", "computed_at", "model", "spot", "meta")
_INSERT_FAIR = _insert_sql("fair_probabilities", _FAIR_COLUMNS)

_SIGNAL_COLUMNS = (
    "market_id",
    "outcome",
    "fair_price",
    "entry_price",
    "edge_after_costs",
    "size",
    "stop_price",
    "take_profit_price",
    "created_at",
    "reason",
)
_INSERT_SIGNAL = _insert_sql("signals", _SIGNAL_COLUMNS)

_ORDER_COLUMNS = (
    "order_id",
    "market_id",
    "token_id",
    "outcome",
    "side",
    "order_type",
    "size",
    "limit_price",
    "created_at",
    "status",
    "filled_size",
    "avg_fill_price",
    "strategy",
)
_INSERT_ORDER = _insert_sql("orders", _ORDER_COLUMNS, ("order_id",))

_FILL_COLUMNS = ("order_id", "price", "size", "fee_usd", "is_maker", "timestamp")
_INSERT_FILL = _insert_sql("fills", _FILL_COLUMNS)

_MERGE_COLUMNS = ("market_ids", "size", "cost_basis_usd", "proceeds_usd", "timestamp")
_INSERT_MERGE = _insert_sql("merges", _MERGE_COLUMNS)

_COMPLETE_SET_COLUMNS = (
    "market_ids",
    "token_ids",
    "size",
    "total_cost_usd",
    "payout_usd",
    "found_at",
)
_INSERT_COMPLETE_SET = _insert_sql("complete_sets", _COMPLETE_SET_COLUMNS)

_OUTCOME_COLUMNS = ("market_id", "resolved_yes", "resolved_at")
_INSERT_OUTCOME = _insert_sql("outcomes", _OUTCOME_COLUMNS, ("market_id",))

_WALLET_COLUMNS = (
    "wallet",
    "market_id",
    "outcome",
    "side",
    "price",
    "size",
    "timestamp",
    "tx_hash",
)
_INSERT_WALLET = _insert_ignore_sql("wallet_trades", _WALLET_COLUMNS)

_QUOTE_COLUMNS = (
    "instrument",
    "asset",
    "strike",
    "expiry",
    "is_call",
    "mark_iv",
    "bid_iv",
    "ask_iv",
    "underlying_price",
    "timestamp",
)
_INSERT_QUOTE = _insert_sql("option_quotes", _QUOTE_COLUMNS)

_ESTIMATE_COLUMNS = (
    "token_id",
    "side",
    "size_requested",
    "size_filled",
    "avg_price",
    "worst_price",
    "slippage",
    "fee_usd",
)
_INSERT_ESTIMATE = _insert_sql("fill_estimates", _ESTIMATE_COLUMNS)

_POSITION_COLUMNS = ("market_id", "outcome", "size", "avg_price", "realized_pnl_usd")
_INSERT_POSITION = _insert_sql("positions", _POSITION_COLUMNS, ("market_id", "outcome"))


def _market_values(market: CryptoMarket) -> tuple[object, ...]:
    return (
        market.market_id,
        market.event_id,
        market.question,
        market.asset.value,
        market.kind.value,
        _opt_float(market.strike_low),
        _opt_float(market.strike_high),
        _dump_dt("expiry", market.expiry),
        market.yes_token_id,
        market.no_token_id,
        market.resolution_source,
        int(market.neg_risk),
        float(market.volume_usd),
    )


def _market_from_row(row: sqlite3.Row) -> CryptoMarket:
    return CryptoMarket(
        market_id=str(row["market_id"]),
        event_id=str(row["event_id"]),
        question=str(row["question"]),
        asset=Asset(str(row["asset"])),
        kind=MarketKind(str(row["kind"])),
        strike_low=_opt_float(row["strike_low"]),
        strike_high=_opt_float(row["strike_high"]),
        expiry=_load_dt(row["expiry"]),
        yes_token_id=str(row["yes_token_id"]),
        no_token_id=str(row["no_token_id"]),
        resolution_source=str(row["resolution_source"]),
        neg_risk=bool(row["neg_risk"]),
        volume_usd=float(row["volume_usd"]),
    )


def _fair_values(fair: FairProbability) -> tuple[object, ...]:
    return (
        fair.market_id,
        float(fair.p_yes),
        _dump_dt("computed_at", fair.computed_at),
        fair.model,
        float(fair.spot),
        _dump_meta(fair.meta),
    )


def _fair_from_row(row: sqlite3.Row) -> FairProbability:
    return FairProbability(
        market_id=str(row["market_id"]),
        p_yes=float(row["p_yes"]),
        computed_at=_load_dt(row["computed_at"]),
        model=str(row["model"]),
        spot=float(row["spot"]),
        meta=_load_meta(row["meta"]),
    )


def _signal_values(signal: Signal) -> tuple[object, ...]:
    return (
        signal.market_id,
        signal.outcome.value,
        float(signal.fair_price),
        float(signal.entry_price),
        float(signal.edge_after_costs),
        float(signal.size),
        _opt_float(signal.stop_price),
        _opt_float(signal.take_profit_price),
        _dump_dt("created_at", signal.created_at),
        signal.reason,
    )


def _signal_from_row(row: sqlite3.Row) -> Signal:
    return Signal(
        market_id=str(row["market_id"]),
        outcome=Outcome(str(row["outcome"])),
        fair_price=float(row["fair_price"]),
        entry_price=float(row["entry_price"]),
        edge_after_costs=float(row["edge_after_costs"]),
        size=float(row["size"]),
        stop_price=_opt_float(row["stop_price"]),
        take_profit_price=_opt_float(row["take_profit_price"]),
        created_at=_load_dt(row["created_at"]),
        reason=str(row["reason"]),
    )


def _order_values(order: Order) -> tuple[object, ...]:
    return (
        order.order_id,
        order.market_id,
        order.token_id,
        order.outcome.value,
        order.side.value,
        order.order_type.value,
        float(order.size),
        _opt_float(order.limit_price),
        _dump_dt("created_at", order.created_at),
        order.status.value,
        float(order.filled_size),
        _opt_float(order.avg_fill_price),
        order.strategy,
    )


def _order_from_row(row: sqlite3.Row) -> Order:
    return Order(
        order_id=str(row["order_id"]),
        market_id=str(row["market_id"]),
        token_id=str(row["token_id"]),
        outcome=Outcome(str(row["outcome"])),
        side=BookSide(str(row["side"])),
        order_type=OrderType(str(row["order_type"])),
        size=float(row["size"]),
        limit_price=_opt_float(row["limit_price"]),
        created_at=_load_dt(row["created_at"]),
        status=OrderStatus(str(row["status"])),
        filled_size=float(row["filled_size"]),
        avg_fill_price=_opt_float(row["avg_fill_price"]),
        strategy=str(row["strategy"]),
    )


def _fill_values(fill: Fill) -> tuple[object, ...]:
    return (
        fill.order_id,
        float(fill.price),
        float(fill.size),
        float(fill.fee_usd),
        int(fill.is_maker),
        _dump_dt("timestamp", fill.timestamp),
    )


def _fill_from_row(row: sqlite3.Row) -> Fill:
    return Fill(
        order_id=str(row["order_id"]),
        price=float(row["price"]),
        size=float(row["size"]),
        fee_usd=float(row["fee_usd"]),
        is_maker=bool(row["is_maker"]),
        timestamp=_load_dt(row["timestamp"]),
    )


def _merge_values(merge: Merge) -> tuple[object, ...]:
    return (
        _dump_strings(merge.market_ids),
        float(merge.size),
        float(merge.cost_basis_usd),
        float(merge.proceeds_usd),
        _dump_dt("timestamp", merge.timestamp),
    )


def _merge_from_row(row: sqlite3.Row) -> Merge:
    return Merge(
        market_ids=_load_strings(row["market_ids"]),
        size=float(row["size"]),
        cost_basis_usd=float(row["cost_basis_usd"]),
        proceeds_usd=float(row["proceeds_usd"]),
        timestamp=_load_dt(row["timestamp"]),
    )


def _complete_set_values(opp: CompleteSetOpportunity) -> tuple[object, ...]:
    return (
        _dump_strings(opp.market_ids),
        _dump_strings(opp.token_ids),
        float(opp.size),
        float(opp.total_cost_usd),
        float(opp.payout_usd),
        _dump_dt("found_at", opp.found_at),
    )


def _complete_set_from_row(row: sqlite3.Row) -> CompleteSetOpportunity:
    return CompleteSetOpportunity(
        market_ids=_load_strings(row["market_ids"]),
        token_ids=_load_strings(row["token_ids"]),
        size=float(row["size"]),
        total_cost_usd=float(row["total_cost_usd"]),
        payout_usd=float(row["payout_usd"]),
        found_at=_load_dt(row["found_at"]),
    )


def _outcome_values(outcome: MarketOutcome) -> tuple[object, ...]:
    return (
        outcome.market_id,
        int(outcome.resolved_yes),
        _dump_dt("resolved_at", outcome.resolved_at),
    )


def _outcome_from_row(row: sqlite3.Row) -> MarketOutcome:
    return MarketOutcome(
        market_id=str(row["market_id"]),
        resolved_yes=bool(row["resolved_yes"]),
        resolved_at=_load_dt(row["resolved_at"]),
    )


def _wallet_values(trade: WalletTrade) -> tuple[object, ...]:
    return (
        trade.wallet,
        trade.market_id,
        trade.outcome.value,
        trade.side.value,
        float(trade.price),
        float(trade.size),
        _dump_dt("timestamp", trade.timestamp),
        trade.tx_hash,
    )


def _wallet_from_row(row: sqlite3.Row) -> WalletTrade:
    return WalletTrade(
        wallet=str(row["wallet"]),
        market_id=str(row["market_id"]),
        outcome=Outcome(str(row["outcome"])),
        side=BookSide(str(row["side"])),
        price=float(row["price"]),
        size=float(row["size"]),
        timestamp=_load_dt(row["timestamp"]),
        tx_hash=str(row["tx_hash"]),
    )


def _quote_values(quote: OptionQuote) -> tuple[object, ...]:
    return (
        quote.instrument,
        quote.asset.value,
        float(quote.strike),
        _dump_dt("expiry", quote.expiry),
        int(quote.is_call),
        float(quote.mark_iv),
        _opt_float(quote.bid_iv),
        _opt_float(quote.ask_iv),
        float(quote.underlying_price),
        _dump_dt("timestamp", quote.timestamp),
    )


def _quote_from_row(row: sqlite3.Row) -> OptionQuote:
    return OptionQuote(
        instrument=str(row["instrument"]),
        asset=Asset(str(row["asset"])),
        strike=float(row["strike"]),
        expiry=_load_dt(row["expiry"]),
        is_call=bool(row["is_call"]),
        mark_iv=float(row["mark_iv"]),
        bid_iv=_opt_float(row["bid_iv"]),
        ask_iv=_opt_float(row["ask_iv"]),
        underlying_price=float(row["underlying_price"]),
        timestamp=_load_dt(row["timestamp"]),
    )


def _estimate_values(estimate: FillEstimate) -> tuple[object, ...]:
    return (
        estimate.token_id,
        estimate.side.value,
        float(estimate.size_requested),
        float(estimate.size_filled),
        float(estimate.avg_price),
        float(estimate.worst_price),
        float(estimate.slippage),
        float(estimate.fee_usd),
    )


def _estimate_from_row(row: sqlite3.Row) -> FillEstimate:
    return FillEstimate(
        token_id=str(row["token_id"]),
        side=BookSide(str(row["side"])),
        size_requested=float(row["size_requested"]),
        size_filled=float(row["size_filled"]),
        avg_price=float(row["avg_price"]),
        worst_price=float(row["worst_price"]),
        slippage=float(row["slippage"]),
        fee_usd=float(row["fee_usd"]),
    )


def _position_values(position: Position) -> tuple[object, ...]:
    return (
        position.market_id,
        position.outcome.value,
        float(position.size),
        float(position.avg_price),
        float(position.realized_pnl_usd),
    )


def _position_from_row(row: sqlite3.Row) -> Position:
    return Position(
        market_id=str(row["market_id"]),
        outcome=Outcome(str(row["outcome"])),
        size=float(row["size"]),
        avg_price=float(row["avg_price"]),
        realized_pnl_usd=float(row["realized_pnl_usd"]),
    )


def _books_from_rows(rows: Iterable[sqlite3.Row]) -> list[tuple[str, OrderBook]]:
    snapshots: list[tuple[str, OrderBook]] = []
    current_id: int | None = None
    market_id = ""
    token_id = ""
    timestamp = ""
    bids: list[PriceLevel] = []
    asks: list[PriceLevel] = []

    def flush() -> None:
        if current_id is None:
            return
        snapshots.append(
            (
                market_id,
                OrderBook(
                    token_id=token_id,
                    bids=tuple(bids),
                    asks=tuple(asks),
                    timestamp=_load_dt(timestamp),
                ),
            )
        )

    for row in rows:
        row_id = int(row["id"])
        if row_id != current_id:
            flush()
            current_id = row_id
            market_id = str(row["market_id"])
            token_id = str(row["token_id"])
            timestamp = str(row["timestamp"])
            bids = []
            asks = []
        if row["side"] is None:
            continue
        level = PriceLevel(float(row["price"]), float(row["size"]))
        side = str(row["side"])
        if side == "bid":
            bids.append(level)
        elif side == "ask":
            asks.append(level)
        else:
            raise ValueError(f"unknown book level side {side!r}")
    flush()
    return snapshots


class Journal:
    """Persist what the bot saw and what it did.

    Append-only, one row per call: book snapshots, fair probabilities, signals,
    fills, merges, complete-set opportunities, option quotes, and fill
    estimates. Fair values and books are kept at every tick so a later report
    can score the model and so a backtest can replay the books.

    Upserted by identity: markets, orders, outcomes, and positions.

    Wallet trades are inserted once per trade identity so polling the data API
    again does not double-count.

    Datetimes are stored as UTC text ``YYYY-MM-DDTHH:MM:SS.ffffff+00:00``.
    ``since`` filters are inclusive.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON").fetchall()
            conn.execute("PRAGMA journal_mode = WAL").fetchall()
            conn.execute("PRAGMA busy_timeout = 5000").fetchall()
            self._conn = conn
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def schema_version(self) -> int:
        """Current ``PRAGMA user_version`` (0 before :meth:`migrate`)."""
        row = self._connection().execute("PRAGMA user_version").fetchone()
        if row is None:
            return 0
        return int(row[0])

    def migrate(self) -> None:
        """Apply any pending migrations and set ``PRAGMA user_version``."""
        conn = self._connection()
        version = self.schema_version()
        while version < SCHEMA_VERSION:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in MIGRATIONS[version]:
                    conn.execute(statement)
                version += 1
                conn.execute(f"PRAGMA user_version = {version}")
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _write(self, sql: str, params: tuple[object, ...]) -> None:
        self.migrate()
        with self._transaction() as conn:
            conn.execute(sql, params)

    def _query(self, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        self.migrate()
        return list(self._connection().execute(sql, params))

    def save_market(self, market: CryptoMarket) -> None:
        self._write(_INSERT_MARKET, _market_values(market))

    def markets(self) -> list[CryptoMarket]:
        rows = self._query("SELECT * FROM markets ORDER BY rowid")
        return [_market_from_row(row) for row in rows]

    def save_book(self, market_id: str, book: OrderBook) -> None:
        """Append one book snapshot. Every tick of every market is kept."""
        self.migrate()
        levels: list[tuple[str, int, float, float]] = []
        levels.extend(
            ("bid", index, float(level.price), float(level.size))
            for index, level in enumerate(book.bids)
        )
        levels.extend(
            ("ask", index, float(level.price), float(level.size))
            for index, level in enumerate(book.asks)
        )
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO book_snapshots (market_id, token_id, timestamp)
                VALUES (?, ?, ?)
                """,
                (market_id, book.token_id, _dump_dt("timestamp", book.timestamp)),
            )
            snapshot_id = cursor.lastrowid
            if snapshot_id is None:
                raise RuntimeError("SQLite did not return a book snapshot id")
            conn.executemany(
                """
                INSERT INTO book_levels (snapshot_id, side, level_index, price, size)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(snapshot_id, side, index, price, size) for side, index, price, size in levels],
            )

    def book_snapshots(
        self,
        market_id: str | None = None,
        since: datetime | None = None,
    ) -> list[tuple[str, OrderBook]]:
        """Snapshots in write order, as ``(market_id, book)``."""
        self.migrate()
        filters: list[str] = []
        params: list[object] = []
        if market_id is not None:
            filters.append("s.market_id = ?")
            params.append(market_id)
        if since is not None:
            filters.append("s.timestamp >= ?")
            params.append(_dump_dt("since", since))
        where = f" WHERE {' AND '.join(filters)}" if filters else ""
        rows = self._connection().execute(
            _BOOK_SELECT + where + " ORDER BY s.id, l.side, l.level_index",
            params,
        )
        return _books_from_rows(rows)

    def save_fair_probability(self, fair: FairProbability) -> None:
        self._write(_INSERT_FAIR, _fair_values(fair))

    def fair_probabilities(self, since: datetime | None = None) -> list[FairProbability]:
        where, params = _since_clause("computed_at", since)
        rows = self._query(
            "SELECT * FROM fair_probabilities" + where + " ORDER BY id",
            params,
        )
        return [_fair_from_row(row) for row in rows]

    def save_signal(self, signal: Signal) -> None:
        self._write(_INSERT_SIGNAL, _signal_values(signal))

    def signals(self, since: datetime | None = None) -> list[Signal]:
        where, params = _since_clause("created_at", since)
        rows = self._query("SELECT * FROM signals" + where + " ORDER BY id", params)
        return [_signal_from_row(row) for row in rows]

    def save_order(self, order: Order) -> None:
        self._write(_INSERT_ORDER, _order_values(order))

    def orders(self) -> list[Order]:
        rows = self._query("SELECT * FROM orders ORDER BY rowid")
        return [_order_from_row(row) for row in rows]

    def save_fill(self, fill: Fill) -> None:
        self._write(_INSERT_FILL, _fill_values(fill))

    def fills(self, since: datetime | None = None) -> list[Fill]:
        where, params = _since_clause("timestamp", since)
        rows = self._query("SELECT * FROM fills" + where + " ORDER BY id", params)
        return [_fill_from_row(row) for row in rows]

    def save_merge(self, merge: Merge) -> None:
        self._write(_INSERT_MERGE, _merge_values(merge))

    def merges(self) -> list[Merge]:
        rows = self._query("SELECT * FROM merges ORDER BY id")
        return [_merge_from_row(row) for row in rows]

    def save_complete_set(self, opp: CompleteSetOpportunity) -> None:
        self._write(_INSERT_COMPLETE_SET, _complete_set_values(opp))

    def complete_sets(self) -> list[CompleteSetOpportunity]:
        rows = self._query("SELECT * FROM complete_sets ORDER BY id")
        return [_complete_set_from_row(row) for row in rows]

    def save_outcome(self, outcome: MarketOutcome) -> None:
        self._write(_INSERT_OUTCOME, _outcome_values(outcome))

    def outcomes(self) -> list[MarketOutcome]:
        rows = self._query("SELECT * FROM outcomes ORDER BY rowid")
        return [_outcome_from_row(row) for row in rows]

    def unresolved_market_ids(self) -> list[str]:
        """Saved markets that do not yet have an outcome."""
        rows = self._query(
            """
            SELECT m.market_id
            FROM markets AS m
            LEFT JOIN outcomes AS o ON o.market_id = m.market_id
            WHERE o.market_id IS NULL
            ORDER BY m.rowid
            """
        )
        return [str(row["market_id"]) for row in rows]

    def save_wallet_trades(self, trades: list[WalletTrade]) -> None:
        self.migrate()
        if not trades:
            return
        rows = [_wallet_values(trade) for trade in trades]
        with self._transaction() as conn:
            conn.executemany(_INSERT_WALLET, rows)

    def wallet_trades(self, wallet: str | None = None) -> list[WalletTrade]:
        if wallet is None:
            rows = self._query("SELECT * FROM wallet_trades ORDER BY id")
        else:
            rows = self._query(
                "SELECT * FROM wallet_trades WHERE wallet = ? ORDER BY id",
                (wallet,),
            )
        return [_wallet_from_row(row) for row in rows]

    def save_option_quote(self, quote: OptionQuote) -> None:
        self._write(_INSERT_QUOTE, _quote_values(quote))

    def option_quotes(self) -> list[OptionQuote]:
        rows = self._query("SELECT * FROM option_quotes ORDER BY id")
        return [_quote_from_row(row) for row in rows]

    def save_fill_estimate(self, estimate: FillEstimate) -> None:
        self._write(_INSERT_ESTIMATE, _estimate_values(estimate))

    def fill_estimates(self) -> list[FillEstimate]:
        rows = self._query("SELECT * FROM fill_estimates ORDER BY id")
        return [_estimate_from_row(row) for row in rows]

    def save_position(self, position: Position) -> None:
        self._write(_INSERT_POSITION, _position_values(position))

    def positions(self) -> list[Position]:
        rows = self._query("SELECT * FROM positions ORDER BY rowid")
        return [_position_from_row(row) for row in rows]
