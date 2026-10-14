"""Task E. Main loop: scan -> fair price -> edge -> risk -> paper execution -> journal."""

from __future__ import annotations

import dataclasses
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx

from polybot.config import Settings
from polybot.deribit.client import DeribitClient
from polybot.deribit.pricing import fair_probability
from polybot.engine.risk import RiskManager
from polybot.execution.complete_sets import find_binary_sets, find_range_sets
from polybot.execution.costs import edge_after_costs, estimate_fill, max_size_for_edge
from polybot.execution.paper import PaperBroker
from polybot.models import (
    Asset,
    BookSide,
    CompleteSetOpportunity,
    CryptoMarket,
    FairProbability,
    Fill,
    OptionQuote,
    Order,
    OrderBook,
    OrderType,
    Outcome,
    Signal,
    _require_utc,
)
from polybot.polymarket.client import PolymarketClient
from polybot.polymarket.parser import parse_market
from polybot.storage.journal import Journal

logger = logging.getLogger(__name__)

DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 0.5
MAX_LOOP_BACKOFF_SECONDS = 60.0
_SOURCE_ERRORS = (httpx.HTTPError, OSError, RuntimeError, ValueError)


def _call_with_retry[T](
    fn: Callable[..., T],
    *args: object,
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """Call an API function, retrying transient HTTP/network failures with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn(*args)
        except (httpx.HTTPError, OSError) as exc:
            last_exc = exc
            if attempt + 1 >= max_attempts:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                "API call %s failed (attempt %d/%d): %s; retrying in %.2fs",
                getattr(fn, "__name__", repr(fn)),
                attempt + 1,
                max_attempts,
                exc,
                delay,
            )
            sleep_fn(delay)
    assert last_exc is not None
    raise last_exc


def _execute_complete_set(
    opp: CompleteSetOpportunity,
    token_lookup: dict[str, tuple[CryptoMarket, Outcome]],
    books: dict[str, OrderBook],
    broker: PaperBroker,
    journal: Journal,
    now: datetime,
    *,
    is_binary: bool,
) -> None:
    """Submit market BUY orders for every leg of a complete set and merge binary sets."""
    if opp.size <= 0.0 or not opp.token_ids:
        return

    leg_items: list[tuple[CryptoMarket, Outcome, str, OrderBook]] = []
    for token_id in opp.token_ids:
        info = token_lookup.get(token_id)
        book = books.get(token_id)
        if info is None or book is None:
            return
        leg_items.append((info[0], info[1], token_id, book))

    fills_by_leg: list[list[Fill]] = []
    for market, outcome, token_id, book in leg_items:
        order = Order(
            order_id=uuid.uuid4().hex,
            market_id=market.market_id,
            token_id=token_id,
            outcome=outcome,
            side=BookSide.BUY,
            order_type=OrderType.MARKET,
            size=opp.size,
            limit_price=None,
            created_at=now,
            strategy="complete_set",
        )
        fills = broker.submit_market(order, book) or []
        journal.save_order(order)
        for fill in fills:
            journal.save_fill(fill)
        fills_by_leg.append(fills)

    if is_binary and len(set(opp.market_ids)) == 1:
        market_id = opp.market_ids[0]
        pos_yes = broker.positions.get((market_id, Outcome.YES))
        pos_no = broker.positions.get((market_id, Outcome.NO))
        if pos_yes is not None and pos_no is not None:
            merge_size = min(pos_yes.size, pos_no.size)
        else:
            leg_sizes = [sum(f.size for f in leg_fills) for leg_fills in fills_by_leg]
            merge_size = min(leg_sizes) if leg_sizes else 0.0

        if merge_size > 0.0:
            merge = broker.merge(market_id, merge_size, now)
            journal.save_merge(merge)


def _with_market_quote(fair: FairProbability, book: OrderBook | None) -> FairProbability:
    """Record the YES book next to the model, so the report scores both at the same moment.

    A missing side is stored as bid 0 / ask 1, which is what an empty side means for a trader.
    """
    if book is None:
        return fair
    meta = {
        **fair.meta,
        "market_yes_bid": book.best_bid if book.best_bid is not None else 0.0,
        "market_yes_ask": book.best_ask if book.best_ask is not None else 1.0,
    }
    return dataclasses.replace(fair, meta=meta)


def run_once(
    settings: Settings,
    *,
    polymarket: PolymarketClient | None = None,
    deribit: DeribitClient | None = None,
    journal: Journal | None = None,
    broker: PaperBroker | None = None,
    risk: RiskManager | None = None,
    now: datetime | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    known_markets: dict[str, CryptoMarket] | None = None,
    save_snapshots: bool = True,
) -> None:
    """One tick. `save_snapshots` persists books and fair probabilities and checks outcomes;
    the loop sets it every `settings.snapshot_seconds`, trading runs on every tick."""
    tick_now = now if now is not None else datetime.now(UTC)
    _require_utc("now", tick_now)
    started = time.monotonic()

    pm = polymarket if polymarket is not None else PolymarketClient()
    db = deribit if deribit is not None else DeribitClient()
    jr = journal if journal is not None else Journal(settings.db_path)
    br = broker if broker is not None else PaperBroker(settings.taker_fee_rate)
    rm = risk if risk is not None else RiskManager(settings)

    jr.migrate()
    if known_markets is None:
        known_markets = {market.market_id: market for market in jr.markets()}
    markets_cache = known_markets
    n_signals = 0
    n_complete_sets = 0

    # 1. Fetch and parse active Polymarket crypto markets.
    raw_markets = (
        _call_with_retry(pm.fetch_crypto_markets, settings.assets, sleep_fn=sleep_fn) or []
    )
    active_markets: list[CryptoMarket] = []
    token_lookup: dict[str, tuple[CryptoMarket, Outcome]] = {}

    for raw in raw_markets:
        market = parse_market(raw)
        if market is None or market.asset not in settings.assets:
            continue
        markets_cache[market.market_id] = market
        if market.expiry <= tick_now:
            continue
        jr.save_market(market)
        active_markets.append(market)
        token_lookup[market.yes_token_id] = (market, Outcome.YES)
        token_lookup[market.no_token_id] = (market, Outcome.NO)

    # 2. Fetch order books in batch and save every snapshot to the journal.
    token_ids: list[str] = []
    seen_tokens: set[str] = set()
    for market in active_markets:
        for tid in (market.yes_token_id, market.no_token_id):
            if tid not in seen_tokens:
                seen_tokens.add(tid)
                token_ids.append(tid)

    books: dict[str, OrderBook] = {}
    if token_ids:
        books = _call_with_retry(pm.fetch_order_books, token_ids, sleep_fn=sleep_fn) or {}
        if save_snapshots:
            for market in active_markets:
                for tid in (market.yes_token_id, market.no_token_id):
                    book = books.get(tid)
                    if book is not None:
                        jr.save_book(market.market_id, book)

    # 3. Fetch Deribit option chains and the Binance basis, compute fair probabilities.
    # One asset's outage skips its pricing for this tick instead of failing the whole tick.
    needed_assets: list[Asset] = []
    for asset in settings.assets:
        if any(m.asset is asset for m in active_markets) and asset not in needed_assets:
            needed_assets.append(asset)

    chains: dict[Asset, list[OptionQuote]] = {}
    bases: dict[Asset, float] = {}
    for asset in needed_assets:
        try:
            chain = _call_with_retry(db.fetch_option_chain, asset, sleep_fn=sleep_fn) or []
            basis = _call_with_retry(db.fetch_resolution_basis, asset, sleep_fn=sleep_fn)
        except _SOURCE_ERRORS as exc:
            logger.warning("No %s pricing this tick: %s", asset.value, exc)
            continue
        chains[asset] = chain
        bases[asset] = basis

    fairs: dict[str, FairProbability] = {}
    for market in active_markets:
        chain = chains.get(market.asset)
        if not chain:
            continue
        fair = fair_probability(
            market, chain, tick_now, resolution_basis=bases[market.asset]
        )
        if fair is None:
            continue
        fair = _with_market_quote(fair, books.get(market.yes_token_id))
        fairs[market.market_id] = fair
        if save_snapshots:
            jr.save_fair_probability(fair)

    # 4. Scan complete sets (binary YES+NO and mutually exclusive range sets).
    if active_markets and books:
        binary_sets = (
            find_binary_sets(active_markets, books, settings.taker_fee_rate, tick_now) or []
        )
        for opp in binary_sets:
            jr.save_complete_set(opp)
            _execute_complete_set(
                opp, token_lookup, books, br, jr, tick_now, is_binary=True
            )

        range_sets = (
            find_range_sets(active_markets, books, settings.taker_fee_rate, tick_now) or []
        )
        for opp in range_sets:
            jr.save_complete_set(opp)
            _execute_complete_set(
                opp, token_lookup, books, br, jr, tick_now, is_binary=False
            )
        n_complete_sets = len(binary_sets) + len(range_sets)

    # 5. Directional signals -> risk sizing -> paper execution -> journal.
    min_trade_tte = timedelta(minutes=settings.min_trade_tte_minutes)
    for market in active_markets:
        fair = fairs.get(market.market_id)
        if fair is None or market.expiry - tick_now < min_trade_tte:
            continue
        for outcome in (Outcome.YES, Outcome.NO):
            token_id = market.token_id(outcome)
            book = books.get(token_id)
            if book is None or not book.asks:
                continue

            fair_p = fair.p(outcome)
            max_book_size = max_size_for_edge(
                book, fair_p, settings.min_edge, settings.taker_fee_rate
            )
            if max_book_size <= 0.0:
                continue

            initial_fill = estimate_fill(
                book, BookSide.BUY, max_book_size, settings.taker_fee_rate
            )
            if initial_fill.size_filled <= 0.0:
                continue

            initial_edge = edge_after_costs(fair_p, initial_fill)
            if initial_edge < settings.min_edge:
                continue

            candidate = Signal(
                market_id=market.market_id,
                outcome=outcome,
                fair_price=fair_p,
                entry_price=initial_fill.avg_price,
                edge_after_costs=initial_edge,
                size=initial_fill.size_filled,
                stop_price=None,
                take_profit_price=fair_p,
                created_at=tick_now,
                reason="model_edge",
            )
            allowed_size = rm.size(candidate, market, tick_now)
            if allowed_size <= 0.0:
                continue

            trade_size = min(allowed_size, initial_fill.size_filled)
            fill_est = estimate_fill(
                book, BookSide.BUY, trade_size, settings.taker_fee_rate
            )
            if fill_est.size_filled <= 0.0:
                continue

            final_edge = edge_after_costs(fair_p, fill_est)
            if final_edge < settings.min_edge:
                continue

            signal = Signal(
                market_id=market.market_id,
                outcome=outcome,
                fair_price=fair_p,
                entry_price=fill_est.avg_price,
                edge_after_costs=final_edge,
                size=fill_est.size_filled,
                stop_price=None,
                take_profit_price=fair_p,
                created_at=tick_now,
                reason="model_edge",
            )
            jr.save_signal(signal)
            n_signals += 1

            order = Order(
                order_id=uuid.uuid4().hex,
                market_id=market.market_id,
                token_id=token_id,
                outcome=outcome,
                side=BookSide.BUY,
                order_type=OrderType.MARKET,
                size=signal.size,
                limit_price=None,
                created_at=tick_now,
                strategy="signal",
            )
            fills = br.submit_market(order, book) or []
            jr.save_order(order)
            for fill in fills:
                jr.save_fill(fill)

    # 6. Settle resolved markets and update risk PnL. Only expired markets can resolve.
    n_settled = 0
    if save_snapshots:
        for market_id in jr.unresolved_market_ids() or []:
            resolved_market = markets_cache.get(market_id)
            if resolved_market is None or resolved_market.expiry > tick_now:
                continue
            try:
                outcome_res = _call_with_retry(pm.fetch_outcome, market_id, sleep_fn=sleep_fn)
            except _SOURCE_ERRORS as exc:
                logger.warning("Outcome check failed for %s: %s", market_id, exc)
                continue
            if outcome_res is None:
                continue
            jr.save_outcome(outcome_res)
            rm.record_pnl(resolved_market, br.settle(outcome_res), tick_now)
            n_settled += 1

    logger.info(
        "tick %s: %d markets, %d books, %d priced, %d signals, %d complete sets, "
        "%d settled%s, %.1fs",
        tick_now.strftime("%H:%M:%S"),
        len(active_markets),
        len(books),
        len(fairs),
        n_signals,
        n_complete_sets,
        n_settled,
        ", snapshot saved" if save_snapshots else "",
        time.monotonic() - started,
    )


def run(
    settings: Settings,
    *,
    polymarket: PolymarketClient | None = None,
    deribit: DeribitClient | None = None,
    journal: Journal | None = None,
    broker: PaperBroker | None = None,
    risk: RiskManager | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Loop forever; survive API errors with backoff; stop cleanly on Ctrl+C."""
    pm = polymarket if polymarket is not None else PolymarketClient()
    db = deribit if deribit is not None else DeribitClient()
    jr = journal if journal is not None else Journal(settings.db_path)
    br = broker if broker is not None else PaperBroker(settings.taker_fee_rate)
    rm = risk if risk is not None else RiskManager(settings)
    jr.migrate()
    known_markets = {market.market_id: market for market in jr.markets()}
    snapshot_every = timedelta(seconds=settings.snapshot_seconds)
    last_snapshot: datetime | None = None

    consecutive_errors = 0
    try:
        while True:
            tick_now = datetime.now(UTC)
            save_snapshots = last_snapshot is None or tick_now - last_snapshot >= snapshot_every
            try:
                run_once(
                    settings,
                    polymarket=pm,
                    deribit=db,
                    journal=jr,
                    broker=br,
                    risk=rm,
                    now=tick_now,
                    sleep_fn=sleep_fn,
                    known_markets=known_markets,
                    save_snapshots=save_snapshots,
                )
                if save_snapshots:
                    last_snapshot = tick_now
                consecutive_errors = 0
                sleep_fn(settings.poll_seconds)
            except KeyboardInterrupt:
                raise
            except Exception:
                consecutive_errors += 1
                backoff = min(
                    MAX_LOOP_BACKOFF_SECONDS,
                    DEFAULT_RETRY_BASE_DELAY * (2 ** (consecutive_errors - 1)),
                )
                logger.exception(
                    "Tick failed (consecutive_errors=%d); backing off for %.2fs",
                    consecutive_errors,
                    backoff,
                )
                sleep_fn(backoff)
    except KeyboardInterrupt:
        logger.info("Shutting down polybot loop cleanly on KeyboardInterrupt.")

