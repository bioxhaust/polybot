"""Task G. Public wallet trades and per-wallet performance stats.

Data API v2 (``/v2/trades``, ``/v2/activity``). No private keys and no orders.

Realized PnL is cash received minus cash spent: sells, redeems, and merges
minus buys and splits. Shares still held are carried at zero, so an open buy
counts against PnL until a sell, merge, or redeem returns the cash. On a flat
book this matches Polymarket's published realized PnL. Drawdown is the
peak-to-trough of that same cash curve. Within one timestamp the order is
split, then buys, then sells, then merges, then redeems.

``share_maker`` needs both the taker feed and the full feed (``taker_only=false``
is a superset). It stays ``None`` when that second pass is skipped or truncated.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import httpx

from polybot.models import BookSide, Outcome, WalletTrade

DATA_API_URL = "https://data-api.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com"

# Cited in the plan as Lennie. The public profile name on this proxy is manwithahat.
ARTICLE_WALLETS: tuple[str, ...] = ("0x0f1b8f81fdcb47b6c3292b8dbf6c2c16b3c4d367",)

MIN_TRADES = 200
MIN_ACTIVE_DAYS = 30
MIN_CRYPTO_SHARE = 0.5

_PAGE_LIMIT = 1000
_MAX_ATTEMPTS = 4
_EPS = 1e-9
_RETRYABLE = frozenset({429, 503})

_ASSET = re.compile(
    r"\b(?:bitcoin|btc|ethereum|ether|eth|solana|sol|xrp|ripple|dogecoin|doge|"
    r"bnb|cardano|ada|litecoin|ltc|avalanche|avax|chainlink|link|hyperliquid|hype)\b",
    re.IGNORECASE,
)
_PRICE = re.compile(
    r"\b(?:price|above|below|between|over|under|less than|greater than|more than|"
    r"dip to|reach|hit|up or down|higher|lower|at least|at most|all[- ]time high)\b",
    re.IGNORECASE,
)
_YES = frozenset({"yes", "up", "higher"})
_NO = frozenset({"no", "down", "lower"})
_ACTION_RANK = {"SPLIT": 0, "TRADE": 1, "MERGE": 2, "REDEEM": 3}


@dataclass(frozen=True, slots=True)
class WalletStats:
    wallet: str
    n_trades: int
    n_markets: int
    realized_pnl_usd: float
    max_drawdown_usd: float
    active_days: int
    share_crypto_markets: float  # fraction of volume in crypto price markets
    share_maker: float | None  # fraction of fills as maker, if derivable
    n_merges: int


@dataclass(frozen=True, slots=True)
class WalletFill(WalletTrade):
    """A public fill or inventory action.

    MERGE and SPLIT imply both YES and NO of a binary market. Their ``outcome``
    is YES only as a placeholder; ``price`` is USDC per complete set.
    """

    title: str = ""
    slug: str = ""
    is_maker: bool | None = None
    action: str = "TRADE"


class WalletClient:
    def __init__(self, http: httpx.Client | None = None, *, max_pages: int = 500) -> None:
        self._http = http or httpx.Client(timeout=10.0)
        self._owns_http = http is None
        self._max_pages = max_pages

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def fetch_trades(
        self,
        wallet: str,
        since: datetime | None = None,
        *,
        label_roles: bool = False,
    ) -> list[WalletTrade]:
        since = _as_utc("since", since)
        params = _trade_params(since, user=wallet, taker_only=False)
        rows, complete = self._paginate(f"{DATA_API_URL}/v2/trades", params)
        flags: list[bool | None]
        if label_roles and complete:
            taker_rows, taker_complete = self._paginate(
                f"{DATA_API_URL}/v2/trades",
                _trade_params(since, user=wallet, taker_only=True),
            )
            flags = _maker_flags(rows, taker_rows) if taker_complete else [None] * len(rows)
        else:
            flags = [None] * len(rows)
        return _collect_trades(rows, flags, since, fallback_wallet=wallet)

    def fetch_market_trades(self, market_id: str, since: datetime | None = None) -> list[WalletTrade]:
        """All public trades in one market, used to find active wallets."""
        since = _as_utc("since", since)
        params = _trade_params(since, condition=market_id, taker_only=False)
        rows, _complete = self._paginate(f"{DATA_API_URL}/v2/trades", params)
        return _collect_trades(rows, [None] * len(rows), since, fallback_wallet=None)

    def fetch_actions(
        self,
        wallet: str,
        since: datetime | None = None,
        *,
        label_roles: bool = False,
    ) -> list[WalletTrade]:
        """Trades plus splits, merges, and redeems for ``compute_stats``."""
        since = _as_utc("since", since)
        actions = list(self.fetch_trades(wallet, since, label_roles=label_roles))
        params: dict[str, Any] = {
            "user": wallet,
            "type": "MERGE,REDEEM,SPLIT",
            "limit": _PAGE_LIMIT,
            "start": _start_param(since),
            "exclude_deposits_withdrawals": True,
        }
        rows, _complete = self._paginate(f"{DATA_API_URL}/v2/activity", params)
        for row in rows:
            parsed = _parse_activity(row, fallback_wallet=wallet)
            if parsed is not None and _in_window(parsed, since):
                actions.append(parsed)
        return actions

    def fetch_crypto_market_ids(self, *, limit_per_tag: int = 20) -> list[str]:
        """Condition ids of active crypto price markets, from Gamma."""
        seen: set[str] = set()
        ordered: list[str] = []
        for tag in ("crypto-prices", "bitcoin", "ethereum"):
            body = self._get(
                f"{GAMMA_URL}/events",
                {
                    "active": "true",
                    "closed": "false",
                    "limit": limit_per_tag,
                    "tag_slug": tag,
                    "order": "volume",
                    "ascending": "false",
                },
            )
            if not isinstance(body, list):
                continue
            for event in body:
                if not isinstance(event, dict):
                    continue
                for market in event.get("markets") or []:
                    if not isinstance(market, dict) or market.get("closed") is True:
                        continue
                    question = str(market.get("question") or "")
                    slug = str(market.get("slug") or "")
                    if not is_crypto_price_market(question, slug):
                        continue
                    condition = market.get("conditionId") or market.get("condition_id")
                    if not isinstance(condition, str) or not condition or condition in seen:
                        continue
                    seen.add(condition)
                    ordered.append(condition)
        return ordered

    def _paginate(self, url: str, params: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _page in range(self._max_pages):
            query = dict(params)
            if cursor is not None:
                query["cursor"] = cursor
            body = self._get(url, query)
            if isinstance(body, list):
                rows.extend(item for item in body if isinstance(item, dict))
                return rows, True
            if not isinstance(body, dict):
                return rows, False
            data = body.get("data") or []
            if not isinstance(data, list):
                return rows, False
            rows.extend(item for item in data if isinstance(item, dict))
            pagination = body.get("pagination") or {}
            if not isinstance(pagination, dict):
                return rows, True
            next_cursor = pagination.get("next_cursor")
            has_more = pagination.get("has_more") is True
            if not next_cursor or not has_more:
                return rows, not has_more
            if not isinstance(next_cursor, str) or next_cursor in seen_cursors:
                return rows, False
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        return rows, False

    def _get(self, url: str, params: dict[str, Any]) -> Any:
        response: httpx.Response | None = None
        for attempt in range(_MAX_ATTEMPTS):
            response = self._http.get(url, params=params)
            if response.status_code not in _RETRYABLE or attempt == _MAX_ATTEMPTS - 1:
                response.raise_for_status()
                return response.json()
            time.sleep(_retry_delay(response))
        raise RuntimeError(f"no response from {url}")


def is_crypto_price_market(title: str, slug: str = "") -> bool:
    """True for BTC/ETH/other spot price questions (above, below, range, up/down)."""
    text = f"{title} {slug.replace('-', ' ')}"
    return _ASSET.search(text) is not None and _PRICE.search(text) is not None


def compute_stats(wallet: str, trades: list[WalletTrade]) -> WalletStats:
    """Cash PnL, drawdown, active days, crypto share, maker share, merges."""
    ordered = sorted(enumerate(trades), key=lambda item: _sort_key(item[0], item[1]))
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    crypto_notional = 0.0
    total_notional = 0.0
    maker_fills = 0
    labeled_fills = 0
    trade_markets: set[str] = set()
    trade_days: set[date] = set()
    n_trades = 0
    n_merges = 0

    for _index, trade in ordered:
        if trade.size <= _EPS:
            continue
        action = _action(trade)
        notional = trade.price * trade.size
        if action == "TRADE":
            cash = -notional if trade.side is BookSide.BUY else notional
            total_notional += notional
            if is_crypto_price_market(_text(trade, "title"), _text(trade, "slug")):
                crypto_notional += notional
            n_trades += 1
            trade_markets.add(trade.market_id)
            trade_days.add(trade.timestamp.astimezone(UTC).date())
            role = getattr(trade, "is_maker", None)
            if role is not None:
                labeled_fills += 1
                if role:
                    maker_fills += 1
        elif action == "REDEEM":
            cash = notional
        elif action == "MERGE":
            cash = notional
            n_merges += 1
        elif action == "SPLIT":
            cash = -notional
        else:
            continue
        equity += cash
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    share_crypto = crypto_notional / total_notional if total_notional > _EPS else 0.0
    share_maker = maker_fills / labeled_fills if labeled_fills else None
    return WalletStats(
        wallet=wallet,
        n_trades=n_trades,
        n_markets=len(trade_markets),
        realized_pnl_usd=equity,
        max_drawdown_usd=max_dd,
        active_days=len(trade_days),
        share_crypto_markets=share_crypto,
        share_maker=share_maker,
        n_merges=n_merges,
    )


def passes_screen(
    stats: WalletStats,
    *,
    min_trades: int = MIN_TRADES,
    min_days: int = MIN_ACTIVE_DAYS,
    min_crypto_share: float = MIN_CRYPTO_SHARE,
) -> bool:
    """Consistently profitable on crypto price markets.

    The wallet needs ``min_trades`` binary fills, ``min_days`` distinct UTC trade
    dates, positive cash PnL, and at least ``min_crypto_share`` of trade notional
    in crypto price markets. Those thresholds are the stability bar: a long
    crypto sample that made money.
    """
    return (
        stats.n_trades >= min_trades
        and stats.active_days >= min_days
        and stats.realized_pnl_usd > 0.0
        and stats.share_crypto_markets >= min_crypto_share
    )


def discover_wallets(
    client: WalletClient,
    market_ids: list[str],
    since: datetime | None = None,
) -> list[tuple[str, int]]:
    """Wallets on these markets, most active first: ``(address, n_prints)``."""
    counts: Counter[str] = Counter()
    labels: dict[str, str] = {}
    for market_id in market_ids:
        for trade in client.fetch_market_trades(market_id, since=since):
            key = trade.wallet.lower()
            counts[key] += 1
            labels.setdefault(key, trade.wallet)
    return [(labels[key], counts[key]) for key in sorted(counts, key=lambda k: (-counts[k], labels[k]))]


def find_profitable_wallets(
    client: WalletClient,
    market_ids: list[str],
    *,
    since: datetime | None = None,
    min_trades: int = MIN_TRADES,
    min_days: int = MIN_ACTIVE_DAYS,
    min_crypto_share: float = MIN_CRYPTO_SHARE,
    extra_wallets: tuple[str, ...] = (),
    max_candidates: int = 25,
    label_roles: bool = True,
) -> list[WalletStats]:
    """Keep wallets that clear :func:`passes_screen`.

    Candidates are ``extra_wallets`` plus the most active addresses on
    ``market_ids`` (at most ``max_candidates`` from the tape). The trade and
    day minimums apply to each wallet's own history, not to one market print.
    """
    ranked = discover_wallets(client, market_ids, since=since)
    chosen: list[str] = []
    seen: set[str] = set()

    def take(wallet: str) -> None:
        key = wallet.lower()
        if key not in seen:
            seen.add(key)
            chosen.append(wallet)

    for wallet in extra_wallets:
        take(wallet)
    added = 0
    for wallet, _count in ranked:
        if added >= max_candidates:
            break
        if wallet.lower() in seen:
            continue
        take(wallet)
        added += 1

    passed: list[WalletStats] = []
    for wallet in chosen:
        stats = compute_stats(
            wallet,
            client.fetch_actions(wallet, since=since, label_roles=label_roles),
        )
        if passes_screen(
            stats,
            min_trades=min_trades,
            min_days=min_days,
            min_crypto_share=min_crypto_share,
        ):
            passed.append(stats)
    passed.sort(key=lambda item: item.realized_pnl_usd, reverse=True)
    return passed


def format_wallet_stats(stats: WalletStats, *, passed: bool) -> str:
    maker = "n/a" if stats.share_maker is None else f"{stats.share_maker:.1%}"
    flag = "PASS" if passed else "FAIL"
    return (
        f"{stats.wallet}  {flag}  trades={stats.n_trades} markets={stats.n_markets} "
        f"pnl={stats.realized_pnl_usd:.2f} dd={stats.max_drawdown_usd:.2f} "
        f"days={stats.active_days} crypto={stats.share_crypto_markets:.1%} "
        f"maker={maker} merges={stats.n_merges}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find wallets consistently profitable on crypto price markets."
    )
    parser.add_argument(
        "wallets",
        nargs="*",
        help="Proxy addresses. Defaults to the article wallet (Lennie).",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Also discover wallets from current crypto price markets.",
    )
    parser.add_argument(
        "--markets",
        nargs="*",
        default=None,
        help="Condition ids to scan. Default with --scan: Gamma crypto price markets.",
    )
    parser.add_argument("--min-trades", type=int, default=MIN_TRADES)
    parser.add_argument("--min-days", type=int, default=MIN_ACTIVE_DAYS)
    parser.add_argument("--min-crypto-share", type=float, default=MIN_CRYPTO_SHARE)
    parser.add_argument("--since", default=None, help="Inclusive UTC ISO timestamp.")
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument(
        "--no-roles",
        action="store_true",
        help="Skip the taker feed, so share_maker stays unknown.",
    )
    args = parser.parse_args(argv)
    since = _parse_since(args.since) if args.since else None
    requested = tuple(args.wallets) if args.wallets else ARTICLE_WALLETS

    with httpx.Client(timeout=30.0) as http:
        client = WalletClient(http)
        market_ids = list(args.markets or [])
        if args.scan and not market_ids:
            market_ids = client.fetch_crypto_market_ids()
        if args.scan:
            found = find_profitable_wallets(
                client,
                market_ids,
                since=since,
                min_trades=args.min_trades,
                min_days=args.min_days,
                min_crypto_share=args.min_crypto_share,
                extra_wallets=requested,
                max_candidates=args.max_candidates,
                label_roles=not args.no_roles,
            )
            if not found:
                print("no wallet passed")
            for stats in found:
                print(format_wallet_stats(stats, passed=True))
            return 0

        n_pass = 0
        for wallet in requested:
            stats = compute_stats(
                wallet,
                client.fetch_actions(wallet, since=since, label_roles=not args.no_roles),
            )
            ok = passes_screen(
                stats,
                min_trades=args.min_trades,
                min_days=args.min_days,
                min_crypto_share=args.min_crypto_share,
            )
            n_pass += int(ok)
            print(format_wallet_stats(stats, passed=ok))
        print(f"{n_pass}/{len(requested)} passed")
    return 0


def _collect_trades(
    rows: list[dict[str, Any]],
    flags: list[bool | None],
    since: datetime | None,
    *,
    fallback_wallet: str | None,
) -> list[WalletTrade]:
    parsed: list[WalletTrade] = []
    for row, flag in zip(rows, flags, strict=True):
        trade = _parse_trade(row, is_maker=flag, fallback_wallet=fallback_wallet)
        if trade is not None and _in_window(trade, since):
            parsed.append(trade)
    return parsed


def _trade_params(
    since: datetime | None,
    *,
    taker_only: bool,
    user: str | None = None,
    condition: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "limit": _PAGE_LIMIT,
        "taker_only": taker_only,
        "start": _start_param(since),
    }
    if user is not None:
        params["user"] = user
    if condition is not None:
        params["condition"] = condition
    return params


def _parse_trade(
    row: dict[str, Any],
    *,
    is_maker: bool | None,
    fallback_wallet: str | None,
) -> WalletFill | None:
    outcome = _outcome(row)
    side = _side(row)
    price = _unit_price(_num(row, "price"))
    size = _num(row, "size")
    timestamp = _timestamp(row)
    market_id = _text_field(row, "condition_id", "conditionId")
    tx_hash = _text_field(row, "transaction_hash", "transactionHash")
    wallet = _text_field(row, "proxy_wallet", "proxyWallet") or fallback_wallet
    if outcome is None or side is None or price is None or size is None or size <= 0:
        return None
    if timestamp is None or not market_id or not tx_hash or not wallet:
        return None
    return WalletFill(
        wallet=wallet,
        market_id=market_id,
        outcome=outcome,
        side=side,
        price=price,
        size=size,
        timestamp=timestamp,
        tx_hash=tx_hash,
        title=_text_field(row, "title") or "",
        slug=_text_field(row, "slug") or "",
        is_maker=is_maker,
        action="TRADE",
    )


def _parse_activity(row: dict[str, Any], *, fallback_wallet: str) -> WalletFill | None:
    action = (_text_field(row, "type") or "").upper()
    if action not in {"MERGE", "REDEEM", "SPLIT"}:
        return None
    size = _num(row, "size")
    usdc = _num(row, "usdc_size", "usdcSize")
    timestamp = _timestamp(row)
    market_id = _text_field(row, "condition_id", "conditionId")
    tx_hash = _text_field(row, "transaction_hash", "transactionHash")
    wallet = _text_field(row, "proxy_wallet", "proxyWallet") or fallback_wallet
    if size is None or size <= 0 or usdc is None or usdc < 0:
        return None
    if timestamp is None or not market_id or not tx_hash or not wallet:
        return None
    per_set = _unit_price(usdc / size)
    if per_set is None:
        return None
    if action == "REDEEM":
        outcome = _outcome(row)
        if outcome is None:
            return None
        side = BookSide.SELL
    else:
        outcome = Outcome.YES
        side = BookSide.BUY
    return WalletFill(
        wallet=wallet,
        market_id=market_id,
        outcome=outcome,
        side=side,
        price=per_set,
        size=size,
        timestamp=timestamp,
        tx_hash=tx_hash,
        title=_text_field(row, "title") or "",
        slug=_text_field(row, "slug") or "",
        is_maker=None,
        action=action,
    )


def _maker_flags(all_rows: list[dict[str, Any]], taker_rows: list[dict[str, Any]]) -> list[bool]:
    """True when a row is absent from the taker feed (the wallet rested the order)."""
    remaining = Counter(_row_key(row) for row in taker_rows)
    flags: list[bool] = []
    for row in all_rows:
        key = _row_key(row)
        if remaining[key] > 0:
            remaining[key] -= 1
            flags.append(False)
        else:
            flags.append(True)
    return flags


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    size = _num(row, "size")
    price = _num(row, "price")
    timestamp = _raw_timestamp(row)
    return (
        _text_field(row, "transaction_hash", "transactionHash"),
        _text_field(row, "condition_id", "conditionId"),
        _text_field(row, "token_id", "asset"),
        (_text_field(row, "side") or "").upper(),
        None if size is None else round(size, 8),
        None if price is None else round(price, 8),
        timestamp,
    )


def _sort_key(index: int, trade: WalletTrade) -> tuple[Any, ...]:
    action = _action(trade)
    side_rank = 0 if action != "TRADE" or trade.side is BookSide.BUY else 1
    return (trade.timestamp, _ACTION_RANK.get(action, 9), side_rank, index)


def _action(trade: WalletTrade) -> str:
    return str(getattr(trade, "action", "TRADE") or "TRADE").upper()


def _text(trade: WalletTrade, name: str) -> str:
    return str(getattr(trade, name, "") or "")


def _outcome(row: dict[str, Any]) -> Outcome | None:
    label = (_text_field(row, "outcome") or "").strip().lower()
    if label in _YES:
        return Outcome.YES
    if label in _NO:
        return Outcome.NO
    return None


def _side(row: dict[str, Any]) -> BookSide | None:
    label = (_text_field(row, "side") or "").strip().upper()
    if label == "BUY":
        return BookSide.BUY
    if label == "SELL":
        return BookSide.SELL
    return None


def _timestamp(row: dict[str, Any]) -> datetime | None:
    raw = _raw_timestamp(row)
    if raw is None:
        return None
    seconds = raw / 1000.0 if raw > 10**12 else float(raw)
    return datetime.fromtimestamp(seconds, UTC)


def _raw_timestamp(row: dict[str, Any]) -> int | None:
    value = _first(row, "timestamp")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _num(row: dict[str, Any], *names: str) -> float | None:
    value = _first(row, *names)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _unit_price(price: float | None) -> float | None:
    if price is None:
        return None
    if -1e-4 <= price < 0.0:
        return 0.0
    if 1.0 < price <= 1.0 + 1e-4:
        return 1.0
    if price < 0.0 or price > 1.0:
        return None
    return price


def _text_field(row: dict[str, Any], *names: str) -> str | None:
    value = _first(row, *names)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _start_param(since: datetime | None) -> int:
    # start=1 asks Data API v2 for full history. 0 is the three-year default.
    if since is None:
        return 1
    return int(since.timestamp())


def _as_utc(name: str, value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware UTC, got naive {value!r}")
    return value.astimezone(UTC)


def _in_window(trade: WalletTrade, since: datetime | None) -> bool:
    return since is None or trade.timestamp >= since


def _parse_since(text: str) -> datetime:
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _retry_delay(response: httpx.Response) -> float:
    raw = response.headers.get("Retry-After", "0")
    try:
        return min(max(float(raw), 0.0), 30.0)
    except ValueError:
        return 1.0


if __name__ == "__main__":
    raise SystemExit(main())
