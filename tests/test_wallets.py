import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from polybot.models import BookSide, Outcome, WalletTrade
from polybot.polymarket.wallets import (
    ARTICLE_WALLETS,
    DATA_API_URL,
    GAMMA_URL,
    WalletClient,
    WalletFill,
    WalletStats,
    compute_stats,
    discover_wallets,
    find_profitable_wallets,
    format_wallet_stats,
    is_crypto_price_market,
    main,
    passes_screen,
)

FIXTURES = Path(__file__).parent / "fixtures" / "wallets"
LENNIE = ARTICLE_WALLETS[0]
DAY = datetime(2025, 8, 1, 12, tzinfo=UTC)


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def replace(stats, **overrides):
    return type(stats)(**{**asdict(stats), **overrides})


def fill(**overrides) -> WalletFill:
    fields = {
        "wallet": "0xabc",
        "market_id": "m1",
        "outcome": Outcome.YES,
        "side": BookSide.BUY,
        "price": 0.4,
        "size": 10.0,
        "timestamp": DAY,
        "tx_hash": "0xtx",
        "title": "Will Bitcoin be above $100k on August 1?",
        "slug": "bitcoin-above-100k",
        "is_maker": None,
        "action": "TRADE",
    }
    fields.update(overrides)
    return WalletFill(**fields)


def page(rows: list[dict], *, more: bool = False, cursor: str | None = None) -> dict:
    return {
        "data": rows,
        "pagination": {
            "limit": 1000,
            "offset": 0,
            "has_more": more,
            "next_cursor": cursor,
        },
    }


def empty_page() -> dict:
    return page([])


def trade_row(**overrides) -> dict:
    row = {
        "proxy_wallet": "0xabc",
        "side": "BUY",
        "token_id": "token",
        "condition_id": "0x" + "ab" * 32,
        "size": 10.0,
        "price": 0.4,
        "timestamp": 1754050000,
        "title": "Will Bitcoin be above $100k on August 1?",
        "slug": "bitcoin-above-100k",
        "outcome": "Yes",
        "outcome_index": 0,
        "transaction_hash": "0xtx",
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("title", "slug", "expected"),
    [
        ("Will the price of Bitcoin be less than $112K on August 1?", "", True),
        ("Will Ethereum dip to $2200 in July?", "", True),
        ("Will Bitcoin reach $150K in July?", "", True),
        ("Will Solana dip to $100 in July?", "", True),
        ("Will XRP dip to $2.0 in July?", "", True),
        ("Ethereum above ___ on October 14?", "", True),
        ("Bitcoin Up or Down - October 14, 5PM ET", "", True),
        ("What price will Bitcoin hit in October?", "", True),
        ("Will ETH be above $4,400 on October 17?", "", True),
        ("", "bitcoin-price-on-august-1", True),
        ("Will Satoshi move any Bitcoin in 2026?", "", False),
        ("Bitcoin ETF approved in 2026?", "", False),
        ("Xi Jinping out before 2027?", "", False),
        ("Will the price of gold be above $3000?", "", False),
    ],
)
def test_crypto_price_market_titles(title, slug, expected):
    assert is_crypto_price_market(title, slug) is expected


@respx.mock
def test_fetch_trades_parses_real_page_and_follows_cursor():
    page1 = load("trades_user_page1.json")
    page2 = load("trades_user_page2.json")
    page2["pagination"] = {**page2["pagination"], "has_more": False, "next_cursor": None}
    calls: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        if "cursor" not in request.url.params:
            return httpx.Response(200, json=page1)
        assert request.url.params["cursor"] == page1["pagination"]["next_cursor"]
        return httpx.Response(200, json=page2)

    respx.get(f"{DATA_API_URL}/v2/trades").mock(side_effect=handler)
    trades = WalletClient(httpx.Client()).fetch_trades(LENNIE)

    assert len(trades) == 4
    assert all(url.params["taker_only"] == "false" for url in calls)
    assert all(url.params["user"] == LENNIE for url in calls)
    assert all(url.params["start"] == "1" for url in calls)
    first = trades[0]
    assert first.wallet == LENNIE
    assert first.market_id == "0xf0ee86ffee78ab8fde56d85e8f8526c6cb37356f0b7cf78b3be29581726fbdd4"
    assert first.outcome is Outcome.NO
    assert first.side is BookSide.SELL
    assert first.price == pytest.approx(0.9904282857)
    assert first.size == pytest.approx(4375.0)
    assert first.timestamp == datetime(2025, 8, 1, 13, 18, 6, tzinfo=UTC)
    assert first.tx_hash == "0x79a090dfd8605688805781847fbf638ed423a03036ed5e2448d779b3215c9269"
    assert first.title.startswith("Will the price of Bitcoin")
    assert is_crypto_price_market(first.title, first.slug)


@respx.mock
def test_fetch_trades_filters_since_and_sends_start():
    page1 = load("trades_user_page1.json")
    page2 = load("trades_user_page2.json")
    page2["pagination"] = {**page2["pagination"], "has_more": False, "next_cursor": None}
    since = datetime(2025, 8, 1, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["start"] == str(int(since.timestamp()))
        if "cursor" not in request.url.params:
            return httpx.Response(200, json=page1)
        return httpx.Response(200, json=page2)

    respx.get(f"{DATA_API_URL}/v2/trades").mock(side_effect=handler)
    trades = WalletClient(httpx.Client()).fetch_trades(LENNIE, since=since)

    assert len(trades) == 1
    assert trades[0].timestamp == datetime(2025, 8, 1, 13, 18, 6, tzinfo=UTC)


def test_fetch_trades_rejects_naive_since():
    with pytest.raises(ValueError, match="timezone-aware"):
        WalletClient(httpx.Client()).fetch_trades(LENNIE, since=datetime(2025, 8, 1))


@respx.mock
def test_fetch_market_trades_keeps_each_wallets_row():
    body = load("trades_market.json")
    body["pagination"] = {**body["pagination"], "has_more": False, "next_cursor": None}
    market_id = body["data"][0]["condition_id"]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["condition"] == market_id
        assert "user" not in request.url.params
        assert request.url.params["taker_only"] == "false"
        return httpx.Response(200, json=body)

    respx.get(f"{DATA_API_URL}/v2/trades").mock(side_effect=handler)
    trades = WalletClient(httpx.Client()).fetch_market_trades(market_id)

    assert [trade.wallet for trade in trades] == [row["proxy_wallet"] for row in body["data"]]
    assert all(trade.market_id == market_id for trade in trades)
    assert [trade.outcome for trade in trades] == [Outcome.YES, Outcome.NO]
    assert [trade.side for trade in trades] == [BookSide.BUY, BookSide.BUY]


@respx.mock
def test_fetch_trades_labels_maker_from_the_taker_feed():
    maker = trade_row(transaction_hash="0xmaker", timestamp=1753922821, price=0.998, size=2.0)
    taker = trade_row(transaction_hash="0xtaker", timestamp=1754054286, price=0.99, size=4.0)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        flag = request.url.params["taker_only"]
        seen.append(flag)
        rows = [maker, taker] if flag == "false" else [taker]
        return httpx.Response(200, json=page(rows))

    respx.get(f"{DATA_API_URL}/v2/trades").mock(side_effect=handler)
    trades = WalletClient(httpx.Client()).fetch_trades("0xabc", label_roles=True)

    assert seen == ["false", "true"]
    by_tx = {trade.tx_hash: trade for trade in trades}
    assert by_tx["0xmaker"].is_maker is True
    assert by_tx["0xtaker"].is_maker is False


@respx.mock
def test_fetch_trades_skips_rows_that_are_not_binary_fills():
    good = trade_row(outcome="Up", outcome_index=0, side="BUY")
    down = trade_row(outcome="Down", outcome_index=1, side="SELL", transaction_hash="0xdown")
    bucket = trade_row(outcome="112000-114000", outcome_index=0, transaction_hash="0xbucket")
    pricey = trade_row(price=1.5, transaction_hash="0xpricey")
    camel = {
        "proxyWallet": "0xabc",
        "side": "BUY",
        "asset": "token",
        "conditionId": "0x" + "cd" * 32,
        "size": "3",
        "price": "0.25",
        "timestamp": "1754050000000",
        "title": "Will ETH be above $4400?",
        "slug": "eth-above",
        "outcome": "Yes",
        "transactionHash": "0xcamel",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if "cursor" not in request.url.params:
            return httpx.Response(200, json=page([good, down, bucket, pricey], more=True, cursor="next"))
        return httpx.Response(200, json=[camel])

    respx.get(f"{DATA_API_URL}/v2/trades").mock(side_effect=handler)
    trades = WalletClient(httpx.Client()).fetch_trades("0xabc")

    assert [(trade.outcome, trade.tx_hash, trade.price) for trade in trades] == [
        (Outcome.YES, "0xtx", pytest.approx(0.4)),
        (Outcome.NO, "0xdown", pytest.approx(0.4)),
        (Outcome.YES, "0xcamel", pytest.approx(0.25)),
    ]
    assert trades[-1].timestamp == datetime.fromtimestamp(1754050000, UTC)
    assert trades[-1].size == pytest.approx(3.0)


@respx.mock
def test_fetch_actions_includes_real_merge_and_redeem():
    merge = load("activity_merge.json")
    redeem = load("activity_redeem.json")
    body = page(merge["data"] + redeem["data"])

    respx.get(f"{DATA_API_URL}/v2/trades").mock(return_value=httpx.Response(200, json=empty_page()))
    respx.get(f"{DATA_API_URL}/v2/activity").mock(return_value=httpx.Response(200, json=body))
    actions = WalletClient(httpx.Client()).fetch_actions(LENNIE)

    kinds = {action.action: action for action in actions}
    assert set(kinds) == {"MERGE", "REDEEM"}
    assert kinds["MERGE"].price == pytest.approx(1.0)
    assert kinds["MERGE"].size == pytest.approx(3992.617)
    assert "Solana" in kinds["MERGE"].title
    assert kinds["REDEEM"].outcome is Outcome.NO
    assert kinds["REDEEM"].side is BookSide.SELL
    assert kinds["REDEEM"].price == pytest.approx(1.0)


@respx.mock
def test_retries_429_then_reads_the_page():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json=empty_page())

    respx.get(f"{DATA_API_URL}/v2/trades").mock(side_effect=handler)
    assert WalletClient(httpx.Client()).fetch_trades("0xabc") == []
    assert calls["n"] == 3


@respx.mock
def test_http_400_is_not_retried():
    route = respx.get(f"{DATA_API_URL}/v2/trades").mock(
        return_value=httpx.Response(400, json={"error": "bad", "code": "invalid_request"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        WalletClient(httpx.Client()).fetch_trades("0xabc")
    assert route.call_count == 1


def test_compute_stats_round_trip_drawdown_and_days():
    trades = [
        fill(side=BookSide.BUY, price=0.40, timestamp=DAY, tx_hash="b1"),
        fill(side=BookSide.SELL, price=0.60, timestamp=DAY, tx_hash="s1"),
        fill(side=BookSide.BUY, price=0.70, timestamp=DAY + timedelta(days=1), tx_hash="b2"),
        fill(side=BookSide.SELL, price=0.50, timestamp=DAY + timedelta(days=1, hours=1), tx_hash="s2"),
    ]
    stats = compute_stats("0xabc", trades)

    assert stats.n_trades == 4
    assert stats.n_markets == 1
    assert stats.realized_pnl_usd == pytest.approx(0.0)
    # Cash went to -4, +2, -5, 0. The trough after the second buy is 7 below the peak.
    assert stats.max_drawdown_usd == pytest.approx(7.0)
    assert stats.active_days == 2
    assert stats.n_merges == 0


def test_compute_stats_average_cost_and_short_cover():
    averaged = compute_stats(
        "0xabc",
        [
            fill(side=BookSide.BUY, price=0.40, tx_hash="b1"),
            fill(side=BookSide.BUY, price=0.60, timestamp=DAY + timedelta(minutes=1), tx_hash="b2"),
            fill(side=BookSide.SELL, price=0.80, size=10, timestamp=DAY + timedelta(minutes=2), tx_hash="s1"),
            fill(side=BookSide.SELL, price=0.50, size=10, timestamp=DAY + timedelta(minutes=3), tx_hash="s2"),
        ],
    )
    assert averaged.realized_pnl_usd == pytest.approx(3.0)

    covered = compute_stats(
        "0xabc",
        [
            fill(side=BookSide.SELL, price=0.60, timestamp=DAY, tx_hash="s"),
            fill(side=BookSide.BUY, price=0.40, timestamp=DAY + timedelta(minutes=1), tx_hash="b"),
        ],
    )
    assert covered.realized_pnl_usd == pytest.approx(2.0)


def test_same_timestamp_spends_buys_before_sells():
    stamp = DAY
    stats = compute_stats(
        "0xabc",
        [
            fill(side=BookSide.SELL, price=0.60, timestamp=stamp, tx_hash="s"),
            fill(side=BookSide.BUY, price=0.40, timestamp=stamp, tx_hash="b1"),
            fill(side=BookSide.BUY, price=0.90, timestamp=stamp, tx_hash="b2"),
        ],
    )
    # Both buys are paid before the sell: -4, -13, then +6.
    assert stats.realized_pnl_usd == pytest.approx(-7.0)
    assert stats.max_drawdown_usd == pytest.approx(13.0)


def test_compute_stats_merge_redeem_and_split():
    merged = compute_stats(
        "0xabc",
        [
            fill(outcome=Outcome.YES, side=BookSide.BUY, price=0.40, tx_hash="y"),
            fill(outcome=Outcome.NO, side=BookSide.BUY, price=0.50, timestamp=DAY, tx_hash="n"),
            fill(action="MERGE", price=1.0, size=10, timestamp=DAY + timedelta(minutes=1), tx_hash="m"),
        ],
    )
    assert merged.n_merges == 1
    assert merged.n_trades == 2
    assert merged.realized_pnl_usd == pytest.approx(1.0)

    partial = compute_stats(
        "0xabc",
        [
            fill(outcome=Outcome.YES, side=BookSide.BUY, price=0.40, size=4, tx_hash="y"),
            fill(outcome=Outcome.NO, side=BookSide.BUY, price=0.50, size=10, tx_hash="n"),
            fill(action="MERGE", price=1.0, size=10, timestamp=DAY + timedelta(minutes=1), tx_hash="m"),
        ],
    )
    assert partial.realized_pnl_usd == pytest.approx(3.4)

    redeemed = compute_stats(
        "0xabc",
        [
            fill(outcome=Outcome.YES, side=BookSide.BUY, price=0.40, tx_hash="y"),
            fill(
                action="REDEEM",
                outcome=Outcome.YES,
                side=BookSide.SELL,
                price=1.0,
                timestamp=DAY + timedelta(minutes=1),
                tx_hash="r",
            ),
        ],
    )
    assert redeemed.realized_pnl_usd == pytest.approx(6.0)
    assert redeemed.n_merges == 0

    held_loser = compute_stats(
        "0xabc",
        [
            fill(outcome=Outcome.YES, side=BookSide.BUY, price=0.40, tx_hash="y"),
            fill(outcome=Outcome.NO, side=BookSide.BUY, price=0.50, tx_hash="n"),
            fill(
                action="REDEEM",
                outcome=Outcome.NO,
                side=BookSide.SELL,
                price=1.0,
                timestamp=DAY + timedelta(minutes=1),
                tx_hash="r",
            ),
        ],
    )
    # The YES buy is still cash out; unredeemed shares are carried at zero.
    assert held_loser.realized_pnl_usd == pytest.approx(1.0)

    split_merge = compute_stats(
        "0xabc",
        [
            fill(action="SPLIT", price=1.0, size=10, timestamp=DAY, tx_hash="s"),
            fill(action="MERGE", price=1.0, size=10, timestamp=DAY + timedelta(minutes=1), tx_hash="m"),
        ],
    )
    assert split_merge.realized_pnl_usd == pytest.approx(0.0)
    assert split_merge.n_trades == 0
    assert split_merge.n_merges == 1
    assert split_merge.active_days == 0


def test_compute_stats_crypto_share_maker_share_and_plain_trades():
    stats = compute_stats(
        "0xabc",
        [
            fill(price=0.50, size=10, is_maker=True, tx_hash="c1"),
            fill(
                market_id="m2",
                price=0.20,
                size=10,
                is_maker=False,
                title="Xi Jinping out before 2027?",
                slug="xi",
                tx_hash="p1",
            ),
            fill(price=0.50, size=10, is_maker=True, timestamp=DAY + timedelta(days=1), tx_hash="c2"),
        ],
    )
    assert stats.share_crypto_markets == pytest.approx(10.0 / 12.0)
    assert stats.share_maker == pytest.approx(2 / 3)
    assert stats.n_markets == 2
    assert stats.active_days == 2

    plain = [
        WalletTrade(
            wallet="0xabc",
            market_id="m1",
            outcome=Outcome.YES,
            side=BookSide.BUY,
            price=0.40,
            size=10,
            timestamp=DAY,
            tx_hash="b",
        ),
        WalletTrade(
            wallet="0xabc",
            market_id="m1",
            outcome=Outcome.YES,
            side=BookSide.SELL,
            price=0.60,
            size=10,
            timestamp=DAY + timedelta(minutes=1),
            tx_hash="s",
        ),
    ]
    bare = compute_stats("0xabc", plain)
    assert bare.realized_pnl_usd == pytest.approx(2.0)
    assert bare.share_maker is None
    assert bare.share_crypto_markets == 0.0
    assert bare.n_merges == 0


def test_compute_stats_empty_and_open_inventory_has_no_pnl():
    empty = compute_stats("0xabc", [])
    assert empty == type(empty)(
        wallet="0xabc",
        n_trades=0,
        n_markets=0,
        realized_pnl_usd=0.0,
        max_drawdown_usd=0.0,
        active_days=0,
        share_crypto_markets=0.0,
        share_maker=None,
        n_merges=0,
    )
    opened = compute_stats("0xabc", [fill()])
    assert opened.realized_pnl_usd == pytest.approx(-4.0)
    assert opened.max_drawdown_usd == pytest.approx(4.0)
    assert opened.n_trades == 1


def test_passes_screen_thresholds():
    base = WalletStats(
        wallet="0xabc",
        n_trades=200,
        n_markets=4,
        realized_pnl_usd=100.0,
        max_drawdown_usd=40.0,
        active_days=30,
        share_crypto_markets=0.6,
        share_maker=0.8,
        n_merges=3,
    )
    assert passes_screen(base) is True
    assert passes_screen(replace(base, max_drawdown_usd=10_000.0)) is True
    assert passes_screen(replace(base, share_crypto_markets=0.5)) is True
    for overrides in (
        {"n_trades": 199},
        {"active_days": 29},
        {"realized_pnl_usd": 0.0},
        {"realized_pnl_usd": -1.0},
        {"share_crypto_markets": 0.49},
    ):
        assert passes_screen(replace(base, **overrides)) is False


@respx.mock
def test_find_profitable_wallets_keeps_only_the_screen():
    market_id = "0x" + "11" * 32

    def history(wallet: str, prices: tuple[float, float], title: str) -> dict:
        rows = []
        for day, (buy, sell) in enumerate((prices, prices)):
            stamp = int(datetime(2025, 6, 1, tzinfo=UTC).timestamp()) + day * 86400
            rows.append(
                trade_row(
                    proxy_wallet=wallet,
                    condition_id=market_id,
                    side="BUY",
                    price=buy,
                    timestamp=stamp,
                    title=title,
                    transaction_hash=f"b{wallet}{day}",
                )
            )
            rows.append(
                trade_row(
                    proxy_wallet=wallet,
                    condition_id=market_id,
                    side="SELL",
                    price=sell,
                    timestamp=stamp + 60,
                    title=title,
                    transaction_hash=f"s{wallet}{day}",
                )
            )
        return page(rows)

    tape = page(
        [
            trade_row(proxy_wallet="0xaaa", condition_id=market_id, transaction_hash="t1"),
            trade_row(proxy_wallet="0xaaa", condition_id=market_id, transaction_hash="t2"),
            trade_row(proxy_wallet="0xbbb", condition_id=market_id, transaction_hash="t3"),
        ]
    )
    books = {
        "0xaaa": history("0xaaa", (0.40, 0.50), "Will Bitcoin be above $100k on August 1?"),
        "0xbbb": history("0xbbb", (0.50, 0.20), "Will Bitcoin be above $100k on August 1?"),
    }

    def trades(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "condition" in params:
            assert params["condition"] == market_id
            return httpx.Response(200, json=tape)
        return httpx.Response(200, json=books[params["user"]])

    respx.get(f"{DATA_API_URL}/v2/trades").mock(side_effect=trades)
    respx.get(f"{DATA_API_URL}/v2/activity").mock(return_value=httpx.Response(200, json=empty_page()))

    found = find_profitable_wallets(
        WalletClient(httpx.Client()),
        [market_id],
        min_trades=4,
        min_days=2,
        min_crypto_share=0.5,
        extra_wallets=(),
        max_candidates=10,
        label_roles=False,
    )

    assert [stats.wallet for stats in found] == ["0xaaa"]
    assert found[0].realized_pnl_usd == pytest.approx(2.0)
    assert found[0].active_days == 2


@respx.mock
def test_discover_wallets_ranks_by_prints():
    market_id = "0x" + "22" * 32
    body = page(
        [
            trade_row(proxy_wallet="0xBBB", condition_id=market_id, transaction_hash="1"),
            trade_row(proxy_wallet="0xaaa", condition_id=market_id, transaction_hash="2"),
            trade_row(proxy_wallet="0xbbb", condition_id=market_id, transaction_hash="3"),
        ]
    )
    respx.get(f"{DATA_API_URL}/v2/trades").mock(return_value=httpx.Response(200, json=body))
    ranked = discover_wallets(WalletClient(httpx.Client()), [market_id])
    assert ranked == [("0xBBB", 2), ("0xaaa", 1)]


@respx.mock
def test_fetch_crypto_market_ids_filters_questions():
    bitcoin = "0x" + "aa" * 32
    payload = [
        {
            "markets": [
                {
                    "question": "Will Bitcoin be above $100k on October 14?",
                    "slug": "bitcoin-above",
                    "conditionId": bitcoin,
                    "closed": False,
                },
                {
                    "question": "Will Satoshi move any Bitcoin in 2026?",
                    "slug": "satoshi",
                    "conditionId": "0x" + "bb" * 32,
                    "closed": False,
                },
                {
                    "question": "Will ETH be above $4400?",
                    "slug": "eth-above",
                    "conditionId": bitcoin,
                    "closed": False,
                },
                {
                    "question": "Closed bitcoin above 1",
                    "slug": "closed",
                    "conditionId": "0x" + "cc" * 32,
                    "closed": True,
                },
            ]
        }
    ]
    respx.get(f"{GAMMA_URL}/events").mock(return_value=httpx.Response(200, json=payload))
    ids = WalletClient(httpx.Client()).fetch_crypto_market_ids()
    assert ids == [bitcoin]


@respx.mock
def test_main_reports_the_article_wallet(capsys):
    row_buy = trade_row(
        proxy_wallet=LENNIE,
        side="BUY",
        price=0.40,
        timestamp=1750000000,
        transaction_hash="0xbuy",
    )
    row_sell = trade_row(
        proxy_wallet=LENNIE,
        side="SELL",
        price=0.60,
        timestamp=1750000060,
        transaction_hash="0xsell",
    )
    respx.get(f"{DATA_API_URL}/v2/trades").mock(
        return_value=httpx.Response(200, json=page([row_buy, row_sell]))
    )
    respx.get(f"{DATA_API_URL}/v2/activity").mock(return_value=httpx.Response(200, json=empty_page()))

    code = main(["--no-roles", "--min-trades", "2", "--min-days", "1"])
    captured = capsys.readouterr().out
    assert code == 0
    assert "PASS" in captured
    assert LENNIE in captured
    assert "1/1 passed" in captured
    assert "pnl=2.00" in captured


def test_format_wallet_stats_marks_unknown_maker():
    stats = compute_stats("0xabc", [])
    text = format_wallet_stats(stats, passed=False)
    assert text.startswith("0xabc  FAIL")
    assert "maker=n/a" in text
