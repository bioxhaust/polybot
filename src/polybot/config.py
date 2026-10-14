from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from polybot.models import Asset


@dataclass(frozen=True, slots=True)
class Settings:
    db_path: Path
    assets: tuple[Asset, ...]
    poll_seconds: float

    bankroll_usd: float
    risk_unit_pct: float
    daily_loss_limit_r: float
    weekly_loss_limit_r: float
    theme_limit_r: float
    max_share_of_volume: float

    min_edge: float
    taker_fee_rate: float

    snapshot_seconds: float = 300.0  # books, fair probabilities and outcome checks
    min_trade_tte_minutes: float = 60.0  # no directional entries closer to expiry than this

    @property
    def risk_unit_usd(self) -> float:
        return self.bankroll_usd * self.risk_unit_pct


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    if env_file is not None:
        load_dotenv(env_file)

    def get(name: str, default: str) -> str:
        return os.environ.get(f"POLYBOT_{name}", default)

    return Settings(
        db_path=Path(get("DB_PATH", "data/polybot.sqlite")),
        assets=tuple(Asset(a.strip().upper()) for a in get("ASSETS", "BTC,ETH").split(",")),
        poll_seconds=float(get("POLL_SECONDS", "30")),
        bankroll_usd=float(get("BANKROLL_USD", "10000")),
        risk_unit_pct=float(get("RISK_UNIT_PCT", "0.005")),
        daily_loss_limit_r=float(get("DAILY_LOSS_LIMIT_R", "3")),
        weekly_loss_limit_r=float(get("WEEKLY_LOSS_LIMIT_R", "7")),
        theme_limit_r=float(get("THEME_LIMIT_R", "3")),
        max_share_of_volume=float(get("MAX_SHARE_OF_VOLUME", "0.01")),
        min_edge=float(get("MIN_EDGE", "0.03")),
        taker_fee_rate=float(get("TAKER_FEE_RATE", "0.0")),
        snapshot_seconds=float(get("SNAPSHOT_SECONDS", "300")),
        min_trade_tte_minutes=float(get("MIN_TRADE_TTE_MINUTES", "60")),
    )
