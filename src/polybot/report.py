"""Task F. Does the model have an edge? Built only from Journal data."""

from __future__ import annotations

from dataclasses import dataclass

from polybot.storage.journal import Journal


@dataclass(frozen=True, slots=True)
class CalibrationBucket:
    p_low: float
    p_high: float
    n: int
    model_mean_p: float
    market_mean_p: float
    realized_freq: float


@dataclass(frozen=True, slots=True)
class Report:
    n_signals: int
    n_resolved: int
    paper_pnl_usd: float
    brier_model: float  # lower is better
    brier_market: float  # the model must beat this to have an edge
    calibration: tuple[CalibrationBucket, ...]
    complete_set_profit_usd: float


def build_report(journal: Journal) -> Report:
    raise NotImplementedError


def format_report(report: Report) -> str:
    raise NotImplementedError
