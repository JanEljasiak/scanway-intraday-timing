"""
Wczytywanie konfiguracji projektu: config.yaml + zmienne srodowiskowe (.env).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    ticker_yf: str
    stooq_symbol_candidates: list
    local_daily_csv: str
    intraday_interval: str
    intraday_period: str
    sell_horizon_bars: int
    sell_target_drop_pct: float
    daily_high_fn_penalty: float
    peak_tolerance_bars: int
    walk_forward_splits: int
    alert_probability_threshold: float
    poll_seconds: int
    market_open: str
    market_close: str
    timezone: str
    random_state: int

    telegram_bot_token: str = field(default="")
    telegram_chat_id: str = field(default="")

    @property
    def local_daily_csv_path(self) -> Path:
        return PROJECT_ROOT / self.local_daily_csv

    @property
    def models_dir(self) -> Path:
        d = PROJECT_ROOT / "models"
        d.mkdir(exist_ok=True)
        return d


def load_config(path: str | Path = None) -> Config:
    load_dotenv(PROJECT_ROOT / ".env")  # nie wywali bledu jesli pliku brak

    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        ticker_yf=raw["ticker_yf"],
        stooq_symbol_candidates=raw["stooq_symbol_candidates"],
        local_daily_csv=raw["local_daily_csv"],
        intraday_interval=raw["intraday_interval"],
        intraday_period=raw["intraday_period"],
        sell_horizon_bars=int(raw["sell_horizon_bars"]),
        sell_target_drop_pct=float(raw["sell_target_drop_pct"]),
        daily_high_fn_penalty=float(raw.get("daily_high_fn_penalty", 12.0)),
        peak_tolerance_bars=int(raw.get("peak_tolerance_bars", 1)),
        walk_forward_splits=int(raw["walk_forward_splits"]),
        alert_probability_threshold=float(raw["alert_probability_threshold"]),
        poll_seconds=int(raw["poll_seconds"]),
        market_open=raw["market_open"],
        market_close=raw["market_close"],
        timezone=raw["timezone"],
        random_state=int(raw["random_state"]),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
    )
