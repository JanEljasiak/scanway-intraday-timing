"""
Zrodla danych:
- dane DZIENNE: lokalny CSV (data/scw_d.csv, od debiutu 2023-10-11) jako
  podstawa + proba dociagniecia swiezszych wierszy ze Stooq live. Jesli
  brak internetu, projekt i tak dziala na lokalnym CSV.
- dane SRODDZIENNE (live): yfinance, 15-min interwal, max ~60 dni wstecz
  (twardy limit darmowego zrodla dla tickerow spoza USA).
"""
from __future__ import annotations

import pandas as pd

from .config import Config

STOOQ_COLUMN_MAP = {
    "Data": "date",
    "Otwarcie": "open",
    "Najwyzszy": "high",
    "Najnizszy": "low",
    "Zamkniecie": "close",
    "Wolumen": "volume",
}


def load_local_daily(cfg: Config) -> pd.DataFrame:
    """Wczytuje lokalny CSV (format eksportu Stooq) jako baze historyczna."""
    path = cfg.local_daily_csv_path
    if not path.exists():
        raise FileNotFoundError(
            f"Brak pliku {path}. Pobierz historie dzienna recznie ze "
            f"https://stooq.pl/q/d/?s={cfg.stooq_symbol_candidates[0]} "
            f"(link 'Dane historyczne' -> CSV) i zapisz jako {path}."
        )
    df = pd.read_csv(path, parse_dates=["Data"])
    df = df.rename(columns=STOOQ_COLUMN_MAP)
    df["date"] = df["date"].dt.date
    return df.set_index("date").sort_index()


def fetch_daily_stooq_live(cfg: Config) -> pd.DataFrame | None:
    """Probuje dociagnac najswiezsze dane ze Stooq. Zwraca None przy braku sieci."""
    for symbol in cfg.stooq_symbol_candidates:
        try:
            url = f"https://stooq.pl/q/d/l/?s={symbol}&i=d"
            df = pd.read_csv(url, parse_dates=["Data"])
            if df.empty or "Zamkniecie" not in df.columns:
                continue
            df = df.rename(columns=STOOQ_COLUMN_MAP)
            df["date"] = df["date"].dt.date
            return df.set_index("date").sort_index()
        except Exception:
            continue
    return None


def get_daily_history(cfg: Config, refresh_live: bool = True) -> pd.DataFrame:
    """
    Glowna funkcja do uzycia w reszcie projektu:
    1. wczytuje lokalny CSV (zawsze dziala, offline),
    2. jesli refresh_live=True, probuje dolaczyc swiezsze wiersze ze Stooq
       i zapisuje zaktualizowany CSV z powrotem na dysk (auto-cache).
    """
    local_df = load_local_daily(cfg)

    if not refresh_live:
        return local_df

    live_df = fetch_daily_stooq_live(cfg)
    if live_df is None:
        print("[data_sources] Brak polaczenia ze Stooq - korzystam z lokalnego CSV.")
        return local_df

    combined = pd.concat([local_df, live_df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    if len(combined) > len(local_df):
        combined.reset_index().rename(columns={
            "date": "Data", "open": "Otwarcie", "high": "Najwyzszy",
            "low": "Najnizszy", "close": "Zamkniecie", "volume": "Wolumen",
        }).to_csv(cfg.local_daily_csv_path, index=False)
        print(f"[data_sources] Zaktualizowano lokalny cache: +{len(combined) - len(local_df)} sesji.")

    return combined


def fetch_intraday_yf(cfg: Config) -> pd.DataFrame:
    """
    Dane sroddzienne live z yfinance. WYMAGA internetu - brak fallbacku,
    bo to jedyne darmowe zrodlo intraday dla SCW dostepne z poziomu Pythona.
    """
    import yfinance as yf

    df = yf.download(cfg.ticker_yf, period=cfg.intraday_period,
                      interval=cfg.intraday_interval, progress=False)
    if df.empty:
        raise ValueError(
            f"Brak danych intraday dla {cfg.ticker_yf}. Sprawdz polaczenie "
            f"z internetem albo czy ticker jest poprawny."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(cfg.timezone)

    df["date"] = df.index.date
    df["time"] = df.index.time
    return df
