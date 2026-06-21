"""
Odtworzenie (replay) pojedynczej sesji swieca-po-swiecy.

Po co to jest:
- `backtest` mowi "jak dobry jest model statystycznie" (walk-forward, ROC AUC),
  ale nie pokazuje JAK to wyglada w praktyce w ciagu jednego dnia.
- `replay` bierze jedna sesje i przechodzi po niej swieca po swiecy, liczac
  dla kazdej 15-min swiecy prawdopodobienstwo "lokalnego szczytu" wedlug
  wytrenowanego modelu. Pokazuje:
    * kiedy model wyslalby ALERT (proba >= prog),
    * gdzie faktycznie byl "dobry moment na sprzedaz" (target=1, ground truth),
    * gdzie byl rzeczywisty szczyt dnia.
  Dzieki temu widac, czy alerty trafiaja w realne szczyty, czy sie spozniaja.

Zrodlo danych (kolejnosc prob):
  1. zapisany snapshot intraday (data/intraday_snapshot.csv) - dziala offline,
  2. yfinance live (wymaga internetu) - i zapisuje snapshot na przyszlosc,
  3. sesja syntetyczna (--synthetic) - czysto ILUSTRACYJNA, do pokazania
     mechanizmu bez internetu; NIE sa to prawdziwe notowania.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .features import build_features

SNAPSHOT_PATH = "data/intraday_snapshot.csv"


def make_synthetic_session(cfg: Config, n_sessions: int = 40, seed: int = 42) -> pd.DataFrame:
    """Generuje ILUSTRACYJNE dane intraday (15-min) dla n_sessions dni.

    Wzorzec celowo odzwierciedla obserwacje z EDA: czesto szczyt wypada przy
    otwarciu (09:00), potem cena dryfuje w dol. To NIE sa prawdziwe notowania
    - sluzy wylacznie do zademonstrowania mechanizmu replay/alertu offline.
    """
    rng = np.random.default_rng(seed)
    bars_per_day = 33  # 09:00 -> 17:00 co 15 min
    times = pd.date_range("09:00", periods=bars_per_day, freq="15min").time
    rows = []
    base_price = 300.0
    start_day = pd.Timestamp("2026-06-01")

    d = 0
    made = 0
    while made < n_sessions:
        day = start_day + pd.Timedelta(days=d)
        d += 1
        if day.weekday() >= 5:  # weekend
            continue
        made += 1
        is_last = made == n_sessions

        # czy dzisiaj "szczyt na otwarciu" (jak w ~41% realnych sesji).
        # Ostatnia sesja jest ZAWSZE typu peak-at-open, zeby replay pokazal
        # wyraznie, jak alert trafia w poranny szczyt (cel ilustracyjny).
        peak_at_open = is_last or rng.random() < 0.55
        open_px = base_price * (1 + rng.normal(0, 0.01))
        if peak_at_open:
            # rano lokalny szczyt, potem wyrazny spadek reszte dnia
            drift = np.concatenate([
                rng.normal(0.004, 0.002, 3),          # mocny ruch w gore na otwarciu
                rng.normal(-0.004, 0.003, bars_per_day - 3),  # spadek reszte dnia
            ])
        else:
            drift = rng.normal(0.0002, 0.004, bars_per_day)  # dzien bez wyraznego wzorca

        px = open_px * np.cumprod(1 + drift)
        for i, t in enumerate(times):
            close = px[i]
            high = close * (1 + abs(rng.normal(0, 0.0015)))
            low = close * (1 - abs(rng.normal(0, 0.0015)))
            vol = max(int(rng.normal(8000, 3000)), 500)
            ts = pd.Timestamp.combine(day.date(), t)
            rows.append((ts, close, high, low, vol, day.date(), t))
        base_price = px[-1]  # ciagnij trend miedzy sesjami

    df = pd.DataFrame(rows, columns=["ts", "Close", "High", "Low", "Volume", "date", "time"])
    df = df.set_index("ts")
    df["Open"] = df["Close"]  # uproszczenie - features nie uzywaja Open
    return df


def load_intraday_for_replay(cfg: Config, synthetic: bool) -> tuple[pd.DataFrame, str]:
    """Zwraca (intraday_df, zrodlo). Patrz docstring modulu po kolejnosc prob."""
    if synthetic:
        return make_synthetic_session(cfg), "syntetyczne (ILUSTRACJA)"

    snap = Path(cfg.local_daily_csv_path).parent.parent / SNAPSHOT_PATH
    if snap.exists():
        df = pd.read_csv(snap, parse_dates=["ts"]).set_index("ts")
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["time"] = pd.to_datetime(df["time"].astype(str)).dt.time
        return df, f"snapshot ({snap.name})"

    # ostatnia deska: yfinance live
    from .data_sources import fetch_intraday_yf
    df = fetch_intraday_yf(cfg)
    # zapisz snapshot na przyszlosc (zeby replay dzialal pozniej offline)
    out = df.copy()
    out.index.name = "ts"
    out.reset_index().to_csv(snap, index=False)
    return df, "yfinance live (zapisano snapshot)"


def replay_session(intraday_df: pd.DataFrame, daily_ctx, cfg: Config,
                   model, feature_cols: list, session_date=None) -> pd.DataFrame:
    """Liczy proba szczytu dla kazdej swiecy WYBRANEJ sesji.

    Zwraca DataFrame z kolumnami: time, close, proba, alert (proba>=prog),
    target (ground truth: czy faktycznie nastapil spadek >drop% w horyzoncie),
    is_session_high (czy ta swieca to szczyt dnia).
    """
    feat_df, cols = build_features(intraday_df, daily_ctx, cfg)
    if feat_df.empty:
        return pd.DataFrame()

    if session_date is None:
        session_date = sorted(feat_df["date"].unique())[-1]  # ostatnia sesja
    day = feat_df[feat_df["date"] == session_date].sort_index()
    if day.empty:
        return pd.DataFrame()

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(day[feature_cols])[:, 1]
    else:
        proba = model.predict(day[feature_cols]).astype(float)

    session_high = day["High"].max() if "High" in day else day["Close"].max()
    out = pd.DataFrame({
        "time": [t.strftime("%H:%M") for t in day["time"]],
        "close": day["Close"].values,
        "proba": proba,
        "alert": proba >= cfg.alert_probability_threshold,
        "target": day["target_local_top"].values.astype(int),
        "is_session_high": (day["High"].values if "High" in day else day["Close"].values) >= session_high,
    })
    return out.reset_index(drop=True)
