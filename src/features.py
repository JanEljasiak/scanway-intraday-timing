"""
Inzynieria cech:
- cechy DZIENNE liczone na pelnej historii (od 2023-10-11) - dostarczaja
  dlugoterminowego kontekstu (trend, zmiennosc, dzien tygodnia, luka).
- cechy SRODDZIENNE liczone na danych live z yfinance (~60 dni) - lokalna
  dynamika danej sesji (VWAP, RSI, momentum, wolumen).
- target: czy w nadchodzacym oknie czasowym cena spadnie ponizej obecnej
  o zadany % - czyli "czy TERAZ byl dobry moment na sprzedaz".

Wszystkie cechy dzienne sa przesuwane (.shift(1)) tak, by na danej sesji
uzywac wylacznie informacji znanej PRZED jej otwarciem - bez tego model
"podgladalby przyszlosc" i wyniki backtestu bylyby falszywie dobre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def daily_seasonality(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Sredni zwrot / zmiennosc / czestosc domkniecia luki wg dnia tygodnia."""
    d = daily_df.copy()
    d["ret_pct"] = d["close"].pct_change() * 100
    d["dow"] = pd.to_datetime(d.index).day_name()
    d["gap_pct"] = (d["open"] / d["close"].shift(1) - 1) * 100
    d["gap_filled_same_day"] = (
        ((d["gap_pct"] > 0) & (d["low"] <= d["close"].shift(1))) |
        ((d["gap_pct"] < 0) & (d["high"] >= d["close"].shift(1)))
    )
    d["intraday_range_pct"] = (d["high"] - d["low"]) / d["open"] * 100

    by_dow = d.groupby("dow").agg(
        avg_ret_pct=("ret_pct", "mean"),
        avg_range_pct=("intraday_range_pct", "mean"),
        gap_fill_rate_pct=("gap_filled_same_day", "mean"),
        n=("ret_pct", "count"),
    )
    by_dow["gap_fill_rate_pct"] *= 100
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    return by_dow.reindex([d for d in order if d in by_dow.index])


def build_daily_context_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    d = daily_df.copy()
    d["prior_day_ret_pct"] = d["close"].pct_change() * 100
    d["realized_vol_10d_pct"] = d["close"].pct_change().rolling(10).std() * 100
    d["dist_from_52w_high_pct"] = (d["close"] / d["close"].rolling(252, min_periods=20).max() - 1) * 100
    up = (d["close"].diff() > 0).astype(int)
    d["up_streak"] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    d["gap_pct"] = (d["open"] / d["close"].shift(1) - 1) * 100
    d["dow_num"] = pd.to_datetime(d.index).dayofweek

    cols_lag = ["prior_day_ret_pct", "realized_vol_10d_pct", "dist_from_52w_high_pct", "up_streak"]
    out = d[cols_lag].shift(1).join(d[["dow_num", "gap_pct"]])
    out.index.name = "date"
    return out.dropna()


def intraday_seasonality(intraday_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, day_df in intraday_df.groupby("date"):
        day_df = day_df.sort_index()
        if len(day_df) < 3:
            continue
        day_open = day_df["Open"].iloc[0]
        day_high = day_df["High"].max()
        out.append(pd.DataFrame({
            "time": day_df["time"],
            "ret_from_open": (day_df["Close"] / day_open - 1) * 100,
            "is_daily_high": (day_df["High"] == day_high).values,
        }))
    all_days = pd.concat(out, ignore_index=True)
    summary = all_days.groupby("time").agg(
        avg_ret_from_open_pct=("ret_from_open", "mean"),
        median_ret_from_open_pct=("ret_from_open", "median"),
        pct_of_days_this_is_daily_high=("is_daily_high", "mean"),
        n_obs=("ret_from_open", "count"),
    ).sort_index()
    summary["pct_of_days_this_is_daily_high"] *= 100
    return summary


def build_features(intraday_df: pd.DataFrame, daily_context: pd.DataFrame,
                    cfg: Config) -> tuple[pd.DataFrame, list]:
    df = intraday_df.copy().sort_index()
    df["vwap_cum"] = (df["Close"] * df["Volume"]).groupby(df["date"]).cumsum() / \
                      df["Volume"].groupby(df["date"]).cumsum()
    df["dist_from_vwap_pct"] = (df["Close"] / df["vwap_cum"] - 1) * 100
    df["ret_5"] = df["Close"].pct_change(5) * 100
    df["ret_1"] = df["Close"].pct_change(1) * 100
    df["vol_zscore"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    df["minute_of_day"] = df.index.hour * 60 + df.index.minute
    df["rsi_14"] = compute_rsi(df["Close"], 14)

    feature_cols = ["dist_from_vwap_pct", "ret_5", "ret_1", "vol_zscore", "minute_of_day", "rsi_14"]

    if daily_context is not None and not daily_context.empty:
        ctx = daily_context.copy()
        ctx.index = pd.to_datetime(ctx.index).date
        df = df.join(ctx, on="date")
        feature_cols += list(ctx.columns)

    horizon = cfg.sell_horizon_bars
    drop_thr = cfg.sell_target_drop_pct / 100
    fut_min = df["Low"].shift(-1).rolling(horizon).min().shift(-(horizon - 1))
    df["target_local_top"] = ((df["Close"] - fut_min) / df["Close"] > drop_thr).astype(int)

    df = df.dropna(subset=feature_cols + ["target_local_top"])
    return df, feature_cols
