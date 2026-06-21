"""
Wysylka alertow + petla live. Domyslnie alert trafia tylko do konsoli;
jesli w .env ustawione sa TELEGRAM_BOT_TOKEN i TELEGRAM_CHAT_ID, wyslemy
tez wiadomosc na Telegrama.

WAZNE OGRANICZENIE: yfinance ma opoznienie danych (typowo ~15 min, dla
malo plynnych spolek GPW czasem wiecej). W polaczeniu z odpytywaniem co
poll_seconds (domyslnie 15 min) realne opoznienie reakcji moze siegac
30 min. To narzedzie orientacyjne, nie system do precyzyjnego tradingu.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, time as dtime

import joblib
import pandas as pd
import requests

from .config import Config
from .data_sources import fetch_intraday_yf, get_daily_history
from .features import build_daily_context_features, build_features


def send_console(message: str) -> None:
    print(f"[ALERT {datetime.now():%Y-%m-%d %H:%M:%S}] {message}")


def send_telegram(message: str, cfg: Config) -> None:
    if not cfg.telegram_bot_token or not cfg.telegram_chat_id:
        return
    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": cfg.telegram_chat_id, "text": message}, timeout=10)
    except Exception as e:
        print(f"[alert] Nie udalo sie wyslac na Telegrama: {e}")


def send_alert(message: str, cfg: Config) -> None:
    send_console(message)
    send_telegram(message, cfg)


def load_trained_model(cfg: Config):
    model_path = cfg.models_dir / "best_model.joblib"
    meta_path = cfg.models_dir / "best_model_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(
            "Brak wytrenowanego modelu. Uruchom najpierw: python main.py backtest"
        )
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def _in_market_hours(cfg: Config) -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    open_h, open_m = map(int, cfg.market_open.split(":"))
    close_h, close_m = map(int, cfg.market_close.split(":"))
    return dtime(open_h, open_m) <= now.time() <= dtime(close_h, close_m)


def run_live_loop(cfg: Config) -> None:
    model, meta = load_trained_model(cfg)
    feature_cols = meta["feature_cols"]
    print(f"Zaladowano model '{meta['model_name']}' (trenowany na {meta['trained_on_rows']} wierszach).")
    print(f"Pilnuje SCW w godzinach {cfg.market_open}-{cfg.market_close} ({cfg.timezone}), "
          f"co {cfg.poll_seconds // 60} min. Ctrl+C aby przerwac.")

    daily_df = get_daily_history(cfg, refresh_live=True)
    daily_ctx = build_daily_context_features(daily_df)
    last_daily_refresh = datetime.now().date()

    while True:
        try:
            if _in_market_hours(cfg):
                if datetime.now().date() != last_daily_refresh:
                    daily_df = get_daily_history(cfg, refresh_live=True)
                    daily_ctx = build_daily_context_features(daily_df)
                    last_daily_refresh = datetime.now().date()

                intraday = fetch_intraday_yf(cfg)
                feat_df, _ = build_features(intraday, daily_ctx, cfg)
                if feat_df.empty:
                    print(f"{datetime.now():%H:%M} - brak wystarczajacych danych do predykcji jeszcze.")
                else:
                    latest = feat_df.iloc[[-1]]
                    proba = model.predict_proba(latest[feature_cols])[0, 1] \
                        if hasattr(model, "predict_proba") else float(model.predict(latest[feature_cols])[0])
                    price = latest["Close"].values[0]
                    if proba >= cfg.alert_probability_threshold:
                        send_alert(
                            f"SCW: prawdopodobienstwo lokalnego szczytu = {proba:.0%} "
                            f"przy cenie {price:.2f} PLN (prog: {cfg.alert_probability_threshold:.0%})",
                            cfg,
                        )
                    else:
                        print(f"{datetime.now():%H:%M} - SCW {price:.2f} PLN, "
                              f"prawdopodobienstwo szczytu {proba:.0%} (ponizej progu)")
            else:
                print(f"{datetime.now():%H:%M} - rynek zamkniety, czekam...")
        except Exception as e:
            print(f"[live] Blad: {e}")

        time.sleep(cfg.poll_seconds)
