"""
Punkt wejscia projektu. Tryby:

  python main.py update-data   - odswieza dane dzienne (Stooq) i pokazuje sezonowosc
  python main.py backtest      - porownuje modele ML (walk-forward) i zapisuje najlepszy
  python main.py live          - uruchamia petle alertowa w godzinach sesji GPW

Zobacz README.md po pelne instrukcje instalacji i uzycia.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.alert import run_live_loop
from src.backtest import compare_models, pick_best_model, train_and_save_best
from src.config import load_config
from src.data_sources import fetch_intraday_yf, get_daily_history
from src.features import build_daily_context_features, build_features, daily_seasonality, intraday_seasonality

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def cmd_update_data(cfg):
    print("Pobieram/aktualizuje dane dzienne...")
    daily = get_daily_history(cfg, refresh_live=True)
    print(f"Zakres danych dziennych: {daily.index.min()} -> {daily.index.max()} ({len(daily)} sesji)\n")

    print("=== Sezonowosc dzienna wg dnia tygodnia (pelna historia) ===")
    print(daily_seasonality(daily))

    print("\nPobieram dane intraday (live, ~60 dni)...")
    try:
        intraday = fetch_intraday_yf(cfg)
        print(f"Pobrano {len(intraday)} swiec z {intraday['date'].nunique()} sesji.\n")
        print("=== Sezonowosc sroddzienna (sredni zwrot od otwarcia wg godziny) ===")
        print(intraday_seasonality(intraday).sort_values("avg_ret_from_open_pct", ascending=False).head(10))
    except Exception as e:
        print(f"Nie udalo sie pobrac danych intraday ({e}). "
              f"To wymaga polaczenia z internetem przy kazdym uruchomieniu.")


def cmd_backtest(cfg):
    print("Buduje zbior cech (dane dzienne jako kontekst + intraday live)...")
    daily = get_daily_history(cfg, refresh_live=True)
    daily_ctx = build_daily_context_features(daily)
    intraday = fetch_intraday_yf(cfg)
    feat_df, feature_cols = build_features(intraday, daily_ctx, cfg)
    print(f"Cechy: {feature_cols}")
    print(f"Liczba probek: {len(feat_df)} z {feat_df['date'].nunique()} sesji\n")

    print("Porownuje modele (walk-forward, n_splits="
          f"{cfg.walk_forward_splits})...\n")
    summary = compare_models(feat_df, feature_cols, cfg)
    print(summary.to_string(index=False))

    if summary.empty:
        print("\nZa malo danych, zeby cokolwiek porownac. Sprobuj zmniejszyc "
              "walk_forward_splits w config.yaml.")
        return

    best_name = pick_best_model(summary)
    print(f"\nNajlepszy model wg sredniego ROC AUC (baseline wykluczony z wyboru): {best_name}")
    path = train_and_save_best(feat_df, feature_cols, cfg, best_name)
    print(f"Zapisano: {path}")
    print("\nUWAGA: 'najlepszy' tu znaczy najlepszy wzgledem pozostalych "
          "kandydatow na tym konkretnym, wciaz ograniczonym zbiorze danych "
          "(~60 dni intraday). To NIE jest gwarancja skutecznosci na zywo.")


def cmd_live(cfg):
    run_live_loop(cfg)


def main():
    parser = argparse.ArgumentParser(description="SCW - intraday timing toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update-data", help="Odswiez dane i pokaz sezonowosc")
    sub.add_parser("backtest", help="Porownaj modele ML i zapisz najlepszy")
    sub.add_parser("live", help="Uruchom petle alertowa")

    args = parser.parse_args()
    cfg = load_config()

    if args.command == "update-data":
        cmd_update_data(cfg)
    elif args.command == "backtest":
        cmd_backtest(cfg)
    elif args.command == "live":
        cmd_live(cfg)


if __name__ == "__main__":
    main()
