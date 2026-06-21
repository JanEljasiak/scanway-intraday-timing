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

from src.alert import load_trained_model, run_live_loop
from src.backtest import compare_models, pick_best_model, train_and_save_best
from src.config import load_config
from src.data_sources import fetch_intraday_yf, get_daily_history
from src.features import build_daily_context_features, build_features, daily_seasonality, intraday_seasonality
from src.replay import load_intraday_for_replay, replay_session

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


def cmd_replay(cfg, synthetic=False):
    """Odtwarza jedna sesje swieca-po-swiecy i pokazuje, gdzie model wyslalby
    alert sprzedazy w porownaniu z faktycznymi szczytami (ground truth)."""
    print("Wczytuje dane intraday do odtworzenia sesji...")
    intraday, source = load_intraday_for_replay(cfg, synthetic=synthetic)
    print(f"Zrodlo danych: {source}")

    if synthetic:
        # tryb ilustracyjny: trenujemy szybki model na samych cechach intraday
        from src.backtest import train_and_save_best
        feat_df, feature_cols = build_features(intraday, None, cfg)
        sessions = sorted(feat_df["date"].unique())
        train_df = feat_df[feat_df["date"].isin(sessions[:-1])]  # bez ostatniej sesji
        train_and_save_best(train_df, feature_cols, cfg, "logistic_regression")
        model, meta = load_trained_model(cfg)
        daily_ctx = None
    else:
        model, meta = load_trained_model(cfg)
        feature_cols = meta["feature_cols"]
        daily = get_daily_history(cfg, refresh_live=True)
        daily_ctx = build_daily_context_features(daily)

    rep = replay_session(intraday, daily_ctx, cfg,
                         model, meta["feature_cols"], session_date=None)
    if rep.empty:
        print("Za malo danych do odtworzenia sesji.")
        return

    print(f"\n=== Odtworzenie ostatniej sesji (prog alertu {cfg.alert_probability_threshold:.0%}) ===")
    view = rep.copy()
    view["proba"] = (view["proba"] * 100).round(0).astype(int).astype(str) + "%"
    view["alert"] = view["alert"].map({True: "ALERT", False: ""})
    view["target"] = view["target"].map({1: "spadek>prog", 0: ""})
    view["szczyt_dnia"] = rep["is_session_high"].map({True: "<-- HIGH", False: ""})
    view = view.drop(columns=["is_session_high"]).rename(columns={
        "time": "godz", "close": "cena", "target": "ground_truth"})
    print(view.to_string(index=False))

    # podsumowanie skutecznosci na tej jednej sesji
    n_alert = int(rep["alert"].sum())
    n_target = int(rep["target"].sum())
    hits = int(((rep["alert"]) & (rep["target"] == 1)).sum())
    prec = hits / n_alert if n_alert else float("nan")
    rec = hits / n_target if n_target else float("nan")
    print(f"\nAlertow: {n_alert} | rzeczywistych momentow sprzedazy (target=1): {n_target}")
    print(f"Trafione alerty: {hits} | precyzja sesji: {prec:.0%} | pokrycie: {rec:.0%}")
    print("\nUWAGA: to jedna sesja - sluzy ZROZUMIENIU dzialania, nie ocenie modelu."
          " Pelna, uczciwa ocena jest w `backtest` (walk-forward, ROC AUC).")


def main():
    parser = argparse.ArgumentParser(description="SCW - intraday timing toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update-data", help="Odswiez dane i pokaz sezonowosc")
    sub.add_parser("backtest", help="Porownaj modele ML i zapisz najlepszy")
    sub.add_parser("live", help="Uruchom petle alertowa")
    p_replay = sub.add_parser("replay", help="Odtworz jedna sesje swieca-po-swiecy")
    p_replay.add_argument("--synthetic", action="store_true",
                          help="Uzyj sesji syntetycznej (ilustracja, bez internetu)")

    args = parser.parse_args()
    cfg = load_config()

    if args.command == "update-data":
        cmd_update_data(cfg)
    elif args.command == "backtest":
        cmd_backtest(cfg)
    elif args.command == "live":
        cmd_live(cfg)
    elif args.command == "replay":
        cmd_replay(cfg, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
