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
from src.evaluate import daywise_evaluation, logistic_formula
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
    """Odtwarza jedna HISTORYCZNA sesje (out-of-sample) swieca-po-swiecy i
    pokazuje, jak zadzialalby alert na sesji, ktorej model NIE widzial."""
    from src.replay import session_roc_auc, threshold_sweep, train_excluding_session

    print("Wczytuje dane intraday do odtworzenia sesji...")
    intraday, source = load_intraday_for_replay(cfg, synthetic=synthetic)
    print(f"Zrodlo danych: {source}")

    # tryb syntetyczny uzywa tylko cech intraday; realny dokłada kontekst dzienny
    if synthetic:
        daily_ctx = None
        model_name = "logistic_regression"
    else:
        daily = get_daily_history(cfg, refresh_live=True)
        daily_ctx = build_daily_context_features(daily)
        try:
            _, meta = load_trained_model(cfg)
            model_name = meta.get("model_name", "logistic_regression")
        except FileNotFoundError:
            model_name = "logistic_regression"

    feat_df, feature_cols = build_features(intraday, daily_ctx, cfg)
    if feat_df.empty:
        print("Za malo danych do odtworzenia sesji.")
        return

    holdout = sorted(feat_df["date"].unique())[-1]   # ostatnia sesja = testowa
    model, model_name, n_train = train_excluding_session(
        feat_df, feature_cols, cfg, model_name, holdout)
    print(f"\nModel '{model_name}' wytrenowany na {n_train} probkach "
          f"BEZ sesji {holdout} (out-of-sample).")

    rep = replay_session(intraday, daily_ctx, cfg, model, feature_cols, session_date=holdout)
    if rep.empty:
        print("Za malo danych do odtworzenia sesji.")
        return

    print(f"\n=== Sesja testowa {holdout} (prog alertu {cfg.alert_probability_threshold:.0%}) ===")
    view = rep.copy()
    view["proba"] = (view["proba"] * 100).round(0).astype(int).astype(str) + "%"
    view["alert"] = view["alert"].map({True: "ALERT", False: ""})
    view["target"] = view["target"].map({1: "spadek>prog", 0: ""})
    view["szczyt_dnia"] = rep["is_session_high"].map({True: "<-- HIGH", False: ""})
    view = view.drop(columns=["is_session_high"]).rename(columns={
        "time": "godz", "close": "cena", "target": "ground_truth"})
    print(view.to_string(index=False))

    # podsumowanie skutecznosci na tej jednej sesji + ROC AUC sesji
    n_alert = int(rep["alert"].sum())
    n_target = int(rep["target"].sum())
    hits = int(((rep["alert"]) & (rep["target"] == 1)).sum())
    prec = hits / n_alert if n_alert else float("nan")
    rec = hits / n_target if n_target else float("nan")
    auc = session_roc_auc(rep)
    print(f"\nAlertow: {n_alert} | rzeczywistych momentow sprzedazy (target=1): {n_target}")
    print(f"Trafione alerty: {hits} | precyzja sesji: {prec:.0%} | pokrycie: {rec:.0%}")
    print(f"ROC AUC tej sesji: {auc:.3f}  (0.5 = losowo, 1.0 = idealnie)")

    print("\n--- Wplyw progu alertu na te sesje (analiza precyzja/pokrycie) ---")
    print(threshold_sweep(rep).to_string(index=False,
          formatters={"precyzja": "{:.0%}".format, "pokrycie": "{:.0%}".format}))

    print("\nUWAGA: to JEDNA sesja out-of-sample - ilustruje dzialanie, ale ma "
          "duza wariancje. Pelna ocena = `backtest` (walk-forward na wielu oknach).")


def cmd_evaluate(cfg, synthetic=False, train_frac=0.8, model_name=None):
    """Ocena dzien-po-dniu na chronologicznym splicie + dowod 'lepiej niz losowo'."""
    print("Wczytuje dane do oceny dzien-po-dniu...")
    intraday, source = load_intraday_for_replay(cfg, synthetic=synthetic)
    print(f"Zrodlo danych: {source}")

    if synthetic:
        daily_ctx = None
        model_name = model_name or "logistic_regression"
    else:
        daily = get_daily_history(cfg, refresh_live=True)
        daily_ctx = build_daily_context_features(daily)
        if model_name is None:
            try:
                _, meta = load_trained_model(cfg)
                model_name = meta.get("model_name", "logistic_regression")
            except FileNotFoundError:
                model_name = "logistic_regression"

    feat_df, feature_cols = build_features(intraday, daily_ctx, cfg)
    per_day, pooled, model = daywise_evaluation(
        feat_df, feature_cols, cfg, model_name, train_frac=train_frac)

    print(f"\nModel: {pooled['model']}  |  split chronologiczny "
          f"{int(train_frac*100)}/{100-int(train_frac*100)}")
    print(f"Sesje treningowe: {pooled['n_train_sessions']}  |  "
          f"testowe: {pooled['n_test_sessions']}")

    print("\n=== Ocena KAZDEGO dnia testowego ===")
    view = per_day.copy()
    for c in ("auc",):
        view[c] = view[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    for c in ("precyzja", "pokrycie"):
        view[c] = view[c].map(lambda v: f"{v:.0%}" if pd.notna(v) else "-")
    print(view.to_string(index=False))

    print("\n=== Czy to lepiej niz losowo? ===")
    print(f"ROC AUC na calym tescie (pooled): {pooled['pooled_auc']:.3f}")
    print(f"AUC losowego modelu (permutacja etykiet): "
          f"{pooled['perm_null_mean']:.3f} +/- {pooled['perm_null_std']:.3f}")
    print(f"p-value (szansa, ze taki AUC wyszedl przez przypadek): {pooled['p_value']:.4f}")
    print(f"Dni testowych z AUC > 0.5: {pooled['days_beat_random']} / {pooled['days_total']}")

    fml = logistic_formula(model, feature_cols)
    if fml:
        print("\n=== Formula modelu (regresja logistyczna) ===")
        print("p(szczyt) = 1 / (1 + exp(-z))")
        print(f"z = {fml['intercept']:+.3f}  +  SUMA[ coef_std_j * (x_j - srednia_j) / odchylenie_j ]")
        print("\nWagi po standaryzacji (im wieksza |waga|, tym wazniejsza cecha;")
        print("znak + => wyzsza wartosc cechy podnosi prawdopodobienstwo szczytu):")
        print(fml["terms"].to_string(index=False,
              formatters={"coef_std": "{:+.3f}".format}))
    print("\nUWAGA: jedna sesja/jeden split ma duza wariancje. Pelna ocena = "
          "`backtest` (walk-forward na wielu oknach).")


def cmd_peak(cfg, synthetic=False, train_frac=0.8):
    """Tryb peak: jeden sygnal dziennie = dzienne maksimum, z asymetryczna kara."""
    from src.peak import daily_peak_evaluation, penalty_sweep

    print("Wczytuje dane do trybu PEAK (jeden szczyt dziennie)...")
    intraday, source = load_intraday_for_replay(cfg, synthetic=synthetic)
    print(f"Zrodlo danych: {source}")
    daily_ctx = None
    if not synthetic:
        daily = get_daily_history(cfg, refresh_live=True)
        daily_ctx = build_daily_context_features(daily)

    feat_df, feature_cols = build_features(intraday, daily_ctx, cfg)
    per_day, summary, _ = daily_peak_evaluation(feat_df, feature_cols, cfg, train_frac=train_frac)

    print(f"\nTarget: dzienne maksimum (1 swieca/sesje) | kara za przegapienie (FN): "
          f"{summary['penalty']}x | split {int(train_frac*100)}/{100-int(train_frac*100)}")
    print(f"Sesje treningowe: {summary['n_train_sessions']} | testowe: {summary['n_test_sessions']}")

    print("\n=== Kazdy dzien testowy: gdzie byl szczyt vs gdzie alert ===")
    view = per_day.copy()
    view["lag_bars"] = view["lag_bars"].map(lambda v: f"{int(v):+d}" if pd.notna(v) else "-")
    view["trafiony"] = view["trafiony"].map({True: "TAK", False: "nie"})
    for c in ("regret_model", "regret_open", "regret_random"):
        view[c] = view[c].map(lambda v: f"{v:.2f}%")
    print(view.to_string(index=False))

    print("\n=== Jak blisko dziennego szczytu sprzedajesz? (sredni regret = % ponizej maksimum) ===")
    print(f"  MODEL (peak):        {summary['mean_regret_model']:.2f}%   <- nasz")
    print(f"  sprzedaz na otwarciu:{summary['mean_regret_open']:.2f}%")
    print(f"  sprzedaz losowa:     {summary['mean_regret_random']:.2f}%")
    print(f"  trzymanie do konca:  {summary['mean_regret_close']:.2f}%")
    print(f"  idealnie (sufit):    0.00%")
    print(f"\nTrafienia w +/-{summary['tolerance_bars']} swiec od szczytu: "
          f"{summary['hit_rate']:.0%} dni | alert odpalil w {summary['days_fired']}/"
          f"{summary['n_test_sessions']} dni")

    print("\n=== Czy to wiarygodne? (test permutacyjny) ===")
    print(f"Sredni regret modelu: {summary['mean_regret_model']:.2f}%")
    print(f"Sredni regret 1000 LOSOWYCH wyborow swiecy: {summary['perm_null_mean']:.2f}%")
    print(f"p-value (szansa, ze losowy wybor jest nie gorszy): {summary['p_value']:.4f}")

    print("\n=== Wplyw asymetrycznej kary (funkcji straty) na lapanie szczytu ===")
    sweep = penalty_sweep(feat_df, feature_cols, cfg, train_frac=train_frac)
    print(sweep.to_string(index=False, formatters={
        "trafione_%": "{:.0f}%".format, "sredni_regret_%": "{:.2f}%".format}))

    print("\nUWAGA: ocena out-of-sample na realnych danych, ale male N - traktuj "
          "jako orientacje, nie gwarancje. Sprzedaz live ma tez ~15 min opoznienie.")


def main():
    parser = argparse.ArgumentParser(description="SCW - intraday timing toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update-data", help="Odswiez dane i pokaz sezonowosc")
    sub.add_parser("backtest", help="Porownaj modele ML i zapisz najlepszy")
    sub.add_parser("live", help="Uruchom petle alertowa")
    p_replay = sub.add_parser("replay", help="Odtworz jedna sesje swieca-po-swiecy")
    p_replay.add_argument("--synthetic", action="store_true",
                          help="Uzyj sesji syntetycznej (ilustracja, bez internetu)")
    p_eval = sub.add_parser("evaluate", help="Ocena dzien-po-dniu + test 'lepiej niz losowo'")
    p_eval.add_argument("--synthetic", action="store_true",
                        help="Uzyj danych syntetycznych (ilustracja, bez internetu)")
    p_eval.add_argument("--train-frac", type=float, default=0.8,
                        help="Udzial sesji treningowych (chronologicznie). Domyslnie 0.8")
    p_peak = sub.add_parser("peak", help="Tryb 'jeden szczyt dziennie' z asymetryczna kara")
    p_peak.add_argument("--synthetic", action="store_true",
                        help="Uzyj danych syntetycznych (ilustracja, bez internetu)")
    p_peak.add_argument("--train-frac", type=float, default=0.8,
                        help="Udzial sesji treningowych (chronologicznie). Domyslnie 0.8")

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
    elif args.command == "evaluate":
        cmd_evaluate(cfg, synthetic=args.synthetic, train_frac=args.train_frac)
    elif args.command == "peak":
        cmd_peak(cfg, synthetic=args.synthetic, train_frac=args.train_frac)


if __name__ == "__main__":
    main()
