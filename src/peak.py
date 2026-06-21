"""
Tryb 'peak': JEDEN sygnal dziennie = DZIENNE MAKSIMUM (a nie kazdy lokalny
moment sprzedazy). To jest to, czego realnie chce uzytkownik: "powiedz mi raz
w ciagu dnia, ze TERAZ jest szczyt, to sprzedam".

Roznice wobec trybu local_top (features.target_local_top):
- TARGET: dokladnie 1 swieca na sesje - ta z najwyzszym Close (osiagalny
  najlepszy moment sprzedazy na zamknieciu swiecy).
- FUNKCJA STRATY: asymetryczna - przegapienie szczytu (false negative) karzemy
  `daily_high_fn_penalty` razy mocniej niz falszywy alarm (class_weight).
- DECYZJA: max 1 alert dziennie - bierzemy swiece z najwyzszym
  prawdopodobienstwem; alarmujemy, jesli przekroczy prog, inaczej brak alertu
  (a brak alertu jest karany w ocenie: "trzymasz do zamkniecia").

Ocena jest skrojona pod laika: glowna metryka to "ile % PONIZEJ dziennego
szczytu sprzedales" (regret). Im mniej, tym lepiej. Porownujemy z prostymi
strategiami (sprzedaz na otwarciu / losowo / na zamknieciu) i z idealem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config


def add_daily_high_target(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje target_daily_high = 1 dla swiecy z najwyzszym Close w sesji.

    Dokladnie jedna '1' na dzien (przy remisie - pierwsze wystapienie).
    """
    df = feat_df.sort_index().copy()
    flags = pd.Series(0, index=df.index)
    for _, g in df.groupby("date"):
        top_idx = g["Close"].idxmax()
        flags.loc[top_idx] = 1
    df["target_daily_high"] = flags.values
    return df


def train_peak_model(train_df: pd.DataFrame, feature_cols: list, cfg: Config,
                     penalty: float = None):
    """Regresja logistyczna z ASYMETRYCZNA kara za przegapienie szczytu.

    class_weight={0:1, 1:penalty} => blad na klasie 'szczyt' (przegapienie)
    kosztuje `penalty` razy wiecej niz falszywy alarm.
    """
    penalty = cfg.daily_high_fn_penalty if penalty is None else penalty
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight={0: 1.0, 1: penalty},
                                    random_state=cfg.random_state)),
    ])
    model.fit(train_df[feature_cols], train_df["target_daily_high"])
    return model


def _sell_metrics_for_day(g: pd.DataFrame, proba: np.ndarray, cfg: Config) -> dict:
    """Liczy decyzje modelu i strategii odniesienia dla JEDNEJ sesji."""
    g = g.sort_index()
    close = g["Close"].values
    best_i = int(close.argmax())                 # idealny moment (najwyzszy Close)
    best_price = close[best_i]

    def regret(sell_price):                       # % ponizej dziennego maksimum
        return (best_price - sell_price) / best_price * 100

    # --- decyzja modelu: max 1 alert dziennie (swieca o najwyzszej proba) ---
    alert_i = int(proba.argmax())
    fired = bool(proba[alert_i] >= cfg.alert_probability_threshold)
    # jesli nie odpalil - "trzymasz do zamkniecia" (sprzedaz na ostatniej swiecy)
    model_sell_i = alert_i if fired else len(close) - 1

    return {
        "n_bars": len(close),
        "peak_i": best_i,
        "alert_i": alert_i,
        "fired": fired,
        "lag_bars": alert_i - best_i,            # +pozno / -wczesnie
        "regret_model": regret(close[model_sell_i]),
        "regret_open": regret(close[0]),         # sprzedaz na otwarciu 09:00
        "regret_close": regret(close[-1]),       # nic nie robisz, trzymasz do konca
        "regret_random": float(np.mean([regret(c) for c in close])),  # losowa godzina
    }


def daily_peak_evaluation(feat_df: pd.DataFrame, feature_cols: list, cfg: Config,
                          train_frac: float = 0.8, penalty: float = None):
    """Ocena dzien-po-dniu trybu peak na chronologicznym splicie + test wiarygodnosci.

    Zwraca (per_day_df, summary, model).
    """
    from .evaluate import chronological_split
    df = add_daily_high_target(feat_df)
    train_dates, test_dates = chronological_split(df, train_frac)
    train = df[df["date"].isin(train_dates)]
    test = df[df["date"].isin(test_dates)]

    model = train_peak_model(train, feature_cols, cfg, penalty)
    proba_all = model.predict_proba(test[feature_cols])[:, 1]
    test = test.assign(_proba=proba_all)

    tol = cfg.peak_tolerance_bars
    rows = []
    for d, g in test.groupby("date"):
        g = g.sort_index()
        m = _sell_metrics_for_day(g, g["_proba"].values, cfg)
        times = [t.strftime("%H:%M") for t in g["time"]]
        rows.append({
            "date": d,
            "szczyt_o": times[m["peak_i"]],
            "alert_o": times[m["alert_i"]] if m["fired"] else "(brak)",
            "lag_bars": m["lag_bars"] if m["fired"] else np.nan,
            "trafiony": bool(m["fired"] and abs(m["lag_bars"]) <= tol),
            "regret_model": m["regret_model"],
            "regret_open": m["regret_open"],
            "regret_random": m["regret_random"],
        })
    per_day = pd.DataFrame(rows)

    # --- test wiarygodnosci: permutacja proba w obrebie dnia (model traci wiedze) ---
    rng = np.random.default_rng(cfg.random_state)
    n_perm = 1000
    obs = per_day["regret_model"].mean()
    null = np.empty(n_perm)
    test_groups = [g.sort_index() for _, g in test.groupby("date")]
    for k in range(n_perm):
        regs = []
        for g in test_groups:
            shuffled = rng.permutation(g["_proba"].values)
            regs.append(_sell_metrics_for_day(g, shuffled, cfg)["regret_model"])
        null[k] = float(np.mean(regs))
    # p-value: jak czesto LOSOWY wybor swiecy daje regret <= naszego (czyli nie gorszy)
    p_value = float((null <= obs).mean())

    summary = {
        "penalty": cfg.daily_high_fn_penalty if penalty is None else penalty,
        "n_train_sessions": len(train_dates),
        "n_test_sessions": len(test_dates),
        "tolerance_bars": tol,
        "mean_regret_model": obs,
        "mean_regret_open": per_day["regret_open"].mean(),
        "mean_regret_random": per_day["regret_random"].mean(),
        "mean_regret_close": float(np.mean([
            _sell_metrics_for_day(g.sort_index(), g["_proba"].values, cfg)["regret_close"]
            for g in test_groups])),
        "hit_rate": float(per_day["trafiony"].mean()),
        "days_fired": int((per_day["alert_o"] != "(brak)").sum()),
        "perm_null": null,
        "perm_null_mean": float(null.mean()),
        "p_value": p_value,
    }
    return per_day, summary, model


def penalty_sweep(feat_df: pd.DataFrame, feature_cols: list, cfg: Config,
                  penalties=(1.0, 3.0, 6.0, 12.0, 30.0), train_frac: float = 0.8) -> pd.DataFrame:
    """Pokazuje wplyw asymetrycznej kary na skutecznosc lapania szczytu."""
    rows = []
    for pen in penalties:
        per_day, summary, _ = daily_peak_evaluation(feat_df, feature_cols, cfg, train_frac, penalty=pen)
        rows.append({
            "kara_FN": pen,
            "trafione_%": summary["hit_rate"] * 100,
            "dni_z_alertem": summary["days_fired"],
            "sredni_regret_%": summary["mean_regret_model"],
        })
    return pd.DataFrame(rows)
