"""
Ocena modelu DZIEN-PO-DNIU na chronologicznym podziale train/test (domyslnie
80/20) + dowod, ze model dziala lepiej niz losowo.

Po co osobny modul (skoro jest backtest):
- `backtest` robi walk-forward (wiele okien) i zwraca usrednione metryki -
  najlepsza ocena statystyczna, ale "srednia z wielu liczb".
- tutaj robimy JEDEN chronologiczny split i pokazujemy KAZDY dzien testowy
  osobno (ROC AUC, precyzja, pokrycie), a potem testujemy formalnie hipoteze
  "to tylko przypadek" (permutacja etykiet + test znakow per dzien).

Dlaczego chronologicznie, a nie losowe k-fold / shuffle:
- to szereg czasowy; losowe mieszanie wstawiloby przyszlosc do treningu
  (przeciek) i zawyzylo wynik. Trening = wczesniejsze sesje, test = pozniejsze.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from .config import Config
from .models import get_candidate_models


def chronological_split(feat_df: pd.DataFrame, train_frac: float = 0.8):
    """Dzieli sesje chronologicznie. Zwraca (train_dates, test_dates)."""
    sessions = sorted(feat_df["date"].unique())
    n_train = max(int(round(len(sessions) * train_frac)), 1)
    return sessions[:n_train], sessions[n_train:]


def daywise_evaluation(feat_df: pd.DataFrame, feature_cols: list, cfg: Config,
                       model_name: str, train_frac: float = 0.8):
    """Trenuje model na czesci treningowej i ocenia KAZDY dzien testowy osobno.

    Zwraca (per_day_df, pooled, model):
      per_day_df: kolumny date, n_bars, n_target, auc, precyzja, pokrycie, n_alertow
      pooled: dict z metrykami na calym tescie (pooled) + wynik permutacji
      model: wytrenowany model (na danych treningowych)
    """
    train_dates, test_dates = chronological_split(feat_df, train_frac)
    train = feat_df[feat_df["date"].isin(train_dates)]
    test = feat_df[feat_df["date"].isin(test_dates)]

    models = get_candidate_models(random_state=cfg.random_state)
    if model_name not in models:
        model_name = "logistic_regression"
    model = clone(models[model_name])
    model.fit(train[feature_cols], train["target_local_top"])

    def proba_of(df):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(df[feature_cols])[:, 1]
        return model.predict(df[feature_cols]).astype(float)

    thr = cfg.alert_probability_threshold
    rows = []
    for d in test_dates:
        day = test[test["date"] == d]
        y = day["target_local_top"].values
        p = proba_of(day)
        alert = p >= thr
        n_alert = int(alert.sum())
        hits = int(((alert) & (y == 1)).sum())
        rows.append({
            "date": d,
            "n_bars": len(day),
            "n_target": int(y.sum()),
            "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
            "precyzja": hits / n_alert if n_alert else np.nan,
            "pokrycie": hits / int(y.sum()) if y.sum() else np.nan,
            "n_alertow": n_alert,
        })
    per_day = pd.DataFrame(rows)

    # --- metryki "pooled" na calym tescie ---
    y_all = test["target_local_top"].values
    p_all = proba_of(test)
    pooled_auc = roc_auc_score(y_all, p_all) if len(np.unique(y_all)) > 1 else np.nan

    # --- permutacja: czy AUC mogl wyjsc taki przez przypadek? ---
    rng = np.random.default_rng(cfg.random_state)
    n_perm = 1000
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = roc_auc_score(rng.permutation(y_all), p_all)
    p_value = float((null >= pooled_auc).mean())

    # --- test znakow: ile dni testowych ma AUC > 0.5 ---
    valid = per_day["auc"].dropna()
    n_days = len(valid)
    n_beat = int((valid > 0.5).sum())

    pooled = {
        "model": model_name,
        "n_train_sessions": len(train_dates),
        "n_test_sessions": len(test_dates),
        "pooled_auc": pooled_auc,
        "perm_null_mean": float(null.mean()),
        "perm_null_std": float(null.std()),
        "p_value": p_value,
        "perm_null": null,
        "days_total": n_days,
        "days_beat_random": n_beat,
    }
    return per_day, pooled, model


def logistic_formula(model, feature_cols: list) -> dict:
    """Wyciaga jawna formule regresji logistycznej (jesli to ten model).

    Pipeline = StandardScaler + LogisticRegression, wiec:
      z = intercept + SUM_j  coef_j * (x_j - mean_j) / scale_j
      p = 1 / (1 + exp(-z))          # prawdopodobienstwo "lokalnego szczytu"
    """
    if not hasattr(model, "named_steps") or "clf" not in model.named_steps:
        return {}
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return {}
    scaler = model.named_steps.get("scale")
    coef = clf.coef_.ravel()
    intercept = float(clf.intercept_[0])

    terms = pd.DataFrame({
        "feature": feature_cols,
        "coef_std": coef,  # waga po standaryzacji (porownywalna sila)
    }).sort_values("coef_std", key=abs, ascending=False).reset_index(drop=True)

    out = {"intercept": intercept, "terms": terms}
    if scaler is not None:
        out["mean"] = dict(zip(feature_cols, scaler.mean_))
        out["scale"] = dict(zip(feature_cols, scaler.scale_))
    return out
