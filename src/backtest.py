"""
Walidacja chronologiczna (walk-forward): dane dzielimy na kolejne okna
TYLKO do przodu w czasie (train zawsze starszy niz test). To jedyny
uczciwy sposob oceny modelu na danych czasowych - losowy train_test_split
dawalby falszywie wysoka skutecznosc (model "widzialby" fragmenty
przyszlosci sasiadujace z probkami treningowymi).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .config import Config
from .models import get_candidate_models


def _walk_forward_splits(dates: list, n_splits: int):
    fold_size = max(len(dates) // (n_splits + 1), 5)
    for i in range(n_splits):
        train_dates = dates[: fold_size * (i + 1)]
        test_dates = dates[fold_size * (i + 1): fold_size * (i + 2)]
        if not test_dates:
            break
        yield train_dates, test_dates


def walk_forward_eval(df: pd.DataFrame, feature_cols: list, model, n_splits: int = 6) -> pd.DataFrame:
    dates = sorted(df["date"].unique())
    rows = []

    for fold, (train_dates, test_dates) in enumerate(_walk_forward_splits(dates, n_splits)):
        train = df[df["date"].isin(train_dates)]
        test = df[df["date"].isin(test_dates)]
        if train["target_local_top"].nunique() < 2 or len(test) < 5:
            continue

        m = clone(model)
        m.fit(train[feature_cols], train["target_local_top"])
        pred = m.predict(test[feature_cols])

        row = {
            "fold": fold,
            "n_train": len(train),
            "n_test": len(test),
            "precision": precision_score(test["target_local_top"], pred, zero_division=0),
            "recall": recall_score(test["target_local_top"], pred, zero_division=0),
            "f1": f1_score(test["target_local_top"], pred, zero_division=0),
        }
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(test[feature_cols])[:, 1]
            try:
                row["roc_auc"] = roc_auc_score(test["target_local_top"], proba)
            except ValueError:
                row["roc_auc"] = np.nan
        else:
            row["roc_auc"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def compare_models(df: pd.DataFrame, feature_cols: list, cfg: Config) -> pd.DataFrame:
    """Uruchamia walk-forward dla kazdego kandydata i zwraca tabele porownawcza."""
    candidates = get_candidate_models(random_state=cfg.random_state)
    summary_rows = []

    for name, model in candidates.items():
        results = walk_forward_eval(df, feature_cols, model, n_splits=cfg.walk_forward_splits)
        if results.empty:
            continue
        summary_rows.append({
            "model": name,
            "avg_precision": results["precision"].mean(),
            "avg_recall": results["recall"].mean(),
            "avg_f1": results["f1"].mean(),
            "avg_roc_auc": results["roc_auc"].mean(),
            "n_folds": len(results),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        "avg_roc_auc", ascending=False, na_position="last"
    ).reset_index(drop=True)
    return summary


def pick_best_model(summary: pd.DataFrame, exclude: tuple[str, ...] = ("baseline_most_frequent",)) -> str:
    """Wybiera najlepszy model wg avg_roc_auc, ignorujac baseline.

    F1 jest zwodniczy przy niezbalansowanych klasach: model przewidujacy
    zawsze klase wiekszosciowa dostaje recall=1.0 i wysokie F1, mimo ze
    ROC AUC=0.5 (brak realnej sily predykcyjnej). ROC AUC mierzy
    rozdzielczosc modelu niezaleznie od progu i nie da sie go "oszukac"
    samym predykowaniem jednej klasy.
    """
    candidates = summary[~summary["model"].isin(exclude)].dropna(subset=["avg_roc_auc"])
    if candidates.empty:
        raise ValueError("Brak kandydatow z policzonym ROC AUC (poza wykluczonymi).")
    return candidates.sort_values("avg_roc_auc", ascending=False).iloc[0]["model"]


def explain_model(model, feature_cols: list) -> pd.DataFrame:
    """Zwraca sile/kierunek wplywu kazdej cechy, jesli model na to pozwala.

    Dla modeli liniowych (Pipeline ze 'clf') zwraca wspolczynnik (kierunek +
    sila po standaryzacji). Dla modeli drzewiastych zwraca feature_importances_
    (sila, bez kierunku). Dla innych (np. KNN, SVM) zwraca puste DataFrame -
    te modele nie dostarczaja prostej interpretacji per-cecha.
    """
    estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(estimator, "coef_"):
        values = estimator.coef_.ravel()
        kind = "coef (standaryzowany)"
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        kind = "feature_importance"
    else:
        return pd.DataFrame(columns=["feature", "value", "kind"])

    out = pd.DataFrame({"feature": feature_cols, "value": values, "kind": kind})
    return out.sort_values("value", key=abs, ascending=False).reset_index(drop=True)


def train_and_save_best(df: pd.DataFrame, feature_cols: list, cfg: Config,
                         model_name: str) -> str:
    """Trenuje wybrany model na CALYM dostepnym zbiorze i zapisuje go na dysk."""
    candidates = get_candidate_models(random_state=cfg.random_state)
    if model_name not in candidates:
        raise ValueError(f"Nieznany model: {model_name}. Dostepne: {list(candidates)}")

    model = candidates[model_name]
    model.fit(df[feature_cols], df["target_local_top"])

    importance = explain_model(model, feature_cols)
    if not importance.empty:
        print(f"\nWplyw cech na decyzje modelu ({importance.iloc[0]['kind']}):")
        print(importance.to_string(index=False))
    else:
        print(f"\nModel '{model_name}' nie udostepnia prostej interpretacji per-cecha "
              "(np. KNN/SVM) - jego decyzje nie sa latwe do rozlozenia na pojedyncze cechy.")

    model_path = cfg.models_dir / "best_model.joblib"
    meta_path = cfg.models_dir / "best_model_meta.json"

    joblib.dump(model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "feature_cols": feature_cols,
            "trained_on_rows": len(df),
            "sell_horizon_bars": cfg.sell_horizon_bars,
            "sell_target_drop_pct": cfg.sell_target_drop_pct,
        }, f, ensure_ascii=False, indent=2)

    return str(model_path)
