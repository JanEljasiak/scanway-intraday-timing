"""
Testy odtworzenia sesji (replay) - dzialaja OFFLINE na danych syntetycznych.
Sprawdzaja, ze mechanizm "swieca-po-swiecy -> proba -> alert" dziala
end-to-end bez internetu. Uruchom: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.features import build_features
from src.replay import (
    make_synthetic_session,
    replay_session,
    session_roc_auc,
    threshold_sweep,
    train_excluding_session,
)


def _train_quick_model(feat_df, feature_cols):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(feat_df[feature_cols], feat_df["target_local_top"])
    return model


def test_synthetic_session_has_expected_shape():
    cfg = load_config()
    df = make_synthetic_session(cfg, n_sessions=10, seed=1)
    assert {"Close", "High", "Low", "Volume", "date", "time"}.issubset(df.columns)
    assert df["date"].nunique() == 10
    # 33 swiece 09:00-17:00 co 15 min
    assert (df.groupby("date").size() == 33).all()


def test_replay_runs_and_produces_probabilities():
    cfg = load_config()
    intraday = make_synthetic_session(cfg, n_sessions=30, seed=42)
    feat_df, cols = build_features(intraday, None, cfg)
    assert not feat_df.empty

    sessions = sorted(feat_df["date"].unique())
    train = feat_df[feat_df["date"].isin(sessions[:-1])]
    model = _train_quick_model(train, cols)

    rep = replay_session(intraday, None, cfg, model, cols)
    assert not rep.empty
    # struktura wyniku
    assert {"time", "close", "proba", "alert", "target", "is_session_high"}.issubset(rep.columns)
    # prawdopodobienstwa w [0,1]
    assert rep["proba"].between(0, 1).all()
    # ostatnia sesja jest wymuszona jako "peak-at-open" -> sa realne momenty sprzedazy
    assert rep["target"].sum() > 0
    # dokladnie jeden (lub wiecej remisowych) szczyt dnia oznaczony
    assert rep["is_session_high"].sum() >= 1


def test_alert_threshold_is_respected():
    cfg = load_config()
    intraday = make_synthetic_session(cfg, n_sessions=30, seed=42)
    feat_df, cols = build_features(intraday, None, cfg)
    sessions = sorted(feat_df["date"].unique())
    train = feat_df[feat_df["date"].isin(sessions[:-1])]
    model = _train_quick_model(train, cols)
    rep = replay_session(intraday, None, cfg, model, cols)
    # alert == (proba >= prog) dla kazdej swiecy
    assert (rep["alert"] == (rep["proba"] >= cfg.alert_probability_threshold)).all()


def test_train_excluding_session_is_out_of_sample():
    """Model nie moze byc trenowany na sesji, ktora potem odtwarzamy."""
    cfg = load_config()
    intraday = make_synthetic_session(cfg, n_sessions=30, seed=42)
    feat_df, cols = build_features(intraday, None, cfg)
    holdout = sorted(feat_df["date"].unique())[-1]
    model, name, n_train = train_excluding_session(feat_df, cols, cfg, "logistic_regression", holdout)
    # liczba probek treningowych = wszystkie OPROCZ sesji holdout
    n_holdout = int((feat_df["date"] == holdout).sum())
    assert n_train == len(feat_df) - n_holdout
    assert name == "logistic_regression"
    rep = replay_session(intraday, None, cfg, model, cols, session_date=holdout)
    assert not rep.empty


def test_threshold_sweep_monotonic_alerts():
    """Wyzszy prog => nie wiecej alertow (monotonicznosc)."""
    cfg = load_config()
    intraday = make_synthetic_session(cfg, n_sessions=30, seed=42)
    feat_df, cols = build_features(intraday, None, cfg)
    holdout = sorted(feat_df["date"].unique())[-1]
    model, _, _ = train_excluding_session(feat_df, cols, cfg, "logistic_regression", holdout)
    rep = replay_session(intraday, None, cfg, model, cols, session_date=holdout)
    sweep = threshold_sweep(rep)
    alerts = sweep["n_alertow"].tolist()
    assert alerts == sorted(alerts, reverse=True)
    # ROC AUC sesji jest liczba w [0,1] albo NaN (gdy jedna klasa)
    auc = session_roc_auc(rep)
    assert auc != auc or 0.0 <= auc <= 1.0  # NaN albo w zakresie


if __name__ == "__main__":
    test_synthetic_session_has_expected_shape()
    test_replay_runs_and_produces_probabilities()
    test_alert_threshold_is_respected()
    test_train_excluding_session_is_out_of_sample()
    test_threshold_sweep_monotonic_alerts()
    print("Testy replay (offline) przeszly OK.")
