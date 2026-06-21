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
from src.replay import make_synthetic_session, replay_session


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


if __name__ == "__main__":
    test_synthetic_session_has_expected_shape()
    test_replay_runs_and_produces_probabilities()
    test_alert_threshold_is_respected()
    print("Testy replay (offline) przeszly OK.")
