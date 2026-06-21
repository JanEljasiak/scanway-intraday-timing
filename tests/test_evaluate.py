"""
Testy oceny dzien-po-dniu (evaluate) - dzialaja OFFLINE na danych
syntetycznych. Uruchom: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.evaluate import chronological_split, daywise_evaluation, logistic_formula
from src.features import build_features
from src.replay import make_synthetic_session


def _prep(n_sessions=40, seed=42):
    cfg = load_config()
    intraday = make_synthetic_session(cfg, n_sessions=n_sessions, seed=seed)
    feat_df, cols = build_features(intraday, None, cfg)
    return cfg, feat_df, cols


def test_chronological_split_is_ordered_and_sized():
    cfg, feat_df, cols = _prep()
    train, test = chronological_split(feat_df, train_frac=0.8)
    # trening i test nie zachodza na siebie, trening jest WCZESNIEJ
    assert set(train).isdisjoint(test)
    assert max(train) < min(test)
    # proporcja ~80/20
    total = len(train) + len(test)
    assert abs(len(train) / total - 0.8) < 0.05


def test_daywise_evaluation_structure_and_stats():
    cfg, feat_df, cols = _prep()
    per_day, pooled, model = daywise_evaluation(feat_df, cols, cfg, "logistic_regression", 0.8)
    assert {"date", "n_bars", "n_target", "auc", "precyzja", "pokrycie", "n_alertow"}.issubset(per_day.columns)
    assert len(per_day) == pooled["n_test_sessions"]
    # p-value w [0,1], dni bijace losowy <= dni ogolem
    assert 0.0 <= pooled["p_value"] <= 1.0
    assert pooled["days_beat_random"] <= pooled["days_total"]
    # rozklad permutacji centruje sie ~0.5 (model losowy)
    assert abs(pooled["perm_null_mean"] - 0.5) < 0.1


def test_logistic_formula_has_terms():
    cfg, feat_df, cols = _prep()
    _, _, model = daywise_evaluation(feat_df, cols, cfg, "logistic_regression", 0.8)
    fml = logistic_formula(model, cols)
    assert "intercept" in fml and "terms" in fml
    assert set(fml["terms"]["feature"]) == set(cols)


if __name__ == "__main__":
    test_chronological_split_is_ordered_and_sized()
    test_daywise_evaluation_structure_and_stats()
    test_logistic_formula_has_terms()
    print("Testy evaluate (offline) przeszly OK.")
