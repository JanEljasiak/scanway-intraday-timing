"""
Testy trybu PEAK (jeden szczyt dziennie) - dzialaja OFFLINE na danych
syntetycznych. Uruchom: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.features import build_features
from src.peak import add_daily_high_target, daily_peak_evaluation, penalty_sweep
from src.replay import make_synthetic_session


def _prep(n_sessions=40, seed=42):
    cfg = load_config()
    intraday = make_synthetic_session(cfg, n_sessions=n_sessions, seed=seed)
    feat_df, cols = build_features(intraday, None, cfg)
    return cfg, feat_df, cols


def test_target_is_exactly_one_per_day():
    cfg, feat_df, cols = _prep()
    df = add_daily_high_target(feat_df)
    counts = df.groupby("date")["target_daily_high"].sum()
    assert (counts == 1).all()  # dokladnie jeden szczyt na sesje


def test_at_most_one_alert_per_day_and_valid_regret():
    cfg, feat_df, cols = _prep()
    per_day, summary, _ = daily_peak_evaluation(feat_df, cols, cfg, train_frac=0.8)
    # max jeden alert dziennie (kazdy wiersz to jeden dzien; alert lub brak)
    assert len(per_day) == summary["n_test_sessions"]
    # regret w sensownym zakresie
    assert per_day["regret_model"].between(0, 100).all()
    # p-value w [0,1]
    assert 0.0 <= summary["p_value"] <= 1.0
    # model nie gorszy niz losowy (na danych z sygnalem)
    assert summary["mean_regret_model"] <= summary["mean_regret_random"]


def test_higher_penalty_fires_more():
    cfg, feat_df, cols = _prep()
    sweep = penalty_sweep(feat_df, cols, cfg, penalties=(1.0, 30.0), train_frac=0.8)
    low = sweep.iloc[0]["dni_z_alertem"]
    high = sweep.iloc[1]["dni_z_alertem"]
    # wieksza kara za przegapienie => model alarmuje czesciej (lub tyle samo)
    assert high >= low


if __name__ == "__main__":
    test_target_is_exactly_one_per_day()
    test_at_most_one_alert_per_day_and_valid_regret()
    test_higher_penalty_fires_more()
    print("Testy peak (offline) przeszly OK.")
