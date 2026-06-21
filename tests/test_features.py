"""
Testy, ktore NIE wymagaja internetu - sprawdzaja tylko, czy pipeline na
lokalnym CSV (dane dzienne) dziala poprawnie. Uruchom: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data_sources import load_local_daily
from src.features import build_daily_context_features, daily_seasonality


def test_local_csv_loads():
    cfg = load_config()
    df = load_local_daily(cfg)
    assert len(df) > 100
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)


def test_daily_context_features_no_lookahead_columns():
    cfg = load_config()
    df = load_local_daily(cfg)
    ctx = build_daily_context_features(df)
    assert not ctx.empty
    expected_cols = {"prior_day_ret_pct", "realized_vol_10d_pct",
                      "dist_from_52w_high_pct", "up_streak", "dow_num", "gap_pct"}
    assert expected_cols.issubset(ctx.columns)
    assert ctx.isna().sum().sum() == 0


def test_daily_seasonality_runs():
    cfg = load_config()
    df = load_local_daily(cfg)
    result = daily_seasonality(df)
    assert len(result) <= 5  # max 5 dni roboczych
    assert "avg_ret_pct" in result.columns


if __name__ == "__main__":
    test_local_csv_loads()
    test_daily_context_features_no_lookahead_columns()
    test_daily_seasonality_runs()
    print("Wszystkie testy offline przeszly OK.")
