"""
Kandydaci do roli modelu "sygnal sprzedazy" - od prostego baseline'u po
kilka klasycznych klasyfikatorow. Cel nie jest "znalezc najlepszy mozliwy
model na swiecie", tylko sprawdzic, czy COKOLWIEK bije prosty baseline
w sposob stabilny na walidacji chronologicznej (walk-forward) - jesli nie,
to znak, ze sygnal jest zbyt slaby/zaszumiony, zeby na nim polegac.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_candidate_models(random_state: int = 42) -> dict:
    models = {
        # baseline - zawsze przewiduje klase wiekszosciowa. Jesli zaden
        # "prawdziwy" model go nie bije, oznacza to brak realnego sygnalu.
        "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),

        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                        random_state=random_state)),
        ]),

        "knn": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15)),
        ]),

        "svm_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced",
                        random_state=random_state)),
        ]),

        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            class_weight="balanced", random_state=random_state,
        ),

        "extra_trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            class_weight="balanced", random_state=random_state,
        ),

        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            random_state=random_state,
        ),
    }

    # XGBoost jest opcjonalny (patrz requirements.txt) - dodajemy tylko
    # jesli jest zainstalowany, zeby projekt dzialal bez niego "z pudelka".
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            eval_metric="logloss", random_state=random_state,
        )
    except ImportError:
        pass

    return models
