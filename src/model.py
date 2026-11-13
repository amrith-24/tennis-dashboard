"""
model.py — Tennis Match Outcome Prediction
-------------------------------------------
Trains and compares models on leakage-free features, picks the one with
the lowest cross-validated log loss, and predicts new matchups.

Cross-validation is grouped by match, so the two mirrored rows of one
match never end up on opposite sides of a train/test split.

Usage:
    python src/model.py        # evaluate models, save best to models/
"""

import os
import pickle
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

try:
    from features import build_features, build_state, FEATURE_COLUMNS
    from data import load_matches
except ImportError:  # imported as a package (e.g. in tests)
    from src.features import build_features, build_state, FEATURE_COLUMNS
    from src.data import load_matches

warnings.filterwarnings("ignore")

DATA_PATH = "matches.csv"
MODEL_PATH = os.path.join("models", "best_model.pkl")
LABEL_COL = "label"
RANDOM_SEED = 42


def load_data(path: str = DATA_PATH):
    """Return X, y, groups (match ids) for training."""
    feats = build_features(load_matches(path))
    return feats[FEATURE_COLUMNS], feats[LABEL_COL], feats["match_id"]


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=3, random_state=RANDOM_SEED
        ),
    }


def evaluate_model(name, model, X, y, groups) -> dict:
    n_splits = max(2, min(5, groups.nunique()))
    cv = GroupKFold(n_splits=n_splits)
    acc = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="accuracy")
    ll = -cross_val_score(model, X, y, groups=groups, cv=cv, scoring="neg_log_loss")
    return {
        "Model": name,
        "Accuracy": float(acc.mean()),
        "Accuracy ±": float(acc.std()),
        "Log loss": float(ll.mean()),
        "Log loss ±": float(ll.std()),
    }


def train_and_select(X, y, groups):
    """Returns (best_fitted_model, best_name, results_df)."""
    models = build_models()
    results = pd.DataFrame(
        [evaluate_model(n, m, X, y, groups) for n, m in models.items()]
    )
    best_name = results.loc[results["Log loss"].idxmin(), "Model"]
    best = models[best_name].fit(X, y)
    return best, best_name, results


def save_model(model, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_match(model, raw_df: pd.DataFrame, player_a: str, player_b: str, surface: str) -> dict:
    """
    Win probabilities for player_a vs player_b, using stats from all
    matches in raw_df. Unknown players fall back to a neutral 0.5 prior.
    The prediction is symmetrised (averaged over both orderings) so that
    swapping A and B always gives mirrored probabilities.
    """
    state = build_state(raw_df)

    def p_a_wins(a, b):
        X = pd.DataFrame([state.features(a, b, surface)])[FEATURE_COLUMNS]
        return float(model.predict_proba(X)[0][1])

    a_prob = (p_a_wins(player_a, player_b) + (1 - p_a_wins(player_b, player_a))) / 2
    return {
        "player_a_prob": round(a_prob, 3),
        "player_b_prob": round(1 - a_prob, 3),
        "predicted_winner": player_a if a_prob >= 0.5 else player_b,
    }


if __name__ == "__main__":
    X, y, groups = load_data()
    print(f"Dataset: {len(X)} rows from {groups.nunique()} matches | {X.shape[1]} features\n")
    best, best_name, results = train_and_select(X, y, groups)
    print(results.round(3).to_string(index=False))
    print(f"\nBest model: {best_name}")
    save_model(best)
    print(f"Saved -> {MODEL_PATH}")
