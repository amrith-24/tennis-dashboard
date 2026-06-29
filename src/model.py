"""
model.py — Tennis Match Outcome Prediction
-------------------------------------------
Trains and evaluates a progression of ML models on engineered features.
Saves the best model to disk for use in the Streamlit app.

Model progression:
  1. Logistic Regression  (interpretable baseline)
  2. Random Forest        (captures non-linear interactions)
  3. XGBoost              (gradient boosting — usually best on tabular data)

Evaluation:
  - Accuracy
  - Log loss  (penalises overconfident wrong predictions)
  - Cross-validated scores (robust with small datasets)

Usage:
  python src/model.py          # trains all models, saves best to models/
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Try to import XGBoost; fall back gracefully if not installed ─────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Run: pip install xgboost")

from features import build_features, get_feature_columns


# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH   = "matches.csv"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.pkl")
FEATURE_COL = get_feature_columns()
LABEL_COL   = "label"
RANDOM_SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV, engineer features, and return X, y."""
    raw = pd.read_csv(path)
    df  = build_features(raw)
    X   = df[FEATURE_COL]
    y   = df[LABEL_COL]
    return X, y


def evaluate_model(name: str, model, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Cross-validate a model and return a results dict.
    Uses StratifiedKFold to preserve label balance across folds.
    Small datasets (< 50 rows) use 3 folds; larger use 5.
    """
    n_splits = 3 if len(y) < 50 else 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    ll_scores  = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")

    results = {
        "name":     name,
        "accuracy": acc_scores.mean(),
        "acc_std":  acc_scores.std(),
        "log_loss": -ll_scores.mean(),   # negate: sklearn returns negative
        "ll_std":   ll_scores.std(),
    }

    print(
        f"  {name:<25} "
        f"Acc: {results['accuracy']:.3f} ± {results['acc_std']:.3f}   "
        f"LogLoss: {results['log_loss']:.3f} ± {results['ll_std']:.3f}"
    )
    return results


def build_models() -> dict:
    """Return a dict of model_name → unfitted sklearn-compatible model."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=RANDOM_SEED
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
        )
    return models


def train_and_select(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train all models, evaluate via cross-validation, and return
    (best_fitted_model, best_model_name, all_results).
    Best is chosen by lowest log loss (most calibrated probability).
    """
    print("\n📊 Model Evaluation (cross-validated)\n" + "─" * 60)
    models  = build_models()
    results = []

    for name, model in models.items():
        r = evaluate_model(name, model, X, y)
        results.append(r)

    # Pick best by log loss
    best_result = min(results, key=lambda r: r["log_loss"])
    best_name   = best_result["name"]
    best_model  = models[best_name]
    best_model.fit(X, y)  # refit on full dataset

    print(f"\n✅ Best model: {best_name} (LogLoss: {best_result['log_loss']:.3f})")
    return best_model, best_name, results


def save_model(model, path: str = MODEL_PATH) -> None:
    """Persist the fitted model to disk with pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Model saved → {path}")


def load_model(path: str = MODEL_PATH):
    """Load a previously saved model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_match(
    model,
    raw_df: pd.DataFrame,
    player_a: str,
    player_b: str,
    surface: str,
) -> dict:
    """
    Predict the outcome of a single match.

    Parameters
    ----------
    model     : fitted sklearn model
    raw_df    : the original matches DataFrame (used to compute live stats)
    player_a  : name of the first player
    player_b  : name of the second player
    surface   : 'Hard', 'Clay', or 'Grass'

    Returns
    -------
    dict with 'player_a_prob', 'player_b_prob', 'predicted_winner'
    """
    from features import build_features, _build_player_stats, _h2h_rate

    stats = _build_player_stats(raw_df)

    def safe_stats(player):
        return stats.get(player, {"win_rate": 0.5, "surface_rates": {}})

    a = safe_stats(player_a)
    b = safe_stats(player_b)

    a_surface = a["surface_rates"].get(surface, 0.5)
    b_surface = b["surface_rates"].get(surface, 0.5)
    h2h       = _h2h_rate(raw_df, player_a, player_b)

    X_pred = pd.DataFrame([{
        "player_a_win_rate":     a["win_rate"],
        "player_b_win_rate":     b["win_rate"],
        "player_a_surface_rate": a_surface,
        "player_b_surface_rate": b_surface,
        "h2h_advantage":         h2h,
        "win_rate_diff":         a["win_rate"] - b["win_rate"],
        "surface_rate_diff":     a_surface - b_surface,
    }])

    proba = model.predict_proba(X_pred)[0]
    return {
        "player_a_prob":    round(proba[1], 3),
        "player_b_prob":    round(proba[0], 3),
        "predicted_winner": player_a if proba[1] >= 0.5 else player_b,
    }


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🎾 Tennis ML Pipeline\n")

    X, y = load_data()
    print(f"Dataset: {len(X)} rows | {X.shape[1]} features | "
          f"Label balance: {y.mean():.2f} (1=win)")

    best_model, best_name, all_results = train_and_select(X, y)
    save_model(best_model)

    # Demo prediction
    print("\n🔮 Example prediction:")
    raw = pd.read_csv(DATA_PATH)
    pred = predict_match(best_model, raw, "Carlos Alcaraz", "Novak Djokovic", "Clay")
    print(f"  Alcaraz vs Djokovic on Clay:")
    print(f"  → {pred['predicted_winner']} wins")
    print(f"  → Alcaraz: {pred['player_a_prob']:.1%} | Djokovic: {pred['player_b_prob']:.1%}")
