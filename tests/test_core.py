import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data import load_matches
from src.features import FEATURE_COLUMNS, build_features
from src.model import load_data, predict_match, train_and_select


@pytest.fixture
def matches():
    return load_matches(os.path.join(ROOT, "matches.csv"))


def test_load_matches_validates_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_matches(str(bad))


def test_two_rows_per_match(matches):
    feats = build_features(matches)
    assert len(feats) == 2 * len(matches)
    assert feats["label"].mean() == 0.5


def test_no_leakage_first_match_is_neutral(matches):
    first = build_features(matches).iloc[0]
    # nothing has happened yet, so every rate must equal the 0.5 prior
    for col in ["player_a_win_rate", "player_b_win_rate", "h2h_advantage"]:
        assert first[col] == 0.5


def test_future_results_do_not_change_past_features(matches):
    full = build_features(matches)
    truncated = build_features(matches.iloc[:10])
    cols = FEATURE_COLUMNS
    pd.testing.assert_frame_equal(
        full[full["match_id"] < 10][cols].reset_index(drop=True),
        truncated[cols].reset_index(drop=True),
    )


def test_predictions_are_symmetric(matches):
    X, y, g = load_data(os.path.join(ROOT, "matches.csv"))
    model, _, _ = train_and_select(X, y, g)
    ab = predict_match(model, matches, "Carlos Alcaraz", "Novak Djokovic", "Clay")
    ba = predict_match(model, matches, "Novak Djokovic", "Carlos Alcaraz", "Clay")
    assert ab["player_a_prob"] == pytest.approx(ba["player_b_prob"], abs=1e-3)
    assert ab["player_a_prob"] + ab["player_b_prob"] == pytest.approx(1.0)


def test_unknown_player_does_not_crash(matches):
    X, y, g = load_data(os.path.join(ROOT, "matches.csv"))
    model, _, _ = train_and_select(X, y, g)
    res = predict_match(model, matches, "Nobody Known", "Carlos Alcaraz", "Hard")
    assert 0 <= res["player_a_prob"] <= 1


def test_app_pages_render_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60).run()
    assert not at.exception
    for page in ["Player Stats", "Head-to-Head", "Predict a Match", "Model Performance", "Data Explorer"]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"
    at.sidebar.radio[0].set_value("Predict a Match").run()
    at.button[0].click().run()
    assert not at.exception
