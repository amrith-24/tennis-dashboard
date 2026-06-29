"""
features.py — Tennis Match Feature Engineering
------------------------------------------------
Transforms raw match data (winner, loser, surface) into
ML-ready features for each matchup.

Features produced per row:
  - player_a_win_rate        : overall win rate for player A
  - player_b_win_rate        : overall win rate for player B
  - player_a_surface_rate    : win rate for player A on this surface
  - player_b_surface_rate    : win rate for player B on this surface
  - h2h_advantage            : player A's head-to-head win rate vs player B
  - win_rate_diff            : player_a_win_rate - player_b_win_rate
  - surface_rate_diff        : player_a_surface_rate - player_b_surface_rate
  - label                    : 1 if player_A won, 0 if player_B won
"""

import pandas as pd
import numpy as np
from itertools import combinations


def _build_player_stats(df: pd.DataFrame) -> dict:
    """
    Build per-player stats from the raw match DataFrame.
    Returns a dict keyed by player name with overall win rates
    and surface-specific win rates.
    """
    players = set(df["winner_name"]).union(df["loser_name"])
    stats = {}

    for player in players:
        won = df[df["winner_name"] == player]
        lost = df[df["loser_name"] == player]
        total = len(won) + len(lost)

        overall_wr = len(won) / total if total > 0 else 0.5

        surface_rates = {}
        for surface in df["surface"].unique():
            s_won = len(won[won["surface"] == surface])
            s_lost = len(lost[lost["surface"] == surface])
            s_total = s_won + s_lost
            surface_rates[surface] = s_won / s_total if s_total > 0 else 0.5

        stats[player] = {
            "win_rate": overall_wr,
            "surface_rates": surface_rates,
        }

    return stats


def _h2h_rate(df: pd.DataFrame, player_a: str, player_b: str) -> float:
    """
    Head-to-head win rate for player_a against player_b.
    Returns 0.5 if they have never played.
    """
    a_wins = len(
        df[(df["winner_name"] == player_a) & (df["loser_name"] == player_b)]
    )
    b_wins = len(
        df[(df["winner_name"] == player_b) & (df["loser_name"] == player_a)]
    )
    total = a_wins + b_wins
    return a_wins / total if total > 0 else 0.5


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw match rows into a feature matrix.

    Each original match generates TWO rows (once with A=winner,
    once with A=loser) so the model sees both perspectives and
    learns a symmetric representation.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: winner_name, loser_name, surface

    Returns
    -------
    pd.DataFrame with feature columns + 'label' column.
    """
    stats = _build_player_stats(df)
    rows = []

    for _, match in df.iterrows():
        winner = match["winner_name"]
        loser = match["loser_name"]
        surface = match["surface"]

        for player_a, player_b, label in [
            (winner, loser, 1),
            (loser, winner, 0),
        ]:
            a_stats = stats[player_a]
            b_stats = stats[player_b]

            a_surface_rate = a_stats["surface_rates"].get(surface, 0.5)
            b_surface_rate = b_stats["surface_rates"].get(surface, 0.5)
            h2h = _h2h_rate(df, player_a, player_b)

            rows.append(
                {
                    "player_a": player_a,
                    "player_b": player_b,
                    "surface": surface,
                    "player_a_win_rate": a_stats["win_rate"],
                    "player_b_win_rate": b_stats["win_rate"],
                    "player_a_surface_rate": a_surface_rate,
                    "player_b_surface_rate": b_surface_rate,
                    "h2h_advantage": h2h,
                    "win_rate_diff": a_stats["win_rate"] - b_stats["win_rate"],
                    "surface_rate_diff": a_surface_rate - b_surface_rate,
                    "label": label,
                }
            )

    return pd.DataFrame(rows)


def get_feature_columns() -> list:
    """Returns the list of feature column names used by the model."""
    return [
        "player_a_win_rate",
        "player_b_win_rate",
        "player_a_surface_rate",
        "player_b_surface_rate",
        "h2h_advantage",
        "win_rate_diff",
        "surface_rate_diff",
    ]


if __name__ == "__main__":
    raw = pd.read_csv("../matches.csv")
    features = build_features(raw)
    print(features[get_feature_columns() + ["label"]].head(10))
    print(f"\nDataset shape: {features.shape}")
    print(f"Label balance:\n{features['label'].value_counts()}")
