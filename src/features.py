"""
features.py — Tennis Match Feature Engineering
-----------------------------------------------
Builds ML features from raw match results WITHOUT data leakage:
every match's features are computed only from matches played *before* it.

All rates are smoothed towards 0.5 (a Laplace-style prior worth
`PRIOR_STRENGTH` matches), so a player with 1 win from 1 match is not
treated as a 100% winner, and unseen players default to 0.5.

Features (A = first player, B = second player):
  player_a_win_rate / player_b_win_rate
  player_a_surface_rate / player_b_surface_rate
  h2h_advantage       : A's smoothed head-to-head rate vs B
  win_rate_diff       : A overall - B overall
  surface_rate_diff   : A surface - B surface
"""

from collections import defaultdict

import pandas as pd

PRIOR_STRENGTH = 2.0

FEATURE_COLUMNS = [
    "player_a_win_rate",
    "player_b_win_rate",
    "player_a_surface_rate",
    "player_b_surface_rate",
    "h2h_advantage",
    "win_rate_diff",
    "surface_rate_diff",
]


def get_feature_columns() -> list:
    return list(FEATURE_COLUMNS)


def _smooth(wins: float, total: float) -> float:
    return (wins + 0.5 * PRIOR_STRENGTH) / (total + PRIOR_STRENGTH)


class PlayerState:
    """Running win/loss counters; the single source of truth for features."""

    def __init__(self):
        self.wins = defaultdict(int)
        self.games = defaultdict(int)
        self.surf_wins = defaultdict(int)
        self.surf_games = defaultdict(int)
        self.h2h_wins = defaultdict(int)  # (winner, loser) -> count

    def win_rate(self, player):
        return _smooth(self.wins[player], self.games[player])

    def surface_rate(self, player, surface):
        key = (player, surface)
        return _smooth(self.surf_wins[key], self.surf_games[key])

    def h2h_rate(self, a, b):
        a_wins, b_wins = self.h2h_wins[(a, b)], self.h2h_wins[(b, a)]
        return _smooth(a_wins, a_wins + b_wins)

    def features(self, a, b, surface) -> dict:
        a_wr, b_wr = self.win_rate(a), self.win_rate(b)
        a_sr, b_sr = self.surface_rate(a, surface), self.surface_rate(b, surface)
        return {
            "player_a_win_rate": a_wr,
            "player_b_win_rate": b_wr,
            "player_a_surface_rate": a_sr,
            "player_b_surface_rate": b_sr,
            "h2h_advantage": self.h2h_rate(a, b),
            "win_rate_diff": a_wr - b_wr,
            "surface_rate_diff": a_sr - b_sr,
        }

    def update(self, winner, loser, surface):
        for p in (winner, loser):
            self.games[p] += 1
            self.surf_games[(p, surface)] += 1
        self.wins[winner] += 1
        self.surf_wins[(winner, surface)] += 1
        self.h2h_wins[(winner, loser)] += 1


def build_state(df: pd.DataFrame) -> PlayerState:
    """State after replaying ALL matches (used for predicting future matches)."""
    state = PlayerState()
    for w, l, s in zip(df["winner_name"], df["loser_name"], df["surface"]):
        state.update(w, l, s)
    return state


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert chronologically ordered matches into a feature matrix.

    Each match yields two rows (A=winner label 1, A=loser label 0) so the
    model sees both perspectives. Both rows share a `match_id`, which must
    be used to group cross-validation folds.
    """
    state = PlayerState()
    rows = []
    for match_id, (w, l, s) in enumerate(
        zip(df["winner_name"], df["loser_name"], df["surface"])
    ):
        for a, b, label in ((w, l, 1), (l, w, 0)):
            row = state.features(a, b, s)
            row.update(
                {"match_id": match_id, "player_a": a, "player_b": b,
                 "surface": s, "label": label}
            )
            rows.append(row)
        state.update(w, l, s)  # update AFTER computing features: no leakage
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data import load_matches

    feats = build_features(load_matches("matches.csv"))
    print(feats[FEATURE_COLUMNS + ["label"]].head(10))
    print(f"\nDataset shape: {feats.shape}")
