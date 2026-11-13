"""
data.py — Loading and validating match data
-------------------------------------------
Required columns : winner_name, loser_name, surface
Optional column  : tourney_date (YYYYMMDD, as in Jeff Sackmann's ATP files).
                   If present, matches are sorted chronologically; otherwise
                   the row order of the file is treated as chronological.
"""

import pandas as pd

REQUIRED_COLUMNS = ["winner_name", "loser_name", "surface"]


def load_matches(path: str = "matches.csv") -> pd.DataFrame:
    """Read, validate and clean a matches CSV."""
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    df = df[df["winner_name"] != df["loser_name"]]

    if "tourney_date" in df.columns:
        df = df.sort_values("tourney_date", kind="stable")

    return df.reset_index(drop=True)
