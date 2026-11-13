# 🎾 Tennis Analytics Dashboard

An interactive tennis analytics app built with **Python, pandas, Plotly and Streamlit**,
with a small machine-learning model that estimates match win probabilities.

## Features

| Page | What it does |
|------|--------------|
| **Player Stats** | Wins, losses, win rate, record by surface, opponents beaten / lost to |
| **Head-to-Head** | Record between two players, with a per-surface breakdown |
| **Predict a Match** | Win probability for any two players on a chosen surface, with an explanation of the inputs |
| **Model Performance** | Cross-validated accuracy and log loss for each candidate model |
| **Data Explorer** | Filter matches by player / surface, download the result as CSV |

## Run locally

```bash
git clone https://github.com/amrith-24/tennis-dashboard.git
cd tennis-dashboard
pip install -r requirements.txt
streamlit run app.py
```

Run the tests with `pip install -r requirements-dev.txt && pytest`.

## How the prediction model works

- **Features** (`src/features.py`): each player's overall and per-surface win rate, and the
  head-to-head rate between the two players. Rates are smoothed toward 50% so a player with
  one match isn't treated as unbeatable.
- **No look-ahead leakage**: a match's features are computed only from matches played *before* it.
- **Training** (`src/model.py`): Logistic Regression and Random Forest are compared with
  cross-validation grouped by match; the model with the lowest log loss is used.
- **Symmetry**: predictions are averaged over both player orderings, so swapping A and B
  gives exactly mirrored probabilities.

> Tennis is hard to predict, so expect modest accuracy. See the **Model Performance** page for the cross-validated numbers on the loaded data.

## Using your own / ATP data

Replace `matches.csv` with a CSV that has these columns:

```
winner_name, loser_name, surface     # required
tourney_date                         # optional, YYYYMMDD; sorts matches chronologically
```

[Jeff Sackmann's `tennis_atp` files](https://github.com/JeffSackmann/tennis_atp) use these
column names, so a season file can be used directly (his data is licensed CC BY-NC-SA 4.0:
attribution required, non-commercial use only). Without a date column, the row order of the
file is assumed to be chronological.

## Project structure

```
app.py            Streamlit UI
matches.csv       Match data
src/data.py       CSV loading and validation
src/features.py   Leak-free feature engineering
src/model.py      Training, evaluation, prediction
tests/            pytest suite (features, model, and a UI smoke test)
.github/          CI workflow
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io>, sign in with GitHub, and click **Create app**.
3. Pick `amrith-24/tennis-dashboard`, branch `main`, main file `app.py`, then **Deploy**.

## Roadmap

- Load full ATP seasons and add Elo ratings
- Recent-form and ranking features
- More models (e.g. gradient boosting) once there is enough data

Match data: "ATP Tennis 2000 - 2026 Daily update" by dissfya on Kaggle — https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull?resource=download
