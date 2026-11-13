"""
app.py — Tennis Analytics Dashboard
-----------------------------------
Run with: streamlit run app.py
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data import load_matches
from features import build_state
from model import load_data, predict_match, train_and_select

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matches.csv")
GREEN, RED = "#2ecc71", "#e74c3c"

st.set_page_config(page_title="Tennis Analytics", page_icon="🎾", layout="wide")


# Streamlit >= 1.50 uses width="stretch"; older versions use use_container_width=True.
def _version_tuple(v):
    parts = []
    for p in v.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


STRETCH = (
    {"width": "stretch"}
    if _version_tuple(st.__version__) >= (1, 50)
    else {"use_container_width": True}
)


# ── Data & model ─────────────────────────────────────────────────────────────
@st.cache_data
def get_matches() -> pd.DataFrame:
    return load_matches(DATA_PATH)


@st.cache_resource
def get_model():
    X, y, groups = load_data(DATA_PATH)
    return train_and_select(X, y, groups)  # (model, name, results_df)


try:
    raw_df = get_matches()
except (FileNotFoundError, ValueError) as err:
    st.error(f"Could not load match data: {err}")
    st.stop()

players = sorted(set(raw_df["winner_name"]).union(raw_df["loser_name"]))
surfaces = sorted(raw_df["surface"].unique())

if len(players) < 2:
    st.error("The dataset needs at least two players.")
    st.stop()

model, model_name, model_results = get_model()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🎾 Tennis Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["Player Stats", "Head-to-Head", "Predict a Match", "Model Performance", "Data Explorer"],
)
st.sidebar.caption(f"{len(raw_df)} matches · {len(players)} players")


# ═════════════════════════════════════════════════════════════════════════════
# Player Stats
# ═════════════════════════════════════════════════════════════════════════════
if page == "Player Stats":
    st.title("Player Statistics")
    player = st.selectbox("Select a player", players)

    won = raw_df[raw_df["winner_name"] == player]
    lost = raw_df[raw_df["loser_name"] == player]
    total = len(won) + len(lost)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wins", len(won))
    c2.metric("Losses", len(lost))
    c3.metric("Win Rate", f"{len(won) / total:.0%}" if total else "—")
    c4.metric("Matches", total)
    st.divider()

    st.subheader("Record by Surface")
    surf_df = pd.DataFrame(
        {
            "Surface": surfaces,
            "Wins": [(won["surface"] == s).sum() for s in surfaces],
            "Losses": [(lost["surface"] == s).sum() for s in surfaces],
        }
    )
    fig = px.bar(
        surf_df, x="Surface", y=["Wins", "Losses"], barmode="group",
        color_discrete_map={"Wins": GREEN, "Losses": RED},
        title=f"{player} — Wins & Losses by Surface",
    )
    fig.update_layout(yaxis_title="Matches", legend_title_text="")
    st.plotly_chart(fig, **STRETCH)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Beaten")
        beaten = won["loser_name"].value_counts().reset_index()
        beaten.columns = ["Opponent", "Times"]
        st.dataframe(beaten, hide_index=True, **STRETCH)
    with col_b:
        st.subheader("Lost To")
        lost_to = lost["winner_name"].value_counts().reset_index()
        lost_to.columns = ["Opponent", "Times"]
        st.dataframe(lost_to, hide_index=True, **STRETCH)


# ═════════════════════════════════════════════════════════════════════════════
# Head-to-Head
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Head-to-Head":
    st.title("Head-to-Head")
    col1, col2 = st.columns(2)
    p1 = col1.selectbox("Player A", players, index=0)
    p2 = col2.selectbox("Player B", players, index=1)

    if p1 == p2:
        st.warning("Select two different players.")
        st.stop()

    h2h = raw_df[
        ((raw_df["winner_name"] == p1) & (raw_df["loser_name"] == p2))
        | ((raw_df["winner_name"] == p2) & (raw_df["loser_name"] == p1))
    ]
    p1_wins = int((h2h["winner_name"] == p1).sum())
    p2_wins = int((h2h["winner_name"] == p2).sum())

    st.subheader(f"{p1} vs {p2}")
    if p1_wins + p2_wins == 0:
        st.info("These two players have not met in this dataset.")
        st.stop()

    c1, c2, c3 = st.columns([2, 1, 2])
    c1.metric(p1, f"{p1_wins} wins")
    c2.markdown("<div style='text-align:center;padding-top:28px'>vs</div>", unsafe_allow_html=True)
    c3.metric(p2, f"{p2_wins} wins")

    fig = go.Figure(
        go.Bar(x=[p1_wins, p2_wins], y=[p1, p2], orientation="h",
               marker_color=[GREEN, RED], text=[p1_wins, p2_wins], textposition="inside")
    )
    fig.update_layout(title="Head-to-Head Record", xaxis_title="Wins", showlegend=False, height=220)
    st.plotly_chart(fig, **STRETCH)

    st.subheader("By Surface")
    rows = [
        {"Surface": s,
         f"{p1} Wins": int(((h2h["surface"] == s) & (h2h["winner_name"] == p1)).sum()),
         f"{p2} Wins": int(((h2h["surface"] == s) & (h2h["winner_name"] == p2)).sum())}
        for s in surfaces
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)


# ═════════════════════════════════════════════════════════════════════════════
# Predict a Match
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Predict a Match":
    st.title("🔮 Match Predictor")
    st.write("Pick two players and a surface to get a model-estimated win probability.")

    col1, col2, col3 = st.columns(3)
    player_a = col1.selectbox("Player A", players, index=0)
    player_b = col2.selectbox("Player B", players, index=1)
    surface = col3.selectbox("Surface", surfaces)

    if player_a == player_b:
        st.warning("Select two different players.")
        st.stop()

    if st.button("Predict", type="primary"):
        result = predict_match(model, raw_df, player_a, player_b, surface)
        a_prob, b_prob, winner = result["player_a_prob"], result["player_b_prob"], result["predicted_winner"]

        st.divider()
        st.subheader(f"Predicted winner: **{winner}** 🏆")

        fig = go.Figure(
            go.Bar(
                x=[a_prob, b_prob], y=[player_a, player_b], orientation="h",
                marker_color=[GREEN if player_a == winner else RED,
                              GREEN if player_b == winner else RED],
                text=[f"{a_prob:.0%}", f"{b_prob:.0%}"], textposition="inside",
            )
        )
        fig.update_layout(
            title=f"Win Probability — {surface}",
            xaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False, height=220,
        )
        st.plotly_chart(fig, **STRETCH)

        with st.expander("📐 How was this calculated?"):
            state = build_state(raw_df)
            feat_df = pd.DataFrame(
                {
                    "Feature": ["Overall win rate", f"Win rate ({surface})", "Head-to-head rate"],
                    player_a: [f"{state.win_rate(player_a):.0%}",
                               f"{state.surface_rate(player_a, surface):.0%}",
                               f"{state.h2h_rate(player_a, player_b):.0%}"],
                    player_b: [f"{state.win_rate(player_b):.0%}",
                               f"{state.surface_rate(player_b, surface):.0%}",
                               f"{state.h2h_rate(player_b, player_a):.0%}"],
                }
            )
            st.dataframe(feat_df, hide_index=True, **STRETCH)
            st.caption(
                f"Model: {model_name}. Rates are smoothed toward 50% so players with few "
                "matches aren't over-rated. See **Model Performance** for how reliable this is."
            )


# ═════════════════════════════════════════════════════════════════════════════
# Model Performance
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance")
    st.write(
        "Models are compared with grouped cross-validation (both perspectives of a match "
        "stay in the same fold) on features built only from *earlier* matches, so there is "
        "no look-ahead leakage."
    )
    st.metric("Selected model", model_name)

    shown = model_results.copy()
    shown["Accuracy"] = shown["Accuracy"].map("{:.1%}".format)
    shown["Accuracy ±"] = shown["Accuracy ±"].map("{:.1%}".format)
    shown["Log loss"] = shown["Log loss"].map("{:.3f}".format)
    shown["Log loss ±"] = shown["Log loss ±"].map("{:.3f}".format)
    st.dataframe(shown, hide_index=True, **STRETCH)

    st.caption("A coin flip scores 50% accuracy and a log loss of 0.693. Lower log loss is better.")
    if len(raw_df) < 200:
        st.warning(
            f"Only {len(raw_df)} matches are loaded, so these estimates are very noisy. "
            "Add more match data (see the README) for meaningful results."
        )


# ═════════════════════════════════════════════════════════════════════════════
# Data Explorer
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.title("Match Data")
    col1, col2 = st.columns(2)
    filter_player = col1.multiselect("Filter by player", players)
    filter_surface = col2.multiselect("Filter by surface", surfaces)

    filtered = raw_df
    if filter_player:
        filtered = filtered[
            filtered["winner_name"].isin(filter_player) | filtered["loser_name"].isin(filter_player)
        ]
    if filter_surface:
        filtered = filtered[filtered["surface"].isin(filter_surface)]

    st.metric("Matches shown", len(filtered))
    st.dataframe(filtered, **STRETCH, hide_index=True)
    st.download_button("Download filtered data (CSV)", filtered.to_csv(index=False),
                       file_name="matches_filtered.csv", mime="text/csv")

    if len(filtered):
        st.subheader("Surface Distribution")
        fig = px.pie(filtered, names="surface", title="Matches by Surface",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, **STRETCH)
    else:
        st.info("No matches for the selected filters.")
