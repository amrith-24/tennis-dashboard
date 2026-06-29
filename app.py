"""
app.py — Tennis Analytics Dashboard (upgraded)
-----------------------------------------------
Run with:  streamlit run app.py
"""

import os
import sys
import pickle
import warnings

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── Path setup so src/ modules are importable ────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from features import build_features, get_feature_columns
from model import predict_match, load_model, train_and_select, save_model, load_data

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tennis Analytics",
    page_icon="🎾",
    layout="wide",
)

st.markdown("""
<style>
    .big-font { font-size: 1.1rem; font-weight: 600; }
    .prob-bar-a { background: #2ecc71; border-radius: 4px; height: 20px; }
    .prob-bar-b { background: #e74c3c; border-radius: 4px; height: 20px; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv("matches.csv")

raw_df = load_raw()

players = sorted(set(raw_df["winner_name"]).union(raw_df["loser_name"]))
surfaces = sorted(raw_df["surface"].unique())

# ── Load or train model ───────────────────────────────────────────────────────
MODEL_PATH = "models/best_model.pkl"

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    X, y = load_data()
    model, _, _ = train_and_select(X, y)
    save_model(model, MODEL_PATH)
    return model

model = get_model()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🎾 Tennis Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["Player Stats", "Head-to-Head", "Predict a Match", "Data Explorer"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Player Stats
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Player Stats":
    st.title("Player Statistics")

    player = st.selectbox("Select a player", players)

    won  = raw_df[raw_df["winner_name"] == player]
    lost = raw_df[raw_df["loser_name"]  == player]
    total = len(won) + len(lost)

    # ── Top metrics ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wins",        len(won))
    col2.metric("Losses",      len(lost))
    col3.metric("Win Rate",    f"{len(won)/total:.0%}" if total else "—")
    col4.metric("Matches",     total)

    st.divider()

    # ── Win rate by surface ──────────────────────────────────────────────────
    st.subheader("Win Rate by Surface")
    surface_data = []
    for surface in surfaces:
        sw = len(won[won["surface"] == surface])
        sl = len(lost[lost["surface"] == surface])
        st_total = sw + sl
        surface_data.append({
            "Surface": surface,
            "Wins": sw,
            "Losses": sl,
            "Win Rate": sw / st_total if st_total else 0,
        })

    surf_df = pd.DataFrame(surface_data)

    fig_surf = px.bar(
        surf_df,
        x="Surface",
        y=["Wins", "Losses"],
        barmode="group",
        color_discrete_map={"Wins": "#2ecc71", "Losses": "#e74c3c"},
        title=f"{player} — Wins & Losses by Surface",
    )
    st.plotly_chart(fig_surf, use_container_width=True)

    # ── Opponents beaten / lost to ────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Beaten")
        beaten = won["loser_name"].value_counts().reset_index()
        beaten.columns = ["Opponent", "Times"]
        st.dataframe(beaten, hide_index=True, use_container_width=True)
    with col_b:
        st.subheader("Lost To")
        lost_to = lost["winner_name"].value_counts().reset_index()
        lost_to.columns = ["Opponent", "Times"]
        st.dataframe(lost_to, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Head-to-Head
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Head-to-Head":
    st.title("Head-to-Head")

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Player A", players, index=0)
    with col2:
        p2 = st.selectbox("Player B", players, index=1)

    if p1 == p2:
        st.warning("Select two different players.")
        st.stop()

    h2h = raw_df[
        ((raw_df["winner_name"] == p1) & (raw_df["loser_name"] == p2)) |
        ((raw_df["winner_name"] == p2) & (raw_df["loser_name"] == p1))
    ].copy()

    p1_wins = len(h2h[h2h["winner_name"] == p1])
    p2_wins = len(h2h[h2h["winner_name"] == p2])
    total   = p1_wins + p2_wins

    # ── Summary bar ──────────────────────────────────────────────────────────
    st.subheader(f"{p1}  vs  {p2}")
    col_a, col_b, col_c = st.columns([2, 1, 2])
    col_a.metric(p1, f"{p1_wins} wins")
    col_b.markdown("<div style='text-align:center;padding-top:28px'>vs</div>", unsafe_allow_html=True)
    col_c.metric(p2, f"{p2_wins} wins")

    if total:
        fig_h2h = go.Figure(go.Bar(
            x=[p1_wins, p2_wins],
            y=[p1, p2],
            orientation="h",
            marker_color=["#2ecc71", "#e74c3c"],
            text=[f"{p1_wins}", f"{p2_wins}"],
            textposition="inside",
        ))
        fig_h2h.update_layout(
            title="Head-to-Head Record",
            xaxis_title="Wins",
            showlegend=False,
            height=200,
        )
        st.plotly_chart(fig_h2h, use_container_width=True)

        # ── Per-surface breakdown ──────────────────────────────────────────────
        st.subheader("By Surface")
        surf_rows = []
        for surf in surfaces:
            s_matches = h2h[h2h["surface"] == surf]
            surf_rows.append({
                "Surface":      surf,
                f"{p1} Wins":   len(s_matches[s_matches["winner_name"] == p1]),
                f"{p2} Wins":   len(s_matches[s_matches["winner_name"] == p2]),
            })
        st.dataframe(pd.DataFrame(surf_rows), hide_index=True, use_container_width=True)
    else:
        st.info("These two players have not met in this dataset.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Predict a Match
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Predict a Match":
    st.title("🔮 Match Predictor")
    st.write("Pick two players and a surface to get an ML-powered win probability.")

    col1, col2, col3 = st.columns(3)
    with col1:
        player_a = st.selectbox("Player A", players, index=0)
    with col2:
        player_b = st.selectbox("Player B", players, index=1)
    with col3:
        surface = st.selectbox("Surface", surfaces)

    if player_a == player_b:
        st.warning("Select two different players.")
        st.stop()

    if st.button("Predict", type="primary"):
        result = predict_match(model, raw_df, player_a, player_b, surface)
        a_prob = result["player_a_prob"]
        b_prob = result["player_b_prob"]
        winner = result["predicted_winner"]

        st.divider()
        st.subheader(f"Predicted winner: **{winner}** 🏆")

        # ── Probability gauge ─────────────────────────────────────────────────
        fig_gauge = go.Figure(go.Bar(
            x=[a_prob, b_prob],
            y=[player_a, player_b],
            orientation="h",
            marker_color=["#2ecc71" if player_a == winner else "#e74c3c",
                          "#2ecc71" if player_b == winner else "#e74c3c"],
            text=[f"{a_prob:.0%}", f"{b_prob:.0%}"],
            textposition="inside",
        ))
        fig_gauge.update_layout(
            title=f"Win Probability — {surface} court",
            xaxis=dict(range=[0, 1], tickformat=".0%"),
            showlegend=False,
            height=200,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Feature breakdown ──────────────────────────────────────────────────
        with st.expander("📐 How was this calculated?"):
            from features import _build_player_stats, _h2h_rate
            stats = _build_player_stats(raw_df)
            a_st = stats.get(player_a, {"win_rate": 0.5, "surface_rates": {}})
            b_st = stats.get(player_b, {"win_rate": 0.5, "surface_rates": {}})
            h2h  = _h2h_rate(raw_df, player_a, player_b)

            feat_df = pd.DataFrame({
                "Feature":  ["Overall win rate", f"Win rate ({surface})", "H2H advantage"],
                player_a:   [
                    f"{a_st['win_rate']:.0%}",
                    f"{a_st['surface_rates'].get(surface, 0.5):.0%}",
                    f"{h2h:.0%}",
                ],
                player_b:   [
                    f"{b_st['win_rate']:.0%}",
                    f"{b_st['surface_rates'].get(surface, 0.5):.0%}",
                    f"{1-h2h:.0%}",
                ],
            })
            st.dataframe(feat_df, hide_index=True, use_container_width=True)
            st.caption("The model combines these (and more) features to estimate win probability.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.title("Match Data")

    col1, col2 = st.columns(2)
    with col1:
        filter_player = st.multiselect(
            "Filter by player",
            players,
            default=[],
        )
    with col2:
        filter_surface = st.multiselect(
            "Filter by surface",
            surfaces,
            default=[],
        )

    filtered = raw_df.copy()
    if filter_player:
        filtered = filtered[
            filtered["winner_name"].isin(filter_player) |
            filtered["loser_name"].isin(filter_player)
        ]
    if filter_surface:
        filtered = filtered[filtered["surface"].isin(filter_surface)]

    st.metric("Matches shown", len(filtered))
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # ── Surface distribution pie ──────────────────────────────────────────────
    st.subheader("Surface Distribution")
    fig_pie = px.pie(
        filtered,
        names="surface",
        title="Matches by Surface",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_pie, use_container_width=True)