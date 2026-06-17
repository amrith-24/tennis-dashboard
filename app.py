import streamlit as st
import pandas as pd


df = pd.read_csv("matches.csv")

st.title("🎾 Tennis Dashboard")

players = sorted(
    set(df["winner_name"]).union(df["loser_name"])
)

player = st.selectbox(
    "Select a player",
    players
)

wins = len(df[df["winner_name"] == player])
losses = len(df[df["loser_name"] == player])

st.metric("Wins", wins)
st.metric("Losses", losses)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.bar(
    ["Wins", "Losses"],
    [wins, losses]
)

ax.set_title(player)

st.pyplot(fig)

st.dataframe(df)