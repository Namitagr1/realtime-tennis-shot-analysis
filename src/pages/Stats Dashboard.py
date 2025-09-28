# src/stats_dashboard.py
import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from navbar import navbar
navbar()


# --------------------------
# Plotly config (user requested)
# --------------------------
plotly_config = {"width": "stretch"}  # user-specified (nonstandard key accepted per request)

st.set_page_config(layout="wide", page_title="Tennis Performance Trends Dashboard")
st.title("🎾 Tennis Performance Trends Dashboard")

# --------------------------
# Paths
# --------------------------
BASE_DIR = os.path.dirname(__file__)
ATP_DIR = os.path.join(BASE_DIR, "../../data/atp")

# --------------------------
# Load all ATP match CSVs (cached)
# --------------------------
@st.cache_data(show_spinner=True)
def load_all_matches(atp_dir):
    files = sorted(glob.glob(os.path.join(atp_dir, "atp_matches_*.csv")))
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f, low_memory=False)
            # derive year from tourney_date if present (YYYYMMDD -> YYYY)
            if "tourney_date" in d.columns:
                td = pd.to_numeric(d["tourney_date"], errors="coerce").fillna(0).astype(int)
                d["year"] = (td // 10000).astype(int)
            elif "year" in d.columns:
                d["year"] = pd.to_numeric(d["year"], errors="coerce").fillna(0).astype(int)
            else:
                # fallback: infer from filename if possible
                try:
                    yr = int(os.path.basename(f).split("_")[-1].split(".")[0])
                except Exception:
                    yr = 0
                d["year"] = yr
            dfs.append(d)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    return out

all_data = load_all_matches(ATP_DIR)
if all_data.empty:
    st.error("No ATP CSVs found in data/atp/. Place files like atp_matches_2010.csv there.")
    st.stop()

# --------------------------
# Basic cleaning & derived cols
# --------------------------
def to_numeric_safe(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

# Ensure columns exist; create defaults if missing
expected_numeric = [
    "w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon",
    "l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon"
]
for col in expected_numeric:
    if col not in all_data.columns:
        all_data[col] = 0
    all_data[col] = to_numeric_safe(all_data[col])

# Canonical string cols
for c in ["winner_name","loser_name","surface","tourney_name","tourney_date"]:
    if c not in all_data.columns:
        all_data[c] = ""

# year guard
if "year" not in all_data.columns:
    all_data["year"] = 0
all_data["year"] = to_numeric_safe(all_data["year"]).astype(int)

# Derived percentages & proxies, NaN-proofed
def safe_div(n, d):
    d = d.replace(0, np.nan)
    return (n / d).fillna(0)

all_data["w_1st_pct"] = safe_div(all_data["w_1stIn"], all_data["w_svpt"]) * 100.0
all_data["l_1st_pct"] = safe_div(all_data["l_1stIn"], all_data["l_svpt"]) * 100.0

# return points proxies
all_data["w_return_pts_won"] = (all_data["l_svpt"] - (all_data["l_1stWon"] + all_data["l_2ndWon"])).clip(lower=0)
all_data["l_return_pts_won"] = (all_data["w_svpt"] - (all_data["w_1stWon"] + all_data["w_2ndWon"])).clip(lower=0)

# total points (proxy for rally length)
all_data["total_points"] = (all_data["w_svpt"] + all_data["l_svpt"]).fillna(0)

# --------------------------
# Sidebar filters & mode
# --------------------------
st.sidebar.header("Filters & Mode")

# surfaces available
surfaces = sorted([s for s in all_data["surface"].unique() if isinstance(s, str) and s.strip() != ""])
sel_surfaces = st.sidebar.multiselect("Surfaces (filter)", surfaces, default=surfaces if surfaces else None)

# year range slider
min_year = int(all_data["year"].replace(0, np.nan).min() or all_data["year"].min())
max_year = int(all_data["year"].max())
yr_start, yr_end = st.sidebar.slider("Year range", min_year, max_year, (max_year-5 if max_year-5>=min_year else min_year, max_year))

mode = st.sidebar.radio("Mode", ["Overview", "Single Player", "Comparison"], index=0)

# Candidate players (top N winners in filtered data)
def top_players_from(df, n=25):
    vc = df["winner_name"].value_counts()
    players = [p for p in vc.head(n).index if isinstance(p, str) and p.strip()]
    return players

# base filtered dataset for macro charts (applies to Overview charts)
base_df = all_data.copy()
if sel_surfaces:
    base_df = base_df[base_df["surface"].isin(sel_surfaces)]
base_df = base_df[(base_df["year"] >= yr_start) & (base_df["year"] <= yr_end)]

if base_df.empty:
    st.warning("No matches in selected surface/year range. Adjust filters.")
    st.stop()

candidates = top_players_from(base_df, n=50) or top_players_from(all_data, n=50)

# --------------------------
# Overview / Macro Trends
# --------------------------
if mode == "Overview":
    st.subheader("Overview — Trends across selected years/surfaces")

    # 1) Aces per match (year)
    aces_year = (base_df.groupby("year")["w_ace"].sum() + base_df.groupby("year")["l_ace"].sum()).fillna(0)
    matches_year = base_df.groupby("year").size().astype(float)
    aces_pm = (aces_year / matches_year.replace(0, np.nan)).fillna(0).reset_index()
    aces_pm.columns = ["year","aces_per_match"]
    fig = px.line(aces_pm, x="year", y="aces_per_match", markers=True, title="Aces per Match (year)")
    st.plotly_chart(fig, config=plotly_config)

    # 2) Serve reliability (first serve % and DF rate) per year
    base_df["first_pct_avg"] = ((base_df["w_1stIn"]/base_df["w_svpt"].replace(0,np.nan)).fillna(0) +
                                (base_df["l_1stIn"]/base_df["l_svpt"].replace(0,np.nan)).fillna(0)) / 2.0
    base_df["df_rate_avg"] = ((base_df["w_df"]/base_df["w_svpt"].replace(0,np.nan)).fillna(0) +
                              (base_df["l_df"]/base_df["l_svpt"].replace(0,np.nan)).fillna(0)) / 2.0
    rel = base_df.groupby("year").agg({"first_pct_avg":"mean","df_rate_avg":"mean"}).reset_index()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=rel["year"], y=rel["first_pct_avg"]*100, mode="lines+markers", name="First Serve %"))
    fig2.add_trace(go.Scatter(x=rel["year"], y=rel["df_rate_avg"]*100, mode="lines+markers", name="Double Fault Rate %"))
    fig2.update_layout(title="Serve Reliability Over Time", xaxis_title="Year", yaxis_title="Percentage")
    st.plotly_chart(fig2, config=plotly_config)

    # 3) Return points won distribution
    ret_list = []
    for _, r in base_df.iterrows():
        try:
            ret_list.append((r["w_return_pts_won"] / max(1.0, r["l_svpt"])) * 100.0)
            ret_list.append((r["l_return_pts_won"] / max(1.0, r["w_svpt"])) * 100.0)
        except Exception:
            continue
    if ret_list:
        ret_df = pd.DataFrame({"return_pts_pct": ret_list})
        fig3 = px.histogram(ret_df, x="return_pts_pct", nbins=40, title="Return Points Won % Distribution")
        st.plotly_chart(fig3, config=plotly_config)
    else:
        st.info("Not enough return data to show distribution.")

    # 4) Rally proxy by surface
    rally = base_df.groupby("surface")["total_points"].mean().reset_index().sort_values("total_points", ascending=False)
    if not rally.empty:
        fig4 = px.bar(rally, x="surface", y="total_points", title="Avg Points per Match (rally proxy) by Surface")
        st.plotly_chart(fig4, config=plotly_config)

    # 5) Win-condition scatter
    winners = base_df.groupby("winner_name").agg({
        "w_1stIn":"sum","w_svpt":"sum","w_ace":"mean","w_df":"mean"
    }).reset_index()
    winners["first_pct"] = safe_div(winners["w_1stIn"], winners["w_svpt"]) * 100.0
    for col in ["first_pct","w_ace","w_df"]:
        winners[col] = winners[col].fillna(0)
    fig5 = px.scatter(winners, x="first_pct", y="w_ace", size="w_df", hover_name="winner_name",
                      title="Winning Serve Profiles (First% vs Aces; bubble = DF)")
    st.plotly_chart(fig5, config=plotly_config)

# --------------------------
# Single Player Mode
# --------------------------
elif mode == "Single Player":
    st.header("Single Player Trends")
    player = st.sidebar.selectbox("Select player", options=candidates, index=0 if candidates else None)
    if not player:
        st.info("No player candidates available for selection with current filters.")
        st.stop()

    # player matches across all years (use all_data so trends cover all years then apply surface/year filters for display)
    player_all = all_data[(all_data["winner_name"] == player) | (all_data["loser_name"] == player)].copy()
    if player_all.empty:
        st.warning(f"No matches for {player} in the full dataset.")
        st.stop()
    # apply surface/year filters chosen earlier for display charts
    if sel_surfaces:
        player_df = player_all[player_all["surface"].isin(sel_surfaces)]
    else:
        player_df = player_all
    player_df = player_df[(player_df["year"] >= yr_start) & (player_df["year"] <= yr_end)]
    if player_df.empty:
        st.warning("No matches for this player with current surface/year filters.")
    else:
        st.subheader(f"Trends for {player}")

        # Wins per year
        wins_year = player_df[player_df["winner_name"] == player].groupby("year").size().reset_index(name="wins")
        if not wins_year.empty:
            fig_w = px.line(wins_year, x="year", y="wins", markers=True, title=f"Wins per Year — {player}")
            st.plotly_chart(fig_w, config=plotly_config)
        else:
            st.info("No wins recorded for this player in the selected filters.")

        # Aces & DF per year (aggregated on matches they played; show per-match averages)
        # compute for matches where player was winner: use w_* columns; for matches where they were loser, use l_* columns but keep label
        def player_year_stats(df_in, player_name):
            # separate winner rows and loser rows
            w = df_in[df_in["winner_name"] == player_name].groupby("year").agg({
                "w_ace":"mean","w_df":"mean","w_1stIn":"sum","w_svpt":"sum"
            }).rename(columns={"w_ace":"aces_won_mean","w_df":"df_won_mean","w_1stIn":"w_1stIn_sum","w_svpt":"w_svpt_sum"})
            l = df_in[df_in["loser_name"] == player_name].groupby("year").agg({
                "l_ace":"mean","l_df":"mean","l_1stIn":"sum","l_svpt":"sum"
            }).rename(columns={"l_ace":"aces_lost_mean","l_df":"df_lost_mean","l_1stIn":"l_1stIn_sum","l_svpt":"l_svpt_sum"})
            merged = w.join(l, how="outer").fillna(0)
            # compute combined per-match averages approx (only using wins side for aces/df is simpler and matches earlier approach)
            merged = merged.reset_index()
            return merged

        p_stats = player_year_stats(player_df, player)
        if not p_stats.empty:
            if "aces_won_mean" in p_stats.columns:
                fig_aces = px.line(p_stats, x="year", y="aces_won_mean", markers=True, title=f"Aces per Match (wins) — {player}")
                st.plotly_chart(fig_aces, config=plotly_config)
            if "df_won_mean" in p_stats.columns:
                fig_dfs = px.line(p_stats, x="year", y="df_won_mean", markers=True, title=f"Double Faults per Match (wins) — {player}")
                st.plotly_chart(fig_dfs, config=plotly_config)

        # First serve % trend (on wins)
        wins_only = player_df[player_df["winner_name"] == player].copy()
        if not wins_only.empty:
            wins_only["first_pct"] = safe_div(wins_only["w_1stIn"], wins_only["w_svpt"]) * 100.0
            sp = wins_only.groupby("year")["first_pct"].mean().reset_index()
            if not sp.empty:
                fig_sp = px.line(sp, x="year", y="first_pct", markers=True, title=f"First-Serve % (wins) — {player}")
                st.plotly_chart(fig_sp, config=plotly_config)

        # Win % by surface (use all matches involving player in selected range)
        surf_rows = []
        for s in sorted(player_df["surface"].dropna().unique()):
            sub = player_df[player_df["surface"] == s]
            total = len(sub)
            wins = (sub["winner_name"] == player).sum()
            if total > 0:
                surf_rows.append({"surface": s, "win_pct": 100.0 * wins / total, "matches": total})
        surf_df = pd.DataFrame(surf_rows)
        if not surf_df.empty:
            fig_surf = px.bar(surf_df, x="surface", y="win_pct", text="matches", title=f"Win % by Surface — {player}")
            st.plotly_chart(fig_surf, config=plotly_config)

# --------------------------
# Comparison Mode
# --------------------------
elif mode == "Comparison":
    st.header("Player Comparison")

    # two selectors side-by-side
    c1, c2 = st.columns(2)
    with c1:
        player_a = st.selectbox("Player A", options=candidates, index=0, key="player_a")
    with c2:
        # limit options for player B to people other than A
        other_options = [p for p in candidates if p != player_a]
        player_b = st.selectbox("Player B", options=other_options, index=0 if other_options else None, key="player_b")

    if not player_a or not player_b:
        st.info("Choose two different players to compare.")
        st.stop()

    # use entire all_data then apply sidebar filters
    cmp_df = all_data.copy()
    if sel_surfaces:
        cmp_df = cmp_df[cmp_df["surface"].isin(sel_surfaces)]
    cmp_df = cmp_df[(cmp_df["year"] >= yr_start) & (cmp_df["year"] <= yr_end)]

    # Head-to-head
    h2h = cmp_df[
        ((cmp_df["winner_name"] == player_a) & (cmp_df["loser_name"] == player_b)) |
        ((cmp_df["winner_name"] == player_b) & (cmp_df["loser_name"] == player_a))
    ]
    wins_a = int((h2h["winner_name"] == player_a).sum())
    wins_b = int((h2h["winner_name"] == player_b).sum())
    st.subheader("Head-to-Head")
    st.write(f"{player_a}: **{wins_a}** wins vs {player_b}")
    st.write(f"{player_b}: **{wins_b}** wins vs {player_a}")
    if h2h.empty:
        st.info("No head-to-head matches found in the selected filters.")

    # Career serve averages (on wins)
    def career_serve_averages(df_src, player_name):
        wins = df_src[df_src["winner_name"] == player_name].copy()
        if wins.empty:
            return {"avg_aces": 0.0, "avg_df": 0.0, "avg_first_pct": 0.0, "matches": 0}
        avg_aces = wins["w_ace"].mean()
        avg_df = wins["w_df"].mean()
        avg_first = safe_div(wins["w_1stIn"], wins["w_svpt"]).mean() * 100.0
        return {"avg_aces": float(avg_aces), "avg_df": float(avg_df), "avg_first_pct": float(avg_first), "matches": len(wins)}

    stats_a = career_serve_averages(cmp_df, player_a)
    stats_b = career_serve_averages(cmp_df, player_b)
    st.subheader("Career Serve Averages (on wins)")
    table = pd.DataFrame({
        "player": [player_a, player_b],
        "avg_aces_per_win": [stats_a["avg_aces"], stats_b["avg_aces"]],
        "avg_df_per_win": [stats_a["avg_df"], stats_b["avg_df"]],
        "avg_first_serve_pct": [stats_a["avg_first_pct"], stats_b["avg_first_pct"]],
        "matches_counted": [stats_a["matches"], stats_b["matches"]]
    }).set_index("player")
    st.table(table.round(2))

    # Wins per year comparison
    wins_by_year = (cmp_df[cmp_df["winner_name"].isin([player_a, player_b])]
                    .groupby(["year", "winner_name"])["winner_name"].count().reset_index(name="wins"))
    if not wins_by_year.empty:
        fig_wy = px.line(wins_by_year, x="year", y="wins", color="winner_name", markers=True,
                         title="Wins per Year — Comparison")
        st.plotly_chart(fig_wy, config=plotly_config)
    else:
        st.info("No wins data to plot for the selected players/filters.")

    # Aces & DF trends per year (wins)
    serve_trends = (cmp_df[cmp_df["winner_name"].isin([player_a, player_b])]
                    .groupby(["year","winner_name"]).agg({"w_ace":"mean","w_df":"mean"}).reset_index())
    if not serve_trends.empty:
        fig_aces = px.line(serve_trends, x="year", y="w_ace", color="winner_name", markers=True,
                           title="Average Aces per Match (wins only) — Yearly")
        st.plotly_chart(fig_aces, config=plotly_config)
        fig_dfs = px.line(serve_trends, x="year", y="w_df", color="winner_name", markers=True,
                          title="Average Double Faults per Match (wins only) — Yearly")
        st.plotly_chart(fig_dfs, config=plotly_config)
    else:
        st.info("No serve trend data for these players in the selected filters.")

    # Win % by surface for both players
    surf_rows = []
    surfaces_list = sorted([s for s in cmp_df["surface"].dropna().unique()])
    for p in [player_a, player_b]:
        p_matches = cmp_df[(cmp_df["winner_name"] == p) | (cmp_df["loser_name"] == p)]
        for s in surfaces_list:
            sub = p_matches[p_matches["surface"] == s]
            total = len(sub)
            wins = (sub["winner_name"] == p).sum()
            if total > 0:
                surf_rows.append({"player": p, "surface": s, "win_pct": 100.0 * wins / total, "matches": total})
    surf_cmp = pd.DataFrame(surf_rows)
    if not surf_cmp.empty:
        fig_surf = px.bar(surf_cmp, x="surface", y="win_pct", color="player", barmode="group",
                          title=f"Win % by Surface — {player_a} vs {player_b}")
        st.plotly_chart(fig_surf, config=plotly_config)
    else:
        st.info("No surface breakdown data available for these players in the selected filters.")
