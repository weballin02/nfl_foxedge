
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FoxEdge NFL Props Lab — v3 (schema-aware from provided zip)
- Tailored to your actual columns:
  * NO: play_action, motion, personnel, rpo flag, routes_run
  * YES: air_yards, pass_location, run_gap/run_location, scramble, qb_hit, sack, complete_pass, touchdown
- Auto-gates tabs accordingly.
"""

import numpy as np
import pandas as pd
import streamlit as st

from nfl_data_py import import_pbp_data, import_weekly_data, import_schedules

st.set_page_config(page_title="NFL Props Lab v3", page_icon="🦊", layout="wide")

# --- session state init (prevents missing-key errors on cold start) ---
if "pbp" not in st.session_state:
    st.session_state["pbp"] = None
if "wk" not in st.session_state:
    st.session_state["wk"] = None


def safe_div(n,d):
    try:
        return n/d if d not in (0,0.0,None) else np.nan
    except Exception:
        return np.nan

@st.cache_data(show_spinner=False)
def load_pbp(seasons):
    pbp = import_pbp_data(seasons)

    # Normalize flags
    for b in ["pass","rush","qb_hit","sack","no_huddle","complete_pass","touchdown"]:
        if b in pbp.columns:
            pbp[b] = pd.to_numeric(pbp[b], errors="coerce").fillna(0.0)

    # Scramble normalization (present per your zip)
    if "scramble" in pbp.columns:
        pbp["scramble"] = pd.to_numeric(pbp["scramble"], errors="coerce")
    elif "qb_scramble" in pbp.columns:
        pbp["scramble"] = pd.to_numeric(pbp["qb_scramble"], errors="coerce")
    else:
        pbp["scramble"] = np.nan

    if "air_yards" not in pbp.columns: pbp["air_yards"] = np.nan
    if "yardline_100" not in pbp.columns: pbp["yardline_100"] = np.nan

    # Red zone buckets
    pbp["rz20"] = (pbp["yardline_100"] <= 20).astype(int)
    pbp["rz10"] = (pbp["yardline_100"] <= 10).astype(int)
    pbp["rz05"] = (pbp["yardline_100"] <= 5).astype(int)

    # Neutral situations
    qtr = pbp.get("qtr", pd.Series(np.nan, index=pbp.index))
    score_diff = pbp.get("score_differential", pbp.get("score_differential_post", pd.Series(np.nan, index=pbp.index)))
    two_min = pbp.get("two_minute_warning", pd.Series(False, index=pbp.index)).fillna(False)
    pbp["neutral_state"] = (qtr.between(1,3, inclusive="both").fillna(False) & score_diff.abs().le(7).fillna(False) & (~two_min)).astype(int)

    # Explosives
    pbp["explosive_rec"]  = ((pbp.get("pass",0)==1.0) & (pbp["air_yards"] >= 15)).astype(int)
    pbp["explosive_rush"] = ((pbp.get("rush",0)==1.0) & (pbp.get("yards_gained",0) >= 12)).astype(int)

    # Participant convenience
    for c in ["posteam","defteam","receiver_player_name","rusher_player_name","passer_player_name"]:
        if c not in pbp.columns: pbp[c]=np.nan
    pbp["receiver"] = pbp.get("receiver", pbp["receiver_player_name"])
    pbp["rusher"]   = pbp.get("rusher",   pbp["rusher_player_name"])
    pbp["passer"]   = pbp.get("passer",   pbp["passer_player_name"])

    return pbp

@st.cache_data(show_spinner=False)
def load_weekly(seasons: list[int]) -> pd.DataFrame:
    """
    Load weekly data per season with graceful fallbacks:
      1) import_weekly_data([season])  -> primary
      2) import_weekly_pfr([season])   -> fallback (schema may differ)
      3) empty DataFrame               -> last resort
    Never raises on 404; returns concatenated frame or empty.
    """
    from nfl_data_py import import_weekly_data, import_weekly_pfr, import_snap_counts

    frames = []
    failures = []

    # Try each season individually so one 404 doesn't kill the rest
    for y in seasons:
        df = None
        err = None
        try:
            df = import_weekly_data([y])
        except Exception as e:
            err = f"weekly_data {y}: {e}"

        if df is None or df.empty:
            # Try PFR weekly as a fallback
            try:
                pfr = import_weekly_pfr([y])
                if pfr is not None and not pfr.empty:
                    # Try to normalize obvious aliases if present
                    pfr = pfr.rename(columns={
                        "routes": "routes_run",
                        "pass_att": "attempts",
                        "pass_attempts": "attempts",
                        "att": "attempts",
                    })
                    df = pfr
                else:
                    if err is None:
                        err = "no weekly_pfr data"
            except Exception as e2:
                err = (err or "") + f" | weekly_pfr {y}: {e2}"

        if df is None or df.empty:
            # Optional: try snap counts just so we have something to join on
            try:
                snaps = import_snap_counts([y])
                if snaps is not None and not snaps.empty:
                    # minimal schema to not break downstream filters
                    df = snaps.rename(columns={"team":"team", "player":"player"})
            except Exception:
                pass

        if df is None or df.empty:
            failures.append((y, err or "unavailable"))
        else:
            frames.append(df)

    if not frames:
        # Return truly empty; downstream code already tolerates empty weekly
        out = pd.DataFrame()
    else:
        out = pd.concat(frames, ignore_index=True)

    # Surface a one-line notice instead of a stacktrace
    if failures:
        missing_years = ", ".join(str(y) for y, _ in failures)
        st.info(f"Weekly aggregates unavailable for: {missing_years}. App will continue with empty weekly for those seasons.")

    return out


# ---------- Derives (only those supported by your schema) ----------

def derive_rroe(pbp):
    df = pbp.loc[pbp["neutral_state"]==1, ["posteam","rush"]].copy()
    if df.empty: return pd.DataFrame(columns=["team","neutral_run_rate","rROE"])
    df["play"]=1
    agg = df.groupby("posteam")[["rush","play"]].sum(min_count=1)
    league_rr = safe_div(agg["rush"].sum(), agg["play"].sum())
    agg["neutral_run_rate"] = agg["rush"]/agg["play"]
    agg["rROE"] = agg["neutral_run_rate"] - league_rr
    return agg.reset_index().rename(columns={"posteam":"team"})[["team","neutral_run_rate","rROE"]]

def derive_pace(pbp):
    cols = ["game_id","posteam"]
    if not set(cols).issubset(pbp.columns): return pd.DataFrame(columns=["team","plays_per_game","seconds_per_play","no_huddle_rate"])
    df = pbp[cols + (["no_huddle"] if "no_huddle" in pbp.columns else [])].copy()
    df["play"]=1
    g = df.groupby(["game_id","posteam"])["play"].sum().reset_index()
    g["seconds_per_play"] = 3600.0/g["play"]
    team = g.groupby("posteam").agg(plays_per_game=("play","mean"),
                                    seconds_per_play=("seconds_per_play","mean")).reset_index()
    if "no_huddle" in df.columns:
        nh = df.groupby(["game_id","posteam"])["no_huddle"].mean().groupby("posteam").mean().reset_index(name="no_huddle_rate")
        team = team.merge(nh, on="posteam", how="left")
    return team.rename(columns={"posteam":"team"})

def derive_adot(pbp):
    df = pbp.loc[pbp.get("pass",0)==1.0, ["posteam","air_yards"]].dropna()
    if df.empty: return pd.DataFrame(columns=["team","aDOT","ay_q25","ay_q50","ay_q75"])
    g = df.groupby("posteam")["air_yards"].agg(aDOT="mean", ay_q25=lambda s: s.quantile(0.25), ay_q50="median", ay_q75=lambda s: s.quantile(0.75)).reset_index()
    return g.rename(columns={"posteam":"team"})

def derive_hhi(pbp):
    df = pbp.loc[pbp.get("pass",0)==1.0, ["posteam","receiver"]].dropna()
    if df.empty: return pd.DataFrame(columns=["team","target_HHI","n_targeted_players"])
    tgt = df.groupby(["posteam","receiver"]).size().reset_index(name="t")
    team = tgt.groupby("posteam")["t"].sum().reset_index(name="T")
    tgt = tgt.merge(team, on="posteam", how="left")
    tgt["share"] = tgt["t"]/tgt["T"]
    hhi = tgt.groupby("posteam")["share"].apply(lambda s: (s**2).sum()).reset_index(name="target_HHI")
    npl = tgt.groupby("posteam").size().reset_index(name="n_targeted_players")
    return hhi.merge(npl, on="posteam", how="left").rename(columns={"posteam":"team"})

def derive_slot_proxy(pbp):
    if not {"pass_location","air_yards"}.issubset(pbp.columns):
        return pd.DataFrame(columns=["team","slot_target_share_proxy","outside_target_share_proxy"])
    df = pbp.loc[pbp.get("pass",0)==1.0, ["posteam","air_yards","pass_location"]].dropna()
    if df.empty:
        return pd.DataFrame(columns=["team","slot_target_share_proxy","outside_target_share_proxy"])
    df["slot_like"] = ((df["air_yards"]<=7) & (df["pass_location"].astype(str).str.lower().eq("middle"))).astype(int)
    team_t = df.groupby("posteam").size().reset_index(name="T")
    slot_t = df.groupby("posteam")["slot_like"].sum().reset_index(name="S")
    out = team_t.merge(slot_t, on="posteam", how="left").fillna({"S":0})
    out["slot_target_share_proxy"] = out["S"]/out["T"]
    out["outside_target_share_proxy"] = 1 - out["slot_target_share_proxy"]
    return out.rename(columns={"posteam":"team"})[["team","slot_target_share_proxy","outside_target_share_proxy"]]

def derive_redzone_usage(pbp):
    cols = ["posteam","rush","pass","receiver","rusher","rz20","rz10","rz05"]
    need = ["posteam"]
    if not set(need).issubset(pbp.columns): return pd.DataFrame(columns=["team","player","bucket","plays","share"])
    # build df safely
    df = pd.DataFrame({"posteam": pbp["posteam"]})
    for c in ["rush","pass","receiver","rusher","rz20","rz10","rz05"]:
        df[c] = pbp.get(c, np.nan)
    df["player"] = np.where(pd.to_numeric(df["rush"], errors="coerce")==1.0, df["rusher"], df["receiver"])
    df = df.dropna(subset=["posteam","player"])
    recs=[]
    for b in ["rz20","rz10","rz05"]:
        sub = df.loc[pd.to_numeric(df[b], errors="coerce")==1.0]
        if sub.empty: continue
        team = sub.groupby("posteam").size().reset_index(name="T")
        plays = sub.groupby(["posteam","player"]).size().reset_index(name="plays")
        out = plays.merge(team, on="posteam", how="left")
        out["share"]=out["plays"]/out["T"]
        out["bucket"]=b
        recs.append(out[["posteam","player","bucket","plays","share"]])
    if not recs:
        return pd.DataFrame(columns=["team","player","bucket","plays","share"])
    return pd.concat(recs).rename(columns={"posteam":"team"})

def derive_protection(pbp):
    df = pbp.loc[pbp.get("pass",0)==1.0, ["posteam","sack","qb_hit"]].copy()
    if df.empty: return pd.DataFrame(columns=["team","dropbacks","sacks","qb_hits","pressure_rate_proxy"])
    df["db"]=1
    g = df.groupby("posteam").agg(dropbacks=("db","sum"), sacks=("sack","sum"), qb_hits=("qb_hit","sum")).reset_index()
    g["pressure_rate_proxy"] = (g["sacks"]+g["qb_hits"])/g["dropbacks"]
    return g.rename(columns={"posteam":"team"})

def derive_rpo_proxy(pbp):
    # rpo flag not present per your schema; use screen/quick proxy
    if "air_yards" not in pbp.columns: return pd.DataFrame(columns=["team","rpo_rate"])
    df = pbp.loc[pbp.get("pass",0)==1.0, ["posteam","air_yards"]].dropna()
    if df.empty: return pd.DataFrame(columns=["team","rpo_rate"])
    df["screen_like"] = (df["air_yards"]<=1).astype(int)
    out = df.groupby("posteam")["screen_like"].mean().reset_index(name="rpo_rate")
    return out.rename(columns={"posteam":"team"})

# Defense
def def_run_eff(pbp):
    df = pbp.loc[pbp.get("rush",0)==1.0, ["defteam","epa","yards_gained"]].copy()
    if df.empty: return pd.DataFrame(columns=["defense","rush_epa_per_play","yards_per_rush","rush_success_rate"])
    df["success_off"]=(df["epa"]>0).astype(int)
    g = df.groupby("defteam").agg(rush_epa_per_play=("epa","mean"),
                                  yards_per_rush=("yards_gained","mean"),
                                  rush_success_rate=("success_off","mean")).reset_index()
    return g.rename(columns={"defteam":"defense"})

def def_passrush(pbp):
    df = pbp.loc[pbp.get("pass",0)==1.0, ["defteam","sack","qb_hit"]].copy()
    if df.empty: return pd.DataFrame(columns=["defense","pressure_rate_proxy","sack_rate"])
    df["db"]=1
    g = df.groupby("defteam").agg(dropbacks=("db","sum"), sacks=("sack","sum"), qb_hits=("qb_hit","sum")).reset_index()
    g["pressure_rate_proxy"]=(g["sacks"]+g["qb_hits"])/g["dropbacks"]
    g["sack_rate"]=g["sacks"]/g["dropbacks"]
    return g.rename(columns={"defteam":"defense"})

def def_explosives(pbp):
    dfp = pbp.loc[pbp.get("pass",0)==1.0, ["defteam","explosive_rec"]].copy()
    dfr = pbp.loc[pbp.get("rush",0)==1.0, ["defteam","explosive_rush"]].copy()
    base = pd.DataFrame({"defense": sorted(pbp["defteam"].dropna().unique())})
    if not dfp.empty:
        ep = dfp.groupby("defteam")["explosive_rec"].mean().reset_index(name="explosive_pass_rate")
        base = base.merge(ep, left_on="defense", right_on="defteam", how="left").drop(columns=["defteam"])
    if not dfr.empty:
        er = dfr.groupby("defteam")["explosive_rush"].mean().reset_index(name="explosive_rush_rate")
        base = base.merge(er, left_on="defense", right_on="defteam", how="left").drop(columns=["defteam"])
    return base

def def_yac_proxy(pbp):
    need = {"defteam","yards_gained","air_yards","complete_pass"}
    if not need.issubset(pbp.columns): return pd.DataFrame(columns=["defense","yac_per_rec_allowed_proxy"])
    df = pbp[list(need)].dropna(subset=["defteam","yards_gained"])
    df = df.loc[df["complete_pass"]==1.0]
    df["yac_proxy"] = df["yards_gained"] - np.clip(df["air_yards"].fillna(0), 0, None)
    g = df.groupby("defteam")["yac_proxy"].mean().reset_index(name="yac_per_rec_allowed_proxy")
    return g.rename(columns={"defteam":"defense"})

def def_screen_quick(pbp):
    if "air_yards" not in pbp.columns: return pd.DataFrame(columns=["defense","screen_quick_rate_allowed_proxy"])
    df = pbp.loc[pbp.get("pass",0)==1.0, ["defteam","air_yards"]].dropna()
    if df.empty: return pd.DataFrame(columns=["defense","screen_quick_rate_allowed_proxy"])
    df["screen_quick"]=(df["air_yards"]<=1).astype(int)
    g = df.groupby("defteam")["screen_quick"].mean().reset_index(name="screen_quick_rate_allowed_proxy")
    return g.rename(columns={"defteam":"defense"})

def def_run_dir(pbp):
    gap = "run_gap" if "run_gap" in pbp.columns else None
    loc = "run_location" if "run_location" in pbp.columns else None
    if not (gap or loc): return pd.DataFrame(columns=["defense","left_edge_sr","middle_sr","right_edge_sr"])
    df = pbp.loc[pbp.get("rush",0)==1.0, ["defteam","epa"] + [c for c in [gap,loc] if c]].copy()
    df["bucket"] = np.where(df.get(gap).notna(), df[gap].astype(str).str.lower(),
                            df.get(loc, pd.Series("unknown", index=df.index)).astype(str).str.lower())
    def map_dir(x):
        if any(k in x for k in ["left_tackle","left_end","left","lt","le"]): return "left_edge"
        if any(k in x for k in ["right_tackle","right_end","right","rt","re"]): return "right_edge"
        return "middle"
    df["dir"]=df["bucket"].apply(map_dir)
    df["success_off"]=(df["epa"]>0).astype(int)
    g = df.groupby(["defteam","dir"])["success_off"].mean().reset_index()
    piv = g.pivot(index="defteam", columns="dir", values="success_off").reset_index().rename(columns={"defteam":"defense"})
    for col in ["left_edge","middle","right_edge"]:
        if col not in piv.columns: piv[col]=np.nan
    piv.columns = ["defense","left_edge_sr","middle_sr","right_edge_sr"]
    return piv

def def_plays_forced(pbp):
    df = pbp[["game_id","defteam"]].dropna()
    if df.empty: return pd.DataFrame(columns=["defense","plays_against_per_game"])
    df["play"]=1
    g = df.groupby(["game_id","defteam"])["play"].sum().reset_index()
    out = g.groupby("defteam")["play"].mean().reset_index(name="plays_against_per_game")
    return out.rename(columns={"defteam":"defense"})

def derive_qb_runs_off(pbp):
    need = {"posteam","passer","rush"}
    if not need.issubset(pbp.columns): return pd.DataFrame(columns=["team","qb","designed_run_rate","scramble_rate"])
    mask = (pd.to_numeric(pbp["rush"], errors="coerce")==1.0) & (pbp["passer"].notna())
    sub = pbp.loc[mask, ["posteam","passer"]].copy()
    if sub.empty: return pd.DataFrame(columns=["team","qb","designed_run_rate","scramble_rate"])
    sub["scramble"]=pd.to_numeric(pbp.loc[mask, "scramble"], errors="coerce")
    agg = sub.groupby(["posteam","passer"]).agg(rushes=("passer","count"),
                                               scrambles=("scramble","sum")).reset_index()
    agg["scramble_rate"] = agg.apply(lambda r: r["scrambles"]/r["rushes"] if pd.notna(r["scrambles"]) else np.nan, axis=1)
    agg["designed_run_rate"]=1.0-agg["scramble_rate"]
    return agg.rename(columns={"posteam":"team","passer":"qb"})[["team","qb","designed_run_rate","scramble_rate"]]

def def_qb_contain(pbp):
    need = {"defteam","passer","rush"}
    if not need.issubset(pbp.columns): return pd.DataFrame(columns=["defense","qb_rush_rate_allowed_proxy","qb_scramble_rate_allowed_proxy"])
    mask = (pd.to_numeric(pbp["rush"], errors="coerce")==1.0) & (pbp["passer"].notna())
    sub = pbp.loc[mask, ["defteam"]].copy()
    if sub.empty: return pd.DataFrame(columns=["defense","qb_rush_rate_allowed_proxy","qb_scramble_rate_allowed_proxy"])
    sub["qb_rush"]=1
    sub["scramble"]=pd.to_numeric(pbp.loc[mask, "scramble"], errors="coerce")
    agg = sub.groupby("defteam").agg(qb_rushes=("qb_rush","sum"), scrambles=("scramble","sum")).reset_index().rename(columns={"defteam":"defense"})
    db = pbp.loc[pbp.get("pass",0)==1.0].groupby("defteam").size().reset_index(name="dropbacks_against").rename(columns={"defteam":"defense"})
    agg = agg.merge(db, on="defense", how="left")
    agg["qb_rush_rate_allowed_proxy"] = agg["qb_rushes"]/agg["dropbacks_against"]
    agg["qb_scramble_rate_allowed_proxy"] = agg.apply(lambda r: r["scrambles"]/r["qb_rushes"] if pd.notna(r["scrambles"]) else np.nan, axis=1)
    return agg[["defense","qb_rush_rate_allowed_proxy","qb_scramble_rate_allowed_proxy"]]

# ---------- UI ----------
st.sidebar.title("FoxEdge NFL Props Lab v3")
years = st.sidebar.multiselect("Seasons", options=list(range(1999, 2026)), default=[2024])
weeks = st.sidebar.slider("Weeks", 1, 22, (1, 18), 1)
team_filter = st.sidebar.text_input("Filter teams (comma-separated)", value="")
btn = st.sidebar.button("Fetch / Refresh", type="primary")

if not years:
    st.stop()

# --- session state init (prevents missing-key errors on cold start) ---
if "pbp" not in st.session_state:
    st.session_state["pbp"] = None
if "wk" not in st.session_state:
    st.session_state["wk"] = None

# --- safe, idempotent data loading ---
NEED_LOAD = (
    (st.session_state.get("pbp") is None) or
    (st.session_state.get("wk") is None) or
    btn
)

if NEED_LOAD:
    with st.status("Loading...", expanded=False):
        _pbp = load_pbp(years)
        _wk  = load_weekly(years)

        # Week-range filters (only if the column exists)
        if "week" in _pbp.columns:
            _pbp = _pbp[(_pbp["week"] >= weeks[0]) & (_pbp["week"] <= weeks[1])]
        if "week" in _wk.columns:
            _wk = _wk[(_wk["week"] >= weeks[0]) & (_wk["week"] <= weeks[1])]

        st.session_state["pbp"] = _pbp
        st.session_state["wk"]  = _wk

# Always read via .get() so a mid-rerun doesn’t explode
pbp = st.session_state.get("pbp")
wk  = st.session_state.get("wk")

# Final fuse: if either is still None, load once more
if pbp is None or wk is None:
    _pbp = load_pbp(years)
    _wk  = load_weekly(years)
    if "week" in _pbp.columns:
        _pbp = _pbp[(_pbp["week"] >= weeks[0]) & (_pbp["week"] <= weeks[1])]
    if "week" in _wk.columns:
        _wk = _wk[(_wk["week"] >= weeks[0]) & (_wk["week"] <= weeks[1])]
    st.session_state["pbp"] = _pbp
    st.session_state["wk"]  = _wk
    pbp, wk = _pbp, _wk

# Optional team filter applied to local copies (won't mutate session cache)
if team_filter.strip():
    teams = [t.strip().upper() for t in team_filter.split(",") if t.strip()]
    if not pbp.empty:
        pbp = pbp[(pbp["posteam"].isin(teams)) | (pbp["defteam"].isin(teams))]
    if (wk is not None) and (not wk.empty) and ("team" in wk.columns):
        wk = wk[wk["team"].isin(teams)]

st.success(f"Data loaded. Plays: {0 if pbp is None else len(pbp):,} | Weekly rows: {0 if wk is None else len(wk):,}")


checks = {
    "air_yards": "air_yards" in pbp.columns,
    "no_huddle": "no_huddle" in pbp.columns,
    "play_action": "play_action" in pbp.columns,      # false per zip
    "personnel": ("personnel_o" in pbp.columns) or ("posteam_personnel" in pbp.columns),  # false per zip
    "pass_location": "pass_location" in pbp.columns,  # true
    "run_gap/location": ("run_gap" in pbp.columns) or ("run_location" in pbp.columns),    # true
    "scramble": ("scramble" in pbp.columns) and not pbp["scramble"].isna().all(),         # true
    "rpo_flag": "rpo" in pbp.columns,                 # false
    "weekly.routes_run": ("routes_run" in wk.columns) if not wk.empty else False,         # false
    "weekly.attempts": any(c in wk.columns for c in ["attempts","pass_attempts","att"]) if not wk.empty else False,
}
st.subheader("Data Health & Availability")
st.table(pd.DataFrame({"field": list(checks.keys()), "available": list(checks.values())}))

with st.spinner("Computing..."):
    off_rroe = derive_rroe(pbp)
    off_pace = derive_pace(pbp)
    off_adot = derive_adot(pbp)
    off_hhi  = derive_hhi(pbp)
    off_slot = derive_slot_proxy(pbp)
    off_rz   = derive_redzone_usage(pbp)
    off_prot = derive_protection(pbp)
    off_rpo  = derive_rpo_proxy(pbp)   # proxy only

    de_run = def_run_eff(pbp)
    de_pr  = def_passrush(pbp)
    de_exp = def_explosives(pbp)
    de_yac = def_yac_proxy(pbp)
    de_scr = def_screen_quick(pbp)
    de_dir = def_run_dir(pbp)
    de_pfg = def_plays_forced(pbp)

    off_qb = derive_qb_runs_off(pbp) if checks["scramble"] else pd.DataFrame(columns=["team","qb","designed_run_rate","scramble_rate"])
    de_qbc = def_qb_contain(pbp)      if checks["scramble"] else pd.DataFrame(columns=["defense","qb_rush_rate_allowed_proxy","qb_scramble_rate_allowed_proxy"])

tabs = [
    ("Offense: rROE", off_rroe),
    ("Offense: Pace & No-Huddle", off_pace),
    ("Offense: Pass Depth", off_adot),
    ("Offense: Target HHI", off_hhi),
    ("Offense: Slot/Wide Proxy", off_slot),
    ("Offense: Red Zone Usage", off_rz),
    ("Offense: Protection", off_prot),
    ("Offense: RPO Proxy", off_rpo),
    ("Defense: Run Eff.", de_run),
    ("Defense: Pass Rush", de_pr),
    ("Defense: Explosives Allowed", de_exp),
    ("Defense: YAC/Tackling Proxy", de_yac),
    ("Defense: Screen/Quick Allowed", de_scr),
    ("Defense: Run Direction", de_dir),
    ("Defense: Plays Forced", de_pfg),
]

# Enable QB tabs because scramble exists per the zip
tabs.insert(8, ("Offense: QB Runs", off_qb))
tabs.insert(15, ("Defense: QB Containment", de_qbc))

tab_objs = st.tabs([t[0] for t in tabs])
for i, (_, df) in enumerate(tabs):
    with tab_objs[i]:
        st.dataframe(df.loc[:, ~df.columns.duplicated()], use_container_width=True)

st.caption("v3 aligned to your provided schemas: personnel/PA/motion/rpo_flag/routes_run not required, QB scramble supported.")
