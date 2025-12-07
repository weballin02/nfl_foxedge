#!/usr/bin/env python3
# nfl_ref_engine_with_assignments.py
#
# Complete Ref Engine:
# - Per-ref cards with EB shrinkage, Wilson CIs, archetypes, prop tags, impact metrics
# - Football Zebras scraper OR assignments CSV reader
# - Clean "Whistle Watch" preview (MD + JSON) that hides missing fields
#
# Deps: pandas, numpy, nfl_data_py  (stdlib otherwise)

import argparse
import csv
import datetime as dt
import html
import json
import math
import os
import re
import sys
import urllib.request
from html.parser import HTMLParser
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import nfl_data_py as nfl

# =========================
# Config (tweakable knobs)
# =========================

YEARS_BACK = 3
USE_MULTIYEAR = True

MIN_GAMES_TOTAL = 20          # ref eligibility for leaderboards/conf tags
CI_HIDE_IF_DECISIONS_LT = 15  # Wilson CI hidden when fewer decisions than this
CI_WIDE_SPAN_GT = 0.25        # tag [WIDE CI] if CI span exceeds this
EB_SHRINK_K = 20              # EB pseudo-counts
EPS_RESPECT = 0.25            # ignore tiny open->close error deltas

LOW_BUCKET_N = 5

PACE_A_TEMPO_DRAGS = -0.8     # plays delta per 1 tempo drag
PACE_B_FLAGS = -0.15          # plays delta per 1 flag/game

IMPLIED_FLAGS_COEF = 0.6      # points per (flags/gm above lg)
IMPLIED_FREEPASS_PER10 = 0.5  # points per 10 free pass yards/gm
LEAGUE_BASE_PPG = 43.5

PROP_THRESH = {
    "qb_protect_pg": 1.2,
    "free_pass_yards_pg": 18.0,
    "drive_killers_pg": 2.0,
    "tempo_drags_pg": 1.5,
    "bailout_3rd_fd_per_game": 1.0,
}

ARCH_RULES = {
    "QB Sanctuary": ("qb_protect_pg", 1.2),
    "Drive Extenders": ("free_pass_yards_pg", 20.0),
    "Clock Vampire": ("tempo_drags_pg", 1.6),
    "Laundry Storm": ("pens_per_game", 16.0),
}

SHOW_MIN = {
    "tax_abs_min_yds": 2.0,
    "uplift_abs_min_pts": 0.3,
}

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
PLOTS_DIR = os.path.join(OUTDIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

TEAM_LASTTOKEN_TO_ABBR = {
    "Cardinals":"ARI","Falcons":"ATL","Ravens":"BAL","Bills":"BUF","Panthers":"CAR","Bears":"CHI","Bengals":"CIN",
    "Browns":"CLE","Cowboys":"DAL","Broncos":"DEN","Lions":"DET","Packers":"GB","Texans":"HOU","Colts":"IND",
    "Jaguars":"JAX","Chiefs":"KC","Raiders":"LV","Chargers":"LAC","Rams":"LAR","Dolphins":"MIA","Vikings":"MIN",
    "Patriots":"NE","Saints":"NO","Giants":"NYG","Jets":"NYJ","Eagles":"PHI","Steelers":"PIT","49ers":"SF",
    "Seahawks":"SEA","Buccaneers":"TB","Titans":"TEN","Commanders":"WAS"
}

REF_ALIASES = {
    # "William Vinovich": "Bill Vinovich",
}

# =========================
# Small helpers
# =========================

def _ref_key(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = REF_ALIASES.get(name.strip(), name.strip())
    return name.lower()

def last_token(s: str) -> str:
    return str(s).strip().split()[-1]

def safe_ratio(num, den):
    return num / den if den else 0.0

def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt((p*(1-p)/n) + (z*z/(4*n*n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def eb(p_raw: float, n: int, p_lg: float, k: int = EB_SHRINK_K) -> float:
    if n <= 0:
        return p_lg
    return (p_raw * n + p_lg * k) / (n + k)

def pct(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    wins = (s == 1).sum()
    decs = ((s == 1) | (s == -1)).sum()
    return safe_ratio(wins, decs)

def norm_team_token(t: str) -> str:
    tok = last_token(str(t))
    return TEAM_LASTTOKEN_TO_ABBR.get(tok, tok)

# =========================
# Assignments scraper (Football Zebras)
# =========================

BASE = "https://www.footballzebras.com"
ASSIGNMENTS_INDEX = f"{BASE}/category/nfl/assignments/"
UA_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}

def _fetch(url: str) -> str:
    url = url.split("#", 1)[0]
    req = urllib.request.Request(url, headers=UA_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", "ignore")

class LinesFromArticle(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_article = False
        self.buf = []
    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.in_article = True
        if self.in_article and tag in ("h1","h2","h3","p","li","br"):
            self.buf.append("\n")
    def handle_endtag(self, tag):
        if tag == "article":
            self.in_article = False
        if self.in_article and tag in ("h1","h2","h3","p","li","div"):
            self.buf.append("\n")
    def handle_data(self, data):
        if self.in_article:
            t = data.replace("\xa0"," ").strip()
            if t:
                self.buf.append(t + " ")
    def lines(self):
        raw = "".join(self.buf)
        raw = html.unescape(raw)
        lines = [re.sub(r"\s+"," ", x).strip() for x in raw.split("\n")]
        return [x for x in lines if x]

def discover_week_url(season: int, week: str | int) -> Optional[str]:
    if week != "latest":
        w = int(week)
        candidates = [
            f"{BASE}/{season:04d}/09/week-{w}-referee-assignments-{season}/",
            f"{BASE}/{season:04d}/09/week-{w}-referee-assignments-{w}/",
            f"{BASE}/{season:04d}/09/week-{w}-referee-assignments-5/",
            f"{BASE}/{season:04d}/09/week-{w}-referee-assignments/",
        ]
        for u in candidates:
            try:
                doc = _fetch(u)
                if "referee assignments" in doc.lower():
                    return u
            except Exception:
                pass
    try:
        idx = _fetch(ASSIGNMENTS_INDEX)
    except Exception:
        return None
    links = re.findall(r'href="([^"]*week-\d+-referee-assignments[^"]*)"', idx, flags=re.I)
    links = [l if l.startswith("http") else BASE + l for l in links]
    links = list(dict.fromkeys(links))
    if not links:
        return None
    if week == "latest":
        def wnum(u):
            m = re.search(r"week-(\d+)-referee-assignments", u, re.I)
            return int(m.group(1)) if m else -1
        links.sort(key=wnum, reverse=True)
        return links[0]
    w = int(week)
    for u in links:
        if re.search(rf"week-{w}-referee-assignments", u, re.I):
            return u
    return links[0]

def parse_assignments(url: str, expected_week: Optional[int] = None) -> Tuple[List[Dict], Optional[int], Optional[int]]:
    doc = _fetch(url)
    p = LinesFromArticle()
    p.feed(doc)
    lines = p.lines()

    season_guess = None
    wk_guess = expected_week
    my = re.search(r"/(\d{4})/", url)
    if my:
        try:
            season_guess = int(my.group(1))
        except Exception:
            season_guess = None
    if wk_guess is None:
        for t in lines[:12]:
            m = re.search(r"Week\s+(\d+)", t, re.I)
            if m:
                wk_guess = int(m.group(1))
                break

    is_date = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,\s+[A-Za-z]+\.?\s+\d{1,2}$", re.I)
    is_match= re.compile(r"^([A-Za-z .]+?)\s+(at|vs\.)\s+([A-Za-z .]+?)$", re.I)
    is_time = re.compile(r"\b\d{1,2}(:\d{2})?\s*[ap]\.m\.\b", re.I)
    is_net  = re.compile(r"\b(Prime|Peacock|FOX|CBS|NFLN|ESPN(?:\s*ESPN\+|2)?|ABC|Amazon|NBC)\b", re.I)

    def looks_like_ref(s: str) -> bool:
        if not s or len(s) > 80:
            return False
        return ("referee" in s.lower()) or bool(re.match(r"^[A-Z][a-z]+(?:[-\s'][A-Z][a-z]+)+", s))

    rows = []
    date = ""
    i = 0
    while i < len(lines):
        t = lines[i]
        if is_date.search(t):
            date = t
            i += 1
            continue
        m = is_match.match(t)
        if not m:
            i += 1
            continue

        away_raw, sep, home_raw = m.group(1), m.group(2).lower(), m.group(3)
        ref, tim, net = "", "", ""
        j = 1
        if i+j < len(lines) and not is_date.search(lines[i+j]) and not is_match.match(lines[i+j]):
            if looks_like_ref(lines[i+j]):
                ref = lines[i+j]
                j += 1
        if i+j < len(lines) and is_time.search(lines[i+j]):
            tim = lines[i+j]
            j += 1
        if tim:
            after = is_time.sub("", tim).strip()
            if after and is_net.search(after):
                net = after
        if not net and i+j < len(lines) and is_net.search(lines[i+j]):
            net = lines[i+j]
            j += 1

        away = norm_team_token(away_raw)
        home = norm_team_token(home_raw)
        site = "neutral" if sep.startswith("vs") else "home"

        rows.append({
            "season": season_guess or "",
            "week": wk_guess or "",
            "game_date": date,
            "kickoff_et": tim,
            "home_team": home,
            "away_team": away,
            "site_type": site,
            "network": net,
            "Referee": re.sub(r"\s*is the referee.*$", "", ref, flags=re.I).strip()
        })
        i += j

    return rows, season_guess, wk_guess

def write_assignments_csv(rows: List[Dict], out_path: str):
    hdr = ["season","week","game_date","kickoff_et","home_team","away_team","site_type","network","Referee"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in hdr})

# =========================
# Engine: data load + metrics
# =========================

def load_years() -> List[int]:
    current = dt.datetime.now().year
    if USE_MULTIYEAR:
        cand = [current - i for i in range(YEARS_BACK)]
    else:
        cand = [current]
    valid = []
    for y in cand:
        try:
            sch = nfl.import_schedules([y])
            if not sch.empty:
                valid.append(y)
        except Exception:
            continue
    if not valid:
        valid = [current - 1]
    return sorted(valid)

def load_frames(years: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    schedules = nfl.import_schedules(years)
    pbp = nfl.import_pbp_data(years)
    officials = nfl.import_officials(years)
    schedules = schedules[schedules.get("game_type") == "REG"].copy()
    if "season_type" in pbp.columns:
        pbp = pbp[pbp["season_type"] == "REG"].copy()
    return schedules, pbp, officials

def build_ref_game_table(schedules: pd.DataFrame, pbp: pd.DataFrame, officials: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    refs = officials[officials.get("off_pos") == "R"][["game_id","name","season"]].rename(columns={"name":"ref_name"})
    pen_cols = [
        "game_id","play_id","penalty","penalty_type","penalty_yards","penalty_team",
        "home_team","away_team","down","ydstogo","first_down","first_down_penalty","penalty_1st_down",
        "epa","wpa","yardline_100","qtr"
    ]
    pens = pbp[[c for c in pen_cols if c in pbp.columns]].copy()
    pens = pens[pens.get("penalty", 0) == 1]

    pens["pen_against_side"] = np.where(
        pens["penalty_team"] == pens["home_team"], "home",
        np.where(pens["penalty_team"] == pens["away_team"], "away", "other")
    )

    agg_pen = pens.groupby(["game_id","pen_against_side"]).agg(
        pen_count=("play_id","count"),
        pen_yards=("penalty_yards","sum"),
    ).reset_index()
    agg_piv = agg_pen.pivot(index="game_id", columns="pen_against_side", values=["pen_count","pen_yards"]).fillna(0)
    agg_piv.columns = [f"{a}_{b}" for a,b in agg_piv.columns]
    agg_piv = agg_piv.reset_index()

    sch_cols = ["game_id","season","week","home_team","away_team","home_score","away_score","spread_line","total_line"]
    sch = schedules[sch_cols].copy()

    g = sch.merge(refs, on=["game_id","season"], how="left").merge(agg_piv, on="game_id", how="left")
    g = g[g["ref_name"].notna()].copy()
    g["ref_name"] = g["ref_name"].astype(str).str.strip()

    g["final_total"] = g["home_score"].fillna(0) + g["away_score"].fillna(0)
    g["home_margin"] = g["home_score"].fillna(0) - g["away_score"].fillna(0)

    # No guessing: nfl_data_py uses home perspective. Negative = home favorite.
    g["spread_norm"] = g["spread_line"]

    # ATS from home perspective
    spread_mask = g["spread_norm"].notna()
    g["home_ats"] = np.nan
    g.loc[spread_mask & ((g["home_margin"] + g["spread_norm"]) > 0), "home_ats"] = 1
    g.loc[spread_mask & ((g["home_margin"] + g["spread_norm"]) == 0), "home_ats"] = 0
    g.loc[spread_mask & ((g["home_margin"] + g["spread_norm"]) < 0), "home_ats"] = -1

    # Over/Under
    total_mask = g["total_line"].notna()
    g["ou_result"] = np.nan
    g.loc[total_mask & (g["final_total"] > g["total_line"]), "ou_result"] = 1
    g.loc[total_mask & (g["final_total"] == g["total_line"]), "ou_result"] = 0
    g.loc[total_mask & (g["final_total"] < g["total_line"]), "ou_result"] = -1

    for col in ["pen_count_home","pen_count_away","pen_yards_home","pen_yards_away"]:
        if col not in g.columns:
            g[col] = 0
    g["pen_count_diff_away_minus_home"] = g["pen_count_away"] - g["pen_count_home"]
    g["pen_yards_diff_away_minus_home"] = g["pen_yards_away"] - g["pen_yards_home"]

    # Angles
    p3 = pens.copy()
    p3["is_3rd"] = (p3.get("down", np.nan) == 3)
    defensive_set = {"Defensive Pass Interference","Defensive Holding","Illegal Contact","Roughing the Passer","Unnecessary Roughness"}
    p3["is_def_flag"] = p3["penalty_type"].isin(defensive_set)
    p3["fd_by_pen"] = False
    for col in ["first_down_penalty","penalty_1st_down"]:
        if col in p3.columns:
            p3["fd_by_pen"] = p3["fd_by_pen"] | p3[col].fillna(False).astype(bool)

    bail = p3.groupby("game_id").apply(
        lambda df: pd.Series({
            "bailout_3rd_def_flags": int((df["is_3rd"] & df["is_def_flag"]).sum()),
            "bailout_3rd_fd":        int((df["is_3rd"] & df["is_def_flag"] & df["fd_by_pen"]).sum())
        }),
        include_groups=False
    ).reset_index()
    g = g.merge(bail, on="game_id", how="left").fillna({"bailout_3rd_def_flags":0, "bailout_3rd_fd":0})

    pass_def_flags = {"Defensive Pass Interference","Defensive Holding","Illegal Contact"}
    fy = pens[pens["penalty_type"].isin(pass_def_flags)].groupby("game_id")["penalty_yards"].sum().rename("free_pass_yards").reset_index()
    g = g.merge(fy, on="game_id", how="left").fillna({"free_pass_yards":0})

    kill_types = {"Offensive Holding","False Start"}
    dk = pens[pens["penalty_type"].isin(kill_types)].copy()
    dk["early_down"] = dk.get("down", np.nan).isin([1,2])
    dk["drive_kill_flag"] = False
    dk.loc[dk["penalty_type"]=="Offensive Holding", "drive_kill_flag"] = dk["early_down"] & (pd.to_numeric(dk["penalty_yards"], errors="coerce").abs() >= 10)
    if "ydstogo" in dk.columns:
        fs_mask = (dk["penalty_type"]=="False Start") & (dk["down"]==2) & (pd.to_numeric(dk["ydstogo"], errors="coerce")<=4)
        dk.loc[fs_mask, "drive_kill_flag"] = True
    dk_agg = dk.groupby("game_id")["drive_kill_flag"].sum().rename("drive_killers").reset_index()
    g = g.merge(dk_agg, on="game_id", how="left").fillna({"drive_killers":0})

    drag_types = {"Delay of Game","Illegal Substitution","Illegal Shift","Illegal Motion"}
    td = pens[pens["penalty_type"].isin(drag_types)].groupby("game_id").size().rename("tempo_drags").reset_index()
    g = g.merge(td, on="game_id", how="left").fillna({"tempo_drags":0})

    fdp = pens.copy()
    fdp["fd_by_pen"] = False
    for col in ["first_down_penalty","penalty_1st_down"]:
        if col in fdp.columns:
            fdp["fd_by_pen"] = fdp["fd_by_pen"] | fdp[col].fillna(False).astype(bool)
    fdp_agg = fdp.groupby("game_id")["fd_by_pen"].sum().rename("first_downs_by_pen").reset_index()
    g = g.merge(fdp_agg, on="game_id", how="left").fillna({"first_downs_by_pen":0})

    gf_types = {"Roughing the Passer","Unnecessary Roughness"}
    gfw = pens[pens["penalty_type"].isin(gf_types)].groupby("game_id").size().rename("qb_protect_flags").reset_index()
    g = g.merge(gfw, on="game_id", how="left").fillna({"qb_protect_flags":0})

    # Impact metrics (EPA/WPA, late, RZ, plays)
    ep_cols = {"epa","wpa","yardline_100","qtr","down"}
    if ep_cols.issubset(set(p3.columns)) or ep_cols.issubset(set(pens.columns)):
        pe = pens[["game_id","epa","wpa","yardline_100","qtr","down","penalty_type"]].copy()
        pe["is_rz"] = pd.to_numeric(pe.get("yardline_100", np.nan), errors="coerce") <= 20
        pe["is_q4"] = pd.to_numeric(pe.get("qtr", 0), errors="coerce").isin([4,5])

        # 3rd-and-7+ bailout EPA specifically
        pe["fd_by_pen"] = False
        for col in ["first_down_penalty","penalty_1st_down"]:
            if col in pens.columns:
                pe["fd_by_pen"] = pe["fd_by_pen"] | pens[col].fillna(False).astype(bool)
        pe["bailout_long"] = (pe["down"] == 3) & (pd.to_numeric(pens.get("ydstogo", np.nan), errors="coerce")>=7) & pe["fd_by_pen"]

        imp = pe.groupby("game_id").agg(
            pen_epa_forced=("epa","sum"),
            pen_wpa_forced=("wpa","sum"),
            rz_flag_fd_rate=("is_rz","mean")
        ).reset_index()

        q4 = pe.groupby("game_id").apply(lambda df: df.loc[df["is_q4"], "wpa"].sum(), include_groups=False).rename("q4_wpa").reset_index()
        tot = pe.groupby("game_id")["wpa"].sum().rename("tot_wpa").reset_index()
        wshare = q4.merge(tot, on="game_id", how="left")
        wshare["q4_wpa_share"] = np.where(wshare["tot_wpa"]!=0, wshare["q4_wpa"]/wshare["tot_wpa"], np.nan)

        plays_pg = pbp.groupby("game_id").size().rename("plays").reset_index()
        q4_flags = pe[pe["is_q4"]].groupby("game_id").size().rename("q4_flags").reset_index()
        late = plays_pg.merge(q4_flags, on="game_id", how="left")
        late["q4_flags_per100"] = 100 * late["q4_flags"].fillna(0) / late["plays"].replace(0, np.nan)

        g = g.merge(imp, on="game_id", how="left") \
             .merge(wshare[["game_id","q4_wpa_share"]], on="game_id", how="left") \
             .merge(late[["game_id","q4_flags_per100","plays"]], on="game_id", how="left")

        g[["pen_epa_forced","pen_wpa_forced","rz_flag_fd_rate","q4_wpa_share","q4_flags_per100"]] = \
            g[["pen_epa_forced","pen_wpa_forced","rz_flag_fd_rate","q4_wpa_share","q4_flags_per100"]].fillna(0.0)
    else:
        for c in ["pen_epa_forced","pen_wpa_forced","rz_flag_fd_rate","q4_wpa_share","q4_flags_per100","plays"]:
            g[c] = 0.0

    # Penalty type table for rates
    type_counts = pens.groupby(["game_id","penalty_type"]).size().rename("type_count").reset_index()
    return g, type_counts

def league_baselines(ref_games: pd.DataFrame, type_counts: pd.DataFrame) -> Dict:
    total_games = ref_games["game_id"].nunique()
    p_home_ats = pct(ref_games["home_ats"])
    p_over = pct(ref_games["ou_result"])
    lg_flags_pg = (ref_games["pen_count_home"].fillna(0) + ref_games["pen_count_away"].fillna(0)).mean()
    plays_pg = ref_games.get("plays", pd.Series(dtype=float)).dropna().mean() if "plays" in ref_games.columns else float("nan")

    type_pg = (type_counts.groupby("penalty_type")["type_count"].sum() / total_games) if total_games else pd.Series(dtype=float)

    return {
        "p_home_ats": p_home_ats,
        "p_over": p_over,
        "penalties_per_game": float(lg_flags_pg),
        "plays_per_game": float(plays_pg) if not np.isnan(plays_pg) else float("nan"),
        "type_per_game": type_pg.to_dict()
    }

def aggregate_refs(ref_games: pd.DataFrame, type_counts: pd.DataFrame, baselines: Dict) -> pd.DataFrame:
    # build per-ref per-game unique
    g0 = ref_games.groupby(["ref_name","game_id"], as_index=False).agg({
        "season":"first","week":"first","home_ats":"first","ou_result":"first",
        "pen_count_home":"first","pen_count_away":"first","pen_yards_home":"first","pen_yards_away":"first",
        "pen_count_diff_away_minus_home":"first","pen_yards_diff_away_minus_home":"first",
        "bailout_3rd_fd":"first","free_pass_yards":"first","drive_killers":"first","tempo_drags":"first",
        "first_downs_by_pen":"first","qb_protect_flags":"first",
        "pen_epa_forced":"first","pen_wpa_forced":"first","q4_flags_per100":"first","q4_wpa_share":"first",
        "rz_flag_fd_rate":"first","plays":"first","spread_norm":"first"
    })

    # penalty type rates per game
    top_types = type_counts["penalty_type"].value_counts().head(10).index.tolist()
    t_piv = type_counts[type_counts["penalty_type"].isin(top_types)] \
        .pivot_table(index="game_id", columns="penalty_type", values="type_count", aggfunc="sum", fill_value=0).reset_index()
    g = g0.merge(t_piv, on="game_id", how="left").fillna(0)

    grp = g.groupby("ref_name").agg(
        games=("game_id","nunique"),
        seasons=("season", lambda s: sorted(set(s))),
        ats_w=("home_ats", lambda s: (s==1).sum()),
        ats_l=("home_ats", lambda s: (s==-1).sum()),
        ats_p=("home_ats", lambda s: (s==0).sum()),
        ou_over=("ou_result", lambda s: (s==1).sum()),
        ou_under=("ou_result", lambda s: (s==-1).sum()),
        ou_push=("ou_result", lambda s: (s==0).sum()),
        pen_home=("pen_count_home","sum"),
        pen_away=("pen_count_away","sum"),
        yards_home=("pen_yards_home","sum"),
        yards_away=("pen_yards_away","sum"),
        pen_cnt_diff_avg=("pen_count_diff_away_minus_home","mean"),
        pen_yds_diff_avg=("pen_yards_diff_away_minus_home","mean"),
        bailout_3rd_fd=("bailout_3rd_fd","sum"),
        free_pass_yards=("free_pass_yards","sum"),
        drive_killers=("drive_killers","sum"),
        tempo_drags=("tempo_drags","sum"),
        first_downs_by_pen=("first_downs_by_pen","sum"),
        qb_protect_flags=("qb_protect_flags","sum"),
        pen_epa_forced_pg=("pen_epa_forced","mean"),
        pen_wpa_forced_pg=("pen_wpa_forced","mean"),
        q4_flags_per100_pg=("q4_flags_per100","mean"),
        q4_wpa_share_pg=("q4_wpa_share","mean"),
        rz_flag_fd_rate_pg=("rz_flag_fd_rate","mean"),
        plays_pg=("plays","mean"),
    ).reset_index()

    # outcomes
    grp["ats_decisions"] = grp["ats_w"] + grp["ats_l"]
    grp["ou_decisions"] = grp["ou_over"] + grp["ou_under"]
    grp["ats_pct_raw"] = grp.apply(lambda r: safe_ratio(r["ats_w"], r["ats_decisions"]), axis=1)
    grp["over_pct_raw"] = grp.apply(lambda r: safe_ratio(r["ou_over"], r["ou_decisions"]), axis=1)

    # EB shrink
    grp["ats_pct_adj"] = grp.apply(lambda r: eb(r["ats_pct_raw"], int(r["ats_decisions"]), baselines["p_home_ats"]), axis=1)
    grp["over_pct_adj"] = grp.apply(lambda r: eb(r["over_pct_raw"], int(r["ou_decisions"]), baselines["p_over"]), axis=1)

    # per-game angles
    denom = grp["games"].replace(0, np.nan)
    grp["pens_per_game"] = (grp["pen_home"] + grp["pen_away"]) / denom
    grp["penalty_yards_pg"] = (grp["yards_home"] + grp["yards_away"]) / denom
    grp["home_bias_ratio"] = (grp["pen_home"] / grp["pen_away"]).replace({np.inf: np.nan})

    grp["bailout_3rd_fd_per_game"] = grp["bailout_3rd_fd"] / denom
    grp["free_pass_yards_pg"] = grp["free_pass_yards"] / denom
    grp["drive_killers_pg"] = grp["drive_killers"] / denom
    grp["tempo_drags_pg"] = grp["tempo_drags"] / denom
    grp["fd_by_pen_pg"] = grp["first_downs_by_pen"] / denom
    grp["qb_protect_pg"] = grp["qb_protect_flags"] / denom

    # Plays delta vs league
    grp["plays_pg_delta_vs_lg"] = grp["plays_pg"] - baselines.get("plays_per_game", np.nan)

    # Implied total uplift
    grp["implied_total_uplift"] = (
        IMPLIED_FLAGS_COEF * (grp["pens_per_game"] - baselines["penalties_per_game"]) +
        IMPLIED_FREEPASS_PER10 * (grp["free_pass_yards_pg"] / 10.0)
    )

    # Wilson CIs and confidence tag
    ats_lo, ats_hi, over_lo, over_hi, tags = [], [], [], [], []
    for p, n, po, no in zip(grp["ats_pct_adj"], grp["ats_decisions"], grp["over_pct_adj"], grp["ou_decisions"]):
        if int(n) >= CI_HIDE_IF_DECISIONS_LT:
            lo, hi = wilson_ci(p, int(n))
        else:
            lo, hi = (float("nan"), float("nan"))
        if int(no) >= CI_HIDE_IF_DECISIONS_LT:
            lo2, hi2 = wilson_ci(po, int(no))
        else:
            lo2, hi2 = (float("nan"), float("nan"))
        ats_lo.append(lo); ats_hi.append(hi)
        over_lo.append(lo2); over_hi.append(hi2)

        # Early-season: more generous thresholds
        games_played = int(grp["games"].max() or 0)
        if games_played < 45:
            HIGH_N, MED_N = 45, 25
        else:
            HIGH_N, MED_N = 60, 35
            
        tag = "LOW"
        if int(n) >= HIGH_N and int(no) >= HIGH_N:
            tag = "HIGH"
        elif int(n) >= MED_N or int(no) >= MED_N:
            tag = "MED"
        tags.append(tag)

    grp["ats_pct_adj_lo"] = ats_lo
    grp["ats_pct_adj_hi"] = ats_hi
    grp["over_pct_adj_lo"] = over_lo
    grp["over_pct_adj_hi"] = over_hi
    grp["confidence_tag"] = tags

    # Archetypes (max 2)
    archs = []
    for _, r in grp.iterrows():
        labels = []
        for name, (col, thr) in ARCH_RULES.items():
            try:
                if float(r.get(col, 0.0)) >= float(thr):
                    labels.append(name)
            except Exception:
                pass
        archs.append(labels[:2])
    grp["archetypes"] = archs

    # Prop tags (max 2)
    ptags = []
    for _, r in grp.iterrows():
        tags = []
        if r.get("qb_protect_pg", 0) >= PROP_THRESH["qb_protect_pg"]:
            tags.append("QB_PASS_YDS_UP")
        if r.get("free_pass_yards_pg", 0) >= PROP_THRESH["free_pass_yards_pg"]:
            tags.append("WR_PEN_BOOST")
        if r.get("drive_killers_pg", 0) >= PROP_THRESH["drive_killers_pg"]:
            tags.append("RB_ATT_UP_UNDER_YDS")
        if r.get("tempo_drags_pg", 0) >= PROP_THRESH["tempo_drags_pg"]:
            tags.append("UNDER_PACE_LEAN")
        if r.get("bailout_3rd_fd_per_game", 0) >= PROP_THRESH["bailout_3rd_fd_per_game"]:
            tags.append("3RD_DOWN_EXTENSIONS")
        ptags.append(tags[:2])
    grp["prop_tags"] = ptags

    # Team tax sheets
    team_side = ref_games.groupby(["ref_name","home_team","away_team"]).agg(
        pen_yds_diff=("pen_yards_diff_away_minus_home","mean")
    ).reset_index()

    tax_rows, ben_rows = [], []
    for ref, df in team_side.groupby("ref_name"):
        team_scores = {}
        for _, r in df.iterrows():
            team_scores[r["home_team"]] = team_scores.get(r["home_team"], 0.0) - float(r["pen_yds_diff"])
            team_scores[r["away_team"]] = team_scores.get(r["away_team"], 0.0) + float(r["pen_yds_diff"])
        ts = pd.Series(team_scores)
        if ts.empty:
            tax_rows.append((ref,""))
            ben_rows.append((ref,""))
            continue
        top_tax = ts.sort_values(ascending=False).head(3)
        top_ben = ts.sort_values().head(3)
        def fmt_top(s):
            # Normalize team codes
            TEAM_FIX = {"LA":"LAR"}  # add any other legacy codes
            def _fix_team_code(t): return TEAM_FIX.get(t, t)
            chunks = [f"{_fix_team_code(idx)} {val:+.1f}" for idx, val in s.items() if abs(val) >= SHOW_MIN["tax_abs_min_yds"]]
            return ", ".join(chunks[:2])
        tax_rows.append((ref, fmt_top(top_tax)))
        ben_rows.append((ref, fmt_top(top_ben)))

    tax_df = pd.DataFrame(tax_rows, columns=["ref_name","tax_sheet_top"])
    ben_df = pd.DataFrame(ben_rows, columns=["ref_name","benefit_sheet_top"])
    grp = grp.merge(tax_df, on="ref_name", how="left").merge(ben_df, on="ref_name", how="left")

    # Sanity: decisions cannot exceed games
    for _, r in grp.iterrows():
        if int(r["ats_decisions"]) > int(r["games"]):
            print(f"WARNING: {r['ref_name']} ATS decisions > games")
        if int(r["ou_decisions"]) > int(r["games"]):
            print(f"WARNING: {r['ref_name']} OU decisions > games")

    return grp.sort_values("ref_name").reset_index(drop=True)

# =========================
# Exports
# =========================

def export_ref_cards_md(ref_summary: pd.DataFrame, baselines: Dict, out_path: str):
    lines = []
    lines.append(f"# NFL Referee Report Cards (blended={USE_MULTIYEAR}, years={YEARS_BACK}, MIN_GAMES_TOTAL={MIN_GAMES_TOTAL})\n")
    lines.append(f"League baselines — Home ATS: {baselines['p_home_ats']:.3f}, Over: {baselines['p_over']:.3f}, Flags/Gm: {baselines['penalties_per_game']:.2f}\n")
    lines.append("Legend: Home−Away diff > 0 = more penalties on road teams. CIs are Wilson 95%. Confidence tags reflect decision counts.\n")

    for _, r in ref_summary.sort_values(["games","ref_name"], ascending=[False,True]).iterrows():
        arch = ", ".join(r["archetypes"]) if r.get("archetypes") else ""
        props = ", ".join(r["prop_tags"]) if r.get("prop_tags") else ""
        lines.append(f"\n---\n## {r['ref_name']} — Games {int(r['games'])} | Confidence: {r['confidence_tag']}" )
        # ATS/Over + CI (hide CI if NaN)
        ats_ci = ""
        if not math.isnan(r.get("ats_pct_adj_lo", float("nan"))) and not math.isnan(r.get("ats_pct_adj_hi", float("nan"))):
            span = r["ats_pct_adj_hi"] - r["ats_pct_adj_lo"]
            wide = " [WIDE CI]" if span > CI_WIDE_SPAN_GT else ""
            ats_ci = f" [{100*r['ats_pct_adj_lo']:.1f}–{100*r['ats_pct_adj_hi']:.1f}]{wide}"
        over_ci = ""
        if not math.isnan(r.get("over_pct_adj_lo", float("nan"))) and not math.isnan(r.get("over_pct_adj_hi", float("nan"))):
            span = r["over_pct_adj_hi"] - r["over_pct_adj_lo"]
            wide = " [WIDE CI]" if span > CI_WIDE_SPAN_GT else ""
            over_ci = f" [{100*r['over_pct_adj_lo']:.1f}–{100*r['over_pct_adj_hi']:.1f}]{wide}"

        lines.append(
            f"- Home ATS {100*r['ats_pct_adj']:.1f}%{ats_ci} (n={int(r['ats_decisions'])}) | "
            f"Over {100*r['over_pct_adj']:.1f}%{over_ci} (n={int(r['ou_decisions'])})"
        )
        lines.append(f"- Flags/Gm {r['pens_per_game']:.2f} | Home−Away pen cnt diff {r['pen_cnt_diff_avg']:+.2f} | yds diff {r['pen_yds_diff_avg']:+.2f}")
        impact_bits = []
        if not np.isnan(r.get("pen_epa_forced_pg", np.nan)): impact_bits.append(f"pen EPA {r['pen_epa_forced_pg']:+.2f}/gm")
        if not np.isnan(r.get("q4_flags_per100_pg", np.nan)): impact_bits.append(f"Q4 flags/100 {r['q4_flags_per100_pg']:.2f}")
        if not np.isnan(r.get("rz_flag_fd_rate_pg", np.nan)): impact_bits.append(f"RZ flag FD% {100*r['rz_flag_fd_rate_pg']:.1f}%")
        if impact_bits:
            lines.append("- Impact: " + " | ".join(impact_bits))
        if arch:
            lines.append(f"- Archetype: {arch}")
        # implied total uplift
        uplift = r.get("implied_total_uplift", 0.0)
        if abs(uplift) >= SHOW_MIN["uplift_abs_min_pts"]:
            lines.append(f"- Implied total uplift: {uplift:+.1f} pts")
        # props
        if props:
            lines.append(f"- Prop hooks: {props}")
        # tax
        tax = r.get("tax_sheet_top", "")
        ben = r.get("benefit_sheet_top","")
        tbits = []
        if isinstance(tax, str) and tax.strip():
            tbits.append("Tax " + tax)
        if isinstance(ben, str) and ben.strip():
            tbits.append("Benefit " + ben)
        if tbits:
            lines.append("- " + " | ".join(tbits))

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Wrote {out_path}")

def export_preview(assign_df: pd.DataFrame, ref_summary: pd.DataFrame, baselines: Dict,
                   out_md: str, out_json: str, week_label: Optional[int]):
    idx = ref_summary.set_index(ref_summary["ref_name"].apply(_ref_key))
    lines = ["# 🚨 Sunday Whistle Watch\n"]
    payload = {"week": week_label, "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(), "matchups": []}
    missing = set()

    for _, r in assign_df.iterrows():
        key = _ref_key(r.get("Referee",""))
        if key not in idx.index:
            missing.add(str(r.get("Referee","")).strip())
            continue
        m = idx.loc[key]

        def _clean_str(x: str) -> str:
            x = str(x).strip()
            return "" if x.lower() in {"nan","none","nat","tbd"} else x

        bits = []
        ko = _clean_str(r.get("kickoff_et",""))
        net = _clean_str(r.get("network",""))
        if ko: bits.append(ko)
        if net: bits.append(net)
        meta_line = " ".join(bits)

        # archetype + confidence
        arche = ", ".join(m["archetypes"]) if isinstance(m["archetypes"], list) and m["archetypes"] else None
        conf = str(m.get("confidence_tag","")).strip()
        lean_label = arche if arche else None

        # CIs formatting (hidden if NaN)
        def fmt_ci(lo, hi, n):
            if (lo is None) or (hi is None) or math.isnan(lo) or math.isnan(hi) or int(n) < CI_HIDE_IF_DECISIONS_LT:
                return None
            span = hi - lo
            wide = " [WIDE CI]" if span > CI_WIDE_SPAN_GT else ""
            return f"[{100*lo:.1f}–{100*hi:.1f}] (n={int(n)}){wide}"

        ats_ci = fmt_ci(m.get("ats_pct_adj_lo"), m.get("ats_pct_adj_hi"), m.get("ats_decisions"))
        over_ci = fmt_ci(m.get("over_pct_adj_lo"), m.get("over_pct_adj_hi"), m.get("ou_decisions"))

        # implied uplift if meaningful
        uplift = m.get("implied_total_uplift", 0.0)
        uplift_str = f"{uplift:+.1f} pts" if abs(uplift) >= SHOW_MIN["uplift_abs_min_pts"] else ""

        # prop hooks (max 2) with parenthetical cause
        tag_explanations = []
        tags = m.get("prop_tags", [])
        if isinstance(tags, list):
            for t in tags[:2]:
                if t == "QB_PASS_YDS_UP":
                    tag_explanations.append(f"QB_PASS_YDS_UP (QB protect {m.get('qb_protect_pg',0):.1f}/gm)")
                elif t == "WR_PEN_BOOST":
                    tag_explanations.append(f"WR_PEN_BOOST (+{m.get('free_pass_yards_pg',0):.1f} free pass yds/gm)")
                elif t == "RB_ATT_UP_UNDER_YDS":
                    tag_explanations.append(f"RB_ATT_UP_UNDER_YDS (drive killers {m.get('drive_killers_pg',0):.1f}/gm)")
                elif t == "UNDER_PACE_LEAN":
                    tag_explanations.append(f"UNDER_PACE_LEAN (tempo drags {m.get('tempo_drags_pg',0):.1f}/gm)")
                elif t == "3RD_DOWN_EXTENSIONS":
                    tag_explanations.append(f"3RD_DOWN_EXTENSIONS (3rd-down FDs {m.get('bailout_3rd_fd_per_game',0):.1f}/gm)")

        # tax lines, hidden if empty
        tax = m.get("tax_sheet_top","")
        ben = m.get("benefit_sheet_top","")
        tax_line = ", ".join([x for x in [tax, ben] if isinstance(x,str) and x.strip()])

        # MD line
        head = f"**{r['away_team']} @ {r['home_team']}** — {r.get('Referee','')}"
        if meta_line: head += f"  \n{meta_line}"
        lines.append(head + "\n")
        lines.append(
            f"Flags/Gm {m['pens_per_game']:.2f}"
            f" | Home ATS {100*m['ats_pct_adj']:.1f}%{(' ' + ats_ci) if ats_ci else ''}"
            f" | Over {100*m['over_pct_adj']:.1f}%{(' ' + over_ci) if over_ci else ''}"
        )
        # Split leans and confidence into separate lines
        lean_bits = []
        if isinstance(m.get("archetypes",""), list) and m["archetypes"]:
            lean_bits.extend(m["archetypes"])
        leans_str = "; ".join(lean_bits) if lean_bits else "—"
        
        conf_str = str(m.get("confidence_tag","")).upper() or "LOW"
        
        lines.append(f"Leans: {leans_str} | Confidence: {conf_str}")
        if uplift_str:
            lines.append(f"Implied total: {uplift_str}")
        if tag_explanations:
            lines.append("Props: " + ", ".join(tag_explanations))
        if tax_line:
            lines.append("Tax: " + tax_line)
        lines.append("")

        # JSON payload - omit null fields
        entry = {
            "matchup": f"{r['away_team']} @ {r['home_team']}",
            "ref": r.get("Referee",""),
            "flags_pg": round(float(m["pens_per_game"]),2),
            "home_ats": f"{100*float(m['ats_pct_adj']):.1f}%",
            "over_rate": f"{100*float(m['over_pct_adj']):.1f}%",
            "impact": {
                "pen_epa_pg": round(float(m.get("pen_epa_forced_pg",0)),2),
                "q4_flags_per100": round(float(m.get("q4_flags_per100_pg",0)),2),
                "rz_flag_fd_rate": round(float(m.get("rz_flag_fd_rate_pg",0)),3)
            }
        }
        
        # Add optional fields only if they have meaningful values
        if ko: entry["kickoff_et"] = ko
        if r.get("venue"): entry["venue"] = r.get("venue")
        if net: entry["network"] = net
        if arche: entry["archetype"] = arche
        if conf: entry["confidence"] = conf
        if ats_ci: entry["home_ats_ci"] = ats_ci
        if over_ci: entry["over_ci"] = over_ci
        if uplift_str: entry["impact"]["implied_total_uplift"] = round(float(uplift),1)
        if tax_line: entry["tax"] = tax_line
        if tag_explanations: entry["props"] = tag_explanations
        payload["matchups"].append(entry)

    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✓ Wrote {out_md} and {out_json}")
    if missing:
        print("Ref names with no metric match (alias needed): " + ", ".join(sorted(missing)))

# =========================
# Main orchestration
# =========================

def main():
    ap = argparse.ArgumentParser(description="Complete NFL Ref Engine with assignments and preview export.")
    ap.add_argument("--season", type=int, default=dt.datetime.now().year, help="Target season for assignments fetch")
    ap.add_argument("--week", default="latest", help="'latest' or week number")
    ap.add_argument("--fetch-assignments", action="store_true", help="Fetch assignments from Football Zebras")
    ap.add_argument("--assign-csv", default=None, help="Read assignments from this CSV instead of fetching")
    ap.add_argument("--assign-out", default=None, help="Output CSV for fetched assignments (default: outputs/assignments_week*.csv)")
    ap.add_argument("--assign-min-rows", type=int, default=8, help="Abort fetch if fewer rows parsed")
    args = ap.parse_args()

    # Load data
    years = load_years()
    schedules, pbp, officials = load_frames(years)

    # Build per-game + per-ref metrics
    ref_games, type_counts = build_ref_game_table(schedules, pbp, officials)
    baselines = league_baselines(ref_games, type_counts)
    ref_summary = aggregate_refs(ref_games, type_counts, baselines)

    # Export paid cards
    export_ref_cards_md(ref_summary, baselines, out_path=os.path.join(OUTDIR, "referee_report_cards.md"))

    # Assignments: either fetch or read CSV
    assign_df = None
    week_label = None
    if args.assign_csv:
        try:
            a = pd.read_csv(args.assign_csv)
            # normalize team codes and header casing
            a["home_team"] = a["home_team"].apply(norm_team_token)
            a["away_team"] = a["away_team"].apply(norm_team_token)
            # tolerate either Referee or referee
            if "Referee" not in a.columns and "referee" in a.columns:
                a["Referee"] = a["referee"]
            assign_df = a
            # try to infer week
            try:
                wk = pd.to_numeric(a.get("week"), errors="coerce")
                if wk.notna().any():
                    week_label = int(wk.max())
            except Exception:
                pass
        except Exception as e:
            print(f"Failed to read --assign-csv: {e}")

    if args.fetch_assignments and assign_df is None:
        url = discover_week_url(args.season, args.week)
        if not url:
            print("Could not discover a Week link from Assignments index.")
            sys.exit(2)
        rows, season_guess, wk_guess = parse_assignments(url, expected_week=None if args.week == "latest" else int(args.week))
        print(f"Parsed {len(rows)} rows from {url}")
        if len(rows) < int(args.assign_min_rows):
            print(f"Only {len(rows)} rows parsed. Site likely hasn't posted the full blocks yet.")
            sys.exit(3)
        # write CSV
        wklabel = "latest" if args.week == "latest" else str(int(args.week))
        out_csv = args.assign_out or os.path.join(OUTDIR, f"assignments_week{wklabel}.csv")
        write_assignments_csv(rows, out_csv)
        print(f"✓ Wrote {out_csv} with {len(rows)} rows")
        a = pd.read_csv(out_csv)
        a["home_team"] = a["home_team"].apply(norm_team_token)
        a["away_team"] = a["away_team"].apply(norm_team_token)
        assign_df = a
        week_label = wk_guess

    # Export preview if we have assignments
    if assign_df is not None and not assign_df.empty:
        export_preview(
            assign_df=assign_df,
            ref_summary=ref_summary,
            baselines=baselines,
            out_md=os.path.join(OUTDIR, "flag_files_preview.md"),
            out_json=os.path.join(OUTDIR, "flag_files_preview.json"),
            week_label=week_label
        )
    else:
        print("No assignments provided/fetched. Preview skipped cleanly.")

if __name__ == "__main__":
    main()
