#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nfl_pbp_model_api.py
Author: FoxEdge Build

# Slightly wider tails
python nfl_pbp_model.py predict --season 2025 --week 14 --n_sims 30000 --lean_features \
  --alpha_grid "2,3,5,8" --sim_sd_scale 1.12

# Slightly narrower if probabilities look mushy
python nfl_pbp_model.py predict --season 2025 --week 14 --n_sims 30000 --lean_features \
  --alpha_grid "2,3,5,8" --sim_sd_scale 1.05

End-to-end NFL projections pulling data directly from nfl_data_py.
No CSVs. It fetches pbp, schedules, and betting lines (when available), then:
- Builds opponent-adjusted rolling features
- Trains ridge point models + calibrated cover/over heads
- Simulates bivariate scores for probabilities
- Runs rolling backtests
- Exports weekly picks
"""

import argparse
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import math

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from numpy.random import default_rng

# --- Canonical team mapping for reliable merges (abbr as canonical) ---
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode, urlparse, parse_qs

TEAM_MAP = {
    'ARI':'ARI','Arizona Cardinals':'ARI','Cardinals':'ARI',
    'ATL':'ATL','Atlanta Falcons':'ATL','Falcons':'ATL',
    'BAL':'BAL','Baltimore Ravens':'BAL','Ravens':'BAL',
    'BUF':'BUF','Buffalo Bills':'BUF','Bills':'BUF',
    'CAR':'CAR','Carolina Panthers':'CAR','Panthers':'CAR',
    'CHI':'CHI','Chicago Bears':'CHI','Bears':'CHI',
    'CIN':'CIN','Cincinnati Bengals':'CIN','Bengals':'CIN',
    'CLE':'CLE','Cleveland Browns':'CLE','Browns':'CLE',
    'DAL':'DAL','Dallas Cowboys':'DAL','Cowboys':'DAL',
    'DEN':'DEN','Denver Broncos':'DEN','Broncos':'DEN',
    'DET':'DET','Detroit Lions':'DET','Lions':'DET',
    'GB':'GB','GNB':'GB','Green Bay Packers':'GB','Packers':'GB',
    'HOU':'HOU','Houston Texans':'HOU','Texans':'HOU',
    'IND':'IND','Indianapolis Colts':'IND','Colts':'IND',
    'JAX':'JAX','JAC':'JAX','Jacksonville Jaguars':'JAX','Jaguars':'JAX',
    'KC':'KC','KAN':'KC','Kansas City Chiefs':'KC','Chiefs':'KC',
    'LV':'LV','LVR':'LV','Las Vegas Raiders':'LV','Raiders':'LV',
    'LAC':'LAC','Los Angeles Chargers':'LAC','Chargers':'LAC',
    'LAR':'LAR','Los Angeles Rams':'LAR','Rams':'LAR',
    'MIA':'MIA','Miami Dolphins':'MIA','Dolphins':'MIA','MIA Dolphins':'MIA',
    'MIN':'MIN','Minnesota Vikings':'MIN','Vikings':'MIN',
    'NE':'NE','NWE':'NE','New England Patriots':'NE','Patriots':'NE',
    'NO':'NO','NOR':'NO','New Orleans Saints':'NO','Saints':'NO',
    'NYG':'NYG','New York Giants':'NYG','Giants':'NYG','NY Giants':'NYG',
    'NYJ':'NYJ','New York Jets':'NYJ','Jets':'NYJ','NY Jets':'NYJ',
    'PHI':'PHI','Philadelphia Eagles':'PHI','Eagles':'PHI',
    'PIT':'PIT','Pittsburgh Steelers':'PIT','Steelers':'PIT',
    'SEA':'SEA','Seattle Seahawks':'SEA','Seahawks':'SEA',
    'SF':'SF','SFO':'SF','San Francisco 49ers':'SF','49ers':'SF',
    'TB':'TB','TAM':'TB','Tampa Bay Buccaneers':'TB','Buccaneers':'TB','Bucs':'TB',
    'TEN':'TEN','Tennessee Titans':'TEN','Titans':'TEN',
    'WAS':'WAS','Washington Commanders':'WAS','Commanders':'WAS','Washington Football Team':'WAS', 'LA Rams':'LAR','Los Angeles Rams':'LAR',
'LA Chargers':'LAC','Los Angeles Chargers':'LAC',
'Washington':'WAS','Washington Redskins':'WAS','WAS Football Team':'WAS',
'New Orleans':'NO','New England':'NE','San Francisco':'SF','Tampa Bay':'TB',
'Green Bay':'GB','Kansas City':'KC','Las Vegas':'LV','LA':'LAR'
}

def canon_team(x: str) -> str:
    s = str(x).strip()
    return TEAM_MAP.get(s, s)

try:
    import nfl_data_py as nfl
except Exception as e:
    raise SystemExit("Install nfl_data_py first: pip install nfl_data_py\n" + str(e))

############################
# Fetchers (nfl_data_py)
############################

def fetch_pbp(years: List[int], downcast: bool = True) -> pd.DataFrame:
    pbp = nfl.import_pbp_data(years, downcast=downcast, cache=False)
    # standardize a few columns we'll rely on (create if missing)
    for col, default in [('qb_dropback', 0), ('sack', 0), ('penalty', 0), ('air_yards', np.nan), ('yac_yards', np.nan), ('pass', 0), ('rush', 0)]:
        if col not in pbp.columns:
            pbp[col] = default
    pbp['qb_dropback'] = pd.to_numeric(pbp['qb_dropback'], errors='coerce').fillna(0).astype(int)
    pbp['sack'] = pd.to_numeric(pbp['sack'], errors='coerce').fillna(0).astype(int)
    pbp['penalty'] = pd.to_numeric(pbp['penalty'], errors='coerce').fillna(0).astype(int)
    return pbp

def fetch_games(years: List[int]) -> pd.DataFrame:
    g = nfl.import_schedules(years)
    # Some nfl_data_py versions use 'season_type', others use 'game_type'. Handle both.
    season_col = 'season_type' if 'season_type' in g.columns else ('game_type' if 'game_type' in g.columns else None)
    if season_col is not None:
        g = g[g[season_col].isin(['REG', 'POST'])].copy()
    # essential columns (keep only those that exist)
    keep = ['game_id','season','week','home_team','away_team','home_score','away_score','game_type','season_type']
    keep = [c for c in keep if c in g.columns]
    g = g[keep].rename(columns={'home_score':'home_points','away_score':'away_points'})
    # Ensure required score columns exist, even if missing in this version
    if 'home_points' not in g.columns:
        g['home_points'] = np.nan
    if 'away_points' not in g.columns:
        g['away_points'] = np.nan
    return g

def fetch_lines(years: List[int]) -> Optional[pd.DataFrame]:
    lines = None
    # nfl_data_py added betting lines in 2023+. Try canonical function names.
    for fn in ['import_betting_lines','import_betting','import_betting_data']:
        if hasattr(nfl, fn):
            try:
                lines = getattr(nfl, fn)(years)
                break
            except Exception:
                lines = None
    if lines is None or lines.empty:
        return None
    # Normalize: choose close if present else last pregame
    cols = lines.columns.str.lower()
    lines.columns = cols
    # Heuristics to pick a line
    # Expect columns like: 'spread_close', 'total_close', or 'spread', 'total'
    guess_cols = {
        'game_id':'game_id',
        'close_spread': None,
        'close_total': None,
        'close_home_ml': None
    }
    # Map plausible column names
    if 'spread_close' in cols:
        guess_cols['close_spread'] = 'spread_close'
    elif 'closing_spread' in cols:
        guess_cols['close_spread'] = 'closing_spread'
    elif 'spread' in cols:
        guess_cols['close_spread'] = 'spread'
    if 'total_close' in cols:
        guess_cols['close_total'] = 'total_close'
    elif 'closing_total' in cols:
        guess_cols['close_total'] = 'closing_total'
    elif 'total' in cols:
        guess_cols['close_total'] = 'total'
    if 'ml_home_close' in cols:
        guess_cols['close_home_ml'] = 'ml_home_close'
    elif 'home_moneyline_close' in cols:
        guess_cols['close_home_ml'] = 'home_moneyline_close'
    elif 'home_moneyline' in cols:
        guess_cols['close_home_ml'] = 'home_moneyline'
    use_cols = [c for c in guess_cols.values() if c is not None] + ['game_id']
    L = lines[use_cols].drop_duplicates('game_id').rename(columns={
        guess_cols['close_spread']: 'close_spread' if guess_cols['close_spread'] else 'close_spread',
        guess_cols['close_total']: 'close_total' if guess_cols['close_total'] else 'close_total',
        guess_cols['close_home_ml']: 'close_home_ml' if guess_cols['close_home_ml'] else 'close_home_ml',
    })
    # Fill safe anchors if missing
    if 'close_spread' not in L.columns: L['close_spread'] = 0.0
    if 'close_total' not in L.columns: L['close_total'] = 44.0
    if 'close_home_ml' not in L.columns: L['close_home_ml'] = -110
    return L[['game_id','close_spread','close_total','close_home_ml']].copy()

# --- New: fetch weekly player-level data ---
def fetch_weekly(years: List[int]) -> pd.DataFrame:
    """Pull weekly player-level data from nfl_data_py and return as-is."""
    try:
        if hasattr(nfl, 'import_weekly_data'):
            return nfl.import_weekly_data(years)
        # legacy aliases guardrail
        if hasattr(nfl, 'import_weekly_player_data'):
            return nfl.import_weekly_player_data(years)
    except Exception:
        pass
    return pd.DataFrame()

# --- New: build team-week features from weekly player data ---
def build_weekly_team_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weekly player stats into compact team-week signals helpful for totals.
    - qb_dakota: starting QB's DAKOTA metric (proxy for passing efficiency)
    - qb_passing_epa: starting QB passing EPA (per game level)
    - wrte_hhi: target concentration among WR/TE (Herfindahl index)
    - top2_wopr_mean: mean WOPR of top-2 targeted receivers
    - rb1_carry_share: lead back's carry share
    """
    if weekly is None or weekly.empty:
        return pd.DataFrame(columns=['season','week','team','qb_dakota','qb_passing_epa','wrte_hhi','top2_wopr_mean','rb1_carry_share'])

    w = weekly.copy()
    # normalize column names
    w.columns = [str(c) for c in w.columns]
    team_col = 'recent_team' if 'recent_team' in w.columns else ('team' if 'team' in w.columns else None)
    if team_col is None:
        return pd.DataFrame(columns=['season','week','team','qb_dakota','qb_passing_epa','wrte_hhi','top2_wopr_mean','rb1_carry_share'])
    w = w.rename(columns={team_col: 'team'})

    # ensure core numeric columns exist
    for c in ['attempts','targets','carries','dakota','wopr','passing_epa']:
        if c not in w.columns:
            w[c] = np.nan

    # Identify starting QB per team-week as the QB with max attempts
    qb = w[w.get('position', w.get('position_group','')).isin(['QB']) | (w.get('position_group','') == 'QB')].copy()
    qb['attempts'] = pd.to_numeric(qb['attempts'], errors='coerce').fillna(0)
    qb_rank = qb.sort_values(['season','week','team','attempts'], ascending=[True,True,True,False])
    qb_top = qb_rank.groupby(['season','week','team']).head(1)
    qb_feat = qb_top.groupby(['season','week','team']).agg(
        qb_dakota=('dakota','mean'),
        qb_passing_epa=('passing_epa','mean')
    ).reset_index()

    # Target concentration among WR/TE
    wrte = w[w.get('position_group','').isin(['WR','TE'])].copy()
    wrte['targets'] = pd.to_numeric(wrte['targets'], errors='coerce').fillna(0)
    tgt_team = wrte.groupby(['season','week','team'])['targets'].sum().rename('tgt_total').reset_index()
    wrte = wrte.merge(tgt_team, on=['season','week','team'], how='left')
    wrte['share'] = _safe_div(wrte['targets'], wrte['tgt_total'].replace(0, np.nan))
    wrte['share_sq'] = wrte['share']**2
    hhi = wrte.groupby(['season','week','team'])['share_sq'].sum().rename('wrte_hhi').reset_index()

    # Top-2 WOPR mean for receivers (if available)
    if 'wopr' in w.columns:
        rec = w[w.get('position_group','').isin(['WR','TE'])].copy()
        rec['wopr'] = pd.to_numeric(rec['wopr'], errors='coerce')
        rec_rank = rec.sort_values(['season','week','team','wopr'], ascending=[True,True,True,False])
        top2 = rec_rank.groupby(['season','week','team']).head(2)
        top2_wopr = top2.groupby(['season','week','team'])['wopr'].mean().rename('top2_wopr_mean').reset_index()
    else:
        top2_wopr = pd.DataFrame(columns=['season','week','team','top2_wopr_mean'])

    # Lead-back carry share
    rb = w[w.get('position_group','') == 'RB'].copy()
    rb['carries'] = pd.to_numeric(rb['carries'], errors='coerce').fillna(0)
    rb_team = rb.groupby(['season','week','team'])['carries'].sum().rename('rb_carries_total').reset_index()
    rb = rb.merge(rb_team, on=['season','week','team'], how='left')
    rb['carry_share'] = _safe_div(rb['carries'], rb['rb_carries_total'].replace(0, np.nan))
    rb1 = rb.sort_values(['season','week','team','carry_share'], ascending=[True,True,True,False]).groupby(['season','week','team']).head(1)
    rb1_share = rb1[['season','week','team','carry_share']].rename(columns={'carry_share':'rb1_carry_share'})

    out = qb_feat.merge(hhi, on=['season','week','team'], how='outer')\
                  .merge(top2_wopr, on=['season','week','team'], how='outer')\
                  .merge(rb1_share, on=['season','week','team'], how='outer')
    return out

############################
# Feature engineering
############################

def _safe_div(a, b):
    """
    Safe division that works for scalars and arrays.
    Returns 0.0 where the result is non-finite.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.true_divide(a, b)
    # If scalar, return a clean float
    if np.isscalar(out) or (hasattr(out, 'shape') and out.shape == ()):  # 0-d array
        return float(out) if np.isfinite(out) else 0.0
    # If array-like, zero-out non-finite entries
    out = np.asarray(out)
    out[~np.isfinite(out)] = 0.0
    return out

def build_team_week_features(pbp: pd.DataFrame, games: pd.DataFrame, roll_weeks: int = 6, use_weekly: bool = False) -> pd.DataFrame:
    df = pbp.copy()
    # Optionally merge weekly player-derived aggregates
    wk_feat = pd.DataFrame()
    if use_weekly:
        weekly = fetch_weekly(sorted(df['season'].dropna().astype(int).unique().tolist()))
        wk_feat = build_weekly_team_features(weekly)
    df = df[df['posteam'].notna() & df['defteam'].notna()]
    df['is_pass'] = ((df.get('pass', 0).fillna(0) == 1) | (df.get('qb_dropback', 0).fillna(0) == 1)).astype(int)
    df['is_rush'] = (df.get('rush', 0).fillna(0) == 1).astype(int)
    df['early_down'] = df['down'].isin([1,2])
    df['success'] = (df['epa'].fillna(0) > 0).astype(int)
    df['epna'] = df['epa'].fillna(0.0)
    # Neutral situation and seconds to snap
    df['score_diff'] = (df.get('home_score', 0).fillna(0) - df.get('away_score', 0).fillna(0))
    mask_away = (df['posteam'] == df.get('away_team')) if 'away_team' in df.columns else False
    if isinstance(mask_away, pd.Series):
        df.loc[mask_away, 'score_diff'] = -df.loc[mask_away, 'score_diff']
    df['neutral'] = df['score_diff'].between(-7, 7) & df['qtr'].isin([1, 2, 3])
    # Exclude obvious non-offense snaps from neutral pace
    exclude_flags = (
        (df.get('special_teams_play', 0) == 1) |
        (df.get('qb_spike', 0) == 1) |
        (df.get('qb_kneel', 0) == 1) |
        (df.get('aborted_play', 0) == 1)
    )
    df.loc[exclude_flags, 'neutral'] = False

    if 'game_seconds_remaining' in df.columns:
        df = df.sort_values(['game_id', 'play_id'])
        df['gsr_next'] = df.groupby('game_id')['game_seconds_remaining'].shift(1)
        sec_elapsed = (df['gsr_next'] - df['game_seconds_remaining']).abs()
        df['sec_to_snap'] = sec_elapsed.clip(lower=0, upper=120).fillna(0)
    else:
        df['sec_to_snap'] = np.nan

    # Use provided posteam-relative score differential if available
    if 'score_differential' in df.columns:
        df['neutral'] = df['score_differential'].between(-7, 7) & df['qtr'].isin([1.0, 2.0, 3.0])

    # Robust early-down indicator with float down values
    df['down_int'] = pd.to_numeric(df['down'], errors='coerce').astype('Int64')
    df['early_down'] = df['down_int'].isin([1, 2])

    # PROE and expected pass rate on neutral downs
    if {'xpass','pass_oe','posteam'}.issubset(df.columns):
        neutral_df = df[df['neutral']].copy()
        proe = neutral_df.groupby(['season','week','posteam']).agg(
            proe=('pass_oe','mean'),
            xpass_mean=('xpass','mean')
        ).reset_index().rename(columns={'posteam':'team'})
    else:
        proe = pd.DataFrame(columns=['season','week','team','proe','xpass_mean'])

    # Series conversion rate: take last play of each series
    if {'series','series_success','posteam'}.issubset(df.columns):
        ser_last = df.dropna(subset=['series']).sort_values(['game_id','series','play_id']).groupby(['game_id','posteam','series']).tail(1)
        ser = ser_last.groupby(['season','week','posteam']).agg(series_fd_rate=('series_success','mean')).reset_index().rename(columns={'posteam':'team'})
    else:
        ser = pd.DataFrame(columns=['season','week','team','series_fd_rate'])

    # Starting field position using drive start play yardline_100
    if {'drive_play_id_started','yardline_100','posteam'}.issubset(df.columns):
        starts = df.dropna(subset=['drive_play_id_started']).merge(
            df[['game_id','play_id','yardline_100','posteam']],
            left_on=['game_id','drive_play_id_started'], right_on=['game_id','play_id'], how='left', suffixes=('','_start')
        )
        sfp = starts.groupby(['season','week','posteam']).agg(start_yardline_100=('yardline_100_start','mean')).reset_index().rename(columns={'posteam':'team'})
    else:
        sfp = pd.DataFrame(columns=['season','week','team','start_yardline_100'])

    # Weather summaries per team-week from PBP
    wx = df.groupby(['season','week','posteam']).agg(
        temp_mean=('temp','mean'),
        wind_mean=('wind','mean')
    ).reset_index().rename(columns={'posteam':'team'})

    pace = df[df['neutral']].groupby(['season', 'week', 'posteam']).agg(
        neutral_sec_per_play=('sec_to_snap', 'median')
    ).reset_index().rename(columns={'posteam': 'team'})

    # Optional columns that may exist in nflfastR
    if 'first_down' not in df.columns:
        df['first_down'] = 0
    if 'no_huddle' not in df.columns:
        df['no_huddle'] = 0

    # Drives and plays-per-drive per team-week
    if {'game_id','drive','posteam'}.issubset(df.columns):
        drv = df.groupby(['season','week','posteam','game_id']).agg(
            drives=('drive','nunique'),
            plays=('play_id','count')
        ).reset_index()
        # ensure numeric dtypes
        drv['drives'] = pd.to_numeric(drv['drives'], errors='coerce').fillna(0).astype(float)
        drv['plays'] = pd.to_numeric(drv['plays'], errors='coerce').fillna(0).astype(float)
        # aggregate to team-week
        drv_team = drv.groupby(['season','week','posteam']).agg(
            drives_sum=('drives','sum'),
            plays_sum=('plays','sum'),
            games=('game_id','nunique')
        ).reset_index()
        drv_team['drives_mean'] = _safe_div(drv_team['drives_sum'], drv_team['games'])
        drv_team['ppd'] = _safe_div(drv_team['plays_sum'], drv_team['drives_sum'])
        drv_team = drv_team.rename(columns={'posteam':'team'})[['season','week','team','drives_mean','ppd']]
    else:
        drv_team = pd.DataFrame(columns=['season','week','team','drives_mean','ppd'])

    # First-down rate and no-huddle rate per team-week
    fd_nh = df.groupby(['season','week','posteam']).agg(
        first_down_rate=('first_down', lambda x: _safe_div(x.sum(), len(x))),
        no_huddle_rate=('no_huddle', lambda x: _safe_div(x.sum(), len(x)))
    ).reset_index().rename(columns={'posteam':'team'})

    # Off aggregates
    agg = df.groupby(['season','week','posteam']).agg(
        plays=('play_id','count'),
        epa_mean=('epna','mean'),
        sr=('success','mean'),
        ed_epa=('epna', lambda x: x[df.loc[x.index, 'early_down']].mean() if (df.loc[x.index,'early_down']).any() else 0.0),
        ed_sr=('success', lambda x: x[df.loc[x.index, 'early_down']].mean() if (df.loc[x.index,'early_down']).any() else 0.0),
        pass_rate=('is_pass','mean'),
        rush_rate=('is_rush','mean'),
        sack_rate=('sack', lambda x: _safe_div((df.loc[x.index, 'sack']).sum() if 'sack' in df.columns else 0, (df.loc[x.index, 'qb_dropback']).sum() if 'qb_dropback' in df.columns else 1)),
        pen_rate=('penalty', lambda x: _safe_div((df.loc[x.index,'penalty']).sum() if 'penalty' in df.columns else 0, len(x))),
        air_mean=('air_yards','mean'),
        yac_mean=('yac_yards','mean'),
    ).reset_index().rename(columns={'posteam':'team'})
    # Def aggregates
    d_agg = df.groupby(['season','week','defteam']).agg(
        def_epa=('epna','mean'),
        def_sr=('success','mean'),
        def_ed_epa=('epna', lambda x: x[df.loc[x.index, 'early_down']].mean() if (df.loc[x.index,'early_down']).any() else 0.0),
        def_ed_sr=('success', lambda x: x[df.loc[x.index, 'early_down']].mean() if (df.loc[x.index,'early_down']).any() else 0.0),
        def_pass_rate=('is_pass','mean'),
        def_rush_rate=('is_rush','mean'),
        def_pen_rate=('penalty', lambda x: _safe_div((df.loc[x.index,'penalty']).sum() if 'penalty' in df.columns else 0, len(x))),
    ).reset_index().rename(columns={'defteam':'team'})

    feat = agg.merge(d_agg, on=['season','week','team'], how='outer').fillna(0.0)
    if use_weekly and wk_feat is not None and not wk_feat.empty:
        feat = feat.merge(wk_feat, on=['season','week','team'], how='left')

    # Merge drive/pace features
    if not drv_team.empty:
        feat = feat.merge(drv_team, on=['season','week','team'], how='left')
    feat = feat.merge(fd_nh, on=['season','week','team'], how='left')
    feat = feat.merge(pace, on=['season','week','team'], how='left')
    if not proe.empty:
        feat = feat.merge(proe, on=['season','week','team'], how='left')
    if not ser.empty:
        feat = feat.merge(ser, on=['season','week','team'], how='left')
    if not sfp.empty:
        feat = feat.merge(sfp, on=['season','week','team'], how='left')
    if not wx.empty:
        feat = feat.merge(wx, on=['season','week','team'], how='left')

    # Opponent mapping
    gm = games[['game_id','season','week','home_team','away_team']].drop_duplicates()
    gm_long = pd.concat([
        gm[['season','week','home_team','away_team']].rename(columns={'home_team':'team','away_team':'opp_team'}),
        gm[['season','week','away_team','home_team']].rename(columns={'away_team':'team','home_team':'opp_team'}),
    ], ignore_index=True)

    feat = feat.merge(gm_long, on=['season','week','team'], how='left')

    # Opponent-adjust core cols
    def _opp_adjust(df0, col):
        out = df0.copy()
        opp_means = out.groupby(['season','week','opp_team'])[col].mean().rename('opp_mean')
        out = out.merge(opp_means, left_on=['season','week','team'], right_on=['season','week','opp_team'], how='left')
        out[f'{col}_oa'] = out[col] - out['opp_mean'].fillna(0.0)
        out.drop(columns=['opp_mean'], inplace=True, errors='ignore')
        return out

    for c in ['epa_mean','sr','ed_epa','ed_sr']:
        feat = _opp_adjust(feat, c)

    # Rolling and season priors
    feat = feat.sort_values(['team','season','week'])
    roll_cols = ['epa_mean','sr','ed_epa','ed_sr','pass_rate','rush_rate','sack_rate','pen_rate',
                 'def_epa','def_sr','def_ed_epa','def_ed_sr','def_pass_rate','def_rush_rate','def_pen_rate',
                 'air_mean','yac_mean','epa_mean_oa','sr_oa','ed_epa_oa','ed_sr_oa',
                 'drives_mean','ppd','first_down_rate','no_huddle_rate','neutral_sec_per_play',
                 'proe','xpass_mean','series_fd_rate','start_yardline_100','temp_mean','wind_mean']
    weekly_cols = ['qb_dakota','qb_passing_epa','wrte_hhi','top2_wopr_mean','rb1_carry_share']
    if use_weekly and (wk_feat is not None) and (not wk_feat.empty):
        roll_cols += weekly_cols
    for c in roll_cols:
        feat[f'{c}_r{roll_weeks}'] = feat.groupby('team')[c].transform(lambda s: s.rolling(roll_weeks, min_periods=1).mean())
    season_means = feat.groupby(['season','team'])[roll_cols].mean().rename(columns=lambda x: f'{x}_season_prior').reset_index()
    feat = feat.merge(season_means, on=['season','team'], how='left')
    for c in roll_cols:
        feat[f'{c}_blend'] = 0.6*feat[f'{c}_r{roll_weeks}'] + 0.4*feat[f'{c}_season_prior']

    keep = ['season','week','team','opp_team'] + [f'{c}_blend' for c in roll_cols]
    feat = feat[keep].fillna(0.0)
    return feat

############################
# Modeling utils
############################

# replace your Models dataclass with this
from dataclasses import dataclass

@dataclass
class Models:
    home_points: Pipeline
    away_points: Pipeline
    cover_cls: CalibratedClassifierCV
    over_cls: CalibratedClassifierCV
    feat_cols: list
    sd_home: float
    sd_away: float
    rho: float
    sd_margin: float
    sd_total: float
    pace_q: Tuple[float, float]
    proe_q: Tuple[float, float]
    sd_total_grid: dict
    iso_cover: Optional[IsotonicRegression] = None
    iso_over: Optional[IsotonicRegression] = None
    sim_sd_scale: float = 1.0

def _merge_features_asof(features: pd.DataFrame, games: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each game row (season, week, home_id/away_id), pull the most recent available
    team-week features where feature.week <= game.week. Uses merge_asof per team.
    Returns (f_home_asof, f_away_asof) with columns aligned to f_home/f_away style.
    """
    f = features.copy()
    f['team_id'] = f['team'].map(canon_team).astype(str)
    # Ensure integer dtypes line up exactly for merge_asof (int64 on both sides)
    f['season'] = pd.to_numeric(f['season'], errors='coerce').astype('Int64')
    f['week'] = pd.to_numeric(f['week'], errors='coerce').astype('Int64')
    f = f.dropna(subset=['season','week','team_id'])
    # Cast to concrete int64 for merge keys
    f['season'] = f['season'].astype('int64')
    f['week'] = f['week'].astype('int64')
    f = f.sort_values(['season','team_id','week'])

    g = games[['game_id','season','week','home_team','away_team']].copy()
    g['home_id'] = g['home_team'].map(canon_team).astype(str)
    g['away_id'] = g['away_team'].map(canon_team).astype(str)
    g['season'] = pd.to_numeric(g['season'], errors='coerce').astype('Int64').astype('int64')
    g['week'] = pd.to_numeric(g['week'], errors='coerce').astype('Int64').astype('int64')

    # Build home asof
    f_home = f.rename(columns=lambda c: f'h_{c}' if c not in ['season','week','team','team_id'] else c)
    # Ensure strict sorting and one row per (season, team_id, week) on the RIGHT side
    f_home = (f_home
              .sort_values(['season','team_id','week'], kind='mergesort')
              .drop_duplicates(['season','team_id','week'], keep='last'))

    gh = (g[['game_id','season','week','home_id']]
          .rename(columns={'home_id':'team_id'})
          .sort_values(['season','team_id','week'], kind='mergesort'))

    try:
        fh = pd.merge_asof(
            gh,
            f_home,
            by=['season','team_id'],
            left_on='week', right_on='week',
            direction='backward',
            allow_exact_matches=True
        )
    except Exception:
        # Manual groupwise as-of fallback
        # Build index per (season, team_id)
        fdict = {}
        for (s, t), d in f_home.groupby(['season','team_id'], sort=False):
            dd = d.sort_values('week')
            fdict[(int(s), str(t))] = (dd['week'].to_numpy(), dd)
        recs = []
        for _, row in gh.iterrows():
            key = (int(row['season']), str(row['team_id']))
            weeks_df = fdict.get(key)
            if weeks_df is None:
                recs.append({**row.to_dict()})
                continue
            wks, dfk = weeks_df
            # find rightmost week <= target
            import numpy as _np
            idx = _np.searchsorted(wks, int(row['week']), side='right') - 1
            if idx >= 0:
                joined = {**row.to_dict(), **dfk.iloc[idx].to_dict()}
                recs.append(joined)
            else:
                recs.append({**row.to_dict()})
        fh = pd.DataFrame.from_records(recs)

    # Build away asof
    f_away = f.rename(columns=lambda c: f'a_{c}' if c not in ['season','week','team','team_id'] else c)
    f_away = (f_away
              .sort_values(['season','team_id','week'], kind='mergesort')
              .drop_duplicates(['season','team_id','week'], keep='last'))

    ga = (g[['game_id','season','week','away_id']]
          .rename(columns={'away_id':'team_id'})
          .sort_values(['season','team_id','week'], kind='mergesort'))

    try:
        fa = pd.merge_asof(
            ga,
            f_away,
            by=['season','team_id'],
            left_on='week', right_on='week',
            direction='backward',
            allow_exact_matches=True
        )
    except Exception:
        fdict = {}
        for (s, t), d in f_away.groupby(['season','team_id'], sort=False):
            dd = d.sort_values('week')
            fdict[(int(s), str(t))] = (dd['week'].to_numpy(), dd)
        recs = []
        for _, row in ga.iterrows():
            key = (int(row['season']), str(row['team_id']))
            weeks_df = fdict.get(key)
            if weeks_df is None:
                recs.append({**row.to_dict()})
                continue
            wks, dfk = weeks_df
            import numpy as _np
            idx = _np.searchsorted(wks, int(row['week']), side='right') - 1
            if idx >= 0:
                joined = {**row.to_dict(), **dfk.iloc[idx].to_dict()}
                recs.append(joined)
            else:
                recs.append({**row.to_dict()})
        fa = pd.DataFrame.from_records(recs)

    # Restore keys
    fh = fh.rename(columns={'team_id':'home_id'})
    fa = fa.rename(columns={'team_id':'away_id'})
    # Defensive: coerce any lingering nullable Ints back to plain ints for merge keys
    for df_ in (fh, fa):
        for col in ['season','week']:
            if col in df_.columns:
                df_[col] = pd.to_numeric(df_[col], errors='coerce').astype('Int64').astype('int64')
    return fh, fa

def assemble_game_matrix(features: pd.DataFrame, games: pd.DataFrame, lines: Optional[pd.DataFrame], require_lines: bool = False, lean_features: bool = False) -> Tuple[pd.DataFrame, list]:
    g = games.copy()
    if lines is not None and not lines.empty:
        g = g.merge(lines[['game_id','close_spread','close_total']], on='game_id', how='left')
    else:
        # Do NOT inject fake lines in inference; leave NaN so probabilities can be skipped if needed
        g['close_spread'] = np.nan
        g['close_total'] = np.nan

    # Canonicalize teams for reliable merges
    f = features.copy()
    f['team_id'] = f['team'].map(canon_team)
    f_home = f.rename(columns=lambda c: f'h_{c}' if c not in ['season','week','team','team_id'] else c)
    f_away = f.rename(columns=lambda c: f'a_{c}' if c not in ['season','week','team','team_id'] else c)

    g['home_id'] = g['home_team'].map(canon_team)
    g['away_id'] = g['away_team'].map(canon_team)

    # First attempt: strict same-week merge
    X = g.merge(f_home, left_on=['season','week','home_id'], right_on=['season','week','team_id'], how='left')
    X = X.merge(f_away, left_on=['season','week','away_id'], right_on=['season','week','team_id'], how='left', suffixes=('', '_a'))

    miss_home = X.filter(like='h_').isna().mean().mean()
    miss_away = X.filter(like='a_').isna().mean().mean()

    # If most features are missing (typical when predicting a future week with no PBP yet),
    # fall back to as-of join (use last available week <= game week).
    if (miss_home > 0.20) or (miss_away > 0.20):
        fh, fa = _merge_features_asof(features, g)
        # Drop duplicated join keys before merging
        base = g.copy()
        X = base.merge(fh.drop(columns=['season','week']), on=['game_id'], how='left')
        X = X.merge(fa.drop(columns=['season','week']), on=['game_id'], how='left', suffixes=('', '_a'))

    # Labels if available
    if 'home_points' in X.columns and 'away_points' in X.columns:
        X['margin'] = X['home_points'] - X['away_points']
        X['total'] = X['home_points'] + X['away_points']

    # If training requires real lines, drop rows without them
    if require_lines:
        X = X[~X['close_spread'].isna() & ~X['close_total'].isna()].copy()

    # Drop pushes for classification heads to avoid label noise
    if 'close_spread' in X.columns and 'close_total' in X.columns:
        push_mask = (X['margin'] + X['close_spread'] == 0) | (X['total'] == X['close_total'])
        if 'margin' in X.columns and 'total' in X.columns:
            X = X.loc[~push_mask].copy()

    # Exclude certain noisy blend features from diffs
    exclude_blends = set(['series_fd_rate_blend','start_yardline_100_blend','temp_mean_blend','wind_mean_blend'])

    feat_cols: list = []
    if lean_features:
        needed = ['neutral_sec_per_play_blend','proe_blend','ed_epa_blend','ed_sr_blend']
        for base0 in needed:
            hcol = f'h_{base0}'
            acol = f'a_{base0}'
            if hcol in X.columns and acol in X.columns:
                dcol = f'diff_{base0}'
                X[dcol] = X[hcol] - X[acol]
                feat_cols.append(dcol)
    else:
        for col in features.columns:
            if col in ['season','week','team','opp_team','team_id']:
                continue
            if col.endswith('_blend') and col not in exclude_blends:
                hcol = f'h_{col}'
                acol = f'a_{col}'
                dcol = f'diff_{col}'
                if hcol in X.columns and acol in X.columns:
                    X[dcol] = X[hcol] - X[acol]
                    feat_cols.append(dcol)

    # Add anchors (may remain NaN if no lines; handled later)
    X['spread_anchor'] = X['close_spread']
    X['total_anchor'] = X['close_total']
    feat_cols += ['spread_anchor','total_anchor']

    # Merge integrity check (after constructing diffs). Allow higher tolerance in prediction mode
    miss_home = X.filter(like='h_').isna().mean().mean()
    miss_away = X.filter(like='a_').isna().mean().mean()
    if (miss_home > 0.80) or (miss_away > 0.80):
        raise ValueError(f'Feature merge failed: {miss_home:.0%} home, {miss_away:.0%} away missing even after as-of fallback. Check TEAM_MAP canonization and input weeks.')

    return X, feat_cols
# Helper to compute simulation-based probabilities from arrays for calibration


# Helper to compute simulation-based probabilities from arrays for calibration
def _train_sim_probs(mu_h: np.ndarray, mu_a: np.ndarray, sd_h: float, sd_a: float, rho: float,
                     spread: np.ndarray, total_line: np.ndarray, n_sims: int = 4000, seed: int = 1234) -> Tuple[np.ndarray, np.ndarray]:
    rng = default_rng(seed)
    n = len(mu_h)
    p_cov = np.zeros(n)
    p_ov = np.zeros(n)
    cov = np.array([[sd_h**2, rho*sd_h*sd_a], [rho*sd_h*sd_a, sd_a**2]])
    # Draw sims in batches to avoid huge matrices
    batch = max(500, min(2000, n_sims))
    for i in range(n):
        sims = rng.multivariate_normal([mu_h[i], mu_a[i]], cov, size=n_sims)
        margin = sims[:,0] - sims[:,1]
        tot = sims[:,0] + sims[:,1]
        p_cov[i] = (margin > -float(spread[i])).mean()
        p_ov[i] = (tot > float(total_line[i])).mean()
    return p_cov, p_ov

def train_models(features: pd.DataFrame, games: pd.DataFrame, lines: Optional[pd.DataFrame], lean_features: bool = False, alpha_grid: Optional[List[float]] = None) -> Models:
    X, feat_cols = assemble_game_matrix(features, games, lines, require_lines=True, lean_features=lean_features)
    if X.empty:
        raise ValueError('No training rows with real betting lines. Cannot train models on placeholders.')
    use = X[feat_cols].fillna(0.0).values

    # Alpha sweep (defaults to [3.0] if not provided)
    if not alpha_grid:
        alpha_grid = [3.0]
    best_alpha = None
    best_mae = np.inf
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    y_h = X['home_points'].values.astype(float)
    y_a = X['away_points'].values.astype(float)
    for a in alpha_grid:
        fold_mae = []
        for tr_idx, va_idx in kf.split(use):
            Xtr, Xva = use[tr_idx], use[va_idx]
            ytr_h, yva_h = y_h[tr_idx], y_h[va_idx]
            ytr_a, yva_a = y_a[tr_idx], y_a[va_idx]
            hp_ = Pipeline([('sc', StandardScaler()), ('rg', Ridge(alpha=float(a), random_state=42))])
            ap_ = Pipeline([('sc', StandardScaler()), ('rg', Ridge(alpha=float(a), random_state=42))])
            hp_.fit(Xtr, ytr_h)
            ap_.fit(Xtr, ytr_a)
            mu_h_va = hp_.predict(Xva)
            mu_a_va = ap_.predict(Xva)
            tot_mu_va = mu_h_va + mu_a_va
            tot_obs_va = yva_h + yva_a
            fold_mae.append(mean_absolute_error(tot_obs_va, tot_mu_va))
        mae_a = float(np.mean(fold_mae)) if fold_mae else np.inf
        if mae_a < best_mae:
            best_mae = mae_a
            best_alpha = float(a)
    # Fit final models with best alpha
    hp = Pipeline([('sc', StandardScaler()), ('rg', Ridge(alpha=best_alpha, random_state=42))])
    ap = Pipeline([('sc', StandardScaler()), ('rg', Ridge(alpha=best_alpha, random_state=42))])
    hp.fit(use, y_h)
    ap.fit(use, y_a)

    mu_h = hp.predict(use)
    mu_a = ap.predict(use)
    res_h = (X['home_points'].values - mu_h).astype(float)
    res_a = (X['away_points'].values - mu_a).astype(float)

    sd_home = float(np.nanstd(res_h, ddof=1)) if np.isfinite(np.nanstd(res_h)) else 9.5
    sd_away = float(np.nanstd(res_a, ddof=1)) if np.isfinite(np.nanstd(res_a)) else 9.5

    # residual correlation, clipped to sane range
    if len(res_h) > 2 and np.nanstd(res_h) > 0 and np.nanstd(res_a) > 0:
        rho = float(np.corrcoef(np.nan_to_num(res_h), np.nan_to_num(res_a))[0, 1])
        if not np.isfinite(rho):
            rho = 0.25
        rho = float(np.clip(rho, -0.1, 0.6))
    else:
        rho = 0.25

    # Residual stds for margin and total using point models
    margin_obs = (X['home_points'].values - X['away_points'].values).astype(float)
    total_obs  = (X['home_points'].values + X['away_points'].values).astype(float)
    margin_mu  = (mu_h - mu_a).astype(float)
    total_mu   = (mu_h + mu_a).astype(float)

    sd_margin = float(np.nanstd(margin_obs - margin_mu, ddof=1)) if     np.isfinite(np.nanstd(margin_obs - margin_mu)) else 13.0
    sd_total  = float(np.nanstd(total_obs  - total_mu,  ddof=1)) if     np.isfinite(np.nanstd(total_obs  - total_mu))  else 14.0
    # De-sharpen totals probabilities and reduce Brier
    sd_margin = max(6.0, sd_margin)
    sd_total  = max(9.0, sd_total)

    # Cover/Over labels
    X['cover'] = (X['margin'] > -X['close_spread']).astype(int)
    X['over'] = (X['total'] > X['close_total']).astype(int)

    # Disable isotonic calibration; use simulation probabilities directly
    iso_cover = None
    iso_over = None

    cov_base = LogisticRegression(max_iter=1000)
    ov_base = LogisticRegression(max_iter=1000)
    cov = CalibratedClassifierCV(cov_base, method='isotonic', cv=5)
    ov = CalibratedClassifierCV(ov_base, method='isotonic', cv=5)
    cov.fit(use, X['cover'].values)
    ov.fit(use, X['over'].values)

    # Build 3x3 bucketed sd_total by neutral pace (sum of home/away) and PROE (sum of home/away)
    # Compute sums from X when available; fall back to global if missing
    if {'h_neutral_sec_per_play_blend','a_neutral_sec_per_play_blend','h_proe_blend','a_proe_blend'}.issubset(set(X.columns)):
        speed = (X['h_neutral_sec_per_play_blend'].values + X['a_neutral_sec_per_play_blend'].values)
        proe_sum = (X['h_proe_blend'].values + X['a_proe_blend'].values)
        total_resid = (total_obs - total_mu)
        # Define tertiles
        pace_q = (np.nanpercentile(speed, 33), np.nanpercentile(speed, 66))
        proe_q = (np.nanpercentile(proe_sum, 33), np.nanpercentile(proe_sum, 66))
        grid = {}
        for i, (lo1, hi1) in enumerate([(None, pace_q[0]), (pace_q[0], pace_q[1]), (pace_q[1], None)]):
            for j, (lo2, hi2) in enumerate([(None, proe_q[0]), (proe_q[0], proe_q[1]), (proe_q[1], None)]):
                m = np.ones_like(speed, dtype=bool)
                if lo1 is not None:
                    m &= speed >= lo1
                if hi1 is not None:
                    m &= speed < hi1
                if lo2 is not None:
                    m &= proe_sum >= lo2
                if hi2 is not None:
                    m &= proe_sum < hi2
                val = np.nanstd(total_resid[m], ddof=1)
                if not np.isfinite(val) or np.sum(m) < 20:
                    val = sd_total
                grid[(i, j)] = float(max(7.0, val))
    else:
        pace_q = (np.nan, np.nan)
        proe_q = (np.nan, np.nan)
        grid = {(i, j): float(sd_total) for i in range(3) for j in range(3)}

    return Models(
        home_points=hp,
        away_points=ap,
        cover_cls=cov,
        over_cls=ov,
        feat_cols=feat_cols,
        sd_home=sd_home,
        sd_away=sd_away,
        rho=rho,
        sd_margin=sd_margin,
        sd_total=sd_total,
        pace_q=(pace_q[0], pace_q[1]),
        proe_q=(proe_q[0], proe_q[1]),
        sd_total_grid=grid,
        iso_cover=iso_cover,
        iso_over=iso_over,
        sim_sd_scale=1.0
    )

def simulate_scores(row: pd.Series, models: Models, n_sims: int = 20000, seed: int = 7):
    rng = default_rng(seed)
    x = row[models.feat_cols].fillna(0.0).values.reshape(1, -1)
    mu_h = models.home_points.predict(x)[0]
    mu_a = models.away_points.predict(x)[0]
    total_anchor = float(row.get('total_anchor', 44.0))

    # Use constant residual SDs to match training-time calibration (inflate by sim_sd_scale)
    sd_h = max(3.0, models.sd_home) * float(getattr(models, 'sim_sd_scale', 1.0))
    sd_a = max(3.0, models.sd_away) * float(getattr(models, 'sim_sd_scale', 1.0))
    rho = float(models.rho)
    cov = np.array([[sd_h**2, rho*sd_h*sd_a],
                    [rho*sd_h*sd_a, sd_a**2]])
    sims = rng.multivariate_normal([mu_h, mu_a], cov, size=n_sims)
    # Use raw (unrounded) for projection means; clip at zero for probabilities where appropriate
    hs_raw = sims[:,0]
    as_raw = sims[:,1]
    margin_raw = hs_raw - as_raw
    total_raw = hs_raw + as_raw
    # For event probabilities, rounding isn’t required; use raw comparisons to thresholds
    spread = float(row.get('close_spread', np.nan))
    total_line = float(row.get('close_total', np.nan))
    out = {
        'mu_home': float(mu_h),
        'mu_away': float(mu_a),
        'spread_proj': float(margin_raw.mean()),
        'total_proj': float(total_raw.mean()),
        'p_home_win': float((margin_raw > 0).mean()),
    }
    if np.isfinite(spread):
        out['p_home_cover'] = float((margin_raw > -spread).mean())
    else:
        out['p_home_cover'] = np.nan
    if np.isfinite(total_line):
        out['p_over'] = float((total_raw > total_line).mean())
    else:
        out['p_over'] = np.nan
    return out

############################
# Backtest and Predict
############################

def rolling_backtest(years: List[int], start_week: int, end_week: int, season: int, roll_weeks: int = 6, use_weekly: bool = False, sim_sd_scale: float = 1.0, lean_features: bool = False, alpha_grid: Optional[List[float]] = None):
    pbp = fetch_pbp([season])
    games = fetch_games([season])
    lines = fetch_lines([season])
    # Fallback: build lines from PBP if official lines unavailable
    if lines is None or lines.empty:
        tmp = pbp[['game_id','spread_line','total_line']].copy()
        tmp = tmp.dropna(subset=['spread_line','total_line'], how='all')
        if not tmp.empty:
            lines = tmp.groupby('game_id').agg(
                close_spread=('spread_line','last'),
                close_total=('total_line','last')
            ).reset_index()
        else:
            lines = None
    # Try DKNetwork splits to populate lines if still missing
    if lines is None or lines.empty:
        dk = fetch_dk_splits(event_group=88808, date_range='today')
        dk_lines = build_lines_from_dk(dk, games)
        if dk_lines is not None and not dk_lines.empty:
            lines = dk_lines

    feats = build_team_week_features(pbp, games, roll_weeks=roll_weeks, use_weekly=use_weekly)

    rows = []
    for wk in range(start_week, end_week + 1):
        tr_mask = (games['season'] == season) & (games['week'] < wk)
        te_mask = (games['season'] == season) & (games['week'] == wk)
        if tr_mask.sum() < 8 or te_mask.sum() == 0:
            continue
        models = train_models(
            feats,
            games[tr_mask],
            lines.merge(games[tr_mask][['game_id']], on='game_id', how='right') if lines is not None else None,
            lean_features=lean_features,
            alpha_grid=alpha_grid
        )
        models.sim_sd_scale = float(sim_sd_scale)
        X_test, feat_cols = assemble_game_matrix(
            feats,
            games[te_mask],
            lines.merge(games[te_mask][['game_id']], on='game_id', how='right') if lines is not None else None,
            lean_features=lean_features
        )
        models.feat_cols = feat_cols

        for _, r in X_test.iterrows():
            s = simulate_scores(r, models, n_sims=20000, seed=wk+7)
            # use simulation probabilities directly
            p_cov = float(s['p_home_cover'])
            p_ov  = float(s['p_over'])
            # realized
            cover_true = int(r['margin'] > -r['close_spread'])
            over_true = int(r['total'] > r['close_total'])
            rows.append({
                'season': int(r['season']), 'week': int(r['week']), 'game_id': r['game_id'],
                'home_team': r['home_team'], 'away_team': r['away_team'],
                'close_spread': float(r['close_spread']), 'close_total': float(r['close_total']),
                'home_points': int(r['home_points']), 'away_points': int(r['away_points']),
                'p_home_cover_sim': s['p_home_cover'], 'p_over_sim': s['p_over'],
                'p_cover_cls': p_cov, 'p_over_cls': p_ov,
                'spread_proj_sim': s['spread_proj'], 'total_proj_sim': s['total_proj'],
                'cover_true': cover_true, 'over_true': over_true,
            })

    bt = pd.DataFrame(rows)
    if bt.empty:
        print("[backtest] No rows produced. Check inputs.")
        return bt
    # Remove pushes for scoring
    push_mask = ((bt['home_points'] - bt['away_points'] + bt['close_spread']) == 0) | ((bt['home_points'] + bt['away_points']) == bt['close_total'])
    bt = bt.loc[~push_mask].copy()
    # Metrics
    brier_cover = brier_score_loss(bt['cover_true'], bt['p_cover_cls'])
    brier_over = brier_score_loss(bt['over_true'], bt['p_over_cls'])
    mae_spread = mean_absolute_error(bt['home_points']-bt['away_points'], bt['spread_proj_sim'])
    mae_total = mean_absolute_error(bt['home_points']+bt['away_points'], bt['total_proj_sim'])
    print(f'[backtest] cover Brier={brier_cover:.3f} over Brier={brier_over:.3f} | spread MAE={mae_spread:.2f} total MAE={mae_total:.2f}')
    return bt

def predict_week(season: int, week: int, roll_weeks: int = 6, n_sims: int = 20000, use_weekly: bool = False, sim_sd_scale: float = 1.0, lean_features: bool = False, alpha_grid: Optional[List[float]] = None) -> pd.DataFrame:
    pbp = fetch_pbp([season])
    games = fetch_games([season])
    lines = fetch_lines([season])
    if lines is None or lines.empty:
        tmp = pbp[['game_id','spread_line','total_line']].copy()
        tmp = tmp.dropna(subset=['spread_line','total_line'], how='all')
        if not tmp.empty:
            lines = tmp.groupby('game_id').agg(
                close_spread=('spread_line','last'),
                close_total=('total_line','last')
            ).reset_index()
        else:
            lines = None
    # Try DKNetwork splits to populate lines if still missing
    if lines is None or lines.empty:
        dk = fetch_dk_splits(event_group=88808, date_range='today')
        dk_lines = build_lines_from_dk(dk, games)
        if dk_lines is not None and not dk_lines.empty:
            lines = dk_lines
    feats = build_team_week_features(pbp, games, roll_weeks=roll_weeks, use_weekly=use_weekly)

    mask = (games['season'] == season) & (games['week'] < week)
    models = train_models(feats, games[mask], lines.merge(games[mask][['game_id']], on='game_id', how='right') if lines is not None else None, lean_features=lean_features, alpha_grid=alpha_grid)
    models.sim_sd_scale = float(sim_sd_scale)

    te_mask = (games['season'] == season) & (games['week'] == week)
    X, feat_cols = assemble_game_matrix(feats, games[te_mask], lines.merge(games[te_mask][['game_id']], on='game_id', how='right') if lines is not None else None, lean_features=lean_features)
    # Backfill if some test games still lack lines
    if X[['close_spread','close_total']].isna().any().any():
        dk_extra = fetch_dk_splits(event_group=88808,     date_ranges=['today','tomorrow','this-week','next-7-days'])
        dk_patch = build_lines_from_dk(dk_extra, games[te_mask])
        if dk_patch is not None and not dk_patch.empty:
            miss_ids = X.loc[X['close_spread'].isna() | X['close_total'].isna(), 'game_id'].unique()
            patch = dk_patch[dk_patch['game_id'].isin(miss_ids)]
            if not patch.empty:
                X = X.drop(columns=['close_spread','close_total'], errors='ignore') \
                 .merge(patch[['game_id','close_spread','close_total']], on='game_id',     how='left')
    models.feat_cols = feat_cols

    out = []
    for _, r in X.iterrows():
        s = simulate_scores(r, models, n_sims=n_sims, seed=13+week)
        # publish simulation probabilities directly
        p_cover_norm = float(s.get('p_home_cover')) if np.isfinite(r.get('close_spread')) else np.nan
        p_over_norm  = float(s.get('p_over')) if np.isfinite(r.get('close_total')) else np.nan

        out.append({
            'game_id': r['game_id'],
            'home_team': r['home_team'],
            'away_team': r['away_team'],
            'close_spread': float(r['close_spread']),
            'close_total': float(r['close_total']),
            'spread_proj': s['spread_proj'],
            'total_proj': s['total_proj'],
            'p_home_win': s['p_home_win'],
            'p_home_cover_sim': s['p_home_cover'],
            'p_home_cover_cls': float(p_cover_norm),
            'p_over_sim': s['p_over'],
            'p_over_cls': float(p_over_norm),
        })
    return pd.DataFrame(out)

# ---------- DKNetwork splits (HTML) ----------
DK_BASE = 'https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/'

def _dk_clean_text(s: str) -> str:
    return re.sub(r"opens?\s+in\s+(?:a\s+)?new\s+tab", "", str(s or ''), flags=re.I).strip()

def fetch_dk_splits(event_group: int = 88808, date_ranges: Optional[List[str]] = None, timeout: int = 15) -> pd.DataFrame:
    """Fetch DKNetwork betting splits across multiple date tabs and pages."""
    if date_ranges is None:
        date_ranges = ['today','tomorrow','this-week','next-7-days']

    def get_html(url: str) -> str:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.text
        except Exception:
            return ''
        return ''

    def page_num(u: str) -> int:
        try:
            return int(parse_qs(urlparse(u).query).get('tb_page', ['1'])[0])
        except Exception:
            return 1

    def parse_page(ht: str) -> list[dict]:
        if not ht:
            return []
        sp = BeautifulSoup(ht, 'html.parser')
        out = []
        for div in sp.select('div.tb-se'):
            matchup = _dk_clean_text(div.select_one('.tb-mt').get_text(' ', strip=True) if div.select_one('.tb-mt') else '')
            when = _dk_clean_text(div.select_one('.tb-gd').get_text(' ', strip=True) if div.select_one('.tb-gd') else '')
            for row in div.select('div.tb-mr-row'):
                tds = [t.get_text(' ', strip=True) for t in row.select('div')]
                if len(tds) < 6:
                    continue
                market, side, odds, spread, pct_handle, pct_bets = tds[:6]
                out.append({
                    'matchup': matchup,
                    'game_time': when,
                    'market': market,
                    'side': side,
                    'odds': odds,
                    'spread': spread,
                    '%handle': pct_handle.replace('%',''),
                    '%bets': pct_bets.replace('%','')
                })
        return out

    records = []
    for dr in date_ranges:
        base = f"{DK_BASE}?{urlencode({'tb_eg': event_group, 'tb_edate': dr, 'tb_emt': '0'})}"
        html = get_html(base)
        if not html:
            continue
        soup = BeautifulSoup(html, 'html.parser')
        pages = {base}
        for a in soup.select('div.tb_pagination a[href]'):
            href = a.get('href','')
            if href and href.startswith(DK_BASE):
                pages.add(href)
        for u in sorted(pages, key=page_num):
            h = html if u == base else get_html(u)
            records.extend(parse_page(h))
    return pd.DataFrame.from_records(records)

def _american_to_prob(odds: int | float | str) -> float:
    try:
        o = str(odds).replace('+','').strip()
        o = float(o) if o not in ('', 'None', 'nan') else np.nan
        if np.isnan(o):
            return np.nan
        if odds and str(odds).startswith('+'):
            return 100.0/(o+100.0)
        if o < 0:
            return (-o)/((-o)+100.0)
        return 100.0/(o+100.0)
    except Exception:
        return np.nan

def build_lines_from_dk(dk: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DK splits to a per-game lines table keyed by game_id with home-close spread/total.
    Expects dk['matchup'] like 'NY Jets @ MIA Dolphins'.
    """
    if dk is None or dk.empty:
        return pd.DataFrame(columns=['game_id','close_spread','close_total','close_home_ml'])

    df = dk.copy()
    for c in ['market','side','odds','matchup']:
        if c in df.columns:
            df[c] = df[c].astype(str)
    df['market'] = df['market'].str.strip()
    df['side'] = df['side'].str.strip()

    parts = df['matchup'].str.split('@')
    df['away_raw'] = parts.str[0].str.strip()
    df['home_raw'] = parts.str[1].str.strip()
    df['home_id'] = df['home_raw'].map(canon_team)
    df['away_id'] = df['away_raw'].map(canon_team)

    gkey = games[['game_id','season','week','home_team','away_team']].copy()
    gkey['home_id'] = gkey['home_team'].map(canon_team)
    gkey['away_id'] = gkey['away_team'].map(canon_team)

    # Spread
    sp = df[df['market'].str.contains('Spread', case=False, na=False)].copy()
    sp[['side_team','side_num']] = sp['side'].str.extract(r'^(.*)\s([+-]?\d+\.?\d*)$')
    sp['side_team_id'] = sp['side_team'].map(canon_team)
    sp['side_num'] = pd.to_numeric(sp['side_num'], errors='coerce')
    sp['home_spread'] = np.where(sp['side_team_id'] == sp['home_id'], sp['side_num'], -sp['side_num'])
    spg = sp.groupby(['home_id','away_id'], as_index=False)['home_spread'].last()

    # Total
    to = df[df['market'].str.contains('Total', case=False, na=False)].copy()
    to['total_val'] = pd.to_numeric(to['side'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    tog = to.groupby(['home_id','away_id'], as_index=False)['total_val'].last()

    # Moneyline per matchup (home side)
    ml = df[df['market'].str.contains('Moneyline', case=False, na=False)].copy()
    ml[['side_team']] = ml['side'].str.extract(r'^(.*?)(?:\s+[+-]?\d+)?$')
    ml['side_team_id'] = ml['side_team'].map(canon_team)
    ml['odds_val'] = pd.to_numeric(ml['odds'].str.replace('+','', regex=False), errors='coerce')
    ml['is_home'] = ml['side_team_id'] == ml['home_id']
    ml_home = ml[ml['is_home']].copy()
    mlg = ml_home.groupby(['home_id','away_id'], as_index=False)['odds_val'].last().rename(columns={'odds_val':'close_home_ml'})

    # Merge to games
    lines = gkey.merge(spg, on=['home_id','away_id'], how='left')
    lines = lines.merge(tog, on=['home_id','away_id'], how='left')
    lines = lines.merge(mlg, on=['home_id','away_id'], how='left')
    lines = lines.rename(columns={'home_spread':'close_spread','total_val':'close_total'})
    return lines[['game_id','close_spread','close_total','close_home_ml']]

############################
# CLI
############################

def _build_arg_parser():
    ap = argparse.ArgumentParser(description='NFL PBP projections using nfl_data_py (no CSVs).')
    sub = ap.add_subparsers(dest='cmd')

    p1 = sub.add_parser('backtest', help='Rolling backtest for a single season range of weeks')
    p1.add_argument('--season', type=int, required=True)
    p1.add_argument('--start_week', type=int, required=True)
    p1.add_argument('--end_week', type=int, required=True)
    p1.add_argument('--roll_weeks', type=int, default=6)
    p1.add_argument('--use_weekly', action='store_true', help='Include weekly player-derived features (default off)')
    p1.add_argument('--sim_sd_scale', type=float, default=1.0, help='Scalar to widen/narrow sim residual SDs (default 1.0)')
    p1.add_argument('--lean_features', action='store_true', help='Use minimal, robust diff feature set for point models')
    p1.add_argument('--alpha_grid', type=str, default='3.0', help='Comma-separated ridge alphas to sweep, e.g., "2,3,5,8"')

    p2 = sub.add_parser('predict', help='Predict a target week (uses prior weeks for training)')
    p2.add_argument('--season', type=int, required=True)
    p2.add_argument('--week', type=int, required=True)
    p2.add_argument('--roll_weeks', type=int, default=6)
    p2.add_argument('--n_sims', type=int, default=20000)
    p2.add_argument('--use_weekly', action='store_true', help='Include weekly player-derived features (default off)')
    p2.add_argument('--sim_sd_scale', type=float, default=1.0, help='Scalar to widen/narrow sim residual SDs (default 1.0)')
    p2.add_argument('--lean_features', action='store_true', help='Use minimal, robust diff feature set for point models')
    p2.add_argument('--alpha_grid', type=str, default='3.0', help='Comma-separated ridge alphas to sweep, e.g., "2,3,5,8"')

    return ap

def main():
    warnings.filterwarnings('ignore')
    ap = _build_arg_parser()
    args = ap.parse_args()
    if args.cmd == 'backtest':
        ag = [float(x) for x in str(args.alpha_grid).split(',') if x.strip()]
        _ = rolling_backtest([args.season], start_week=args.start_week, end_week=args.end_week, season=args.season,
                              roll_weeks=args.roll_weeks, use_weekly=args.use_weekly, sim_sd_scale=args.sim_sd_scale,
                              lean_features=args.lean_features, alpha_grid=ag)
    elif args.cmd == 'predict':
        ag = [float(x) for x in str(args.alpha_grid).split(',') if x.strip()]
        df = predict_week(args.season, args.week, roll_weeks=args.roll_weeks, n_sims=args.n_sims,
                          use_weekly=args.use_weekly, sim_sd_scale=args.sim_sd_scale,
                          lean_features=args.lean_features, alpha_grid=ag)
        out = f'picks_{args.season}_wk{args.week}.csv'
        df.to_csv(out, index=False)
        print(f'[predict] wrote {out}, rows={len(df)}')
    else:
        ap.print_help()

if __name__ == '__main__':
    main()
