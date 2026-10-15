import os
import re
import requests
import numpy as np
import pandas as pd
import nflreadpy as nfl

# Standardize team abbreviation mappings between Sleeper API and nflreadpy
SLEEPER_TEAM_MAP = {
    'JAX': 'JAX', 'JAC': 'JAX',
    'LA': 'LAR',  'LAR': 'LAR',
    'WSH': 'WAS', 'WAS': 'WAS',
    'OAK': 'LV',  'LV': 'LV',
    'SD': 'LAC',  'LAC': 'LAC'
}

# Roster/Injury statuses that disqualify a player from being an active depth chart starter
INACTIVE_STATUSES = {
    "ir", "pup", "sus", "out", "reserve/injured", 
    "injured reserve", "opt_out", "dnp", "physician_physically_unable_to_perform",
    "free_agent", "inactive"
}

# Hardcoded name fixes.
RENAME_MAP = {
    "Zonovan Knight": "Bam Knight",
}

# Helper to normalize names (removes suffixes, punctuation, and casing)
def clean_player_name(name):
    if not isinstance(name, str):
        return ""
    name = RENAME_MAP.get(name, name)
    name = name.lower()
    name = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b', '', name)  # Strip common suffixes
    name = re.sub(r'[^a-z0-9\s]', '', name)              # Strip special chars/punctuation
    return " ".join(name.split())                        # Normalize spacing


def make_player_key(name: str, team: str, position: str) -> tuple:
    """Returns a standardized player key tuple (cleaned_name, team, position)."""
    clean_name = clean_player_name(name)
    clean_team = str(team or "").upper().strip()
    clean_pos = str(position or "").upper().strip()
    return (clean_name, clean_team, clean_pos)

# -------------------------------------------------------------------
# Positional Starter Workload Capture Rates & Calculation Types
# -------------------------------------------------------------------
STARTER_CAPTURE_CONFIG = {
    'QB1': {'rate': 0.88, 'type': 'replacement'},
    'RB1': {'rate': 0.65, 'type': 'replacement'},
    'WR1': {'rate': 0.35, 'type': 'additive'},
    'WR2': {'rate': 0.30, 'type': 'additive'},
    'TE1': {'rate': 0.50, 'type': 'replacement'}
}

# -------------------------------------------------------------------
# 1. Fetch Active Rosters & Player Metadata from Sleeper API
# -------------------------------------------------------------------
def get_sleeper_rostered_players():
    print("\n" + "="*80)
    print("STEP 1: FETCHING ACTIVE ROSTERS FROM SLEEPER API")
    print("="*80)

    url = "https://api.sleeper.app/v1/players/nfl"
    response = requests.get(url)
    data = response.json()

    print(f"[LOG] Raw catalog records retrieved: {len(data):,}")

    rostered_players = []
    filtered_out_counts = {"inactive_status": 0, "wrong_pos": 0, "no_team": 0}

    for player_id, p in data.items():
        raw_team = p.get("team")
        status = str(p.get("status") or "").lower()
        injury_status = str(p.get("injury_status") or "").lower()

        is_injured_or_inactive = (
            status in INACTIVE_STATUSES or 
            injury_status in INACTIVE_STATUSES
        )

        if not p.get("active") or is_injured_or_inactive:
            filtered_out_counts["inactive_status"] += 1
            continue

        if not raw_team:
            filtered_out_counts["no_team"] += 1
            continue

        if p.get("position") not in ["QB", "RB", "WR", "TE"]:
            filtered_out_counts["wrong_pos"] += 1
            continue

        clean_team = SLEEPER_TEAM_MAP.get(raw_team, raw_team)
        full_name = p.get("full_name", "")
        position = str(p.get("position")).upper().strip()

        rostered_players.append({
            "sleeper_id": str(player_id),
            "full_name": full_name,
            "full_name_clean": clean_player_name(full_name),
            "team": clean_team,
            "position": position,
            "player_key": make_player_key(full_name, clean_team, position),
            "depth_chart_order": p.get("depth_chart_order", 99)
        })

    df_sleeper = pd.DataFrame(rostered_players)

    print(f"[LOG] Active/Eligible skill players retained: {len(df_sleeper):,}")
    print(f"[LOG] Excluded breakdown: {filtered_out_counts}")
    print(f"[LOG] Positional distribution:\n{df_sleeper['position'].value_counts().to_string(header=False)}")
    return df_sleeper

# -------------------------------------------------------------------
# 2. Fetch Multi-Season Historical Volume via nflreadpy
# -------------------------------------------------------------------
def get_historical_opportunity():
    print("\n" + "="*80)
    print("STEP 2: COMPUTING HISTORICAL OPPORTUNITY METRICS (nflreadpy)")
    print("="*80)

    try:
        current_season = int(nfl.get_current_season())
    except Exception:
        current_season = 2026

    seasons_to_try = [current_season, current_season - 1, current_season - 2]
    print(f"[LOG] Target seasons for volume evaluation: {seasons_to_try}")

    weekly_stats = pd.DataFrame()
    loaded_seasons = []

    for yr in seasons_to_try:
        try:
            df_yr = nfl.load_player_stats(seasons=[yr])
            if hasattr(df_yr, "to_pandas"):
                df_yr = df_yr.to_pandas()

            if df_yr is not None and not df_yr.empty:
                weekly_stats = pd.concat([weekly_stats, df_yr], ignore_index=True)
                loaded_seasons.append(yr)
                print(f"[LOG]  -> Loaded {len(df_yr):,} weekly records for season {yr}")
        except Exception as e:
            print(f"[LOG]  -> Failed to load season {yr}: {e}")

    if weekly_stats.empty:
        print("[LOG] Fallback: Loading default nflreadpy player stats dataset...")
        weekly_stats = nfl.load_player_stats()
        if hasattr(weekly_stats, "to_pandas"):
            weekly_stats = weekly_stats.to_pandas()

    team_col = 'team' if 'team' in weekly_stats.columns else ('recent_team' if 'recent_team' in weekly_stats.columns else 'UNK')
    if team_col == 'UNK':
        weekly_stats['team'] = 'UNK'

    if 'position' in weekly_stats.columns:
        weekly_stats['position'] = weekly_stats['position'].astype(str).str.upper().str.strip()
    else:
        weekly_stats['position'] = 'UNK'

    for col in ['carries', 'targets', 'attempts']:
        if col not in weekly_stats.columns:
            weekly_stats[col] = 0

    weekly_stats['carries'] = weekly_stats['carries'].fillna(0)
    weekly_stats['targets'] = weekly_stats['targets'].fillna(0)
    weekly_stats['attempts'] = weekly_stats['attempts'].fillna(0)
    
    weekly_stats['opportunities'] = weekly_stats['carries'] + weekly_stats['targets'] + (weekly_stats['attempts'] * 0.5)

    max_season = max(loaded_seasons) if loaded_seasons else current_season
    print(f"[LOG] Base anchor season established as: {max_season}")

    weekly_stats['weight'] = np.where(
        weekly_stats['season'] == max_season, 3.0, 
        np.where(weekly_stats['season'] == max_season - 1, 1.0, 0.2)
    )

    for col in ['opportunities', 'carries', 'targets']:
        weekly_stats[f'weighted_{col}'] = weekly_stats[col] * weekly_stats['weight']

    name_col = 'player_display_name' if 'player_display_name' in weekly_stats.columns else ('player_name' if 'player_name' in weekly_stats.columns else 'player_id')
    weekly_stats['full_name_clean'] = weekly_stats[name_col].apply(clean_player_name)
    weekly_stats['team'] = weekly_stats[team_col].map(lambda x: SLEEPER_TEAM_MAP.get(x, x))

    group_cols = ['full_name_clean', 'team', 'position']

    agg_dict = {
        'weight': ('weight', 'sum'),
        'weighted_opps': ('weighted_opportunities', 'sum'),
        'weighted_carries': ('weighted_carries', 'sum'),
        'weighted_targets': ('weighted_targets', 'sum'),
        'raw_player_name': (name_col, 'last')
    }

    player_summary = weekly_stats.groupby(group_cols).agg(**agg_dict).reset_index()

    player_summary['opportunities_pg'] = player_summary['weighted_opps'] / player_summary['weight']
    player_summary['carries_pg'] = player_summary['weighted_carries'] / player_summary['weight']
    player_summary['targets_pg'] = player_summary['weighted_targets'] / player_summary['weight']

    print(f"[LOG] Aggregated metrics across {len(player_summary):,} distinct (Name + Team + Position) player tuples.")
    return player_summary[['full_name_clean', 'team', 'position', 'opportunities_pg', 'carries_pg', 'targets_pg']]

# -------------------------------------------------------------------
# 3. Model Handcuffs Streamlined Output
# -------------------------------------------------------------------
def calculate_handcuffs(df_sleeper, df_stats):
    print("\n" + "="*80)
    print("STEP 3: CALCULATING HANDCUFFS")
    print("="*80)

    merged = pd.merge(
        df_sleeper,
        df_stats,
        on=['full_name_clean', 'team', 'position'],
        how='left'
    )

    unmatched_sleeper = merged[merged['opportunities_pg'].isna()]
    matched_sleeper = merged[merged['opportunities_pg'].notna()]

    print(f"[LOG] Correlation Results: {len(matched_sleeper)} matched / {len(unmatched_sleeper)} unmatched Sleeper active players.")

    if not unmatched_sleeper.empty:
        print("\n" + "-"*60)
        print("CORRELATION WARNING: Active Sleeper players missing historical nflreadpy data:")
        print("-"*60)
        for _, row in unmatched_sleeper.iterrows():
            print(f"  - [{row['team']} {row['position']}] {row['full_name']} (Cleaned: '{row['full_name_clean']}')")
        print("-"*60 + "\n")

    merged['opportunities_pg'] = merged['opportunities_pg'].fillna(0.5)
    merged['carries_pg'] = merged['carries_pg'].fillna(0.2)
    merged['targets_pg'] = merged['targets_pg'].fillna(0.3)
    
    results = []
    teams = sorted([t for t in merged['team'].unique() if isinstance(t, str) and len(t) <= 3])
    print(f"[LOG] Processing contingency modeling for {len(teams)} teams...")

    for team in teams:
        team_players = merged[merged['team'] == team].copy()
        if team_players.empty:
            continue

        starters = {}

        qbs = team_players[team_players['position'] == 'QB'].sort_values(
            by=['depth_chart_order', 'opportunities_pg'], ascending=[True, False]
        )
        if not qbs.empty: starters['QB1'] = qbs.iloc[0]

        rbs = team_players[team_players['position'] == 'RB'].sort_values(
            by=['depth_chart_order', 'opportunities_pg'], ascending=[True, False]
        )
        if not rbs.empty: starters['RB1'] = rbs.iloc[0]

        wrs = team_players[team_players['position'] == 'WR'].sort_values(
            by=['depth_chart_order', 'targets_pg'], ascending=[True, False]
        )
        if len(wrs) > 0: starters['WR1'] = wrs.iloc[0]
        if len(wrs) > 1: starters['WR2'] = wrs.iloc[1]

        tes = team_players[team_players['position'] == 'TE'].sort_values(
            by=['depth_chart_order', 'targets_pg'], ascending=[True, False]
        )
        if not tes.empty: starters['TE1'] = tes.iloc[0]

        for pos_key, starter in starters.items():
            candidates = team_players[team_players['sleeper_id'] != starter['sleeper_id']].copy()
            if candidates.empty:
                continue

            pos_filter = starter['position']
            same_pos_candidates = candidates[candidates['position'] == pos_filter].copy()
            if same_pos_candidates.empty:
                same_pos_candidates = candidates

            same_pos_candidates = same_pos_candidates.sort_values(
                by=['depth_chart_order', 'opportunities_pg'], ascending=[True, False]
            )

            # --- HANDCUFF SELECTION ---
            hc_player = same_pos_candidates.iloc[0]
            hc_cfg = STARTER_CAPTURE_CONFIG.get(pos_key, {'rate': 0.65, 'type': 'replacement'})

            results.append({
                'team': team,
                'absent_starter_role': pos_key,
                'absent_starter_name': starter['full_name'],
                'absent_starter_position': starter['position'],
                'player_name': hc_player['full_name'],
                'position': hc_player['position'],
                'sleeper_id': hc_player['sleeper_id'],
                'starter_decay_multiplier': round(hc_cfg['rate'], 2),
                'calculation_type': hc_cfg['type']
            })

    df_results = pd.DataFrame(results)

    df_results = df_results.sort_values(
        by=['team', 'absent_starter_role'],
        ascending=[True, True]
    ).reset_index(drop=True)

    print(f"[LOG] Successfully generated {len(df_results):,} contingent outcomes.")
    return df_results

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)

    df_sleeper = get_sleeper_rostered_players()
    df_stats = get_historical_opportunity()
    contingency_df = calculate_handcuffs(df_sleeper, df_stats)

    output_filename = "./data/handcuff.csv"
    contingency_df.to_csv(output_filename, index=False)

    print("\n" + "="*80)
    print(f"COMPLETE: Output saved to '{output_filename}'")
    print(f"Total rows: {len(contingency_df)} across {contingency_df['team'].nunique()} teams.")
    print("="*80 + "\n")
