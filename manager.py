import csv
import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import feedparser
from rich.console import Console
from rich.markdown import Markdown

from discord import (
    format_embed_to_markdown,
    generate_discord_embed,
    post_to_webhook,
)
from espn import (
    TEAM_ID,
    fetch_league_data,
)
from gemini import (
    SCHEDULED_RUN_REASON,
    evaluate_rotowire_news,
    evaluate_team,
    print_debug_prompt,
    start_interactive_mode,
)
from handcuff import clean_player_name, make_player_key
from solver import (
    calculate_upside_flyers,
    determine_bench_policies,
    optimize_roster,
    unlock_lineups_if_tuesday,
)

# ---------------------------------------------------------------------------
# CONFIGURATION & ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------
ROTOWIRE_RSS_URL = "https://www.rotowire.com/rss/news.php?sport=NFL"
HANDCUFF_CSV = os.getenv("HANDCUFF_CSV", "./data/handcuff.csv")
INJURY_CSV = os.getenv("INJURY_CSV", "./data/injury.csv")
TRIGGER_REASON = os.getenv("TRIGGER_REASON")


# ---------------------------------------------------------------------------
# LOAD A FAKE_NEWS.TXT TO TEST NEWS EVALUATION
# ---------------------------------------------------------------------------
def load_fake_news_file(filepath: str) -> List[Dict[str, str]]:
    """Parses custom key-value block format separated by double newlines into news dictionaries."""
    if not os.path.exists(filepath):
        print(f"Error: Fake news file '{filepath}' not found.")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    stories = []
    blocks = content.split("\n\n")

    for block in blocks:
        story = {}
        for line in block.strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                story[key.strip()] = val.strip()

        if story:
            story.setdefault("guid", f"fake-{hash(block)}")
            story.setdefault("title", "")
            story.setdefault("link", "")
            story.setdefault("description", "")
            story.setdefault("published", "")
            stories.append(story)

    print(f"✓ Successfully loaded {len(stories)} fake news item(s) from '{filepath}'.")
    return stories


# ---------------------------------------------------------------------------
# LOAD A ROSTER.TXT TO OVERRIDE ROSTER SPOTS AND FREE AGENT LIST FOR TESTING
# ---------------------------------------------------------------------------
def load_roster_file(filepath: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Parses custom roster text files supporting operations on user roster and free agency.

    Syntax format:
      Section 1 (User Roster Operations):
        -Player A,+Player B -> Drops 'Player A' to FA and adds 'Player B' to that same vacated spot.
        -Player C           -> Drops 'Player C' to FA, leaving the spot open.

      Section 2 (Free Agent Operations, separated by an empty newline):
        -Player D           -> Removes 'Player D' from free agency.
        +Player E           -> Forces 'Player E' to be added directly to free agency.
    """
    if not os.path.exists(filepath):
        print(f"Error: Roster file '{filepath}' not found.")
        return [], []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return [], []

    parts = re.split(r"\n\s*\n", content, maxsplit=1)
    roster_lines = parts[0].strip().splitlines()
    fa_lines = parts[1].strip().splitlines() if len(parts) > 1 else []

    roster_ops = []
    for line in roster_lines:
        line = line.strip()
        if not line:
            continue

        drop_player = None
        add_player = None

        tokens = line.split(",")
        for token in tokens:
            token = token.strip()
            if token.startswith("-"):
                drop_player = token[1:].strip()
            elif token.startswith("+"):
                add_player = token[1:].strip()

        if drop_player or add_player:
            roster_ops.append({
                "drop_player": drop_player,
                "add_player": add_player
            })

    fa_overrides = []
    for line in fa_lines:
        line = line.strip()
        if line:
            fa_overrides.append(line)

    print(f"✓ Loaded roster file '{filepath}': {len(roster_ops)} user roster operation(s), "
          f"{len(fa_overrides)} FA override(s).")
    return roster_ops, fa_overrides


def apply_roster_overrides(
    league_data: Dict[str, Any],
    roster_file: str,
) -> None:
    """
    Applies user roster and free agent overrides directly to league_data
    using the new inline + / - operation syntax.
    """
    roster_ops, fa_overrides = load_roster_file(roster_file)
    if not roster_ops and not fa_overrides:
        return

    # 1. Build a lookup pool across user team, opponents, and free agents
    player_pool: Dict[str, Dict[str, Any]] = {}

    all_teams = []
    if league_data.get("user_team"):
        all_teams.append(league_data["user_team"])
    if league_data.get("opponent_teams"):
        all_teams.extend(league_data["opponent_teams"])

    for team in all_teams:
        roster = team.get("roster", {})
        for list_key in ["starters", "bench", "IR"]:
            players = roster.get(list_key, [])
            for p in players:
                norm = clean_player_name(p.get("name", ""))
                if norm:
                    player_pool[norm] = p

    for p in league_data.get("free_agents", []):
        norm = clean_player_name(p.get("name", ""))
        if norm:
            player_pool[norm] = p

    # 2. Process Section 1: User Roster Operations
    for op in roster_ops:
        drop_name = op.get("drop_player")
        add_name = op.get("add_player")

        drop_norm = clean_player_name(drop_name) if drop_name else None
        add_norm = clean_player_name(add_name) if add_name else None

        # Find where the player being dropped is currently located on the user team
        vacated_slot = "BE"
        user_team = league_data.get("user_team", {})
        user_roster = user_team.get("roster", {})

        if drop_norm:
            for list_key in ["starters", "bench", "IR"]:
                for p in list(user_roster.get(list_key, [])):
                    if clean_player_name(p.get("name", "")) == drop_norm:
                        vacated_slot = p.get("slot_position", "BE")
                        user_roster[list_key].remove(p)

                        # Return dropped player to free agency
                        p["slot_position"] = "FA"
                        p["is_user_team"] = False
                        p["owner_team_id"] = 0
                        p.pop("keeper_price", None)
                        league_data.setdefault("free_agents", []).append(p)
                        break

        # If a replacement player is added (+), put them into the vacated slot
        if add_norm:
            # Check if player exists in the global pool or create a stub player
            player_to_add = player_pool.get(add_norm, {
                "name": add_name,
                "position": "FLEX",
                "projected_points": 0.0,
                "status": "ACTIVE",
            })

            # Remove player from any other team's roster or free agency list
            for team in all_teams:
                roster = team.get("roster", {})
                for list_key in ["starters", "bench", "IR"]:
                    roster[list_key] = [
                        p for p in roster.get(list_key, [])
                        if clean_player_name(p.get("name", "")) != add_norm
                    ]

            league_data["free_agents"] = [
                p for p in league_data.get("free_agents", [])
                if clean_player_name(p.get("name", "")) != add_norm
            ]

            # Assign to user team at the vacated spot
            player_to_add["slot_position"] = vacated_slot
            player_to_add["is_user_team"] = True
            player_to_add["owner_team_id"] = TEAM_ID

            if vacated_slot == "IR":
                user_roster.setdefault("IR", []).append(player_to_add)
            elif vacated_slot == "BE":
                user_roster.setdefault("bench", []).append(player_to_add)
            else:
                user_roster.setdefault("starters", []).append(player_to_add)

    # 3. Process Section 2: Free Agent Overrides
    for fa_line in fa_overrides:
        is_add = fa_line.startswith("+")
        is_remove = fa_line.startswith("-")
        clean_name = fa_line.lstrip("+-").strip()
        norm_fa = clean_player_name(clean_name)

        if not norm_fa:
            continue

        if is_remove:
            # Remove player from free agents list
            league_data["free_agents"] = [
                p for p in league_data.get("free_agents", [])
                if clean_player_name(p.get("name", "")) != norm_fa
            ]
        elif is_add:
            # Steal player from anywhere and add to free agency
            fa_player = player_pool.get(norm_fa, {
                "name": clean_name,
                "position": "FLEX",
                "projected_points": 0.0,
                "status": "ACTIVE",
            })

            # Remove from all team rosters if stolen
            for team in all_teams:
                roster = team.get("roster", {})
                for list_key in ["starters", "bench", "IR"]:
                    roster[list_key] = [
                        p for p in roster.get(list_key, [])
                        if clean_player_name(p.get("name", "")) != norm_fa
                    ]

            fa_player["slot_position"] = "FA"
            fa_player["is_user_team"] = False
            fa_player["is_lineup_locked"] = False
            fa_player["owner_team_id"] = 0
            fa_player.pop("keeper_price", None)

            # Avoid duplicates in free_agents list
            league_data["free_agents"] = [
                p for p in league_data.get("free_agents", [])
                if clean_player_name(p.get("name", "")) != norm_fa
            ]
            league_data["free_agents"].insert(0, fa_player)


# ---------------------------------------------------------------------------
# LOAD INJURY RELATED DATA FROM CSV (re-computed a few times per week)
# ---------------------------------------------------------------------------
def load_injury_map(filepath: str = INJURY_CSV) -> Dict[tuple, Dict[str, Any]]:
    """Reads injury predictions from CSV and returns a lookup dictionary keyed by (clean_name, team, position)."""
    injury_map = {}
    if not os.path.exists(filepath):
        print(f"Warning: Injury CSV '{filepath}' not found. Skipping injury predictions mapping.")
        return injury_map

    try:
        with open(filepath, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p_name = row.get("player_name", "").strip()
                if not p_name:
                    continue

                p_team = row.get("team", "").strip()
                p_pos = row.get("position", "").strip()
                pkey = make_player_key(p_name, p_team, p_pos)

                def parse_float(val: Any) -> Optional[float]:
                    try:
                        return float(val) if val is not None and val != "" else None
                    except ValueError:
                        return None

                injury_map[pkey] = {
                    "projected_games_missed": parse_float(row.get("projected_games_missed")),
                    "in_game_injury_probability_pct": parse_float(row.get("in_game_injury_probability_pct")),
                    "expected_snap_pct_if_active": parse_float(row.get("projected_snap_pct_if_active")),
                    "injury_risk": row.get("injury_risk"),
                }
        print(f"✓ Successfully mapped injury predictions for {len(injury_map)} players.")
    except Exception as e:
        print(f"Error reading injury predictions file '{filepath}': {e}")

    return injury_map


# ---------------------------------------------------------------------------
# LOAD HANDCUFF DATA FROM CSV (computed prior to every run)
# ---------------------------------------------------------------------------
def load_handcuff_map(
    league_data: Dict[str, Any],
    csv_path: str = HANDCUFF_CSV,
) -> Tuple[Dict[tuple, Any], Dict[tuple, Any]]:
    """Loads handcuffs from CSV and maps them to absent starters using (clean_name, team, position) tuple keys."""
    if not os.path.exists(csv_path):
        raise RuntimeError(
            f"Error: Handcuff CSV '{csv_path}' not found. Did you forget to run `python handcuff.py` first?"
        )
        return {}, {}

    proj_points_map = {}
    all_teams = league_data.get("teams", [])

    for team in all_teams:
        if not team:
            continue
        player_list = team.get("roster", {})
        for player in player_list:
            p_team = player.get("team") or player.get("sleeper_nfl_team", "")
            pkey = make_player_key(player.get("name", ""), p_team, player.get("position", ""))
            if pkey[0]:
                proj_points_map[pkey] = float(player.get("projected_points", 0.0))

    for player in league_data.get("free_agents", []):
        p_team = player.get("team") or player.get("sleeper_nfl_team", "")
        pkey = make_player_key(player.get("name", ""), p_team, player.get("position", ""))
        if pkey[0]:
            proj_points_map[pkey] = float(player.get("projected_points", 0.0))

    handcuff_map = {}
    reverse_handcuff_map = {}

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s_team = row.get("team", "")
                starter_key = make_player_key(row.get("absent_starter_name", ""), s_team, row.get("absent_starter_position", row.get("position", "")))
                target_key = make_player_key(row.get("player_name", ""), s_team, row.get("position", ""))

                if not starter_key[0] or not target_key[0]:
                    continue

                multiplier = float(row.get("starter_decay_multiplier", 0.0))
                calc_type = row.get("calculation_type", "replacement")

                starter_proj = proj_points_map.get(starter_key, 0.0)
                target_base_proj = proj_points_map.get(target_key, 0.0)

                contingent_proj = 0.0
                if calc_type == "additive":
                    contingent_proj = target_base_proj + (starter_proj * multiplier)
                else:
                    contingent_proj = starter_proj * multiplier

                if contingent_proj <= 0.0:
                    continue

                # it shouldn't go down
                contingent_proj = max(contingent_proj, target_base_proj)

                entry = {
                    "starter_name": row.get("absent_starter_name"),
                    "starter_team": s_team,
                    "starter_position": row.get("absent_starter_position", row.get("position", "")),
                    "handcuff_name": row.get("player_name"),
                    "handcuff_team": s_team,
                    "handcuff_position": row.get("position"),
                    "contingent_projected_points": round(contingent_proj, 2),
                }

                handcuff_map[starter_key] = entry
                reverse_handcuff_map[target_key] = entry

    except Exception as e:
        print(f"Error loading handcuff CSV '{csv_path}': {e}")

    print(f"✓ Successfully mapped {len(handcuff_map)} handcuffs.")
    return handcuff_map, reverse_handcuff_map


# ---------------------------------------------------------------------------
# FETCH SLEEPER METADATA & TRENDS
# ---------------------------------------------------------------------------
def fetch_sleeper_player_map(
    lookback_hours: int = 24,
    limit: int = 1000,
) -> Dict[tuple, Dict[str, Any]]:
    """Fetches Sleeper trending data and player metadata, returning a lookup map keyed by (clean_name, team, position) tuples."""
    headers = {"User-Agent": "Mozilla/5.0"}

    def fetch_json(url: str) -> Any:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))

    print("Fetching trending player data...")
    adds_url = (
        f"https://api.sleeper.app/v1/players/nfl/trending/add"
        f"?lookback_hours={lookback_hours}&limit={limit}"
    )
    drops_url = (
        f"https://api.sleeper.app/v1/players/nfl/trending/drop"
        f"?lookback_hours={lookback_hours}&limit={limit}"
    )

    trending_adds_data = fetch_json(adds_url) or []
    trending_drops_data = fetch_json(drops_url) or []

    trending_map = {}
    for item in trending_adds_data:
        trending_map[item["player_id"]] = {
            "type": "add",
            "count": item.get("count", 0),
        }
    for item in trending_drops_data:
        player_id = item["player_id"]
        if player_id not in trending_map:
            trending_map[player_id] = {
                "type": "drop",
                "count": item.get("count", 0),
            }

    print("Fetching player data from Sleeper API...")
    all_players = fetch_json("https://api.sleeper.app/v1/players/nfl")
    if not all_players:
        raise RuntimeError("Failed to retrieve Sleeper player data.")
        return {}

    player_map = {}

    for player_id, p in all_players.items():
        first_name = p.get("first_name", "")
        last_name = p.get("last_name", "")
        full_name = p.get("full_name") or f"{first_name} {last_name}".strip()

        if not full_name:
            continue

        trend_info = trending_map.get(player_id, {})
        team = p.get("team", "")
        position = p.get("position", "")
        pkey = make_player_key(full_name, team, position)

        player_map[pkey] = {
            "name": full_name,
            "sleeper_nfl_team": team,
            "sleeper_depth_order": p.get("depth_chart_order"),
            "sleeper_depth_position": p.get("depth_chart_position"),
            "sleeper_position": position,
            "sleeper_trending": trend_info.get("type"),
            "sleeper_trending_count": trend_info.get("count", 0),
            "sleeper_years_exp": p.get("years_exp"),
        }

    print(f"✓ Successfully mapped {len(player_map)} Sleeper players.")
    return player_map


# ---------------------------------------------------------------------------
# POLL ROTOWIRE RSS FEED
# ---------------------------------------------------------------------------
def fetch_rotowire_news() -> List[Dict[str, str]]:
    """Polls the RotoWire NFL RSS feed for recent news articles."""
    feed = feedparser.parse(ROTOWIRE_RSS_URL)
    articles = []

    for entry in feed.entries:
        articles.append({
            "guid": entry.guid,
            "title": entry.title,
            "link": entry.link,
            "description": entry.description,
            "published": getattr(entry, "published", ""),
        })
    return articles


# ---------------------------------------------------------------------------
# DECORATE LEAGUE DATA
# ---------------------------------------------------------------------------
def decorate_player_entry(
    player: Dict[str, Any],
    owner_team_id: int,
    sleeper_map: Dict[tuple, Dict[str, Any]],
    handcuff_map: Dict[tuple, Dict[str, Any]],
    reverse_handcuff_map: Dict[tuple, Dict[str, Any]],
    injury_map: Dict[tuple, Dict[str, Any]],
    news_map: Dict[str, Dict[str, Any]],
) -> None:
    """Decorates a single player entry in place using (clean_name, team, position) key tuples."""
    player_name = player.get("name", "")
    player_team = player.get("team") or player.get("sleeper_nfl_team", "")
    player_pos = player.get("position", "")
    pkey = make_player_key(player_name, player_team, player_pos)

    sleeper_info = sleeper_map.get(pkey, {})

    if owner_team_id != 0:
        player["owner_team_id"] = owner_team_id
    player["is_user_team"] = owner_team_id == TEAM_ID
    player["sleeper_nfl_team"] = sleeper_info.get("sleeper_nfl_team", player_team)
    player["sleeper_depth_order"] = sleeper_info.get("sleeper_depth_order")
    player["sleeper_depth_position"] = sleeper_info.get("sleeper_depth_position")
    player["sleeper_years_exp"] = sleeper_info.get("sleeper_years_exp")
    trending = sleeper_info.get("sleeper_trending")
    if trending in ("add", "drop"):
        player["sleeper_trending"] = trending
        player["sleeper_trending_count"] = sleeper_info.get("sleeper_trending_count", 0)

    if pkey in injury_map:
        injury_info = injury_map[pkey]
        player["projected_games_missed"] = injury_info.get("projected_games_missed")
        player["expected_snap_pct_if_active"] = injury_info.get("expected_snap_pct_if_active")
        player["injury_risk"] = injury_info.get("injury_risk")

    if pkey in handcuff_map:
        hc_entry = handcuff_map[pkey]
        hc_name = hc_entry.get("handcuff_name")
        if hc_name:
            player["handcuff_name"] = hc_name

    contingent_proj = 0.0
    if pkey in reverse_handcuff_map:
        hc_entry = reverse_handcuff_map[pkey]
        starter_name = hc_entry.get("starter_name")
        starter_team = hc_entry.get("starter_team", player_team)
        starter_pos = hc_entry.get("starter_position", player_pos)
        contingent_proj = hc_entry.get("contingent_projected_points", 0.0)

        if starter_name:
            player["handcuff_for_starter_name"] = starter_name
        if contingent_proj > 0.0:
            player["contingent_projected_points"] = round(contingent_proj, 2)

        # If our starter is in the news in a negative way, set is_gaining_volume
        starter_key = make_player_key(starter_name, player_team, player_pos)
        if starter_key in news_map:
            news = news_map[starter_key]
            if news.get("is_missing_game", False) or news.get("is_losing_volume", False):
                print(f" - Starter {starter_name} is missing time/losing volume; {player_name} is set to gain volume.")
                player["is_gaining_volume"] = True

    if pkey in news_map:
        news = news_map[pkey]
        if news.get("is_missing_game", False):
            player["is_missing_game"] = True
            # Nuke point projection
            player["projected_points"] = 0.0
            print(f" - Player {player_name} is missing a game.")
        if news.get("is_losing_volume", False):
            player["is_losing_volume"] = True
            print(f" - Player {player_name} is losing volume.")
        if news.get("is_gaining_volume", False):
            print(f" - Player {player_name} is gaining volume.")
            player["is_gaining_volume"] = True


# ---------------------------------------------------------------------------
# NORMALIZE ALL PLAYER VALUES USED WITH SOLVER
# ---------------------------------------------------------------------------
def normalize_player_values(
    league_data: Dict[str, Any],
    keys_to_normalize: List[str] = [
        "projected_points",
        "projected_ceiling_points",
        "projected_ppg",
        "ytd_ppg",
        "sleeper_trending_count",
        "contingent_projected_points",
    ],
) -> Dict[str, Any]:
    """Dynamically normalizes specified numerical keys on all player objects

    (rostered players and free agents) to a [0.0, 1.0] scale using min-max scaling.

    Treats sleeper_trending_count as 0.0 if sleeper_trending is not 'add'.
    """
    reg_season_count = league_data.get("reg_season_count", 0)
    denominator = float(max(reg_season_count - 1, 1))

    # 1. Collect all player dictionaries across user team, opponent teams, and free agents
    all_players: List[Dict[str, Any]] = []

    teams = []
    if league_data.get("user_team"):
        teams.append(league_data["user_team"])
    if league_data.get("opponent_teams"):
        teams.extend(league_data["opponent_teams"])
    if league_data.get("teams"):
        teams.extend(league_data["teams"])

    for team in teams:
        roster = team.get("roster", {})
        if isinstance(roster, dict):
            for list_key in ["starters", "bench", "IR"]:
                all_players.extend(roster.get(list_key, []))
        elif isinstance(roster, list):
            all_players.extend(roster)

    # Extract free agents
    all_players.extend(league_data.get("free_agents", []))

    # 2. Helper function to pull/derive raw values with trend gate
    def get_raw_value(player: Dict[str, Any], key: str) -> float:
        if key == "projected_ppg":
            total_pts = float(player.get("projected_total_points") or player.get("total_points") or 0.0)
            player["projected_ppg"] = round(total_pts / denominator, 2)
            return player["projected_ppg"]

        if key == "sleeper_trending_count":
            # Check both possible key names for trending type ('sleeper_trending' or 'sleeper_trending_type')
            trending_type = player.get("sleeper_trending") or player.get("sleeper_trending_type")
            if str(trending_type).lower() != "add":
                return 0.0

        return float(player.get(key) or 0.0)

    # 3. Find global min/max bounds across all players for each key
    min_max_map: Dict[str, Tuple[float, float]] = {}

    for key in keys_to_normalize:
        values = [get_raw_value(p, key) for p in all_players]
        min_val = min(values) if values else 0.0
        max_val = max(values) if values else 0.0
        min_max_map[key] = (min_val, max_val)

    # 4. Perform dynamic min-max normalization in-place on each player object
    for player in all_players:
        for key in keys_to_normalize:
            raw_val = get_raw_value(player, key)
            min_val, max_val = min_max_map[key]

            # Avoid division by zero if all player values are equal
            if max_val > min_val:
                norm_val = (raw_val - min_val) / (max_val - min_val)
            else:
                norm_val = 0.0

            player[f"norm_{key}"] = round(norm_val, 4)

    return league_data


def process_league_data(
    league_data: Dict[str, Any],
    sleeper_map: Dict[tuple, Dict[str, Any]],
    handcuff_map: Dict[tuple, Dict[str, Any]],
    reverse_handcuff_map: Dict[tuple, Dict[str, Any]],
    injury_map: Dict[tuple, Dict[str, Any]],
    news_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """In-place decorates league team rosters and free agency list."""
    user_team = None
    opponent_teams = []

    for team in league_data["teams"]:
        starters = []
        bench = []
        ir = []

        # Handle roster whether it is a list or pre-parsed dictionary
        roster_input = team.get("roster", [])
        if isinstance(roster_input, dict):
            raw_players = (
                roster_input.get("starters", [])
                + roster_input.get("bench", [])
                + roster_input.get("IR", [])
            )
        else:
            raw_players = roster_input

        for player in raw_players:
            decorate_player_entry(
                player,
                team.get("team_id"),
                sleeper_map,
                handcuff_map,
                reverse_handcuff_map,
                injury_map,
                news_map
            )

            slot = player.get("slot_position", "BE")

            if slot == "IR":
                ir.append(player)
            elif slot == "BE":
                bench.append(player)
            else:
                starters.append(player)

        team["roster"] = {
            "starters": starters,
            "bench": bench,
            "IR": ir,
        }

        if str(team.get("team_id")) == str(TEAM_ID) or team.get("id") == TEAM_ID:
            user_team = team
        else:
            opponent_teams.append(team)

    del league_data["teams"]
    league_data["user_team"] = user_team
    league_data["opponent_teams"] = opponent_teams

    for player in league_data.get("free_agents", []):
        decorate_player_entry(
            player,
            0,
            sleeper_map,
            handcuff_map,
            reverse_handcuff_map,
            injury_map,
            news_map
        )

    free_agents = league_data.pop("free_agents", None)
    recent_activity = league_data.pop("recent_activity", None)
    if free_agents is not None:
        league_data["free_agents"] = free_agents
    if recent_activity is not None:
        league_data["recent_activity"] = recent_activity

    return normalize_player_values(league_data)


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def manage_team(
    interactive: bool = False,
    debug: bool = False,
    fake_news_file: Optional[str] = None,
    roster_file: Optional[str] = None,
) -> None:
    print("Polling Fantasy League Information...")
    league_data = fetch_league_data()

    print("Polling Sleeper Player Metadata And Trends Information...")
    sleeper_map = fetch_sleeper_player_map()

    print("Loading Injury Predictions Data...")
    injury_map = load_injury_map(INJURY_CSV)

    print("Loading Handcuff Data...")
    handcuff_map, reverse_handcuff_map = load_handcuff_map(league_data, HANDCUFF_CSV)

    print("Polling RotoWire's RSS Feed...")
    news_data = fetch_rotowire_news()

    # Inject fake news stories prior to 1st pass inference if specified
    if fake_news_file:
        fake_stories = load_fake_news_file(fake_news_file)
        news_data = fake_stories + news_data

    # Pass 1: Perform deterministic pre-evaluation on news articles
    print("Executing Pass 1: Evaluating RotoWire news items with Gemini...")
    news_map = {}
    try:
        news_map = evaluate_rotowire_news(news_data, clean_player_name)
    except Exception as e:
        # Don't blow up though. There's still useful work we could do without Gemini
        print(f"❌ Failed to evaluate RotoWire News: {e}")

    print("Processing League Data and Decorating with Sleeper attributes, Injury Predictions, Handcuffs and News Impacts...")
    league_data = process_league_data(league_data, sleeper_map, handcuff_map, reverse_handcuff_map, injury_map, news_map)

    # Inject roster file overrides if specified
    if roster_file:
        apply_roster_overrides(league_data, roster_file)

    settings = league_data.get("settings", {})
    position_slot_counts = settings.get("position_slot_counts", {})
    is_te_premium = settings.get("is_te_premium", False)

    user_team = league_data.get("user_team")
    roster = user_team.get("roster", {}) if user_team else {}
    free_agents = league_data.get("free_agents", [])
    opponent_teams = league_data.get("opponent_teams")

    # Determine bench slot policies given position slot counts and number of teams in the league
    num_teams = len(opponent_teams) + 1
    bench_slot_policies = determine_bench_policies(position_slot_counts, num_teams, is_te_premium)

    # Override lineup locks if it's Tuesday so we can find out which waiver claims we need to make
    unlock_lineups_if_tuesday()

    # Calculate optimal roster
    print("Optimizing Roster...")
    starters_res, bench_res, ir_res, drops_res = optimize_roster(
        roster.get("starters", []),
        roster.get("bench", []),
        roster.get("IR", []),
        free_agents,
        position_slot_counts,
        bench_slot_policies,
        max_weekly_moves=8,
    )

    upside_flyers = []
    if TRIGGER_REASON == SCHEDULED_RUN_REASON:
        upside_flyers = calculate_upside_flyers(free_agents, is_te_premium, 10)

    # Pare down data we send to inference
    free_agents_limit = 25 if TRIGGER_REASON == SCHEDULED_RUN_REASON else 10
    league_data["free_agents"] = free_agents[:free_agents_limit]
    if TRIGGER_REASON != SCHEDULED_RUN_REASON:
        league_data.pop("recent_activity", None)
        for team in league_data.get("opponent_teams", []):
            team.pop("roster", None)

    # JSON data blob appended to our system prompt
    payload = {
        "hints": {
            "optimal_roster": {
                "starters": starters_res,
                "bench": bench_res,
                "ir": ir_res,
                "drops": drops_res,
            },
        },
        "league_data": league_data,
        "rotowire_news": news_data,
    }

    if upside_flyers:
        payload["hints"].update({"upside_flyers": upside_flyers})

    if debug:
        print_debug_prompt(payload, interactive=interactive)
        return

    if interactive:
        start_interactive_mode(payload)
        return

    print("Executing Pass 2: Sending complete snapshot to Gemini for evaluation...")
    evaluation_json = evaluate_team(payload)

    # Post-process the JSON result to create a Discord embed
    discord_embed_json = generate_discord_embed(
        evaluation_json,
        starters_res,
        bench_res,
        ir_res,
        drops_res,
        roster.get("starters", []),
        roster.get("bench", []),
        bench_slot_policies,
        upside_flyers,
    )

    console = Console()
    try:
        markdown_summary = format_embed_to_markdown(discord_embed_json)
        console.print(Markdown(markdown_summary))
    except Exception:
        console.print(discord_embed_json)

    print("Sending embed to Webhook...")
    post_to_webhook(discord_embed_json)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NFL Fantasy Team Manager")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Print full payload without executing LLM pass 2.",
    )
    parser.add_argument(
        "-f",
        "--fake-news",
        dest="fake_news_file",
        metavar="<input_file.txt>",
        type=str,
        help="Path to text file containing fake news stories to inject before 1st pass inference.",
    )
    parser.add_argument(
        "-r",
        "--roster",
        dest="roster_file",
        metavar="<roster.txt>",
        type=str,
        help="Path to text file containing custom roster overrides.",
    )

    args = parser.parse_args()
    manage_team(
        interactive=args.interactive,
        debug=args.debug,
        fake_news_file=args.fake_news_file,
        roster_file=args.roster_file,
    )
