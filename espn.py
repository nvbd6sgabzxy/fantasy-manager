import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import requests
from espn_api.football import League

# ESPN Environment Configurations
LEAGUE_ID = int(os.getenv("ESPN_LEAGUE_ID", "0"))
TEAM_ID = int(os.getenv("ESPN_TEAM_ID", "0"))
YEAR = int(os.getenv("ESPN_YEAR", "2026"))
ESPN_S2 = os.getenv("ESPN_S2")
SWID = os.getenv("ESPN_SWID")

DEFAULT_POSITION_MAP = {
    0: "QB",     # Generic / Unassigned QB
    1: "QB",     # Quarterback
    2: "RB",     # Running Back
    3: "WR",     # Wide Receiver
    4: "TE",     # Tight End
    5: "K",      # Kicker
    6: "TQB",    # Team Quarterback
    7: "EDR",    # Edge Rusher (IDP)
    8: "DT",     # Defensive Tackle (IDP)
    9: "DE",     # Defensive End (IDP)
    10: "DE",    # Defensive End (IDP) (really OLB)
    11: "LB",    # Linebacker (IDP)
    12: "CB",    # Cornerback (IDP)
    13: "S",     # Safety (IDP)
    14: "P",     # Punter
    15: "DL",    # Defensive Line (Generic IDP)
    16: "D/ST",  # Team Defense / Special Teams
    17: "DB",    # Defensive Back (Generic IDP)
}

PRO_TEAM_MAP = {
    0: "FA",   # Free Agent / Unassigned
    1: "ATL",  # Atlanta Falcons
    2: "BUF",  # Buffalo Bills
    3: "CHI",  # Chicago Bears
    4: "CIN",  # Cincinnati Bengals
    5: "CLE",  # Cleveland Browns
    6: "DAL",  # Dallas Cowboys
    7: "DEN",  # Denver Broncos
    8: "DET",  # Detroit Lions
    9: "GB",   # Green Bay Packers
    10: "TEN", # Tennessee Titans
    11: "IND", # Indianapolis Colts
    12: "KC",  # Kansas City Chiefs
    13: "LV",  # Las Vegas Raiders
    14: "LAR", # Los Angeles Rams
    15: "MIA", # Miami Dolphins
    16: "MIN", # Minnesota Vikings
    17: "NE",  # New England Patriots
    18: "NO",  # New Orleans Saints
    19: "NYG", # New York Giants
    20: "NYJ", # New York Jets
    21: "PHI", # Philadelphia Eagles
    22: "ARI", # Arizona Cardinals
    23: "PIT", # Pittsburgh Steelers
    24: "LAC", # Los Angeles Chargers
    25: "SF",  # San Francisco 49ers
    26: "SEA", # Seattle Seahawks
    27: "TB",  # Tampa Bay Buccaneers
    28: "WAS", # Washington Commanders
    29: "CAR", # Carolina Panthers
    30: "JAX", # Jacksonville Jaguars
    33: "BAL", # Baltimore Ravens
    34: "HOU", # Houston Texans
}


def detect_league_features_and_prices(
    league: League,
) -> Tuple[bool, int, bool, Dict[int, float]]:
    """Detects draft settings, keeper status, and player draft/keeper prices via API endpoints."""
    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        f"seasons/{league.year}/segments/0/leagues/{league.league_id}"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36"
        )
    }

    # Include mTransactions2 to catch keeper transaction events shown in player activity logs
    params = [
        ("view", "mSettings"),
        ("view", "mKeeper"),
        ("view", "mDraftDetail"),
        ("view", "mTransactions2"),
    ]

    player_prices: Dict[int, float] = {}
    is_auction = False
    auction_budget = 200
    has_keepers = False

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            cookies=league.espn_request.cookies,
        )
        response.raise_for_status()
        data = response.json()

        settings = data.get("settings", {})

        # 1. Auction Draft & Budget Detection
        draft_settings = settings.get("draftSettings", {})
        draft_type = str(draft_settings.get("type", "")).upper()
        if "AUCTION" in draft_type:
            is_auction = True

        auction_budget = draft_settings.get("auctionBudget", 200)

        # 2. Check Native keeperSettings
        keeper_settings = settings.get("keeperSettings", {})
        if keeper_settings.get("keeperCount", 0) > 0 or keeper_settings.get("enabled", False):
            has_keepers = True

        # 3. Check Transactions Endpoint (Matches screenshot log: "Selected as Keeper for $X")
        transactions = data.get("transactions", [])
        for tx in transactions:
            tx_type = str(tx.get("type", "")).upper()
            execution_type = str(tx.get("executionType", "")).upper()

            # Check for keeper transaction types or descriptions
            if "KEEPER" in tx_type or "KEEPER" in execution_type:
                has_keepers = True

            # Extract player keeper prices from transaction items if available
            for item in tx.get("items", []):
                if item.get("type") == "KEEPER":
                    has_keepers = True
                    p_id = item.get("playerId")
                    price = item.get("price") or item.get("bidAmount")
                    if p_id and price is not None:
                        player_prices[p_id] = float(price)

        # 4. Check Draft Pick Items for Reserved Keepers and Bid Amounts
        draft_detail = data.get("draftDetail", {})
        picks = draft_detail.get("picks", [])

        for pick in picks:
            p_id = pick.get("playerId")
            bid = pick.get("bidAmount", 0)
            is_keeper = pick.get("reservedForKeeper", False) or pick.get("keeper", False)

            if is_keeper:
                has_keepers = True

            if p_id and bid > 0:
                player_prices[p_id] = float(bid)

    except Exception as e:
        print(f"Warning: Error inspecting keeper settings and transactions: {e}")

    return is_auction, auction_budget, has_keepers, player_prices


def _parse_player_item(item: Dict[str, Any], current_week: int, season_year: int) -> Optional[Dict[str, Any]]:
    """Helper function to parse a raw ESPN kona_player_info item into a unified player dictionary."""
    p_entry = item.get("playerPoolEntry", {})
    p_data = p_entry.get("player") or item.get("player") or {}

    player_id = p_data.get("id")
    player_name = p_data.get("fullName")
    if not player_name or not player_id:
        return None

    # Position mapping from ESPN defaultPositionId
    pos_id = p_data.get("defaultPositionId")
    position = DEFAULT_POSITION_MAP.get(pos_id, "N/A")

    # Team mapping from ESPN proTeamId
    pro_team_id = p_data.get("proTeamId", 0)
    team_abbrev = PRO_TEAM_MAP.get(pro_team_id, "FA")

    # Projected points calculation for current scoring period / week and YTD PPG
    projected_points = 0.0
    projected_ceiling_points = 0.0
    projected_total_points = 0.0
    ytd_ppg = 0.0

    for stats in p_data.get("stats", []):
        if stats.get("seasonId", 0) != season_year:
            continue

        stat_source = stats.get("statSourceId")
        stat_split = stats.get("statSplitTypeId", 0)
        scoring_period = stats.get("scoringPeriodId", 0)
        applied_total = float(stats.get("appliedTotal", 0.0))
        applied_total_ceiling = float(stats.get("appliedTotalCeiling", 0.0))

        if stat_source == 1 and scoring_period == 0:
            projected_total_points = applied_total
        if stat_source == 1 and scoring_period == current_week:
            projected_points = applied_total
            projected_ceiling_points = applied_total_ceiling

        # statSourceId == 0 represents actual played game statistics
        # Extract pre-computed YTD PPG directly from ESPN's full-season summary block
        if stat_source == 0 and stat_split == 0 and scoring_period == 0:
            if "appliedAverage" in stats:
                ytd_ppg = round(float(stats.get("appliedAverage", 0.0)), 2)

    # Extract Ownership Stats
    ownership = p_data.get("ownership", {})
    percent_owned = float(ownership.get("percentOwned", 0.0))
    percent_started = float(ownership.get("percentStarted", 0.0))

    # Extract Waiver and Availability Status
    status = item.get("status") or p_entry.get("status", "ONTEAM")
    wpd_ms = item.get("waiverProcessDate", 0) or p_entry.get("waiverProcessDate", 0)

    # Injury Status
    injury_status = p_data.get("injuryStatus", "NORMAL")

    # Lineup locked
    is_lineup_locked = item.get("lineupLocked") if "lineupLocked" in item else p_entry.get("lineupLocked", False)

    entry = {
        "player_id": player_id,
        "name": player_name,
        "team": team_abbrev,
        "position": position,
        "projected_points": round(projected_points, 2),
        "projected_ceiling_points": round(projected_ceiling_points, 2),
        "total_points": round(projected_total_points, 2),
        "ytd_ppg": ytd_ppg,
        "injury_status": injury_status,
        "percent_owned": percent_owned,
        "percent_started": percent_started,
        "is_lineup_locked": is_lineup_locked,
        "status": status,
    }

    if wpd_ms != 0:
        dt_utc = datetime.fromtimestamp(wpd_ms / 1000, tz=timezone.utc)
        entry["waiver_process_date"] = dt_utc.strftime("%a, %b %d, %Y at %I:%M %p %Z")

    return entry


def fetch_all_league_players(league: League, limit: int = 2000) -> List[Dict[str, Any]]:
    """Fetches all players (rostered, free agents, waivers) using ESPN's robust kona_player_info endpoint in a single request."""
    # Build array of all played/current scoring period IDs up to the current week
    past_and_current_weeks = list(range(1, max(league.current_week + 1, 2)))

    filter_header = {
        "players": {
            "filterSlotIds": {"value": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24]},
            "limit": limit,
            "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
            "filterStats": {
                "filterStatsForSeasonIds": {"value": [league.year]},
                "filterStatsForTopScoringPeriodIds": {
                    "value": len(past_and_current_weeks),
                    "additionalValue": past_and_current_weeks
                }
            }
        }
    }

    headers = {
        "x-fantasy-filter": json.dumps(filter_header),
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36"
        ),
    }

    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        f"seasons/{league.year}/segments/0/leagues/{league.league_id}"
    )

    response = requests.get(
        url,
        params=[("view", "kona_player_info"), ("view", "mMatchupScore")],
        headers=headers,
        cookies=league.espn_request.cookies,
    )
    response.raise_for_status()
    raw_data = response.json()

    players = []
    for item in raw_data.get("players", []):
        parsed = _parse_player_item(item, league.current_week, league.year)
        if parsed:
            players.append(parsed)

    return players


def fetch_free_agents_with_waiver_status(league: League, players_cache: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Filters free agents and waiver wire players from the pre-fetched player data list."""
    if players_cache is None:
        players_cache = fetch_all_league_players(league)

    free_agents = [
        p for p in players_cache
        if p.get("status") in ("FREEAGENT", "WAIVERS")
    ]

    # Sort free agents descending by projected points
    free_agents.sort(key=lambda x: x["projected_points"], reverse=True)
    return free_agents


def fetch_player_outlooks(
    league: League,
    limit: int = 1000,
) -> Dict[str, str]:
    """Fetches season and weekly outlook text directly from ESPN's player payload."""
    filter_header = {
        "players": {
            "limit": limit,
            "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
            "filterIncludeEssentially": {"value": ["outlooks"]},
        }
    }

    headers = {
        "x-fantasy-filter": json.dumps(filter_header),
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36"
        ),
    }

    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        f"seasons/{league.year}/segments/0/leagues/{league.league_id}"
    )

    response = requests.get(
        url,
        params={"view": "kona_player_info"},
        headers=headers,
        cookies=league.espn_request.cookies,
    )

    response.raise_for_status()
    raw_data = response.json()

    outlooks = {}

    for item in raw_data.get("players", []):
        player_data = (
            item.get("player")
            or item.get("playerPoolEntry", {}).get("player")
            or {}
        )

        player_name = player_data.get("fullName")
        if not player_name:
            continue

        # 1. Grab 'seasonOutlook' directly from player_data
        outlook_text = player_data.get("seasonOutlook")

        # 2. Fallback to weekly outlooks dictionary if seasonOutlook is not present
        if not outlook_text and "outlooks" in player_data:
            outlooks_dict = player_data["outlooks"].get("outlooksByWeek", {})
            outlook_text = outlooks_dict.get("0") or next(
                iter(outlooks_dict.values()), None
            )

        # 3. Fallback to general 'outlook' key
        if not outlook_text:
            outlook_text = player_data.get("outlook")

        if outlook_text:
            outlooks[player_name] = outlook_text.strip()

    return outlooks


def fetch_pending_waiver_claims(
    league: League, team_id: int, player_name_map: Dict[int, str]
) -> List[Dict[str, Any]]:
    """Fetches pending waiver wire claims for a specific team ID."""
    if not team_id:
        print("Warning: ESPN_TEAM_ID environment variable is not set or is 0.")
        return []

    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        f"seasons/{league.year}/segments/0/leagues/{league.league_id}"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36"
        ),
    }

    # Pass mTeam, mRoster, and mPendingTransactions view params
    params = [
        ("view", "mTeam"),
        ("view", "mRoster"),
        ("view", "mPendingTransactions"),
        # Query next scoring period if claims are scheduled for the upcoming week
        ("scoringPeriodId", str(league.current_week + 1)),
    ]

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            cookies=league.espn_request.cookies,
        )
        response.raise_for_status()
        data = response.json()

        # Check for top-level transactions array or team-level pendingTransactions
        pending_transactions = data.get("transactions", [])
        if not pending_transactions:
            for team in data.get("teams", []):
                if team.get("id") == team_id:
                    pending_transactions = (
                        team.get("pendingTransactions")
                        or team.get("transactions")
                        or []
                    )
                    break

        # Filter active claims for the specified team
        my_claims = [
            tx for tx in pending_transactions
            if tx.get("teamId") == team_id or not tx.get("teamId")
        ]

        def resolve_player_name(p_id: int) -> str:
            if not p_id:
                return "Unknown Player"

            # Primary lookup from existing name map
            if p_id in player_name_map:
                return player_name_map[p_id]

            # Fallback: On-demand API lookup only if p_id is unmapped
            try:
                info = league.player_info(playerId=p_id)
                if info and hasattr(info, "name") and info.name:
                    player_name_map[p_id] = info.name
                    return info.name
            except Exception:
                pass

            return f"Player #{p_id}"

        formatted_claims = []
        for claim in my_claims:
            adds = []
            drops = []
            for item in claim.get("items", []):
                p_id = item.get("playerId")
                p_name = resolve_player_name(p_id)

                # Check move types (ADD / DROP or moveTypeId mappings)
                move_type = item.get("type")
                if move_type == "ADD" or item.get("moveTypeId") == 1:
                    adds.append(p_name)
                elif move_type == "DROP" or item.get("moveTypeId") == 2:
                    drops.append(p_name)

            formatted_claims.append(
                {
                    "transaction_id": claim.get("id"),
                    "team_id": claim.get("teamId", team_id),
                    "type": claim.get("type"),
                    "status": claim.get("status"),
                    "bid_amount": claim.get("bidAmount", 0),
                    "execution_type": claim.get("executionType"),
                    "adds": adds,
                    "drops": drops,
                }
            )

        return formatted_claims
    except Exception as e:
        print(f"Warning: Could not fetch pending waiver claims: {e}")
        return []


def fetch_league_data() -> Dict[str, Any]:
    """Polls ESPN Fantasy League data including rosters, matchups, and free agents."""
    # Initialize League (supports both public and private leagues via cookies)
    kwargs = {"league_id": LEAGUE_ID, "year": YEAR}
    if ESPN_S2 and SWID:
        kwargs["espn_s2"] = ESPN_S2
        kwargs["swid"] = SWID

    league = League(**kwargs)

    print(f"Current League Week: {league.current_week}")

    # Single primary request to fetch all player data across the league
    all_players = fetch_all_league_players(league)
    all_players_map = {p["player_id"]: p for p in all_players}

    # Detect auction, default auction budget, keeper settings & player draft prices
    is_auction, draft_budget, has_keepers, player_prices = (
        detect_league_features_and_prices(league)
    )

    raw_team_url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        f"seasons/{league.year}/segments/0/leagues/{league.league_id}"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36"
        )
    }
    params = [("view", "mTeam"), ("view", "mSettings"), ("view", "mRoster")]

    raw_response = requests.get(
        raw_team_url,
        params=params,
        headers=headers,
        cookies=league.espn_request.cookies,
    )
    raw_response.raise_for_status()
    raw_json_data = raw_response.json()

    settings = raw_json_data.get("settings")
    if not settings:
        raise KeyError(
            "API payload missing 'settings' object. Ensure 'view=mSettings' parameter is passed."
        )

    acquisition_settings = settings.get("acquisitionSettings")
    if not acquisition_settings:
        raise KeyError(
            "API payload missing 'settings.acquisitionSettings'. Check league configuration."
        )

    if "acquisitionBudget" not in acquisition_settings:
        raise KeyError(
            "Field 'acquisitionBudget' not found in 'acquisitionSettings'. Verify FAAB is enabled."
        )

    default_starting_budget = acquisition_settings["acquisitionBudget"]

    teams_payload = raw_json_data.get("teams")
    if not teams_payload:
        raise ValueError("API payload missing 'teams' list in raw league response.")

    faab_map = {}
    for team_data in teams_payload:
        t_id = team_data.get("id")
        if t_id is None:
            raise KeyError("Found team object missing an 'id' field in payload.")

        # 1. Direct root key read
        budget = team_data.get("acquisitionBudget")

        # 2. Check transactionCounter dictionary key
        if budget is None or budget == 0:
            budget = team_data.get("transactionCounter", {}).get(
                "acquisitionBudget"
            )

        # 3. Calculate from exact starting budget if ESPN defaults direct keys to 0
        if budget is None or budget == 0:
            tx_counter = team_data.get("transactionCounter")
            if tx_counter is None or "acquisitionBudgetSpent" not in tx_counter:
                raise KeyError(
                    f"Team ID {t_id} missing 'transactionCounter.acquisitionBudgetSpent'. "
                    "Cannot compute remaining FAAB budget."
                )
            spent = tx_counter["acquisitionBudgetSpent"]
            budget = default_starting_budget - spent

        faab_map[t_id] = budget

    te_ppr = 0.0
    standard_ppr = 0.0

    scoring_items = settings.get("scoringSettings", {}).get("scoringItems", [])

    for item in scoring_items:
        # Stat ID 53 represents Receptions across ESPN Fantasy FFL
        if str(item.get("statId")) == "53":
            base_points = float(item.get("points", 0.0))
            raw_overrides = item.get("pointsOverrides", {})

            # Normalize keys to strings to prevent int/str mismatch
            overrides = {str(k): float(v) for k, v in raw_overrides.items()}

            # Position IDs: "3" = WR, "2" = RB, "4" = TE
            wr_ppr = overrides.get("3", overrides.get("2", base_points))
            te_ppr = overrides.get("4", base_points)

            standard_ppr = float(wr_ppr)
            te_ppr = float(te_ppr)
            break

    # Fallback to espn_api league settings if standard_ppr remained 0
    if standard_ppr == 0.0 and te_ppr == 0.0 and hasattr(league, "settings"):
        scoring_dict = getattr(league.settings, "scoring_settings", {})
        if isinstance(scoring_dict, dict):
            standard_ppr = float(
                scoring_dict.get(53, scoring_dict.get("53", scoring_dict.get("REC", 0.0)))
            )
            te_ppr = standard_ppr

    is_te_premium = te_ppr > standard_ppr

    # Scoring parameters
    scoring_settings = {
        "scoring_type": getattr(league.settings, "scoring_type", "N/A"),
        "reg_season_count": getattr(league.settings, "reg_season_count", 0),
        "position_slot_counts": getattr(
            league.settings, "position_slot_counts"
        ),
        "is_te_premium": is_te_premium,
    }

    # Matchup Schedule Data
    schedule_data = []
    try:
        box_scores = league.box_scores(week=league.current_week)
        for box in box_scores:
            home_name = box.home_team.team_name if box.home_team else "N/A"
            away_name = box.away_team.team_name if box.away_team else "N/A"

            schedule_data.append(
                {
                    "home_team": home_name,
                    "home_score": box.home_score,
                    "away_team": away_name,
                    "away_score": box.away_score,
                }
            )
    except Exception as e:
        raise RuntimeError(f"Warning: Could not fetch schedule/box scores: {e}")

    # Map player IDs to names for resolving pending claim IDs
    player_id_to_name = {}

    # Teams and Rosters populated via the single kona_player_info player lookup
    teams_data = []
    for team in league.teams:
        roster_list = []
        for p in team.roster:
            p_id = getattr(p, "playerId", None)
            player_id_to_name[p_id] = p.name

            # Lookup pre-fetched robust stats or construct baseline fallback
            p_details = all_players_map.get(p_id, {})

            # Resolve team abbreviation
            p_team_id = getattr(p, "proTeamId", 0)
            team_abbrev = p_details.get("team") or PRO_TEAM_MAP.get(p_team_id, "FA")

            player_entry = {
                "player_id": p_id,
                "name": p.name,
                "team": team_abbrev,
                "position": p.position,
                "slot_position": p.lineupSlot,
                "acquisition_type": p.acquisitionType,
                "projected_points": p_details.get("projected_points", 0.0),
                "projected_ceiling_points": p_details.get("projected_ceiling_points", 0.0),
                "total_points": p_details.get("total_points", 0.0),
                "ytd_ppg": p_details.get("ytd_ppg", 0.0),
                "injury_status": p_details.get("injury_status", p.injuryStatus),
                "percent_owned": p_details.get("percent_owned", p.percent_owned),
                "percent_started": p_details.get("percent_started", p.percent_started),
                "is_lineup_locked": p_details.get("is_lineup_locked", False),
            }

            if has_keepers and p.acquisitionType != "ADD":
                price = player_prices.get(p_id, None)
                if price:
                    player_entry["keeper_price"] = price

            roster_list.append(player_entry)

        if team.team_id not in faab_map:
            raise KeyError(
                f"Team ID {team.team_id} ({team.team_name}) not resolved in faab_map."
            )

        teams_data.append(
            {
                "team_id": team.team_id,
                "team_name": team.team_name,
                "wins": team.wins,
                "losses": team.losses,
                "acquisition_budget": faab_map[team.team_id],
                "roster": roster_list,
            }
        )

    # Available Free Agents / Waiver Players filtered from pre-fetched player cache
    free_agents = fetch_free_agents_with_waiver_status(league, players_cache=all_players)
    for fa in free_agents:
        player_id_to_name[fa["player_id"]] = fa["name"]

    # Recent League Activity
    activity_data = []
    try:
        activities = league.recent_activity(size=50)
        for act in activities:
            activity_data.append(str(act))
    except Exception as e:
        print(f"Warning: Could not fetch recent activity: {e}")

    # Pending Waiver Claims
    pending_claims = fetch_pending_waiver_claims(
        league=league, team_id=TEAM_ID, player_name_map=player_id_to_name
    )

    return {
        "league_id": LEAGUE_ID,
        "year": YEAR,
        "is_auction_draft": is_auction,
        "default_auction_budget": draft_budget,
        "has_keepers": has_keepers,
        "settings": scoring_settings,
        "teams": teams_data,
        "free_agents": free_agents,
        "recent_activity": activity_data,
        "current_matchups": schedule_data,
        "pending_waiver_claims": pending_claims,
        "reg_season_count": league.settings.reg_season_count,
    }