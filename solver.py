import math
from enum import Enum
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import zoneinfo

import pulp
from handcuff import clean_player_name, make_player_key

# ---------------------------------------------------------------------------
# GLOBALS & HELPER FUNCTIONS
# ---------------------------------------------------------------------------
NOW_UTC = datetime.now(timezone.utc)
OVERRIDE_LINEUP_LOCKED = False

SLOT_ELIGIBILITY_MAP: Dict[str, List[str]] = {
    "K": ["K"],
    "P": ["P"],
    "S": ["S"],
    "QB": ["QB"],
    "RB": ["RB"],
    "WR": ["WR"],
    "TE": ["TE"],
    "HC": ["HC"],
    "DT": ["DT"],
    "DE": ["DE"],
    "LB": ["LB"],
    "CB": ["CB"],
    "OP": ["QB", "RB", "WR", "TE"],
    "ER": ["RB", "WR", "TE"],
    "DL": ["DT", "DE", "DL"],
    "DB": ["CB", "S", "DB"],
    "DP": ["DT", "DE", "LB", "CB", "S", "DL", "DB"],
    "BE": [],
    "IR": [],
    "TQB": ["QB"],
    "D/ST": ["D/ST"],
    "RB/WR": ["RB", "WR"],
    "WR/TE": ["WR", "TE"],
    "RB/WR/TE": ["RB", "WR", "TE"],
}

def get_player_key_from_dict(player: Dict[str, Any]) -> tuple:
    """Extracts a standardized player key tuple from a player dictionary."""
    p_name = player.get("name", "")
    p_team = player.get("team") or player.get("sleeper_nfl_team") or player.get("nfl_team", "")
    p_pos = player.get("position", "")
    return make_player_key(p_name, p_team, p_pos)


# ---------------------------------------------------------------------------
# WAIVER & LINEUP LOCK HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def unlock_lineups_if_tuesday() -> None:
    """
    Sets a global boolean OVERRIDE_LINEUP_LOCKED to True if it's after Tuesday 6 AM
    Pacific time and before Wednesday 6am Pacific time. Otherwise sets it to True.
    """
    global OVERRIDE_LINEUP_LOCKED
    pacific_tz = zoneinfo.ZoneInfo("America/Los_Angeles")
    now_pacific = NOW_UTC.astimezone(pacific_tz)

    days_since_tuesday = (now_pacific.weekday() - 1) % 7
    tuesday_6am = (now_pacific - timedelta(days=days_since_tuesday)).replace(
        hour=6, minute=0, second=0, microsecond=0
    )
    wednesday_6am = tuesday_6am + timedelta(days=1)

    if tuesday_6am <= now_pacific < wednesday_6am:
        OVERRIDE_LINEUP_LOCKED = True
    else:
        OVERRIDE_LINEUP_LOCKED = False


def is_lineup_locked(player: Dict[str, Any]) -> bool:
    """
    Determines if a player's lineup status is currently locked.
    If OVERRIDE_LINEUP_LOCKED is True, returns False; otherwise returns player's
    'is_lineup_locked' status (defaulting to False).
    """
    if OVERRIDE_LINEUP_LOCKED:
        return False
    return player.get("is_lineup_locked", False)


# ---------------------------------------------------------------------------
# BENCH POLICIES & SUITABILITY SCORING
# ---------------------------------------------------------------------------
class BenchSlotPolicy(Enum):
    SUPERFLEX_INSURANCE = "Maintain high floor with starting QB coverage for OP/Superflex slot"
    HIGH_FLOOR_HANDCUFF = "High-floor handcuff to secure workload floor during RB injuries"
    FLEX_DEPTH = "Consistent volume replacement for bye weeks and flex match-ups"
    IDP_STREAM_DEPTH = "High-tackle floor or matchup streamer for IDP positions"
    UPSIDE_FLYER = "Late-season high-volatility/ceiling breakout candidate"
    IR = "Injury Reserve Slot for out/injured players"


def determine_bench_policies(
    position_slot_counts: Dict[str, int],
    num_teams: int,
    is_te_premium: bool = False,
) -> List[Dict[str, Any]]:
    bench_capacity = position_slot_counts.get("BE", 0)
    ir_capacity = position_slot_counts.get("IR", 0)

    # 1. Identify primary flex slot type present in starter settings
    active_flex_slots = [
        slot for slot, count in position_slot_counts.items()
        if count > 0 and slot in ["RB/WR/TE", "RB/WR", "WR/TE"]
    ]

    # Priority order: RB/WR/TE > RB/WR > WR/TE
    if "RB/WR/TE" in active_flex_slots:
        primary_flex_label = "RB/WR/TE"
    elif "RB/WR" in active_flex_slots:
        primary_flex_label = "RB/WR"
    elif "WR/TE" in active_flex_slots:
        primary_flex_label = "WR/TE"
    else:
        primary_flex_label = "WR"

    # 2. Key positional indicators
    has_superflex = position_slot_counts.get("OP", 0) > 0 or position_slot_counts.get("TQB", 0) > 0
    idp_starter_count = sum(position_slot_counts.get(pos, 0) for pos in ["DL", "LB", "DB", "DT", "DE", "CB", "S", "DP"])
    is_deep_league = True if num_teams >= 12 else False

    policies: List[Dict[str, Any]] = []

    def add_slot(position: str, policy: BenchSlotPolicy):
        policies.append({
            "bench_slot": len(policies) + 1,
            "target_position": position,
            "policy_type": policy.name,
            "rationale": policy.value
        })

    remaining = bench_capacity

    # 1. Superflex / OP QB Coverage
    if has_superflex and remaining > 0:
        add_slot("OP", BenchSlotPolicy.SUPERFLEX_INSURANCE)
        remaining -= 1

    # 2. RB Handcuffs / Floor Depth (Scaled for 12+ team leagues)
    rb_handcuff_target = 1
    if is_deep_league and remaining >= 4:
        if (position_slot_counts.get("RB", 0) > 2) and remaining >= 8:
            rb_handcuff_target = 3
        else:
            rb_handcuff_target = 2
    for _ in range(min(rb_handcuff_target, remaining)):
        add_slot("RB", BenchSlotPolicy.HIGH_FLOOR_HANDCUFF)
        remaining -= 1

    # 3. Flexible Starter Depth (Uses detected flex label)
    flex_depth_target = 2 if remaining >= 4 else 1
    for _ in range(min(flex_depth_target, remaining)):
        add_slot(primary_flex_label, BenchSlotPolicy.FLEX_DEPTH)
        remaining -= 1

    # 4. IDP Coverage (If 4+ IDP starters required)
    if idp_starter_count >= 4 and remaining > 0:
        add_slot("IDP", BenchSlotPolicy.IDP_STREAM_DEPTH)
        remaining -= 1

    # 5. Fill remaining bench spots with high-upside flyers
    while remaining > 0:
        pos = "RB/WR/TE" if is_te_premium else "RB/WR"
        add_slot(pos, BenchSlotPolicy.UPSIDE_FLYER)
        remaining -= 1

    # 6. Add IR bench slots with sequential integer numbers
    for _ in range(ir_capacity):
        add_slot("IR", BenchSlotPolicy.IR)

    return policies


def get_eligible_positions_for_slot(slot_name: str) -> List[str]:
    """Resolves eligible player positions for any standard or slash-separated flex slot."""
    if slot_name in SLOT_ELIGIBILITY_MAP:
        return SLOT_ELIGIBILITY_MAP[slot_name]

    if "/" in slot_name:
        return [pos.strip() for pos in slot_name.split("/")]

    return [slot_name]


def calculate_bench_suitability_score(
    metrics: Dict[str, float],
    knobs: Dict[str, float],
) -> float:
    """Calculates suitability score (0-100 scale) using normalized primary metrics [0.0, 1.0]
    and knob weights passed in as dictionaries.
    """
    total_weight = sum(knobs.values())

    if total_weight == 0.0:
        return 0.0

    weighted_sum = sum(
        metrics.get(metric_key, 0.0) * weight
        for metric_key, weight in knobs.items()
    )

    return (weighted_sum / total_weight) * 100.0


def calculate_bench_suitability_scores(
    rostered_players: List[Dict[str, Any]],
    free_agents: List[Dict[str, Any]],
    slot_policies: List[Dict[str, Any]],
) -> Dict[Tuple[int, tuple], float]:
    """Calculates suitability scores for rostered and free-agent players
    against each BenchSlotPolicy using normalized player values and knob weights.
    Returns lookup dictionary keyed by (bench_slot, player_key_tuple).
    """
    candidate_pool = rostered_players + free_agents
    lookup_map = {}
    idp_positions = {"DL", "LB", "DB", "DT", "DE", "CB", "S", "DP"}
    ir_eligible_statuses = {"OUT", "INJURY_RESERVE"}

    for player in candidate_pool:
        pkey = get_player_key_from_dict(player)
        p_pos = player.get("position", "K")
        p_depth = player.get("sleeper_depth_order", 3)
        p_is_gaining = player.get("is_gaining_volume", False)
        p_is_handcuff = player.get("handcuff_for_starter_name", "") != ""
        p_is_my_handcuff = player.get("handcuff_mine", False)
        p_inj_status = str(player.get("injury_status", "")).upper()

        # Pull normalized attributes computed during normalization step
        norm_pts = float(player.get("norm_projected_points", 0.0))
        norm_ceiling_pts = float(player.get("norm_projected_ceiling_points", 0.0))
        norm_proj_ppg = float(player.get("norm_projected_ppg", 0.0))
        norm_ytd_ppg = float(player.get("norm_ytd_ppg", 0.0))
        norm_contingent_pts = float(player.get("norm_contingent_projected_points", 0.0))
        norm_trend_count = float(player.get("norm_sleeper_trending_count", 0.0))

        # Snap percentage and ownership percentage scaled to [0.0, 1.0]
        raw_snap = float(player.get("expected_snap_pct_if_active", 0.0))
        norm_snap = min(max(raw_snap / 100.0, 0.0), 1.0)
        # Raw percent owned (0.0 to 100.0 scale) normalized to [0.0, 1.0]
        raw_owned = float(player.get("percent_owned", 0.0))
        norm_owned = min(max(raw_owned / 100.0, 0.0), 1.0)
        # Raw percent started (0.0 to 100.0 scale) normalized to [0.0, 1.0]
        raw_started = float(player.get("percent_started", 0.0))
        norm_started = min(max(raw_started / 100.0, 0.0), 1.0)

        for policy_info in slot_policies:
            policy_type = policy_info.get("policy_type")
            target_pos = policy_info.get("target_position")
            bench_slot = policy_info.get("bench_slot")

            # Explicitly exclude IDP players from non-IDP bench policies
            if policy_type != "IDP_STREAM_DEPTH" and p_pos in idp_positions:
                lookup_map[(bench_slot, pkey)] = 0.0
                continue

            score = 0.0

            # Standardized dictionary of primary normalized metrics for this candidate
            metrics = {
                "projected_points": norm_pts,
                "projected_ceiling_points": norm_ceiling_pts,
                "projected_ppg": norm_proj_ppg,
                "ytd_ppg": norm_ytd_ppg,
                "contingent_projected_points": norm_contingent_pts,
                "snap_pct": norm_snap,
                "percent_owned": norm_owned,
                "percent_start": norm_started,
                "trending_count": norm_trend_count,
            }

            # -----------------------------------------------------------------
            # 1. SUPERFLEX_INSURANCE Policy
            # -----------------------------------------------------------------
            if policy_type == "SUPERFLEX_INSURANCE":
                eligible_positions = get_eligible_positions_for_slot(target_pos)
                if p_pos == "QB":
                    if p_depth == 1:
                        if norm_snap == 0.0:
                            metrics["snap_pct"] = 0.8
                        knobs = {
                            "projected_points": 8.0,
                            "ytd_ppg": 5.0,
                            "projected_ppg": 5.0,
                            "percent_owned": 4.0,
                            "percent_start": 3.0,
                            "snap_pct": 2.0,
                            "trending_count": 1.0,
                        }
                        score = calculate_bench_suitability_score(metrics, knobs)
                    elif p_depth == 2:
                        if norm_snap == 0.0:
                            metrics["snap_pct"] = 0.2
                        knobs = {
                            "contingent_projected_points": 6.0 if p_is_gaining else 2.0,
                            "percent_owned": 5.0,
                            "trending_count": 4.0,
                            "projected_points": 2.0,
                            "ytd_ppg": 2.0,
                            "snap_pct": 2.0,
                            "projected_ppg": 1.0
                        }
                        score = calculate_bench_suitability_score(metrics, knobs)
                    else:
                        score = 1.0
                elif p_pos in eligible_positions:
                    # This is pretty hand-wavy but it will never outscore a valuable QB.
                    # Maybe at some point we could score them as a penalized FLEX_DEPTH as a fallback.
                    score = norm_pts * 15.0
                else:
                    score = 0.0

            # -----------------------------------------------------------------
            # 2. HIGH_FLOOR_HANDCUFF Policy
            # -----------------------------------------------------------------
            elif policy_type == "HIGH_FLOOR_HANDCUFF":
                if p_pos == "RB":
                    if norm_snap == 0.0:
                        depth_snap_defaults = {1: 0.65, 2: 0.25, 3: 0.10}
                        metrics["snap_pct"] = depth_snap_defaults.get(p_depth, 0.20)

                    # Heavily boost our own handcuff (or a 3rd party's handcuff if they're being promoted)
                    # Otherwise we want to optimize this policy for players we can drop in to replace a
                    # starter as best as possible during injury/bye.
                    conting_k = 10.0 if p_is_my_handcuff or (p_is_handcuff and p_is_gaining) else 2.0

                    knobs = {
                        "contingent_projected_points": conting_k,
                        "ytd_ppg": 7.0,
                        "projected_points": 6.0,
                        "projected_ppg": 5.0,
                        "percent_owned": 5.0,
                        "snap_pct": 3.0,
                        "percent_start": 2.0,
                    }
                    score = calculate_bench_suitability_score(metrics, knobs)
                else:
                    score = 0.0

            # -----------------------------------------------------------------
            # 3. FLEX_DEPTH Policy
            # -----------------------------------------------------------------
            elif policy_type == "FLEX_DEPTH":
                eligible_positions = get_eligible_positions_for_slot(target_pos)
                if p_pos in eligible_positions:
                    # Mild boost if they're a handcuff to someone and being promoted
                    conting_k = 3.0 if p_is_handcuff and p_is_gaining else 1.0

                    knobs = {
                        "ytd_ppg": 7.0,
                        "projected_points": 6.0,
                        "projected_ppg": 5.0,
                        "percent_owned": 4.0,
                        "snap_pct": 3.0,
                        "projected_ceiling_points": 3.0,
                        "contingent_projected_points": conting_k,
                        "percent_start": 2.0,
                    }
                    score = calculate_bench_suitability_score(metrics, knobs)
                else:
                    score = 0.0

            # -----------------------------------------------------------------
            # 4. IDP_STREAM_DEPTH Policy
            # -----------------------------------------------------------------
            elif policy_type == "IDP_STREAM_DEPTH":
                if p_pos in idp_positions or target_pos in idp_positions:
                    knobs = {
                        "projected_points": 6.0,
                        "ytd_ppg": 5.0,
                        "projected_ppg": 5.0,
                        "percent_owned": 4.0,
                        "snap_pct": 3.0,
                        "trending_count": 3.0,
                        "percent_start": 2.0,
                    }
                    score = calculate_bench_suitability_score(metrics, knobs)
                else:
                    score = 0.0

            # -----------------------------------------------------------------
            # 5. UPSIDE_FLYER Policy
            # -----------------------------------------------------------------
            elif policy_type == "UPSIDE_FLYER":
                eligible_positions = get_eligible_positions_for_slot(target_pos)
                if p_pos in eligible_positions:
                    # Heavy boost if they are a true handcuff to someone and being promoted
                    conting_k = 8.0 if p_is_handcuff and p_is_gaining else 2.0

                    knobs = {
                        "contingent_projected_points": conting_k,
                        "projected_ceiling_points": 7.0,
                        "trending_count": 5.0,
                        "percent_owned": 3.0,
                        "snap_pct": 2.0,
                        "projected_points": 1.0,
                        "ytd_ppg": 1.0,
                        "percent_start": 1.0,
                    }
                    score = calculate_bench_suitability_score(metrics, knobs)
                else:
                    score = 0.0

            # -----------------------------------------------------------------
            # 6. IR Policy
            # -----------------------------------------------------------------
            elif policy_type == "IR":
                if p_inj_status in ir_eligible_statuses:
                    knobs = {
                        "projected_ppg": 8.0,
                        "ytd_ppg": 6.0,
                        "percent_owned": 6.0,
                        "trending_count": 3.0,
                    }
                    score = calculate_bench_suitability_score(metrics, knobs)
                else:
                    score = 0.0

            else:
                score = 0.0

            rounded_score = round(score, 2)
            lookup_map[(bench_slot, pkey)] = rounded_score

    return lookup_map


def calculate_current_bench_suitability(
    current_bench: List[Dict[str, Any]],
    bench_slot_policies: List[Dict[str, Any]],
) -> float:
    """Calculates total bench suitability score for the current bench using Pulp solver."""
    if not current_bench or not bench_slot_policies:
        return 0.0

    scores = calculate_bench_suitability_scores(
        rostered_players=current_bench,
        free_agents=[],
        slot_policies=bench_slot_policies,
    )

    bench_slots = [p["bench_slot"] for p in bench_slot_policies]
    player_keys = [get_player_key_from_dict(p) for p in current_bench]

    prob = pulp.LpProblem("Current_Bench_Suitability", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("b", ((b_id, pk) for b_id in bench_slots for pk in player_keys), cat=pulp.LpBinary)

    prob += pulp.lpSum(x[(b_id, pk)] * scores.get((b_id, pk), 0.0) for b_id in bench_slots for pk in player_keys)

    for b_id in bench_slots:
        prob += pulp.lpSum(x[(b_id, pk)] for pk in player_keys) <= 1
    for pk in player_keys:
        prob += pulp.lpSum(x[(b_id, pk)] for b_id in bench_slots) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return float(pulp.value(prob.objective) or 0.0)


def display_bench_suitability_scores(
    suitability_scores: Dict[Tuple[int, tuple], float],
    slot_policies: List[Dict[str, Any]]
) -> None:
    """
    Displays the top 10 candidate players for each bench slot sorted by suitability score descending.

    :param suitability_scores: Lookup map keyed by (bench_slot, player_key_tuple) mapping to float score
    :param slot_policies: List of policy dictionaries containing bench_slot and policy_type.
    """
    policy_map = {p["bench_slot"]: p.get("policy_type") for p in slot_policies}

    slot_candidates: Dict[int, List[Tuple[tuple, float]]] = {}
    for (bench_slot, pkey), score in suitability_scores.items():
        slot_candidates.setdefault(bench_slot, []).append((pkey, score))

    for bench_slot in sorted(slot_candidates.keys()):
        policy_type_str = policy_map.get(bench_slot)

        policy_description = "Unknown Policy"
        if policy_type_str and policy_type_str in BenchSlotPolicy.__members__:
            policy_description = BenchSlotPolicy[policy_type_str].value

        top_candidates = sorted(
            slot_candidates[bench_slot],
            key=lambda item: item[1],
            reverse=True
        )[:10]

        print(f"Bench Slot {bench_slot}: {policy_description}")
        print("-" * 60)
        for rank, (pkey, score) in enumerate(top_candidates, start=1):
            disp_name = f"{pkey[0]} [{pkey[1]} {pkey[2]}]"
            print(f"  {rank:2d}. {disp_name:<35} Score: {score:.2f}")
        print("\n")


def calculate_upside_flyers(
    free_agents: List[Dict[str, Any]],
    is_te_premium: bool,
    limit: int
) -> List[Dict[str, Any]]:
    slot_policies = [
        {
            "bench_slot": 1,
            "target_position": "RB/WR/TE" if is_te_premium else "RB/WR",
            "policy_type": "UPSIDE_FLYER",
            "rationale": "Late-season high-volatility/ceiling breakout candidate"
        }
    ]

    scores_map = calculate_bench_suitability_scores(
        rostered_players=[],
        free_agents=free_agents,
        slot_policies=slot_policies,
    )

    fa_map = {get_player_key_from_dict(player): player for player in free_agents if player.get("name")}

    flyers = [
        {
            "name": fa_map[pkey].get("name"),
            "nfl_team": fa_map[pkey].get("sleeper_nfl_team") or fa_map[pkey].get("team"),
            "position": fa_map[pkey].get("position"),
            "score": score,
        }
        for (bench_slot, pkey), score in scores_map.items()
        if bench_slot == 1 and pkey in fa_map
    ]

    flyers.sort(key=lambda x: x["score"], reverse=True)
    return flyers[:limit]


def format_policy_type(policy_type: Optional[Any]) -> Optional[str]:
    """Converts policy_type to a human-readable string by substituting spaces for underscores

    and capitalizing words except known acronyms.
    """
    if not policy_type:
        return policy_type

    policy_type_str = str(policy_type)
    acronyms = {"IDP", "IR"}
    words = policy_type_str.split("_")
    formatted_words = []

    for word in words:
        if word.upper() in acronyms:
            formatted_words.append(word.upper())
        else:
            formatted_words.append(word.capitalize())

    return " ".join(formatted_words)


def on_starters_selected(
    starters_res: List[Dict[str, Any]],
    remaining_rostered: List[Dict[str, Any]],
    remaining_fa: List[Dict[str, Any]],
) -> None:
    """Marks 'handcuff_mine' to True for any remaining rostered or free agent player

    who acts as a handcuff for a selected starting player.
    """
    starter_names = {clean_player_name(s.get("name")) for s in starters_res if s.get("name")}

    for player in remaining_rostered + remaining_fa:
        handcuff_target = clean_player_name(player.get("handcuff_for_starter_name"))
        if handcuff_target and handcuff_target in starter_names:
            player["handcuff_mine"] = True


# ---------------------------------------------------------------------------
# ROSTER SOLVER
# ---------------------------------------------------------------------------
def optimize_roster(
    starters: List[Dict[str, Any]],
    bench: List[Dict[str, Any]],
    ir: List[Dict[str, Any]],
    free_agents: List[Dict[str, Any]],
    position_slot_counts: Dict[str, int],
    bench_slot_policies: List[Dict[str, Any]],
    max_weekly_moves: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Optimizes the roster using a two-pass linear programming model:

    1. Solves for optimal starter slots to maximize projected points.
    2. Solves for bench and IR slots using remaining candidates to maximize suitability scores.
    Returns a tuple of (starters_results, bench_results, ir_results, drops_results).
    """
    if ir is None:
        ir = []

    # Exclude free agents who have is_lineup_locked set
    unlocked_free_agents = [fa for fa in free_agents if not is_lineup_locked(fa)]

    rostered_players = starters + bench + ir
    candidate_pool_list = rostered_players + unlocked_free_agents
    candidate_pool = {get_player_key_from_dict(p): p for p in candidate_pool_list}
    player_keys = list(candidate_pool.keys())

    # Record initial roster status for existing team players
    initial_starter_keys = {get_player_key_from_dict(p) for p in starters}
    initial_bench_keys = {get_player_key_from_dict(p) for p in bench}
    initial_ir_keys = {get_player_key_from_dict(p) for p in ir}

    full_bench_policies = list(bench_slot_policies)

    # Build unique starter slot instances with positional tracking
    starter_slot_assignments = {}  # Map starter player key if locked
    slot_type_counters = {}

    for p in starters:
        slot_type = p.get("slot_position", "BE")
        slot_type_counters[slot_type] = slot_type_counters.get(slot_type, 0) + 1
        s_id = f"{slot_type}_{slot_type_counters[slot_type]}"
        if is_lineup_locked(p):
            starter_slot_assignments[get_player_key_from_dict(p)] = s_id

    start_slots = []
    for slot_type, count in position_slot_counts.items():
        if slot_type not in ["BE", "IR", ""] and count > 0:
            for i in range(1, int(count) + 1):
                start_slots.append((f"{slot_type}_{i}", slot_type))

    start_slot_ids = [s[0] for s in start_slots]
    pass2_slots = [p["bench_slot"] for p in full_bench_policies]

    # Track free agent moves across both solves
    current_roster_keys = {get_player_key_from_dict(p) for p in rostered_players}
    moves_used = 0

    # =========================================================================
    # PASS 1: SOLVE FOR STARTER SLOTS
    # =========================================================================
    print("Solving for starter slots...")
    prob_starters = pulp.LpProblem("Starter_Optimization", pulp.LpMaximize)

    x_start = pulp.LpVariable.dicts(
        "start", ((s_id, pk) for s_id in start_slot_ids for pk in player_keys), cat=pulp.LpBinary
    )

    # Objective 1: Maximize total starting projected points
    prob_starters += pulp.lpSum(
        x_start[(s_id, pk)] * float(candidate_pool[pk].get("projected_points", 0.0))
        for s_id in start_slot_ids for pk in player_keys
    )

    # Position eligibility for starting slots
    for s_id, base_type in start_slots:
        eligible_positions = get_eligible_positions_for_slot(base_type)
        for pk in player_keys:
            if candidate_pool[pk].get("position") not in eligible_positions:
                prob_starters += x_start[(s_id, pk)] == 0

    # Enforce locked roster entries constraints for starters
    for p_info in rostered_players:
        pk = get_player_key_from_dict(p_info)
        if is_lineup_locked(p_info):
            if pk in initial_starter_keys:
                assigned_sid = starter_slot_assignments.get(pk)
                if assigned_sid and assigned_sid in start_slot_ids:
                    prob_starters += x_start[(assigned_sid, pk)] == 1
                else:
                    prob_starters += pulp.lpSum(x_start[(s_id, pk)] for s_id in start_slot_ids) == 1
            else:
                for s_id in start_slot_ids:
                    prob_starters += x_start[(s_id, pk)] == 0

    # Slot capacity: Exactly 1 player per starting slot
    for s_id in start_slot_ids:
        prob_starters += pulp.lpSum(x_start[(s_id, pk)] for pk in player_keys) == 1

    # Player uniqueness: At most 1 starting slot assignment per player
    for pk in player_keys:
        prob_starters += pulp.lpSum(x_start[(s_id, pk)] for s_id in start_slot_ids) <= 1

    # Transaction budget constraint for starters
    starter_fa_adds = [
        x_start[(s_id, pk)]
        for pk in player_keys if not candidate_pool[pk].get("is_user_team", False)
        for s_id in start_slot_ids
    ]
    if starter_fa_adds:
        prob_starters += pulp.lpSum(starter_fa_adds) <= max_weekly_moves

    prob_starters.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract Starter Results
    starters_res = []
    selected_starters = set()

    for s_id, base_type in start_slots:
        for pk in player_keys:
            if pulp.value(x_start[(s_id, pk)]) == 1.0:
                selected_starters.add(pk)
                if pk not in current_roster_keys:
                    moves_used += 1

                is_fa = not candidate_pool[pk].get("is_user_team", False)
                p_status = str(candidate_pool[pk].get("status", "")).strip().upper()
                if is_fa:
                    transaction_type = "Claim" if p_status == "WAIVERS" else "Add"
                elif pk in initial_bench_keys or pk in initial_ir_keys:
                    transaction_type = "Promoted"
                else:
                    transaction_type = None

                starters_res.append({
                    "slot": base_type,
                    "slot_id": s_id,
                    "name": candidate_pool[pk].get("name"),
                    "position": candidate_pool[pk].get("position"),
                    "injury_status": candidate_pool[pk].get("injury_status"),
                    "nfl_team": candidate_pool[pk].get("sleeper_nfl_team") or candidate_pool[pk].get("team"),
                    "projected_points": float(candidate_pool[pk].get("projected_points", 0.0)),
                    "is_free_agent": is_fa,
                    "transaction_type": transaction_type,
                    "is_lineup_locked": is_lineup_locked(candidate_pool[pk]),
                })

    # =========================================================================
    # PASS 2: SOLVE FOR BENCH AND IR SLOTS
    # =========================================================================
    print("Solving for bench and IR slots...")
    remaining_player_keys = [pk for pk in player_keys if pk not in selected_starters]
    remaining_rostered = [p for p in rostered_players if get_player_key_from_dict(p) not in selected_starters]
    remaining_fa = [p for p in unlocked_free_agents if get_player_key_from_dict(p) not in selected_starters]

    # Perform any processing/decoration to the rest of the candidates now that starters are solved
    on_starters_selected(
        starters_res=starters_res,
        remaining_rostered=remaining_rostered,
        remaining_fa=remaining_fa,
    )

    bench_scores = calculate_bench_suitability_scores(
        rostered_players=remaining_rostered,
        free_agents=remaining_fa,
        slot_policies=full_bench_policies,
    )

    display_bench_suitability_scores(bench_scores, full_bench_policies)

    prob_bench = pulp.LpProblem("Bench_IR_Optimization", pulp.LpMaximize)

    x_pass2 = pulp.LpVariable.dicts(
        "pass2", ((b_id, pk) for b_id in pass2_slots for pk in remaining_player_keys), cat=pulp.LpBinary
    )

    # Objective 2: Maximize suitability scores across bench and IR slots
    prob_bench += pulp.lpSum(
        x_pass2[(b_id, pk)] * bench_scores.get((b_id, pk), 0.0)
        for b_id in pass2_slots for pk in remaining_player_keys
    )

    # Enforce locked roster entries constraints for bench and IR slots
    for p_info in remaining_rostered:
        pk = get_player_key_from_dict(p_info)
        if is_lineup_locked(p_info):
            if pk in initial_bench_keys:
                standard_bench_slots = [p["bench_slot"] for p in full_bench_policies if p.get("policy_type") != "IR"]
                prob_bench += pulp.lpSum(x_pass2[(b_id, pk)] for b_id in standard_bench_slots) == 1
            elif pk in initial_ir_keys:
                ir_slots = [p["bench_slot"] for p in full_bench_policies if p.get("policy_type") == "IR"]
                if ir_slots:
                    prob_bench += pulp.lpSum(x_pass2[(b_id, pk)] for b_id in ir_slots) == 1

    # Slot capacity: At most 1 player per slot
    for b_id in pass2_slots:
        prob_bench += pulp.lpSum(x_pass2[(b_id, pk)] for pk in remaining_player_keys) <= 1

    # Player uniqueness: At most 1 slot assignment per player
    for pk in remaining_player_keys:
        prob_bench += pulp.lpSum(x_pass2[(b_id, pk)] for b_id in pass2_slots) <= 1

    # Remaining transaction budget constraint across bench and IR slots
    remaining_moves = max_weekly_moves - moves_used
    pass2_fa_adds = [
        x_pass2[(b_id, pk)]
        for pk in remaining_player_keys if not candidate_pool[pk].get("is_user_team", False)
        for b_id in pass2_slots
    ]
    max_remaining_moves = max(0, remaining_moves)
    if pass2_fa_adds:
        prob_bench += pulp.lpSum(pass2_fa_adds) <= max_remaining_moves

    prob_bench.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract Bench and IR Results
    bench_res = []
    ir_res = []
    selected_pass2 = set()

    for policy_info in full_bench_policies:
        b_id = policy_info["bench_slot"]
        p_type = policy_info.get("policy_type")

        for pk in remaining_player_keys:
            if pulp.value(x_pass2[(b_id, pk)]) == 1.0:
                selected_pass2.add(pk)
                is_fa = not candidate_pool[pk].get("is_user_team", False)
                p_status = str(candidate_pool[pk].get("status", "")).strip().upper()

                if p_type == "IR":
                    if is_fa:
                        transaction_type = "Claim" if p_status == "WAIVERS" else "Add"
                    elif pk in initial_starter_keys or pk in initial_bench_keys:
                        transaction_type = "Demoted"
                    else:
                        transaction_type = None

                    ir_res.append({
                        "bench_slot": b_id,
                        "policy_type": format_policy_type(p_type),
                        "name": candidate_pool[pk].get("name"),
                        "position": candidate_pool[pk].get("position"),
                        "injury_status": candidate_pool[pk].get("injury_status"),
                        "nfl_team": candidate_pool[pk].get("sleeper_nfl_team") or candidate_pool[pk].get("team"),
                        "projected_points": float(candidate_pool[pk].get("projected_points", 0.0)),
                        "suitability_score": bench_scores.get((b_id, pk), 0.0),
                        "is_free_agent": is_fa,
                        "is_new_add": is_fa or pk not in initial_ir_keys,
                        "transaction_type": transaction_type,
                        "is_lineup_locked": is_lineup_locked(candidate_pool[pk]),
                    })
                else:
                    if is_fa:
                        transaction_type = "Claim" if p_status == "WAIVERS" else "Add"
                    elif pk in initial_starter_keys:
                        transaction_type = "Demoted"
                    elif pk in initial_ir_keys:
                        transaction_type = "Promoted"
                    else:
                        transaction_type = None

                    bench_res.append({
                        "bench_slot": b_id,
                        "policy_type": format_policy_type(p_type),
                        "name": candidate_pool[pk].get("name"),
                        "position": candidate_pool[pk].get("position"),
                        "injury_status": candidate_pool[pk].get("injury_status"),
                        "nfl_team": candidate_pool[pk].get("sleeper_nfl_team") or candidate_pool[pk].get("team"),
                        "suitability_score": bench_scores.get((b_id, pk), 0.0),
                        "is_free_agent": is_fa,
                        "transaction_type": transaction_type,
                        "is_lineup_locked": is_lineup_locked(candidate_pool[pk]),
                    })

    # =========================================================================
    # IDENTIFY DROPPED PLAYERS
    # =========================================================================
    new_roster_keys = selected_starters.union(selected_pass2)
    drops_res = []

    for p_info in rostered_players:
        pk = get_player_key_from_dict(p_info)
        if not pk[0] or pk in new_roster_keys:
            continue

        p_locked = is_lineup_locked(p_info)
        if not p_locked:
            drops_res.append({
                "name": p_info.get("name"),
                "position": p_info.get("position"),
                "injury_status": p_info.get("injury_status"),
                "nfl_team": p_info.get("sleeper_nfl_team") or p_info.get("team"),
                "projected_points": float(p_info.get("projected_points", 0.0)),
                "is_lineup_locked": False,
            })

    print(f"✓ Successfully optimized {len(starters_res)} starting slots, "
          f"{len(bench_res)} bench slots, assigned {len(ir_res)} IR player(s), "
          f"and identified {len(drops_res)} dropped player(s).")

    return starters_res, bench_res, ir_res, drops_res
