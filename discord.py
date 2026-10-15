import json
import os
import urllib.request
from typing import Any, Dict, List

from espn import LEAGUE_ID, TEAM_ID
from gemini import GEMINI_UNAVAILABLE_MESSAGE, SCHEDULED_RUN_REASON
from solver import calculate_current_bench_suitability, is_lineup_locked

# ---------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------------------------
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
TRIGGER_REASON = os.getenv("TRIGGER_REASON")

GREEN_COLOR = 5880468  # Hex #59BA74 best-effort match of Discord "+" diff color


def resolve_team(player: Dict[str, Any]) -> str:
    """Helper to resolve team abbreviation across various dictionary keys."""
    return (
        player.get("team")
        or player.get("nfl_team")
        or player.get("sleeper_nfl_team")
        or "FA"
    )


# ---------------------------------------------------------------------------
# DIFF CODE BLOCK GENERATORS
# ---------------------------------------------------------------------------
def generate_lineup_diff(
    starters: List[Dict[str, Any]],
    current_starters: List[Dict[str, Any]],
) -> str:
    """Generates the pre-formatted diff code block for the Optimized Lineup section."""
    lines = ["```diff"]
    optimized_points = 0.0

    for s in starters:
        slot = s.get("slot", "")
        name = s.get("name", "")
        if len(name) >= 15:
            name_parts = name.split()
            if len(name_parts) > 1:
                name = f"{name_parts[0][0]}. {' '.join(name_parts[1:])}"

        team_str = resolve_team(s)
        team = f" ({team_str})" if team_str and team_str != "FA" else ""
        trans_type = s.get("transaction_type")
        pts = float(s.get("projected_points", 0.0))
        optimized_points += pts

        symbol = "+" if trans_type in ("Add", "Claim", "Promote", "Promoted") else " "
        symbol = "🔒" if is_lineup_locked(s) else symbol

        flag = ""
        if trans_type == "Add":
            flag = " 🆓"
        elif trans_type == "Claim":
            flag = " 🏷️"
        elif trans_type in ("Promote", "Promoted"):
            flag = " ▲"

        lines.append(f"{symbol} {slot:>5}: {name}{team}{flag}")

    if current_starters is None:
        current_starters = []

    projected_points = sum(float(s.get("projected_points", 0.0)) for s in current_starters)

    lines.append(f"\nProjected points: {projected_points:.2f}")
    lines.append(f"Optimized points: {optimized_points:.2f}")
    lines.append("```")
    return "\n".join(lines)


def generate_bench_diff(
    bench: List[Dict[str, Any]],
    current_bench: List[Dict[str, Any]],
    bench_slot_policies: List[Dict[str, Any]],
) -> str:
    """Generates the pre-formatted diff code block for the Optimized Bench section."""
    lines = ["```diff"]
    optimized_suitability_score = 0.0

    for b in bench:
        name = b.get("name", "")
        team_str = resolve_team(b)
        team = f" ({team_str})" if team_str and team_str != "FA" else ""
        trans_type = b.get("transaction_type")
        idx = b.get("bench_slot", "")
        policy = b.get("policy_type", "")
        optimized_suitability_score += float(b.get("suitability_score", 0.0))

        symbol = "+" if trans_type in ("Add", "Claim", "Demote", "Demoted", "Promote", "Promoted") else " "
        symbol = "🔒" if is_lineup_locked(b) else symbol

        flag = ""
        if trans_type == "Add":
            flag = " 🆓"
        elif trans_type == "Claim":
            flag = " 🏷️"
        elif trans_type in ("Promote", "Promoted"):
            flag = " ▲"
        elif trans_type in ("Demote", "Demoted"):
            flag = " ▼"

        lines.append(f"{symbol} BE: {name}{team}{flag}")
        lines.append(f"     [{idx}: {policy}]")

    current_suitability_score = 0.0
    if current_bench and bench_slot_policies:
        current_suitability_score = calculate_current_bench_suitability(
            current_bench=current_bench,
            bench_slot_policies=bench_slot_policies,
        )

    lines.append(f"\nSuitability score: {current_suitability_score:.2f}")
    lines.append(f"  Optimized score: {optimized_suitability_score:.2f}")
    lines.append("```")
    return "\n".join(lines)


def generate_ir_diff(
    ir: List[Dict[str, Any]],
) -> str:
    """Generates the pre-formatted diff code block for the Injured Reserve section."""
    if not ir:
        return (
            "```diff\n"
            "No players in Injured Reserve.\n"
            "```"
        )

    lines = ["```diff"]
    for player in ir:
        name = player.get("name", "")
        team = resolve_team(player)
        trans_type = player.get("transaction_type")

        symbol = "+" if trans_type in ("Demote", "Demoted") or player.get("is_new_add") else " "
        symbol = "🔒" if is_lineup_locked(player) else symbol

        flag = ""
        if trans_type == "Add":
            flag = " 🆓"
        elif trans_type == "Claim":
            flag = " 🏷️"
        elif trans_type in ("Demote", "Demoted"):
            flag = " ▼"

        lines.append(f"{symbol} IR: {name} ({team}){flag}")

    lines.append("```")
    return "\n".join(lines)


def generate_drops_diff(drops: List[Dict[str, Any]]) -> str:
    """Generates the pre-formatted diff code block for the Drops section."""
    if not drops:
        return (
            "```diff\n"
            "No drops required.\n"
            "```"
        )

    lines = ["```diff"]
    for d in drops:
        name = d.get("name", "Unknown")
        position = d.get("position", "Unknown")
        team = resolve_team(d)
        lines.append(f"- {name} ({team}, {position})")

    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EMBED CREATION & FORMATTING
# ---------------------------------------------------------------------------
def generate_discord_embed(
    evaluation_json: str,
    starters_res: List[Dict[str, Any]],
    bench_res: List[Dict[str, Any]],
    ir_res: List[Dict[str, Any]],
    drops_res: List[Dict[str, Any]],
    current_starters: List[Dict[str, Any]],
    current_bench: List[Dict[str, Any]],
    bench_slot_policies: List[Dict[str, Any]],
    upside_flyers: List[Dict[str, Any]],
) -> str:
    """Constructs the full Discord embed JSON structure entirely in Python after inference

    by extracting fields returned from Gemini and generating the appropriate formatted diff blocks.
    """
    try:
        data = json.loads(evaluation_json)
        managers_thoughts = data.get("managers_thoughts", "")

        has_starter_changes = any(s.get("transaction_type") for s in starters_res)
        has_bench_changes = any(b.get("transaction_type") for b in bench_res)
        has_ir_changes = any(i.get("transaction_type") or i.get("is_new_add") for i in ir_res)
        has_drops = len(drops_res) > 0

        no_changes = not (has_starter_changes or has_bench_changes or has_ir_changes or has_drops)

        if TRIGGER_REASON == SCHEDULED_RUN_REASON:
            description = "*Regularly scheduled 5x daily • 6am - 6pm*"
            fields = [
                {"name": "Manager's Thoughts 🤖💭", "value": f"```\n{managers_thoughts}\n```", "inline": False},
            ]
            if managers_thoughts != GEMINI_UNAVAILABLE_MESSAGE:
                trade_ideas = data.get("trade_ideas", "No suggested trade proposals at this time.")
                fields.append({"name": "Trade Ideas 🤝💡", "value": f"```\n{trade_ideas}\n```", "inline": False})
            upside_flyers_string = "\n".join(
                f"{idx:>2}. {player['name']} ({resolve_team(player)}, {player['position']})"
                for idx, player in enumerate(upside_flyers, start=1)
            )
            fields.append({"name": "Upside Flyers 🚀📈", "value": f"```\n{upside_flyers_string}\n```", "inline": False})
        else:
            description = f"⚡ {TRIGGER_REASON}"
            fields = [
                {"name": "Manager's Thoughts 🤖💭", "value": f"```\n{managers_thoughts}\n```", "inline": False},
            ]

        if no_changes:
            fields.append({
                "name": "No suggested roster moves. ✔️",
                "value": "",
                "inline": False,
            })
        else:
            lineup_diff = generate_lineup_diff(starters_res, current_starters)
            bench_diff = generate_bench_diff(bench_res, current_bench, bench_slot_policies)
            ir_diff = generate_ir_diff(ir_res)
            drops_diff = generate_drops_diff(drops_res)

            fields.extend([
                {"name": "Optimized Lineup 💎", "value": lineup_diff, "inline": False},
                {"name": "Optimized Bench 🪑", "value": bench_diff, "inline": False},
                {"name": "Injured Reserve 🩼", "value": ir_diff, "inline": False},
                {"name": "Drops 🚮", "value": drops_diff, "inline": False},
            ])

        if TRIGGER_REASON == SCHEDULED_RUN_REASON:
            title = "🏈 Fantasy Football • Analysis ⚙️"
        else:
            title = "🏈 Fantasy Football • News 📡"

        embed_payload = {
            "embeds": [
                {
                    "title": title,
                    "url": f"https://fantasy.espn.com/football/team?leagueId={LEAGUE_ID}&teamId={TEAM_ID}",
                    "color": GREEN_COLOR,
                    "description": description,
                    "fields": fields,
                    "footer": {
                        "text": "Click the title above to open your roster 📋"
                    }
                }
            ]
        }

        return json.dumps(embed_payload)
    except Exception as e:
        print(f"Warning: Failed to construct Discord embed payload: {e}")
        return evaluation_json


def truncate_embed_fields(payload_data: Dict[str, Any], max_length: int = 1020) -> Dict[str, Any]:
    """Truncates embed field values to max_length characters to satisfy Discord limits."""
    for embed in payload_data.get("embeds", []):
        for field in embed.get("fields", []):
            if len(field.get("value", "")) > max_length:
                field["value"] = field["value"][: max_length - 6] + "...```"
    return payload_data


def format_embed_to_markdown(evaluation_json: str) -> str:
    """Converts a Discord embed JSON string into formatted Markdown text for console rendering."""
    data = json.loads(evaluation_json)
    embed = data["embeds"][0]
    title = embed.get("title", "Report")
    if url := embed.get("url"):
        title_str = f"[{title}]({url})"
    else:
        title_str = title

    markdown_summary = f"# {title_str}\n\n*{embed.get('description', '')}*\n\n"
    for field in embed.get("fields", []):
        markdown_summary += f"### {field['name']}\n{field['value']}\n\n"
    if footer := embed.get("footer"):
        markdown_summary += f"---\n*{footer.get('text', '')}*"
    return markdown_summary


def post_to_webhook(embed_json: str) -> None:
    """Helper to post recommendation embed payload to the target Webhook."""
    payload_data = json.loads(embed_json)
    payload_data = truncate_embed_fields(payload_data)
    payload = json.dumps(payload_data).encode("utf-8")

    req = urllib.request.Request(
        WEBHOOK_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Python-Webhook",
        },
    )

    with urllib.request.urlopen(req) as response:
        print(f"Status Code: {response.status}")
