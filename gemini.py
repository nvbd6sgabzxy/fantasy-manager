import json
import os
import re
from typing import Any, Dict, List, Type

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    wait_fixed,
)

from espn import TEAM_ID
from handcuff import make_player_key

# ---------------------------------------------------------------------------
# CONFIGURATION & ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TRIGGER_REASON = os.getenv("TRIGGER_REASON")

SCHEDULED_RUN_REASON = "Regularly scheduled deep analysis"
GEMINI_UNAVAILABLE_MESSAGE = "Gemini is unavailable at the moment. 👾"


# ---------------------------------------------------------------------------
# STRUCTURED OUTPUT SCHEMAS
# ---------------------------------------------------------------------------
class NewsStoryEvaluation(BaseModel):
    guid: str = Field(description="The guid element that identifies the news item.")
    subject_player: str = Field(description="The primary NFL player featured in the news item.")
    subject_team: str = Field(description="The standard 2 or 3-letter NFL team abbreviation for the subject player (e.g. 'KC', 'LAR', 'SF').")
    subject_position: str = Field(description="The primary position for the subject player (e.g. 'QB', 'RB', 'WR', 'TE', 'K', 'D/ST', 'LB').")
    is_missing_game: bool = Field(description="True if story states or implies player is likely missing at least one game.")
    is_losing_volume: bool = Field(description="True if story states or implies player is demoted or losing snaps/volume.")
    is_gaining_volume: bool = Field(description=(
        "True if story states or implies player has an expanded role, starter opportunity, or increased volume."
    ))


class NewsEvaluationPayload(BaseModel):
    evaluations: List[NewsStoryEvaluation] = Field(description="Evaluations for each news item.")

    def to_guid_map(self) -> Dict[str, Dict[str, Any]]:
        """Returns evaluations as a dictionary keyed by guid."""
        return {
            item.guid: item.model_dump(exclude={"guid"})
            for item in self.evaluations
            if item.guid
        }


class SortedPlayerNamesPayload(BaseModel):
    sorted_player_names: List[str] = Field(description="List of player names sorted by IR slot value.")


class ManagerThoughtsScheduled(BaseModel):
    managers_thoughts: str = Field(description="Concise strategic explanation for Manager's Thoughts.")
    trade_ideas: str = Field(description="Trade proposals with letter grades for both teams or default string.")


class ManagerThoughtsNews(BaseModel):
    managers_thoughts: str = Field(description="Concise explanation of immediate lineup and news impact for Manager's Thoughts.")


def get_schema() -> Type[BaseModel]:
    """Returns the appropriate Pydantic output schema depending on run trigger type."""
    if TRIGGER_REASON == SCHEDULED_RUN_REASON:
        return ManagerThoughtsScheduled
    return ManagerThoughtsNews


# ---------------------------------------------------------------------------
# SYSTEM INSTRUCTIONS
# ---------------------------------------------------------------------------
def get_base_instructions() -> str:
    """Returns common system directives shared by all run types."""
    return f"""You are a world-class expert fantasy football manager for team ID {TEAM_ID}.

CORE DIRECTIVES:
1. **Data Snapshot**:
   - Evaluate provided JSON data containing ESPN rosters, Sleeper metadata, injury risk, handcuff information,
     RotoWire news and optimization hints.
2. **Lineup & Waiver Optimization**:
   - Make use of `hints.optimal_roster` when constructing "Manager's Thoughts".
   - Maximize total team points using eligible bench or available free agents.
   - Execute recommended operations forcefully to maximize bench upside.
   - Note that `free_agents` is a limited subset of the true list of available free agents. If you are explicitly told someone
     is a FREE AGENT but they are not in the list this is not a contradiction. Believe it when told.
3. **Roster Boundaries**:
    - Active roster and bench are strictly under `user_team`. Players under `opponent_teams` are NOT on your roster.
    - This doesn't mean that you shouldn't attempt to grab an opponent's starter's handcuff when news suggests the starter
      is out.
4. **Breaking News & Waiver Sniping**:
   - Treat `hints.optimal_roster` as real-time ground truth.
"""


def get_system_instruction() -> str:
    """Generates run-type-specific system instruction."""
    base = get_base_instructions()

    if TRIGGER_REASON == SCHEDULED_RUN_REASON:
        return base + f"""
RUN TYPE: Scheduled Deep Analysis

TASK:
- Scan free agency, waiver additions, and available handcuffs across the league.
- Evaluate `hints.optimal_roster` to understand the roster moves we're suggesting. Correlate these moves with
  the provided news and also with any other real-time news sources you may have access to when applicable.
- Propose at least one creative trade offer to give the team an advantage. Include letter grades for both sides of the trade.
- Evaluate `hints.upside_flyers` and say a little something about some of the names in there and why they might be
  there in your commentary.
- Provide a concise strategic rationale and general commentary for `managers_thoughts` and detailed trade ideas for `trade_ideas`.
- When mentioning specific players always indicate their team and position using (Team, Position) notation (e.g. (KC, RB)).

KEEPER & AUCTION TRADE VALUATION RULES:
- Primarily value players based on `proj_ppg` and secondly on `ytd_ppg`.
- If `proj_ppg` is significantly higher than `ytd_ppg` for a given player that indicates the player is under-performing relative
  to pre-season projections. This could indicate a good "buy low" candidate if we're in the first half of the season and they've
  had a tough schedule or there are other extenuating circumstances.
- If `ytd_ppg` is significantly higher than `proj_ppg` for a given player that indicates the player is over-performing relative
  to pre-season projections. This could indicate a good "sell high" candidate if we're in the first half of the season and
  they've had a relatively easy schedule or other circumstances that are likely to be temporary.
- If `league_data.has_keepers` and `league_data.is_auction_draft` are true, evaluate player trade values using their
  `price` relative to `league_data.default_auction_budget`:
    * A **low keeper price relative to the budget** (e.g., $5–$15 on a $200 budget for a star player) represents
      **high surplus value**, making the player significantly more valuable in trades.
    * A **high keeper price relative to the budget** (e.g., $45+ on a $200 budget for a mid-tier player) represents
      **negative or inflated valuation**, making the player far less valuable to keep going forward because they consume too
      much budget space.
    * Factor keeper surplus value into your trade rationale and proposed trade terms whenever applicable.

OUTPUT SCHEMA & REQUIREMENTS:
- JSON output containing:
  1. `managers_thoughts`: Concise strategic explanation. Strictly <= 1020 characters.
  2. `trade_ideas`: Trade proposals with letter grades for both teams. Strictly <= 1020 characters.
"""
    else:
        return base + f"""
RUN TYPE: Breaking News Trigger ({TRIGGER_REASON})

TASK:
- Evaluate `hints.optimal_roster` to understand the roster moves we're suggesting.
  Correlate these moves with the provided news when applicable.
- DO NOT waste tokens evaluating trades, dynasty strategy, or keeper planning.
- Note that opponent teams aren't being provided in the payload to allow you to focus on our team and your tasks.
- Provide a concise explanation of immediate lineup and news impact for `managers_thoughts`.
- When mentioning specific players always indicate their team and position using (Team, Position) notation (e.g. (KC, RB)).

OUTPUT SCHEMA & REQUIREMENTS:
- JSON output containing:
  1. `managers_thoughts`: Concise explanation of immediate lineup and news impact. Strictly <= 1020 characters.
"""


def get_interactive_system_instruction() -> str:
    """Generates system instruction tailored for interactive console chat."""
    return get_base_instructions() + """
RUN TYPE: Interactive Console Chat

RESPONSE FORMAT:
- Output clean Markdown directly (do NOT use JSON/Discord schemas).
- Use section headers, bold text, and bullet points.

STRUCTURE:
1. **Analysis**: Rationale based on recent news and team state.
2. **Suggested Roster Moves**: Actions, swaps, and free-agent pickups.
3. **Suggested Trade Proposals**: Target, terms, grades, and rationale.
4. **Strategy Notes**: General management commentary.
"""


# ---------------------------------------------------------------------------
# GEMINI API INTERACTION HELPERS
# ---------------------------------------------------------------------------
def execute_chat_turn(
    client: genai.Client,
    model: str,
    system_instruction: str,
    prompt: str,
    temperature: float,
    response_schema: Type[BaseModel],
):
    """Executes chat turn enforcing structured JSON response based on the selected schema."""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=response_schema,
    )

    chat = client.chats.create(model=model, config=config)
    return chat.send_message(prompt)


# ---------------------------------------------------------------------------
# PASS 1: PRE-EVALUATION LOGIC
# ---------------------------------------------------------------------------
def evaluate_rotowire_news(
    news_data: List[Dict[str, Any]], normalize_player_name_fn
) -> Dict[tuple, Dict[str, Any]]:
    """Pass 1: Uses Gemini Structured Outputs to deterministically evaluate RotoWire news items."""
    if not news_data or not GEMINI_API_KEY:
        return {}

    client = genai.Client(api_key=GEMINI_API_KEY)

    system_instruction = (
        "You are a precise NFL fantasy news classifier. Evaluate every news story accurately."
    )
    prompt = (
        "Analyze the following RotoWire NFL news items. Identify the primary subject player for each story, "
        "their NFL team abbreviation, their position, and evaluate whether the news indicates an injury/out status, "
        "volume reduction/demotion, or expanded role/opportunity gain.\n\n"
        f"News Items:\n{json.dumps(prune_rotowire_news(news_data), separators=(',', ':'))}"
    )

    retryer = Retrying(
        retry=retry_if_exception_type(errors.ServerError),
        wait=wait_exponential_jitter(initial=2.0, max=30.0, jitter=2.0),
        stop=stop_after_attempt(3),
        reraise=False,
    )

    for attempt in retryer:
        with attempt:
            attempt_num = attempt.retry_state.attempt_number
            print(f"Executing news pre-evaluation with {GEMINI_FALLBACK_MODEL} (Attempt #{attempt_num})...")
            try:
                response = execute_chat_turn(
                    client=client,
                    model=GEMINI_FALLBACK_MODEL,
                    system_instruction=system_instruction,
                    prompt=prompt,
                    temperature=0.1,
                    response_schema=NewsEvaluationPayload,
                )
                print(f"✓ Success! Usage metadata -> Prompt: {response.usage_metadata.prompt_token_count}, "
                      f"Output: {response.usage_metadata.candidates_token_count}, "
                      f"Total: {response.usage_metadata.total_token_count}")
                news_map = {}
                news_evaluations = NewsEvaluationPayload.model_validate_json(response.text).to_guid_map()
                # decorate the news data with the results
                for article in news_data:
                    guid = article.get("guid").rsplit('/', 1)[-1]
                    if guid and guid in news_evaluations:
                        eval_data = news_evaluations[guid]
                        article["subject_player"] = eval_data.get("subject_player")
                        article["subject_team"] = eval_data.get("subject_team")
                        article["subject_position"] = eval_data.get("subject_position")
                        article["is_missing_game"] = eval_data.get("is_missing_game", False)
                        article["is_losing_volume"] = eval_data.get("is_losing_volume", False)
                        article["is_gaining_volume"] = eval_data.get("is_gaining_volume", False)

                        # Add tuple-keyed entry (clean_name, team, position)
                        subj_key = make_player_key(
                            article["subject_player"],
                            article.get("subject_team", ""),
                            article.get("subject_position", "")
                        )
                        entry = news_map.get(subj_key, {})

                        if article["is_missing_game"]:
                            entry["is_missing_game"] = True
                        if article["is_losing_volume"]:
                            entry["is_losing_volume"] = True
                        if article["is_gaining_volume"]:
                            entry["is_gaining_volume"] = True

                        news_map[subj_key] = entry
                return news_map
            except Exception as e:
                print(f"Attempt #{attempt_num} with {GEMINI_FALLBACK_MODEL} failed during news pre-evaluation: {e}")
                raise

    print("Error during news pre-evaluation pass after retries.")
    return {}


def evaluate_ir_candidates(player_names: List[str]) -> List[str]:
    """Pass 1a: Sorts candidate players by IR slot value using the fallback model."""
    if not player_names or not GEMINI_API_KEY:
        return player_names

    client = genai.Client(api_key=GEMINI_API_KEY)
    system_instruction = "You are a fantasy football expert evaluating injured players for IR slot allocation."
    prompt = (
        "The following players are being evaluated for IR slot value for fantasy football. "
        "Please simply sort the list from highest IR slot value to lowest IR slot value based on their long-term value and expected return.\n\n"
        f"Players:\n{json.dumps(player_names, indent=2)}"
    )

    retryer = Retrying(
        retry=retry_if_exception_type(errors.ServerError),
        wait=wait_exponential_jitter(initial=2.0, max=30.0, jitter=2.0),
        stop=stop_after_attempt(3),
        reraise=False,
    )

    for attempt in retryer:
        with attempt:
            attempt_num = attempt.retry_state.attempt_number
            print(f"Executing IR slot evaluation with {GEMINI_FALLBACK_MODEL} (Attempt #{attempt_num})...")
            try:
                response = execute_chat_turn(
                    client=client,
                    model=GEMINI_FALLBACK_MODEL,
                    system_instruction=system_instruction,
                    prompt=prompt,
                    temperature=0.1,
                    response_schema=SortedPlayerNamesPayload,
                )
                sorted_payload = SortedPlayerNamesPayload.model_validate_json(response.text)
                return sorted_payload.sorted_player_names
            except Exception as e:
                print(f"Attempt #{attempt_num} with {GEMINI_FALLBACK_MODEL} failed during IR slot evaluation: {e}")
                raise

    print("Error during IR slot evaluation pass after retries.")
    return player_names


# ---------------------------------------------------------------------------
# PASS 2: FULL EVALUATION LOGIC
# ---------------------------------------------------------------------------
def evaluate_team(payload: Dict[str, Any]) -> str:
    """Invokes Gemini model to analyze fantasy payload and produce structured embed outputs."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    system_instruction = get_system_instruction()
    schema = get_schema()
    pruned_payload = prune_players(payload)
    pruned_payload["rotowire_news"] = prune_rotowire_news(payload["rotowire_news"])
    prompt = (
        f"Here is the latest data snapshot in JSON format:\n\n"
        f"{json.dumps(pruned_payload, separators=(',', ':'))}"
    )

    if TRIGGER_REASON == SCHEDULED_RUN_REASON:
        selected_models = [GEMINI_MODEL, GEMINI_FALLBACK_MODEL]
        selected_temp = 0.4
        print(f"Scheduled run detected. Using primary model: {selected_models[0]}")
    else:
        selected_models = [GEMINI_FALLBACK_MODEL, GEMINI_FALLBACK_MODEL]
        selected_temp = 0.2
        print(f"News/Event trigger detected. Using high-quota model: {selected_models[0]}")

    retryer = Retrying(
        retry=retry_if_exception_type(errors.ServerError),
        wait=wait_fixed(10),
        stop=stop_after_attempt(2),
        reraise=False,
    )

    try:
        for attempt in retryer:
            with attempt:
                attempt_num = attempt.retry_state.attempt_number
                selected_model = selected_models[attempt_num-1]
                print(f"Executing request with {selected_model} (Attempt #{attempt_num})...")
                try:
                    response = execute_chat_turn(
                        client, selected_model, system_instruction, prompt, selected_temp, schema
                    )
                    print(f"✓ Success! Usage metadata -> Prompt: {response.usage_metadata.prompt_token_count}, "
                          f"Output: {response.usage_metadata.candidates_token_count}, "
                          f"Total: {response.usage_metadata.total_token_count}")
                    return schema.model_validate_json(response.text).model_dump_json()
                except Exception as e:
                    print(f"Attempt #{attempt_num} with {selected_model} failed: {e}")
                    raise
    except RetryError as e:
        print(f"Gemini evaluation failed after retries: {e}")
    except errors.ClientError as e:
        print(f"Gemini evaluation failed: {e}")

    fallback_thoughts = GEMINI_UNAVAILABLE_MESSAGE
    if schema == ManagerThoughtsScheduled:
        fallback_obj = schema(
            managers_thoughts=fallback_thoughts,
            trade_ideas="No suggested trade proposals at this time."
        )
    else:
        fallback_obj = schema(
            managers_thoughts=fallback_thoughts
        )
    return fallback_obj.model_dump_json()


# ---------------------------------------------------------------------------
# INTERACTIVE MODE SESSION
# ---------------------------------------------------------------------------
def start_interactive_mode(payload: Dict[str, Any]) -> None:
    """Launches an interactive chat session using the fallback model."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory
    except ImportError:
        raise ImportError(
            "The 'prompt_toolkit' package is required for interactive mode. "
            "Please install it using 'pip install prompt_toolkit'."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)
    console = Console()
    session = PromptSession(history=InMemoryHistory())
    pruned_payload = prune_players(payload)
    pruned_payload["rotowire_news"] = prune_rotowire_news(payload["rotowire_news"])
    system_instruction = get_interactive_system_instruction()
    prompt = (
        f"Here is the latest data snapshot in JSON format:\n\n"
        f"{json.dumps(pruned_payload, separators=(',', ':'))}\n\n"
        f"Please provide your initial assessment and recommendations for my team."
    )

    print(f"\n--- Starting Interactive Chat Mode using {GEMINI_FALLBACK_MODEL} ---")

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.4,
    )

    chat = client.chats.create(model=GEMINI_FALLBACK_MODEL, config=config)

    print("Sending initial prompt and data payload to fallback model...\n")

    retryer = Retrying(
        retry=retry_if_exception_type(errors.ServerError),
        wait=wait_exponential_jitter(initial=2.0, max=30.0, jitter=2.0),
        stop=stop_after_attempt(5),
        reraise=False,
    )

    def send_silent_retry(msg: str):
        try:
            for attempt in retryer:
                with attempt:
                    return chat.send_message(msg)
        except RetryError:
            return None

    response = send_silent_retry(prompt)
    if response:
        console.print(Markdown(response.text))
    else:
        console.print("[bold red]Gave up retrying after multiple server errors. Please try asking again.[/bold red]")

    print("\nInteractive session started. Type your follow-up questions (type 'exit' or 'quit' to stop):\n")

    while True:
        try:
            user_input = session.prompt("\nUser > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting interactive session.")
                break

            response = send_silent_retry(user_input)
            if response:
                console.print("\n[bold green]Manager >[/bold green]")
                console.print(Markdown(response.text))
            else:
                console.print("\n[bold red]Gave up retrying after multiple server errors. Please try asking again.[/bold red]")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive session.")
            break


# ---------------------------------------------------------------------------
# PAYLOAD PRUNING FOR TOKEN EFFICIENCY
# ---------------------------------------------------------------------------
def prune_player_entry(p: Dict[str, Any]) -> Dict[str, Any]:
    """Prunes player attributes and renames keys to minimize token usage for Gemini."""
    if not isinstance(p, dict):
        return p

    pruned = {}

    # Standard Identifiers & Positional Data
    if "name" in p:
        pruned["name"] = p["name"]
    if "position" in p:
        pruned["pos"] = p["position"]
    if "slot_position" in p:
        pruned["slot"] = p["slot_position"]
    if "sleeper_nfl_team" in p:
        pruned["team"] = p["sleeper_nfl_team"]
    if "nfl_team" in p:
        pruned["team"] = p["nfl_team"]

    # Status & Projections
    if "projected_points" in p:
        pruned["proj"] = p["projected_points"]
    if "projected_ceiling_points" in p:
        pruned["ceil"] = p["projected_ceiling_points"]
    if "projected_ppg" in p:
        pruned["proj_ppg"] = p["projected_ppg"]
    if "ytd_ppg" in p:
        pruned["ytd_ppg"] = p["ytd_ppg"]
    if "injury_status" in p and p["injury_status"]:
        pruned["inj"] = p["injury_status"]
    if "status" in p and p["status"]:
        pruned["status"] = p["status"]

    # Percentages & Advanced Stats
    if "percent_owned" in p:
        pruned["own_pct"] = p["percent_owned"]
    if "percent_started" in p:
        pruned["start_pct"] = p["percent_started"]
    if "expected_snap_pct_if_active" in p and p["expected_snap_pct_if_active"] is not None:
        pruned["snap_pct"] = p["expected_snap_pct_if_active"]
    if "projected_games_missed" in p and p["projected_games_missed"] is not None:
        pruned["miss_gms"] = p["projected_games_missed"]
    if "injury_risk" in p and p["injury_risk"]:
        pruned["inj_risk"] = p["injury_risk"]

    # Financials & Ownership Flags
    if "keeper_price" in p and p["keeper_price"] is not None:
        pruned["price"] = p["keeper_price"]
    if "is_lineup_locked" in p:
        pruned["locked"] = p["is_lineup_locked"]

    # Market Trends & Role Indicators
    if "sleeper_trending" in p and p["sleeper_trending"]:
        pruned["trend"] = p["sleeper_trending"]
    if "sleeper_trending_count" in p and p["sleeper_trending_count"]:
        pruned["trend_count"] = p["sleeper_trending_count"]
    if "is_gaining_volume" in p:
        pruned["gain_vol"] = p["is_gaining_volume"]
    if "is_losing_volume" in p:
        pruned["lose_vol"] = p["is_losing_volume"]
    if "is_missing_game" in p:
        pruned["miss_gm"] = p["is_missing_game"]

    # Handcuffs & Contingency Data
    if "handcuff_for_starter_name" in p and p["handcuff_for_starter_name"]:
        pruned["hc_for"] = p["handcuff_for_starter_name"]
    if "contingent_projected_points" in p and p["contingent_projected_points"] is not None:
        pruned["hc_pts"] = p["contingent_projected_points"]

    # Optimization Output Flags
    if "bench_slot" in p:
        pruned["slot"] = p["bench_slot"]
    if "policy_type" in p:
        pruned["policy"] = p["policy_type"]
    if "suitability_score" in p:
        pruned["score"] = p["suitability_score"]
    if "transaction_type" in p and p["transaction_type"]:
        pruned["tx"] = p["transaction_type"]
    if "is_free_agent" in p:
        pruned["fa"] = p["is_free_agent"]

    return pruned


def prune_players(combined_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a new lighter payload by recursively pruning player entries."""
    def process_item(item: Any) -> Any:
        if isinstance(item, dict):
            # If entry resembles a player dictionary with a name and positional marker
            if "name" in item and ("position" in item or "slot_position" in item or "slot" in item):
                return prune_player_entry(item)
            return {k: process_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [process_item(x) for x in item]
        return item

    return process_item(combined_payload)


def prune_rotowire_news(news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prunes RotoWire news items by removing URL/GUID metadata, stripping recurring
    boilerplate text, abbreviating keys, and dropping default false/null flags.
    """
    pruned_news = []

    # Regex to catch the standard RotoWire suffix text & trailing whitespace
    boilerplate_regex = re.compile(r"\s*Visit RotoWire\.com for more analysis on this update\.\s*", re.IGNORECASE)

    for item in news_list:
        if not isinstance(item, dict):
            continue

        pruned_item = {}

        # Core Fields
        if "guid" in item and item["guid"]:
            pruned_item["guid"] = item["guid"].rsplit('/', 1)[-1]
        if "title" in item and item["title"]:
            pruned_item["title"] = item["title"]

        if "description" in item and item["description"]:
            # Clean boilerplate suffix and trim whitespace
            clean_desc = boilerplate_regex.sub("", item["description"]).strip()
            pruned_item["desc"] = clean_desc

        if "subject_player" in item and item["subject_player"]:
            pruned_item["player"] = item["subject_player"]
        if "subject_team" in item and item["subject_team"]:
            pruned_item["team"] = item["subject_team"]
        if "subject_position" in item and item["subject_position"]:
            pruned_item["pos"] = item["subject_position"]

        if "published" in item and item["published"]:
            pruned_item["pub"] = item["published"]

        # Evaluated Flags (Only keep if True to save space)
        if item.get("is_missing_game"):
            pruned_item["miss_gm"] = True
        if item.get("is_losing_volume"):
            pruned_item["lose_vol"] = True
        if item.get("is_gaining_volume"):
            pruned_item["gain_vol"] = True

        pruned_news.append(pruned_item)

    return pruned_news


# ---------------------------------------------------------------------------
# DEBUG DISPLAY
# ---------------------------------------------------------------------------
def print_debug_prompt(payload: Dict[str, Any], interactive: bool = False) -> None:
    """Prints system instruction and JSON payload to console without making API calls."""
    system_instruction = (
        get_interactive_system_instruction()
        if interactive
        else get_system_instruction()
    )
    pruned_payload = prune_players(payload)
    pruned_payload["rotowire_news"] = prune_rotowire_news(payload["rotowire_news"])
    prompt = f"Here is the latest data snapshot in JSON format:\n\n{json.dumps(pruned_payload, indent=2)}"

    print("\n" + "=" * 80)
    print(" DEBUG MODE: FULL PROMPT AND PAYLOAD (PASS 2 INFERENCE SKIPPED)")
    print("=" * 80 + "\n")
    print("--- SYSTEM INSTRUCTION ---")
    print(system_instruction.strip())
    print("\n--- USER PROMPT / PAYLOAD ---")
    print(prompt)
    print("\n" + "=" * 80)
    print(" DEBUG COMPLETE: Exiting without making API calls or posting webhooks.")
    print("=" * 80 + "\n")
