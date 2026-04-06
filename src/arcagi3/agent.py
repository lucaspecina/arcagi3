"""LLM Agent for ARC-AGI-3 using Azure AI Foundry.

Architecture: Analyzer-Reflector-Actor loop.
- Analyzer: focused perception (what changed, what do I see?)
- Reflector: forced meta-cognition — validate EVERY belief against new evidence
- Actor: picks the best action based on validated beliefs
- Trackers: deterministic avatar/bar detection
"""

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
from arcengine import GameAction, GameState

from PIL import Image

from .grid_utils import (
    compute_diff,
    describe_frame,
    grid_hash,
    grid_to_base64,
    grid_to_image,
    grid_to_text_compact,
    image_to_base64,
)
from .exploration import ExplorationController
from .trackers import AvatarTracker, BarTracker

# Map action names to GameAction enum
ACTION_MAP = {
    "RESET": GameAction.RESET,
    "ACTION1": GameAction.ACTION1,
    "ACTION2": GameAction.ACTION2,
    "ACTION3": GameAction.ACTION3,
    "ACTION4": GameAction.ACTION4,
    "ACTION5": GameAction.ACTION5,
    "ACTION6": GameAction.ACTION6,
    "ACTION7": GameAction.ACTION7,
}

ACTION_LABELS = {
    "RESET": "RESET",
    "ACTION1": "A1", "ACTION2": "A2", "ACTION3": "A3", "ACTION4": "A4",
    "ACTION5": "A5", "ACTION6": "A6(click)", "ACTION7": "A7",
}

# --- ANALYZER PROMPT ---
ANALYZER_PROMPT = """\
You are analyzing a frame from an ARC-AGI-3 game — a 64x64 abstract visual puzzle.
Your job is ONLY to perceive, categorize, and interpret. You do NOT choose actions.

The harness provides you with:
- The current frame (image)
- Programmatic diff (exact pixel changes, object movements)
- Tracker data (avatar detection, resource bar warnings)
- Your previous analysis (if any)

You MUST answer ALL of these sections with SPECIFIC details:

== 1. FULL SCENE DESCRIPTION ==
Describe EVERYTHING you see as if explaining to someone who cannot see the image.
What is the overall structure? What distinct areas or regions exist?
What colors dominate? What is the general layout?

== 2. ELEMENT CLASSIFICATION ==
Classify EVERY visual element you can see into one of these categories:

  BACKGROUND/WALLS: Static elements that form the level structure.
    What color are they? Where are they? Do they form corridors, rooms, barriers?

  WALKABLE AREAS / PATHS: Where can the player move?
    What color represents the floor/path? How are paths connected?

  PLAYER / CONTROLLED OBJECT: What do you control?
    What color? What shape? Where is it? What objects move WITH it?
    How do you know this is the player? (cite evidence from action effects)

  META-INFORMATION / HUD: Elements that show game state, NOT part of the level.
    Health bars, resource indicators, score displays, timers, progress bars.
    For each: what does it represent? Is depleting it BAD (cost/health) or
    GOOD (progress)? At what rate is it changing? How much is left?

  POTENTIAL TARGETS / GOALS: Objects that MIGHT be objectives.
    Why do they look like goals? (distinctive color, isolated position, etc.)
    Have you interacted with them yet? What happened when you did?

  POTENTIAL INTERACTIVE OBJECTS: Things you might be able to interact with.
    Doors, switches, items, NPCs, collectibles, keys, etc.
    What makes you think they are interactive? Have you tested them?

  UNKNOWN / UNCLASSIFIED: Things you see but dont understand yet.
    Describe them precisely. What experiments would reveal their purpose?

== 3. SPATIAL MAP ==
Describe the level as a map: where are walls, where are openings,
where are corridors, rooms, dead ends. Where is the player right now?
What areas are reachable? What areas havent you explored yet?
Where are the interesting objects relative to the player?

== 4. GOAL HYPOTHESES AND PLANS ==
List 3-5 hypotheses for what the objective is. Confidences must sum to ~100%.
For EACH hypothesis, include a concrete PLAN — a sequence of actions to test it.

== 5. BIGGEST UNKNOWNS ==
The 3 things you are MOST uncertain about. For each: what specific
action or experiment would reduce that uncertainty?

Respond with JSON:
{
  "scene_description": "full visual description of everything you see",
  "classification": {
    "background_walls": [
      {"color": "gray", "description": "walls forming corridors", "positions": "top half of screen"}
    ],
    "walkable_paths": [
      {"color": "black", "description": "corridors between walls", "positions": "center area"}
    ],
    "player": {
      "color": "blue", "shape": "small square", "position": "(36,43)",
      "attached_objects": "orange pixels move with it",
      "evidence": "moves consistently when actions are taken"
    },
    "meta_info": [
      {"element": "yellow bar at bottom", "type": "health/cost",
       "current_value": "37px", "rate_of_change": "-5px per move",
       "remaining": "about 7 moves left", "interpretation": "each move costs resources"}
    ],
    "potential_targets": [
      {"color": "white", "position": "(10,15)", "why_target": "isolated, distinctive color",
       "tested": false, "interaction_result": "not yet tested"}
    ],
    "potential_interactive": [
      {"color": "red", "position": "(25,30)", "why_interactive": "small isolated object near path",
       "tested": false}
    ],
    "unknown": [
      {"description": "small green pixel at (50,5)", "experiment": "navigate near it"}
    ]
  },
  "spatial_map": "description of level layout as a map, with player position and key landmarks",
  "goal_hypotheses": [
    {
      "rank": 1, "goal": "navigate to the white object at top-left",
      "confidence_pct": 40,
      "evidence_for": "it is isolated and distinctive, common pattern for goals",
      "evidence_against": "no evidence it triggers anything",
      "plan": ["ACTION3", "ACTION3", "ACTION1", "ACTION1", "ACTION1"],
      "plan_reasoning": "move left then up to reach the white object"
    },
    {
      "rank": 2, "goal": "collect all red objects scattered in the level",
      "confidence_pct": 30,
      "evidence_for": "there are multiple red objects at different positions",
      "evidence_against": "havent tried touching one yet",
      "plan": ["ACTION2", "ACTION4", "ACTION4"],
      "plan_reasoning": "move toward the nearest red object to test interaction"
    }
  ],
  "unknowns": [
    {"question": "what does the red object do when touched?",
     "experiment": "navigate to it and step on it"}
  ],
  "contradictions": "what changed or was wrong vs previous analysis"
}
"""

# --- REFLECTOR PROMPT ---
# This is the base prompt. The actual beliefs are injected dynamically
# so the LLM must review each one individually.
REFLECTOR_PROMPT = """\
You are the REFLECTION module of an ARC-AGI-3 agent. You perform MANDATORY \
meta-cognition after every single action.

## STEP 1: WHAT HAPPENED
What exactly changed after this action? Be specific: which objects moved, \
which pixels changed, which bars decreased, what appeared/disappeared.

## STEP 2: PREDICTION vs REALITY
What did you predict? What actually happened? If different, explain WHY \
your prediction was wrong.

## STEP 3: REVIEW EACH BELIEF (MANDATORY)
Below you will see your current beliefs listed one by one.
For EACH belief, you MUST provide:
- VERDICT: KEEP (still true), CHANGE (partially wrong, here's the correction), \
or DROP (completely wrong)
- JUSTIFICATION: What specific evidence from THIS step supports your verdict? \
If no new evidence, say so — but also say how many steps this belief has gone \
unverified and whether you should test it.

You CANNOT skip any belief. Every single one needs a verdict + justification.

## STEP 4: CAUSAL ANALYSIS — THE "WHY" (MOST IMPORTANT STEP)
For EVERY action result, ask yourself WHY:
- If an action WORKED: WHY did it work? What conditions made it possible?
- If an action FAILED or had no effect: WHY? What BLOCKED it? What caused the failure?
  - Is there a wall? WHERE is the wall? What color is it?
  - Is there an object in the way? WHAT object? Could you interact with it?
  - Does the action only work in certain conditions? WHAT conditions?
- If you DIED: WHY? What killed you? What object/position/condition caused death?
  - Was it a color? A position? A timing? A specific object?
- Form CAUSAL HYPOTHESES: "X happened BECAUSE Y"
  - Example: "ACTION3 had no effect BECAUSE there is a gray wall at (30,20) blocking leftward movement"
  - Example: "I died BECAUSE the blue object touched the red object at (15,25)"
- These causal hypotheses MUST be saved in your beliefs and reviewed each step.
  They can be changed or dropped if new evidence contradicts them.

Look at the ACTION EFFECT MAP. If an action had DIFFERENT effects at different \
positions (worked sometimes, failed other times):
- WHAT WAS DIFFERENT about the positions where it worked vs didn't?
- Were there adjacent objects to interact with? Walls blocking? Different surroundings?
- Form a RULE about WHEN/WHERE this action works, not just WHAT it does.

## STEP 5: NEW DISCOVERIES
What did you learn that isn't captured in any existing belief?
What new CAUSAL relationships did you observe?

## STEP 6: GOAL HYPOTHESES (MANDATORY — maintain 3-5 at all times)
You MUST maintain 3-5 goal hypotheses ranked by confidence. For EACH:
- State the hypothesis clearly
- Confidence percentage (0-100%). All confidences must sum to ~100%.
- Key evidence FOR and AGAINST
- What would CONFIRM or REFUTE it (specific test)
**If steps_without_progress >= 5, your TOP hypothesis has FAILED. Promote \
your #2 hypothesis to #1 and demote the old #1 below 20%.**

## STEP 7: UNCERTAINTY REDUCTION
- What are your TOP 3 unknowns right now?
- For EACH unknown: what specific action/experiment would resolve it?
- What is the SINGLE most valuable thing you could learn right now?
- What action would give you the most INFORMATION (not necessarily progress)?

## STEP 8: UPDATED BELIEFS
Output the COMPLETE updated belief set incorporating all changes.
**Controls MUST include WHEN/WHERE each action works, not just what it does.**

Respond with JSON:
{
  "what_happened": "specific changes observed",
  "prediction_vs_reality": "expected X, got Y, because Z",
  "belief_reviews": [
    {"id": 0, "belief": "original text", "verdict": "KEEP|CHANGE|DROP", \
"justification": "why, citing evidence from this step", "corrected": "new text if CHANGE"}
  ],
  "causal_analysis": "WHY did things happen the way they did? What CAUSED each result?",
  "causal_hypotheses": [
    {"observation": "what happened", "cause": "WHY it happened", "confidence_pct": 70,
     "test": "how to verify this causal link"}
  ],
  "new_discoveries": ["discovery 1"],
  "goal_hypotheses": [
    {"rank": 1, "goal": "what to do", "confidence_pct": 50,
     "evidence_for": "specific evidence", "evidence_against": "specific evidence",
     "confirm_test": "action that would prove this", "refute_test": "action that would disprove this"},
    {"rank": 2, "goal": "alternative", "confidence_pct": 30, "...": "..."},
    {"rank": 3, "goal": "alternative", "confidence_pct": 20, "...": "..."}
  ],
  "uncertainty_reduction": {
    "top_unknowns": [
      {"question": "what is X?", "experiment": "try Y to find out"}
    ],
    "most_valuable_info": "the single thing that would help most",
    "best_info_action": "the action that maximizes information gain"
  },
  "strategy_check": {
    "making_progress": true/false,
    "steps_without_progress": N,
    "untried_sequences": ["action sequences not yet attempted"]
  },
  "updated_beliefs": {
    "controls": {"ACTION1": "WHEN it works + WHAT it does + WHY it fails when it does", "...": "..."},
    "rules": ["confirmed rule with evidence count"],
    "causal_model": ["X happens BECAUSE Y (confidence N%, tested N times)"],
    "goal": "top hypothesis (confidence: N%, evidence: X)",
    "objects": ["object descriptions with locations and WHAT ROLE they play"],
    "dangers": ["what causes damage/death and WHY"],
    "unknowns": ["things I still need to test and HOW I would test them"],
    "failed_approaches": ["what I tried, what happened, and WHY it failed"]
  }
}
"""

# --- ACTOR PROMPT ---
ACTOR_PROMPT = """\
You are the ACTION module of an ARC-AGI-3 agent. You receive VALIDATED beliefs \
(already checked against evidence) and choose the BEST next action.

RULES:
- EFFICIENCY MATTERS: score = (human_actions/your_actions)². Fewer = better.
- TRUST YOUR BELIEFS: they were just validated by the reflector. Act on them.
- If there are UNTESTED actions, prioritize testing them — one at a time.
- ACTION EFFECTS ARE POSITION-DEPENDENT: check the ACTION EFFECT MAP. An action \
that worked at one position may not work at another. Plan based on WHERE you are.
- RESET restarts the level. NEVER use RESET as a movement action.
- NEVER repeat a failed approach. Check "failed_approaches" in your beliefs.
- If the same action had no effect, DO NOT try it again from the same position.
- If stuck, try a COMPLETELY DIFFERENT action or sequence.
- Be SPECIFIC in expected_result — the reflector will check your prediction.

Respond with a JSON object:
{
  "reasoning": "Why this action, based on validated beliefs",
  "action": "ACTION1|...|ACTION7|RESET",
  "x": 0, "y": 0,
  "expected_result": "SPECIFIC prediction: what object moves where, what changes"
}

"x" and "y" are ONLY for ACTION6. Keep reasoning SHORT and focused.
"""


@dataclass
class AgentConfig:
    """Configuration for the LLM agent."""

    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-4o"
    max_actions: int = 80
    use_vision: bool = True
    use_text: bool = True
    message_history_limit: int = 10
    temperature: float = 0.7
    delay: float = 0.0
    save_frames: bool = False
    frames_dir: str = "frames"
    step_mode: bool = False
    show_raw_prompt: bool = False
    show_window: bool = False
    analyze_every: int = 3  # Run analyzer every N steps (1 = every step)
    systematic_explore: bool = True  # Phase 1: harness-driven exploration
    prior_beliefs: str = ""  # Inject beliefs from a previous run

    def __post_init__(self):
        if not self.base_url:
            self.base_url = os.environ.get("AZURE_FOUNDRY_BASE_URL", "")
        if not self.api_key:
            self.api_key = os.environ.get("AZURE_INFERENCE_CREDENTIAL", "")
        if not self.model:
            self.model = os.environ.get("AZURE_MODEL", "gpt-4o")


@dataclass
class StepRecord:
    """Record of a single agent step."""

    action_num: int
    action: str
    x: int | None = None
    y: int | None = None
    reasoning: str = ""
    state: str = ""
    levels_completed: int = 0
    diff_summary: str = ""
    avatar_pos: tuple[int, int] | None = None  # Position BEFORE action
    had_effect: bool = False


@dataclass
class AgentState:
    """Mutable state of the agent during a game session."""

    steps: list[StepRecord] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    prev_grid: np.ndarray | None = None
    memory: str = ""
    human_feedback: str = ""
    available_actions: list[int] = field(default_factory=list)
    # Harness-tracked state
    state_hashes: set = field(default_factory=set)
    actions_tested: set = field(default_factory=set)
    no_progress_count: int = 0
    frame_analysis: str = ""
    diff_text: str = ""
    # Analyzer state
    last_analysis: str = ""
    analysis_history: list[str] = field(default_factory=list)
    # Contextual action tracking: action → [(position, had_effect, summary)]
    action_context_log: dict = field(default_factory=dict)


def create_client(config: AgentConfig):
    """Create OpenAI client configured for Azure Foundry."""
    from openai import OpenAI

    if not config.base_url:
        raise ValueError("Set AZURE_FOUNDRY_ENDPOINT env var or pass base_url.")
    if not config.api_key:
        raise ValueError("Set AZURE_INFERENCE_CREDENTIAL env var or pass api_key.")

    return OpenAI(base_url=config.base_url, api_key=config.api_key)


def prompt_human(prompt_text: str) -> str | None:
    """Prompt the human observer for input. Returns None on quit."""
    try:
        user_input = input(f"\n  > {prompt_text} (Enter to continue, q to quit): ")
        if user_input.strip().lower() == "q":
            return None
        return user_input.strip()
    except (KeyboardInterrupt, EOFError):
        return None


def build_action_context_summary(state: AgentState) -> str:
    """Build a position-aware summary of action effects.

    Shows WHERE each action worked vs didn't, so the LLM can infer
    position-dependent mechanics (e.g., swaps only with adjacent objects).
    """
    if not state.action_context_log:
        return ""

    lines = []
    for action, entries in sorted(state.action_context_log.items()):
        if action == "RESET":
            continue  # RESET handled separately
        worked = [e for e in entries if e["had_effect"]]
        failed = [e for e in entries if not e["had_effect"]]
        line = f"  {action}: worked {len(worked)}/{len(entries)} times"
        if worked:
            examples = worked[-2:]  # Last 2 successes
            wheres = [f"at ({e['pos'][0]},{e['pos'][1]}): {e['summary'][:40]}" for e in examples if e.get("pos")]
            if wheres:
                line += f" — " + "; ".join(wheres)
        if failed:
            fail_pos = [f"({e['pos'][0]},{e['pos'][1]})" for e in failed[-2:] if e.get("pos")]
            if fail_pos:
                line += f" | NO effect at: {', '.join(fail_pos)}"
        lines.append(line)

    if not lines:
        return ""
    return "=== ACTION EFFECT MAP (position-aware) ===\n" + "\n".join(lines) + "\n==="


def build_context_text(
    grid: np.ndarray,
    state: AgentState,
    config: AgentConfig,
    avatar_tracker: AvatarTracker,
    bar_tracker: BarTracker,
    exploration: ExplorationController | None = None,
) -> str:
    """Build the context text block shared by both analyzer and actor."""
    parts = []
    parts.append(f"Step {len(state.steps) + 1}/{config.max_actions}")

    if state.steps:
        last = state.steps[-1]
        parts.append(f"Last action: {last.action} -> {last.state}")
        parts.append(f"Levels completed: {last.levels_completed}")

    # Available actions
    if state.available_actions:
        action_names = []
        for aid in state.available_actions:
            name = f"ACTION{aid}" if aid != 0 else "RESET"
            action_names.append(name)
        if "RESET" not in action_names:
            action_names.append("RESET")
        parts.append(f"Available actions: {', '.join(action_names)}")

        untested = [n for n in action_names if n not in state.actions_tested and n != "RESET"]
        if untested:
            parts.append(f"UNTESTED actions: {', '.join(untested)}")

    # Programmatic diff — add RESET annotation
    if state.diff_text:
        diff_label = state.diff_text
        if state.steps and state.steps[-1].action == "RESET":
            diff_label = (
                "WARNING: RESET was used — level restarted to initial state. "
                "Any position changes are from RETURNING TO START, not movement.\n"
                + diff_label
            )
        parts.append(f"\n=== DIFF (computed, accurate) ===\n{diff_label}\n===")

    # Avatar tracker — make action map PROMINENT and authoritative
    avatar_info = avatar_tracker.get_avatar_info()
    action_map = avatar_tracker.get_action_map()
    if action_map:
        parts.append("\n--- CONFIRMED ACTION EFFECTS (computed from observations, trust this) ---")
        for a, desc in action_map.items():
            parts.append(f"  {a}: {desc}")
        # Show actions with no observed movement
        if state.available_actions:
            all_actions = {f"ACTION{aid}" for aid in state.available_actions if aid != 0}
            mapped = set(action_map.keys())
            unmapped = all_actions - mapped - {"RESET"}
            for a in sorted(unmapped):
                if a in state.actions_tested:
                    parts.append(f"  {a}: NO avatar movement observed (tested)")
                else:
                    parts.append(f"  {a}: UNTESTED")
        parts.append("  RESET: RESTARTS the level (returns to initial state, NOT movement)")
        parts.append("--- END ACTION EFFECTS ---")
    elif avatar_info:
        parts.append(f"\n=== AVATAR TRACKER ===\n{avatar_info}\n===")

    # Bar tracker warnings
    bar_warnings = bar_tracker.get_bar_warnings()
    if bar_warnings:
        parts.append(f"\n=== BAR TRACKER ===\n{bar_warnings}\n===")

    # Frame analysis
    if state.frame_analysis:
        parts.append(f"\n=== FRAME OBJECTS ===\n{state.frame_analysis}\n===")

    # Position-aware action effects (critical for learning mechanics)
    action_ctx = build_action_context_summary(state)
    if action_ctx:
        parts.append(f"\n{action_ctx}")

    # State novelty
    current_hash = grid_hash(grid)
    is_novel = current_hash not in state.state_hashes
    parts.append(f"State: {'NEW' if is_novel else 'REPEATED'}")
    if state.no_progress_count >= 3:
        parts.append(f"WARNING: NO PROGRESS for {state.no_progress_count} turns! Change approach!")

    # Structured action history (last 10 steps as table)
    if state.steps:
        parts.append("\n=== ACTION HISTORY (most recent) ===")
        parts.append("Step | Action  | Pos(x,y)  | Result      | What changed")
        parts.append("-----|---------|-----------|-------------|---------------------------")
        for s in state.steps[-10:]:
            diff = s.diff_summary[:45] if s.diff_summary else "no data"
            status = "ok" if s.state == "IN_PROGRESS" else s.state
            pos = f"({s.avatar_pos[0]},{s.avatar_pos[1]})" if s.avatar_pos else "(?)"
            parts.append(f" {s.action_num:>3} | {s.action:<7} | {pos:<9} | {status:<11} | {diff}")
        parts.append("===")

    # Exploration controller report
    if exploration:
        report = exploration.get_exploration_report()
        if report:
            parts.append(f"\n=== EXPLORATION CONTROLLER ===\n{report}\n===")

    # Human feedback
    if state.human_feedback:
        parts.append(f"\n=== HUMAN OBSERVER ===\n{state.human_feedback}\n===")

    return "\n".join(parts)


def run_analyzer(
    client,
    grid: np.ndarray,
    state: AgentState,
    config: AgentConfig,
    avatar_tracker: AvatarTracker,
    bar_tracker: BarTracker,
    exploration: ExplorationController | None = None,
) -> str:
    """Run the ANALYZER to perceive and interpret the current state."""
    context = build_context_text(grid, state, config, avatar_tracker, bar_tracker, exploration)

    content = []
    text = context
    if state.last_analysis:
        text += f"\n\n=== YOUR PREVIOUS ANALYSIS ===\n{state.last_analysis}\n==="
    content.append({"type": "text", "text": text})

    # Add image
    if config.use_vision:
        b64 = grid_to_base64(grid, scale=2)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })

    messages = [
        {"role": "system", "content": ANALYZER_PROMPT},
        {"role": "user", "content": content},
    ]

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=0.3,  # Lower temp for perception accuracy
            max_completion_tokens=3000,
            timeout=180,
        )
        analysis = response.choices[0].message.content or ""
    except Exception as e:
        print(f"  [ANALYZER] API error: {e}")
        analysis = state.last_analysis or "{}"

    state.last_analysis = analysis
    state.analysis_history.append(analysis)
    return analysis


def _enumerate_beliefs(memory_json: str) -> str:
    """Flatten beliefs into a numbered list so the LLM must address each one."""
    try:
        beliefs = json.loads(memory_json)
    except (json.JSONDecodeError, TypeError):
        return f"[0] {memory_json}"

    items = []
    idx = 0

    # Controls
    controls = beliefs.get("controls", {})
    if isinstance(controls, dict):
        for action, effect in controls.items():
            items.append(f"[{idx}] CONTROL: {action} → {effect}")
            idx += 1

    # Rules
    for rule in beliefs.get("rules", []):
        items.append(f"[{idx}] RULE: {rule}")
        idx += 1

    # Goal
    goal = beliefs.get("goal", "")
    if goal:
        items.append(f"[{idx}] GOAL: {goal}")
        idx += 1

    # Objects
    for obj in beliefs.get("objects", []):
        items.append(f"[{idx}] OBJECT: {obj}")
        idx += 1

    # Dangers
    for danger in beliefs.get("dangers", []):
        items.append(f"[{idx}] DANGER: {danger}")
        idx += 1

    # Unknowns
    for unk in beliefs.get("unknowns", []):
        items.append(f"[{idx}] UNKNOWN: {unk}")
        idx += 1

    # Failed approaches
    for fa in beliefs.get("failed_approaches", []):
        items.append(f"[{idx}] FAILED: {fa}")
        idx += 1

    if not items:
        return "(no beliefs yet)"
    return "\n".join(items)


def run_reflector(
    client,
    grid: np.ndarray,
    state: AgentState,
    config: AgentConfig,
    analysis: str,
    last_action: str,
    expected_result: str,
    avatar_tracker: AvatarTracker,
    bar_tracker: BarTracker,
    exploration: ExplorationController | None = None,
) -> str:
    """Run the REFLECTOR to validate beliefs against new evidence."""
    context = build_context_text(grid, state, config, avatar_tracker, bar_tracker, exploration)

    # Enumerate beliefs as numbered list
    belief_list = _enumerate_beliefs(state.memory) if state.memory else "(no beliefs yet — first step)"

    content = []
    text = f"ACTION TAKEN: {last_action}\n"
    text += f"YOUR PREDICTION: {expected_result}\n\n"
    text += context
    if analysis:
        text += f"\n\n=== PERCEPTION ===\n{analysis}\n==="
    text += f"\n\n=== YOUR BELIEFS — Review EACH one. Verdict + justification required ===\n{belief_list}\n==="

    # Harness-enforced stagnation warning
    if state.no_progress_count >= 5:
        text += (
            f"\n\n*** STAGNATION ALERT: {state.no_progress_count} steps without progress! ***\n"
            "Your current goal hypothesis has FAILED. You MUST change it.\n"
            "Pick your best alternative goal and commit to it in updated_beliefs.\n"
            "Also list what you tried that didn't work in failed_approaches."
        )

    content.append({"type": "text", "text": text})

    if config.use_vision:
        b64 = grid_to_base64(grid, scale=2)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })

    messages = [
        {"role": "system", "content": REFLECTOR_PROMPT},
        {"role": "user", "content": content},
    ]

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=3000,
            timeout=180,
        )
        reflection = response.choices[0].message.content or ""
    except Exception as e:
        print(f"  [REFLECTOR] API error: {e}")
        reflection = "{}"

    # Extract updated beliefs from reflection
    parsed = parse_response(reflection)
    if "updated_beliefs" in parsed:
        beliefs = parsed["updated_beliefs"]
        if isinstance(beliefs, dict):
            # HARNESS ENFORCEMENT: if stagnant and goal didn't change, force it
            if state.no_progress_count >= 5:
                old_goal = ""
                try:
                    old_beliefs = json.loads(state.memory)
                    old_goal = old_beliefs.get("goal", "")
                except (json.JSONDecodeError, TypeError):
                    pass
                new_goal = beliefs.get("goal", "")
                # Check if goal is essentially the same (fuzzy match)
                if old_goal and new_goal and old_goal[:40] == new_goal[:40]:
                    # Goal didn't change despite stagnation — force #2 hypothesis
                    goal_hyps = parsed.get("goal_hypotheses", [])
                    alt_goal = None
                    for gh in goal_hyps:
                        if isinstance(gh, dict) and gh.get("rank", 0) == 2:
                            alt_goal = gh.get("goal", "")
                            break
                    if not alt_goal and len(goal_hyps) >= 2:
                        alt_goal = goal_hyps[1].get("goal", "") if isinstance(goal_hyps[1], dict) else str(goal_hyps[1])
                    if alt_goal:
                        beliefs["goal"] = f"{alt_goal} (FORCED by harness — old goal stagnated)"
                        if "failed_approaches" not in beliefs:
                            beliefs["failed_approaches"] = []
                        beliefs["failed_approaches"].append(
                            f"Goal '{old_goal[:80]}' — stagnated for {state.no_progress_count} steps"
                        )
                        print(f"  [HARNESS] FORCED goal change: {beliefs['goal'][:100]}")
            state.memory = json.dumps(beliefs, indent=2)

    return reflection


def run_actor(
    client,
    grid: np.ndarray,
    state: AgentState,
    config: AgentConfig,
    analysis: str,
    avatar_tracker: AvatarTracker,
    bar_tracker: BarTracker,
    exploration: ExplorationController | None = None,
) -> tuple[GameAction, dict | None, str, dict]:
    """Run the ACTOR to choose the next action based on validated beliefs."""
    context = build_context_text(grid, state, config, avatar_tracker, bar_tracker, exploration)

    content = []
    text = context
    if analysis:
        text += f"\n\n=== PERCEPTION ===\n{analysis}\n==="
    if state.memory:
        text += f"\n\n=== VALIDATED BELIEFS ===\n{state.memory}\n==="
    content.append({"type": "text", "text": text})

    if config.use_vision:
        b64 = grid_to_base64(grid, scale=2)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
        })

    # Build actor messages with history
    messages = [{"role": "system", "content": ACTOR_PROMPT}]
    history_start = max(0, len(state.messages) - config.message_history_limit * 2)
    messages.extend(state.messages[history_start:])
    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_completion_tokens=500,
        timeout=120,
    )

    reply_text = response.choices[0].message.content or ""
    parsed = parse_response(reply_text)

    # Store actor exchange in history
    state.messages.append({"role": "user", "content": content})
    state.messages.append({"role": "assistant", "content": reply_text})

    # Map to GameAction
    action_name = parsed.get("action", "ACTION1").upper()
    game_action = ACTION_MAP.get(action_name, GameAction.ACTION1)
    state.actions_tested.add(action_name)

    # Prepare data for ACTION6
    data = None
    if game_action == GameAction.ACTION6:
        x = int(parsed.get("x", 32))
        y = int(parsed.get("y", 32))
        data = {"x": max(0, min(63, x)), "y": max(0, min(63, y))}

    reasoning = parsed.get("reasoning", reply_text[:200])
    return game_action, data, reasoning, parsed


def parse_response(text: str) -> dict:
    """Extract JSON action from LLM response text."""
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    text_upper = text.upper()
    for action_name in ["RESET", "ACTION7", "ACTION6", "ACTION5", "ACTION4", "ACTION3", "ACTION2", "ACTION1"]:
        if action_name in text_upper:
            return {"action": action_name, "reasoning": text}

    return {"action": "ACTION1", "reasoning": f"Failed to parse: {text[:200]}"}


def format_memory(mem: dict) -> str:
    """Format the agent memory dict for terminal output."""
    lines = []
    if "phase" in mem:
        lines.append(f"  Phase: {mem['phase']}")
    if "controls" in mem and isinstance(mem["controls"], dict):
        lines.append("  Controls:")
        for k, v in mem["controls"].items():
            lines.append(f"    {k}: {v}")
    if "rules" in mem and mem["rules"]:
        lines.append("  Rules:")
        rules = mem["rules"] if isinstance(mem["rules"], list) else [mem["rules"]]
        for r in rules:
            lines.append(f"    - {r}")
    if "goal" in mem:
        lines.append(f"  Goal: {mem['goal']}")
    if "plan" in mem:
        lines.append(f"  Plan: {mem['plan']}")
    if "level" in mem:
        lines.append(f"  Level: {mem['level']}")
    if "deaths" in mem:
        lines.append(f"  Deaths: {mem['deaths']}")
    if "lessons" in mem and mem["lessons"]:
        lines.append("  Lessons:")
        lessons = mem["lessons"] if isinstance(mem["lessons"], list) else [mem["lessons"]]
        for l in lessons:
            lines.append(f"    - {l}")
    return "\n".join(lines)


class LiveDisplay:
    """Persistent matplotlib window that updates each step."""

    def __init__(self):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self._plt = plt
        plt.ion()
        self._fig, self._ax = plt.subplots(1, 1, figsize=(6, 6))
        self._fig.canvas.manager.set_window_title("ARC-AGI-3")
        self._im = None
        self._ax.set_axis_off()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    def update(self, grid: np.ndarray, title: str = ""):
        img = grid_to_image(grid, scale=4)
        arr = np.array(img)
        if self._im is None:
            self._im = self._ax.imshow(arr)
        else:
            self._im.set_data(arr)
        self._ax.set_title(title, fontsize=10, loc="left")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.01)

    def close(self):
        self._plt.close(self._fig)


def _print_analysis(raw_analysis: str) -> None:
    """Pretty-print the Analyzer's response."""
    parsed = parse_response(raw_analysis)
    if parsed.get("error"):
        print(f"  [ANALYZER] (raw) {raw_analysis[:300]}...")
        return

    sep = "-" * 70

    # Scene description
    scene = parsed.get("scene_description", "")
    if scene:
        print(f"\n  SCENE")
        for line in scene[:300].split(". "):
            if line.strip():
                print(f"  | {line.strip()}.")

    # Classification
    classif = parsed.get("classification", {})
    if isinstance(classif, dict):
        print(f"\n  ELEMENT CLASSIFICATION")

        bg = classif.get("background_walls", [])
        if bg:
            print(f"  | WALLS/BACKGROUND:")
            for item in bg[:5]:
                if isinstance(item, dict):
                    print(f"  |   {item.get('color', '?')}: {item.get('description', '?')} ({item.get('positions', '?')})")

        paths = classif.get("walkable_paths", [])
        if paths:
            print(f"  | WALKABLE PATHS:")
            for item in paths[:5]:
                if isinstance(item, dict):
                    print(f"  |   {item.get('color', '?')}: {item.get('description', '?')} ({item.get('positions', '?')})")

        player = classif.get("player", {})
        if isinstance(player, dict) and player:
            print(f"  | PLAYER: {player.get('color', '?')} {player.get('shape', '?')} at {player.get('position', '?')}")
            att = player.get("attached_objects", "")
            if att:
                print(f"  |   attached: {att}")

        meta = classif.get("meta_info", [])
        if meta:
            print(f"  | META-INFO (HUD):")
            for item in meta[:5]:
                if isinstance(item, dict):
                    el = item.get("element", "?")
                    typ = item.get("type", "?")
                    val = item.get("current_value", "?")
                    rate = item.get("rate_of_change", "?")
                    rem = item.get("remaining", "?")
                    print(f"  |   {el}: {typ}, value={val}, rate={rate}, remaining={rem}")

        targets = classif.get("potential_targets", [])
        if targets:
            print(f"  | POTENTIAL TARGETS:")
            for item in targets[:5]:
                if isinstance(item, dict):
                    print(f"  |   {item.get('color', '?')} at {item.get('position', '?')}: {item.get('why_target', '?')[:60]}")
                    if item.get("tested"):
                        print(f"  |     result: {item.get('interaction_result', '?')[:60]}")

        interactive = classif.get("potential_interactive", [])
        if interactive:
            print(f"  | INTERACTIVE:")
            for item in interactive[:5]:
                if isinstance(item, dict):
                    print(f"  |   {item.get('color', '?')} at {item.get('position', '?')}: {item.get('why_interactive', '?')[:60]}")

        unknown = classif.get("unknown", [])
        if unknown:
            print(f"  | UNKNOWN:")
            for item in unknown[:5]:
                if isinstance(item, dict):
                    print(f"  |   {item.get('description', '?')[:60]} -> test: {item.get('experiment', '?')[:40]}")

    # Spatial map
    spatial = parsed.get("spatial_map", "")
    if spatial:
        print(f"\n  SPATIAL MAP")
        print(f"  | {spatial[:200]}")

    # Goal hypotheses with plans
    goals = parsed.get("goal_hypotheses", [])
    if goals:
        print(f"\n  GOAL HYPOTHESES")
        for gh in goals[:5]:
            if isinstance(gh, dict):
                rank = gh.get("rank", "?")
                conf = gh.get("confidence_pct", "?")
                goal = gh.get("goal", "?")
                print(f"  | #{rank} ({conf}%) {goal[:80]}")
                ef = gh.get("evidence_for", "")
                ea = gh.get("evidence_against", "")
                if ef:
                    print(f"  |     FOR: {ef[:80]}")
                if ea:
                    print(f"  |     AGAINST: {ea[:80]}")
                plan = gh.get("plan", [])
                pr = gh.get("plan_reasoning", "")
                if plan:
                    print(f"  |     PLAN: {plan[:8]}{'...' if len(plan) > 8 else ''}")
                if pr:
                    print(f"  |     WHY: {pr[:80]}")

    # Unknowns
    unknowns = parsed.get("unknowns", [])
    if unknowns:
        print(f"\n  UNKNOWNS")
        for u in unknowns[:5]:
            if isinstance(u, dict):
                print(f"  | ? {u.get('question', '?')[:60]}")
                print(f"  |   -> {u.get('experiment', '?')[:60]}")

    print(f"  {sep}")


def run_systematic_exploration(
    env,
    state: AgentState,
    config: AgentConfig,
    avatar_tracker: AvatarTracker,
    bar_tracker: BarTracker,
    exploration: ExplorationController,
    grid: np.ndarray,
    display=None,
    frames_path=None,
    saved_frames=None,
) -> tuple[np.ndarray, int]:
    """Phase 1: Harness-driven systematic exploration.

    Tests each available action from the starting position. After each test,
    RESET to start so every action is tested from the SAME state. Uses
    diff-based movement detection (not avatar_tracker which needs multiple obs).

    Returns: (current_grid, steps_used)
    """
    print("\n  === PHASE 1: SYSTEMATIC EXPLORATION (harness-driven, no LLM) ===")

    actions_to_test = []
    for aid in state.available_actions:
        if aid == 0:
            continue  # Skip RESET
        actions_to_test.append(f"ACTION{aid}")

    step_count = 0
    initial_grid = grid.copy()
    explore_results: dict[str, dict] = {}  # action → {movements, direction, had_effect, etc}

    # Test each action from the SAME starting position
    for action_name in actions_to_test:
        game_action = ACTION_MAP[action_name]
        label = ACTION_LABELS.get(action_name, action_name)

        print(f"\n  [EXPLORE] Testing {action_name} from start...")

        state.prev_grid = grid
        obs = env.step(game_action, data={})
        step_count += 1

        if obs is None:
            continue

        if obs.frame:
            grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
        if hasattr(obs, "available_actions") and obs.available_actions:
            state.available_actions = list(obs.available_actions)

        diff_result = compute_diff(state.prev_grid, grid)
        state.diff_text = diff_result["description"]
        avatar_tracker.update(action_name, diff_result)
        bar_tracker.update(grid, step_count)
        state.actions_tested.add(action_name)

        # Detect movement from diff (any object that moved)
        movements = diff_result.get("movements", [])
        swaps = diff_result.get("swaps", [])
        had_movement = len(movements) > 0
        n_changed = diff_result.get("changed_cells", 0)
        effect_desc = "NO effect"
        direction = None
        moved_from = None
        moved_to = None

        if swaps:
            # Swap detected — report the first swap
            s = swaps[0]
            effect_desc = (
                f"SWAP: {s['color_a_name']} at ({s['pos_a_x']},{s['pos_a_y']}) ↔ "
                f"{s['color_b_name']} at ({s['pos_b_x']},{s['pos_b_y']})"
            )
            # Compute direction from the first object's movement
            for m in movements:
                if m["color"] == s["color_a"]:
                    dx, dy = m["dx"], m["dy"]
                    if abs(dy) > abs(dx):
                        direction = "DOWN" if dy > 0 else "UP"
                    elif abs(dx) > abs(dy):
                        direction = "RIGHT" if dx > 0 else "LEFT"
                    else:
                        direction = f"dx={dx},dy={dy}"
                    moved_from = (m["from_x"], m["from_y"])
                    moved_to = (m["to_x"], m["to_y"])
                    effect_desc = f"SWAP {direction}: object moved ({m['from_x']},{m['from_y']})→({m['to_x']},{m['to_y']})"
                    break
            had_movement = True
        elif movements:
            # Direct movement
            m = movements[0]  # Most prominent movement
            dx, dy = m["dx"], m["dy"]
            if abs(dy) > abs(dx):
                direction = "DOWN" if dy > 0 else "UP"
            elif abs(dx) > abs(dy):
                direction = "RIGHT" if dx > 0 else "LEFT"
            else:
                direction = f"dx={dx},dy={dy}"
            moved_from = (m["from_x"], m["from_y"])
            moved_to = (m["to_x"], m["to_y"])
            effect_desc = f"{m['color_name']} moved {direction} ({m['from_x']},{m['from_y']})→({m['to_x']},{m['to_y']})"
        elif n_changed > 0 and n_changed <= 5:
            effect_desc = f"{n_changed} pixels changed (bar/minor)"

        explore_results[action_name] = {
            "direction": direction,
            "had_movement": had_movement,
            "effect_desc": effect_desc,
            "movements": movements,
            "swaps": swaps,
        }

        # Log to context
        if action_name not in state.action_context_log:
            state.action_context_log[action_name] = []
        state.action_context_log[action_name].append({
            "pos": moved_from,
            "had_effect": had_movement,
            "summary": effect_desc,
        })

        # Record step
        record = StepRecord(
            action_num=step_count,
            action=action_name,
            state=obs.state.name if obs.state else "UNKNOWN",
            levels_completed=obs.levels_completed or 0,
            diff_summary=effect_desc,
            avatar_pos=moved_from,
            had_effect=had_movement,
        )
        state.steps.append(record)

        h = grid_hash(grid)
        state.state_hashes.add(h)

        print(f"    → {action_name}: {effect_desc}")

        if config.save_frames and frames_path:
            img = grid_to_image(grid, scale=4)
            img.save(frames_path / f"step_{step_count:03d}_{label.lower()}.png")
            if saved_frames is not None:
                saved_frames.append(img.copy())

        if display:
            display.update(grid, f"Explore: {label}")

        # Check for game over
        if obs.state == GameState.GAME_OVER:
            print(f"    GAME OVER during exploration!")
            obs = env.reset()
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
                state.prev_grid = None
            continue

        # Don't reset between actions — some games need sequential
        # state evolution for actions to have effect.

    # Round 2: If we found actions that move, try untested directions
    # from the current position (which has changed from round 1)
    moved_actions = [a for a, r in explore_results.items() if r.get("had_movement")]
    if moved_actions:
        print(f"\n  [EXPLORE] Round 2: retesting from new position...")
        for action_name in actions_to_test:
            if explore_results.get(action_name, {}).get("had_movement"):
                continue  # Already know this one works
            game_action = ACTION_MAP[action_name]
            label = ACTION_LABELS.get(action_name, action_name)

            state.prev_grid = grid
            obs = env.step(game_action, data={})
            step_count += 1

            if obs is None:
                continue
            if obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
            if hasattr(obs, "available_actions") and obs.available_actions:
                state.available_actions = list(obs.available_actions)

            diff_result = compute_diff(state.prev_grid, grid)
            avatar_tracker.update(action_name, diff_result)
            bar_tracker.update(grid, step_count)

            movements = diff_result.get("movements", [])
            swaps = diff_result.get("swaps", [])
            had_movement = len(movements) > 0
            effect_desc = "NO effect"
            direction = None

            if swaps:
                s = swaps[0]
                for m in movements:
                    if m["color"] == s["color_a"]:
                        dx, dy = m["dx"], m["dy"]
                        if abs(dy) > abs(dx):
                            direction = "DOWN" if dy > 0 else "UP"
                        elif abs(dx) > abs(dy):
                            direction = "RIGHT" if dx > 0 else "LEFT"
                        else:
                            direction = f"dx={dx},dy={dy}"
                        effect_desc = f"SWAP {direction} (from new position)"
                        break
                had_movement = True
            elif movements:
                m = movements[0]
                dx, dy = m["dx"], m["dy"]
                if abs(dy) > abs(dx):
                    direction = "DOWN" if dy > 0 else "UP"
                elif abs(dx) > abs(dy):
                    direction = "RIGHT" if dx > 0 else "LEFT"
                else:
                    direction = f"dx={dx},dy={dy}"
                effect_desc = f"{m['color_name']} moved {direction} (from new position)"

            if had_movement and direction:
                explore_results[action_name] = {
                    "direction": direction,
                    "had_movement": True,
                    "effect_desc": effect_desc,
                    "movements": movements,
                    "swaps": swaps,
                }

            if action_name not in state.action_context_log:
                state.action_context_log[action_name] = []
            state.action_context_log[action_name].append({
                "pos": None, "had_effect": had_movement, "summary": effect_desc,
            })

            record = StepRecord(
                action_num=step_count, action=action_name,
                state=obs.state.name if obs.state else "UNKNOWN",
                levels_completed=obs.levels_completed or 0,
                diff_summary=effect_desc, had_effect=had_movement,
            )
            state.steps.append(record)
            state.state_hashes.add(grid_hash(grid))
            print(f"    → {action_name}: {effect_desc}")

            if obs.state == GameState.GAME_OVER:
                obs = env.reset()
                if obs and obs.frame:
                    grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
                break

    # Build exploration summary using our own results (not avatar_tracker)
    print(f"\n  === EXPLORATION COMPLETE ({step_count} steps used) ===")
    print("  Action → Effect (from exploration):")
    for action_name in actions_to_test:
        r = explore_results.get(action_name, {})
        effect = r.get("effect_desc", "not tested")
        direction = r.get("direction")
        if direction:
            print(f"    {action_name}: {direction} — {effect}")
        else:
            print(f"    {action_name}: {effect}")
    print("    RESET: Restarts the level (harness fact)")
    print()

    return grid, step_count


def run_agent(env, config: AgentConfig | None = None) -> AgentState:
    """Run the LLM agent on an ARC-AGI-3 environment."""
    if config is None:
        config = AgentConfig()

    client = create_client(config)
    state = AgentState()
    avatar_tracker = AvatarTracker()
    bar_tracker = BarTracker()
    exploration = ExplorationController()
    saved_frames: list[Image.Image] = []
    display = None

    if config.show_window:
        try:
            display = LiveDisplay()
        except Exception as e:
            print(f"  WARNING: Could not open display window: {e}")

    if config.save_frames:
        from pathlib import Path
        frames_path = Path(config.frames_dir)
        frames_path.mkdir(parents=True, exist_ok=True)
        print(f"  Saving frames to: {frames_path.resolve()}")

    # Initial observation
    obs = env.reset()
    if obs is None or not obs.frame:
        print("ERROR: No initial observation from environment")
        return state

    grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

    if hasattr(obs, "available_actions") and obs.available_actions:
        state.available_actions = list(obs.available_actions)

    # Initial analysis
    state.frame_analysis = describe_frame(grid)
    state.diff_text = "First frame — no previous frame to compare."
    state.state_hashes.add(grid_hash(grid))
    bar_tracker.update(grid, 0)

    # Inject prior beliefs from a previous run
    if config.prior_beliefs:
        state.memory = config.prior_beliefs
        print(f"\n  [MULTI-RUN] Injected prior beliefs from previous run:")
        try:
            beliefs = json.loads(config.prior_beliefs)
            if beliefs.get("goal"):
                print(f"    Goal: {beliefs['goal']}")
            if beliefs.get("controls"):
                for k, v in beliefs["controls"].items():
                    print(f"    {k}: {v}")
        except (json.JSONDecodeError, TypeError):
            print(f"    {config.prior_beliefs[:200]}")

    if display:
        display.update(grid, "Initial observation")

    if config.save_frames:
        img = grid_to_image(grid, scale=4)
        img.save(frames_path / "step_000_initial.png")
        saved_frames.append(img.copy())

    if config.step_mode:
        feedback = prompt_human("Comment/question before first action")
        if feedback is None:
            print("  Stopped by user.")
            return state
        if feedback:
            state.human_feedback = feedback

    # Phase 1: Systematic exploration (harness-driven, no LLM calls)
    explore_steps = 0
    if config.systematic_explore and state.available_actions:
        grid, explore_steps = run_systematic_exploration(
            env, state, config, avatar_tracker, bar_tracker, exploration,
            grid, display,
            frames_path if config.save_frames else None,
            saved_frames if config.save_frames else None,
        )
        state.frame_analysis = describe_frame(grid)

    # Phase 2: Initial LLM analysis with ALL exploration data
    print("\n  ANALYZER: Running initial perception (with exploration data)...")
    analysis = run_analyzer(client, grid, state, config, avatar_tracker, bar_tracker, exploration)
    _print_analysis(analysis)

    last_expected = "Unknown — first action"
    remaining_steps = config.max_actions - explore_steps

    # Phase 3: LLM-driven execution
    for step_num in range(remaining_steps):
        actual_step = explore_steps + step_num + 1  # Total step number

        # Run analyzer periodically or on important events
        should_analyze = (
            step_num > 0 and (
                step_num % config.analyze_every == 0
                or state.no_progress_count >= 3
                or (state.steps and state.steps[-1].state == "GAME_OVER")
            )
        )
        if should_analyze:
            print(f"\n  ANALYZER: Re-analyzing (step {actual_step})...")
            analysis = run_analyzer(client, grid, state, config, avatar_tracker, bar_tracker, exploration)
            _print_analysis(analysis)

        # === 1. RUN ACTOR — decide action (don't print yet) ===
        game_action, data, reasoning, parsed = run_actor(
            client, grid, state, config, analysis, avatar_tracker, bar_tracker, exploration
        )

        action_name = game_action.name
        label = ACTION_LABELS.get(action_name, action_name)
        coords = f" x={data['x']},y={data['y']}" if data else ""
        last_expected = parsed.get("expected_result", "no prediction")

        # === 2. EXECUTE ACTION ===
        obs = env.step(game_action, data=data or {})

        if obs is None:
            print("  WARNING: No observation returned")
            continue

        # Update grid
        state.prev_grid = grid
        if obs.frame:
            grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
        if hasattr(obs, "available_actions") and obs.available_actions:
            state.available_actions = list(obs.available_actions)

        # HARNESS INTELLIGENCE
        diff_result = compute_diff(state.prev_grid, grid)
        state.diff_text = diff_result["description"]
        state.frame_analysis = describe_frame(grid)

        # Update trackers
        avatar_tracker.update(action_name, diff_result)
        bar_tracker.update(grid, actual_step)

        # Annotate diff with avatar movement info
        if avatar_tracker.avatar_candidate:
            ac = avatar_tracker.avatar_candidate["color"]
            avatar_moved = any(
                m["color"] == ac for m in diff_result.get("movements", [])
            )
            if not avatar_moved and diff_result.get("changed_cells", 0) > 0:
                state.diff_text += " (NOTE: avatar did NOT move — only non-avatar pixels changed)"

        # Update exploration controller
        movements = diff_result.get("movements", [])
        avatar_pos = None
        prev_avatar_pos = None
        if avatar_tracker.avatar_candidate:
            ac = avatar_tracker.avatar_candidate["color"]
            for m in movements:
                if m["color"] == ac:
                    avatar_pos = (m["to_x"], m["to_y"])
                    prev_avatar_pos = (m["from_x"], m["from_y"])
                    break
        health_len = None
        if bar_tracker.detected_bars:
            health_len = bar_tracker.detected_bars[0].get("current_length")
        exploration.update(action_name, avatar_pos, prev_avatar_pos, movements, health_len)

        # State novelty
        h = grid_hash(grid)
        is_novel = h not in state.state_hashes
        state.state_hashes.add(h)

        if is_novel or diff_result.get("changed_cells", 0) > 0:
            state.no_progress_count = 0
        else:
            state.no_progress_count += 1

        # ============================================================
        # DISPLAY — chronological flow: observe → reflect → update → decide
        # ============================================================
        sep = "=" * 70

        print(f"\n{sep}")
        print(f"  STEP {actual_step}/{config.max_actions}")
        print(sep)

        # --- OBSERVE: what happened ---
        print(f"\n  1. OBSERVE (action taken: {label}{coords})")
        print(f"  | What happened: {state.diff_text}")
        avatar_info = avatar_tracker.get_avatar_info()
        if avatar_info:
            print(f"  | Avatar: {avatar_info}")
        bar_warn = bar_tracker.get_bar_warnings()
        if bar_warn:
            for line in bar_warn.split("\n"):
                print(f"  | Bar: {line}")
        print(f"  | State: {'NEW' if is_novel else 'REPEATED'}  |  Stale: {state.no_progress_count} steps")

        # --- REFLECT: what the agent thinks about it ---
        print(f"\n  2. REFLECT")
        reflection = run_reflector(
            client, grid, state, config, analysis, action_name, last_expected,
            avatar_tracker, bar_tracker, exploration,
        )
        reflection_parsed = parse_response(reflection)

        # Prediction check
        pvr = reflection_parsed.get("prediction_vs_reality", "")
        if pvr:
            print(f"  | Predicted: {last_expected[:80]}")
            print(f"  | Reality:   {pvr[:120]}")

        # Causal analysis — the WHY
        causal = reflection_parsed.get("causal_analysis", "")
        if causal:
            print(f"  | WHY: {causal[:200]}")
        causal_hyps = reflection_parsed.get("causal_hypotheses", [])
        for ch in causal_hyps[:3]:
            if isinstance(ch, dict):
                obs_text = ch.get("observation", "?")[:40]
                cause_text = ch.get("cause", "?")[:50]
                conf = ch.get("confidence_pct", "?")
                print(f"  |   {obs_text} BECAUSE {cause_text} ({conf}%)")

        # What's new / what changed
        new_disc = reflection_parsed.get("new_discoveries", [])
        if new_disc:
            print(f"  | Discoveries:")
            for d in new_disc[:3]:
                print(f"  |   + {d[:100]}")

        # --- UPDATE: belief changes ---
        reviews = reflection_parsed.get("belief_reviews", [])
        kept = [r for r in reviews if r.get("verdict") == "KEEP"]
        changed = [r for r in reviews if r.get("verdict") == "CHANGE"]
        dropped = [r for r in reviews if r.get("verdict") == "DROP"]

        if changed or dropped or new_disc:
            print(f"\n  3. UPDATE ({len(kept)} kept, {len(changed)} changed, {len(dropped)} dropped)")
            for r in changed:
                print(f"  |   ~ {r.get('belief', '?')[:40]} -> {r.get('corrected', '?')[:50]}")
            for r in dropped:
                print(f"  |   x {r.get('belief', '?')[:50]} -- {r.get('justification', '?')[:40]}")
        else:
            print(f"\n  3. UPDATE (no changes — {len(kept)} beliefs confirmed)")

        # Goal hypotheses
        goal_hyps = reflection_parsed.get("goal_hypotheses", [])
        if goal_hyps:
            print(f"\n  4. GOALS")
            for gh in goal_hyps[:5]:
                if isinstance(gh, dict):
                    rank = gh.get("rank", "?")
                    conf = gh.get("confidence_pct", "?")
                    goal = gh.get("goal", "?")
                    print(f"  | #{rank} ({conf}%) {goal[:80]}")
                    ev_for = gh.get("evidence_for", "")
                    ev_against = gh.get("evidence_against", "")
                    confirm = gh.get("confirm_test", "")
                    refute = gh.get("refute_test", "")
                    if ev_for:
                        print(f"  |     + {ev_for[:80]}")
                    if ev_against:
                        print(f"  |     - {ev_against[:80]}")
                    if confirm:
                        print(f"  |     confirm: {confirm[:70]}")
                    if refute:
                        print(f"  |     refute:  {refute[:70]}")

        # Unknowns
        unc_red = reflection_parsed.get("uncertainty_reduction", {})
        if isinstance(unc_red, dict):
            unknowns = unc_red.get("top_unknowns", [])
            if unknowns:
                print(f"\n  5. UNKNOWNS")
                for u in unknowns[:3]:
                    if isinstance(u, dict):
                        print(f"  | ? {u.get('question', '?')[:60]}")
                        print(f"  |   experiment: {u.get('experiment', '?')[:60]}")
            mvi = unc_red.get("most_valuable_info", "")
            if mvi:
                print(f"  | >> Most valuable: {mvi[:100]}")

        # Strategy
        strategy = reflection_parsed.get("strategy_check", {})
        if isinstance(strategy, dict):
            prog = strategy.get("making_progress", "?")
            stale = strategy.get("steps_without_progress", "?")
            untried = strategy.get("untried_sequences", [])
            if not prog or stale and int(stale) >= 2 if isinstance(stale, int) else False:
                print(f"\n  STRATEGY WARNING")
                print(f"  | Progress: {prog}  |  Stale: {stale} steps")
                if untried:
                    print(f"  | Untried: {untried[:3]}")

        # Current beliefs summary
        if state.memory:
            try:
                beliefs = json.loads(state.memory)
                print(f"\n  CURRENT BELIEFS")
                if beliefs.get("goal"):
                    print(f"  | Goal: {beliefs['goal'][:100]}")
                if beliefs.get("causal_model"):
                    cm = beliefs["causal_model"]
                    if isinstance(cm, list):
                        for c in cm[:3]:
                            print(f"  | Cause: {c[:100]}")
                if beliefs.get("failed_approaches"):
                    fa = beliefs["failed_approaches"]
                    if isinstance(fa, list) and fa:
                        print(f"  | Failed: {fa[-1][:100]}")
            except (json.JSONDecodeError, AttributeError):
                pass

        # --- DECIDE: what the agent chose to do (shown LAST) ---
        print(f"\n  6. DECIDE")
        print(f"  | Reasoning: {reasoning[:150]}")
        print(f"  | Action:    {label}{coords}")
        print(f"  | Expected:  {last_expected[:150]}")

        # Record step with position context
        # "had_effect" = avatar moved (not just bar pixels changing)
        had_effect = avatar_pos is not None and prev_avatar_pos is not None
        record = StepRecord(
            action_num=actual_step,
            action=action_name,
            x=data.get("x") if data else None,
            y=data.get("y") if data else None,
            reasoning=reasoning,
            state=obs.state.name if obs.state else "UNKNOWN",
            levels_completed=obs.levels_completed or 0,
            diff_summary=state.diff_text,
            avatar_pos=prev_avatar_pos,
            had_effect=had_effect,
        )
        state.steps.append(record)

        # Update contextual action log (position → effect mapping)
        if action_name not in state.action_context_log:
            state.action_context_log[action_name] = []
        state.action_context_log[action_name].append({
            "pos": prev_avatar_pos,
            "had_effect": had_effect,
            "summary": state.diff_text[:60],
        })

        # Update display
        if display:
            status = obs.state.name if obs.state else "?"
            display.update(grid, f"Step {actual_step}: {label}{coords}  [{status}]")

        if config.save_frames:
            img = grid_to_image(grid, scale=4)
            img.save(frames_path / f"step_{actual_step:03d}_{label.lower()}.png")
            saved_frames.append(img.copy())

        if config.step_mode:
            feedback = prompt_human("Comment/question")
            if feedback is None:
                print("  Stopped by user.")
                break
            if feedback:
                state.human_feedback = feedback

        if config.delay > 0:
            time.sleep(config.delay)

        # Check terminal states
        if obs.state == GameState.WIN:
            print(f"\n  WIN after {actual_step} steps! "
                  f"Levels completed: {obs.levels_completed}")
            break
        elif obs.state == GameState.GAME_OVER:
            print(f"\n  *** GAME OVER at step {actual_step} ***")

            # Force deep reflection on death
            death_reflection = run_reflector(
                client, grid, state, config, analysis,
                action_name, last_expected,
                avatar_tracker, bar_tracker, exploration,
            )
            print(f"  [REFLECTOR] Post-death reflection: {death_reflection[:200]}")

            obs = env.reset()
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
                state.prev_grid = None
                state.diff_text = "Level reset after GAME_OVER."
                state.frame_analysis = describe_frame(grid)
                state.no_progress_count = 0
                bar_tracker.update(grid, actual_step)
                if hasattr(obs, "available_actions") and obs.available_actions:
                    state.available_actions = list(obs.available_actions)
                if display:
                    display.update(grid, "GAME OVER — RESET")

                # Re-analyze after death
                print("  ANALYZER: Re-analyzing after death...")
                analysis = run_analyzer(client, grid, state, config, avatar_tracker, bar_tracker, exploration)
                _print_analysis(analysis)

    # Save GIF
    if config.save_frames and len(saved_frames) > 1:
        gif_path = frames_path / "replay.gif"
        saved_frames[0].save(
            gif_path,
            save_all=True,
            append_images=saved_frames[1:],
            duration=500,
            loop=0,
        )
        print(f"  Replay GIF saved: {gif_path.resolve()}")

    if display:
        display.close()

    return state
