"""LLM Agent for ARC-AGI-3 using Azure AI Foundry.

Architecture: Analyzer-Actor split.
- Analyzer: focused perception (what do I control, what are resources, what's the goal?)
- Actor: picks the best action based on the analysis
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
Your job is ONLY to perceive and interpret. You do NOT choose actions.

The harness provides you with:
- The current frame (image)
- Programmatic diff (exact pixel changes, object movements)
- Tracker data (avatar detection, resource bar warnings)
- Your previous analysis (if any)

Answer these questions precisely:

1. CONTROLLED OBJECT: What object do you control? What color/shape is it? Where is it now?
2. RESOURCE INDICATORS: Are there bars, counters, or indicators? Which are health/cost vs progress?
3. SALIENT TARGETS: What objects look like goals, pickups, or interactive elements? Where?
4. SPATIAL LAYOUT: Describe the layout — walls, paths, rooms, doors, connections.
5. CURRENT HYPOTHESIS: What do you think the goal is? What evidence supports this?
6. CONTRADICTIONS: Does anything contradict your previous analysis? What needs revision?

Respond with a JSON object:
{
  "controlled_object": "description of what you control and where it is",
  "resource_bars": ["bar description and whether it's health/cost or progress"],
  "targets": ["potential goal/interactive objects with locations"],
  "layout": "spatial description of the level",
  "goal_hypothesis": "what you think wins the level, with evidence",
  "contradictions": "what changed or was wrong in previous analysis",
  "confidence": "low|medium|high"
}
"""

# --- ACTOR PROMPT ---
ACTOR_PROMPT = """\
You are the ACTION module of an ARC-AGI-3 agent. You receive a perception analysis \
and choose the BEST next action.

RULES:
- EFFICIENCY MATTERS: score = (human_actions/your_actions)². Fewer = better.
- If there are UNTESTED actions, test them first — one at a time.
- Do NOT repeat the same action more than 2 times in a row unless you have strong reason.
- Consuming a resource bar is COST, not progress. Do NOT optimize for depleting bars.
- If stuck 3+ turns, CHANGE APPROACH completely.
- Navigate toward identified targets when you have a goal hypothesis.

PHASE GUIDE:
- DISCOVER: Test each action once. Record effects.
- PROBE: Interact with objects. Test hypotheses about goals.
- PURSUE: Execute a plan toward the goal.

Respond with a JSON object:
{
  "reasoning": "Why this action, based on the analysis",
  "action": "ACTION1|...|ACTION7|RESET",
  "x": 0, "y": 0,
  "expected_result": "What I expect to see after this action",
  "phase": "discover|probe|pursue",
  "memory": {
    "controls": {"ACTION1": "effect (VERIFIED/unverified/untested)", ...},
    "rules": ["confirmed rule 1", ...],
    "goal": "current hypothesis",
    "plan": "current strategy",
    "level": 1,
    "deaths": 0,
    "lessons": ["key learnings"]
  }
}

"x" and "y" are ONLY for ACTION6.
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
            parts.append(f"⚠ UNTESTED actions: {', '.join(untested)}")

    # Programmatic diff
    if state.diff_text:
        parts.append(f"\n=== DIFF (computed, accurate) ===\n{state.diff_text}\n===")

    # Avatar tracker
    avatar_info = avatar_tracker.get_avatar_info()
    if avatar_info:
        parts.append(f"\n=== AVATAR TRACKER ===\n{avatar_info}")
        action_map = avatar_tracker.get_action_map()
        if action_map:
            parts.append("Action → Effect mapping:")
            for a, desc in action_map.items():
                parts.append(f"  {a}: {desc}")
        parts.append("===")

    # Bar tracker warnings
    bar_warnings = bar_tracker.get_bar_warnings()
    if bar_warnings:
        parts.append(f"\n=== BAR TRACKER ===\n{bar_warnings}\n===")

    # Frame analysis
    if state.frame_analysis:
        parts.append(f"\n=== FRAME OBJECTS ===\n{state.frame_analysis}\n===")

    # State novelty
    current_hash = grid_hash(grid)
    is_novel = current_hash not in state.state_hashes
    parts.append(f"State: {'NEW' if is_novel else 'REPEATED'}")
    if state.no_progress_count >= 3:
        parts.append(f"⚠ NO PROGRESS for {state.no_progress_count} turns! Change approach!")

    # Recent actions
    if state.steps:
        recent = state.steps[-5:]
        history = " | ".join(
            f"{s.action}({'ok' if s.state == 'IN_PROGRESS' else s.state})"
            for s in recent
        )
        parts.append(f"Recent: {history}")

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

    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=0.3,  # Lower temp for perception accuracy
        max_completion_tokens=1000,
    )

    analysis = response.choices[0].message.content or ""
    state.last_analysis = analysis
    state.analysis_history.append(analysis)
    return analysis


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
    """Run the ACTOR to choose the next action based on analysis."""
    context = build_context_text(grid, state, config, avatar_tracker, bar_tracker, exploration)

    content = []
    text = context
    text += f"\n\n=== ANALYZER OUTPUT ===\n{analysis}\n==="
    if state.memory:
        text += f"\n\n=== YOUR MEMORY ===\n{state.memory}\n==="
    content.append({"type": "text", "text": text})

    # Add image for spatial reference
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
        max_completion_tokens=1000,
    )

    reply_text = response.choices[0].message.content or ""
    parsed = parse_response(reply_text)

    # Store actor exchange in history
    state.messages.append({"role": "user", "content": content})
    state.messages.append({"role": "assistant", "content": reply_text})

    # Update memory
    if "memory" in parsed:
        mem = parsed["memory"]
        if isinstance(mem, dict):
            state.memory = json.dumps(mem, indent=2)
        else:
            state.memory = str(mem)

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

    # Run initial analyzer
    print("\n  [ANALYZER] Running initial perception...")
    analysis = run_analyzer(client, grid, state, config, avatar_tracker, bar_tracker, exploration)
    print(f"  [ANALYZER] {analysis[:300]}...")

    for step_num in range(config.max_actions):
        # Run analyzer periodically or on important events
        should_analyze = (
            step_num > 0 and (
                step_num % config.analyze_every == 0
                or state.no_progress_count >= 3
                or (state.steps and state.steps[-1].state == "GAME_OVER")
            )
        )
        if should_analyze:
            print(f"\n  [ANALYZER] Re-analyzing (step {step_num + 1})...")
            analysis = run_analyzer(client, grid, state, config, avatar_tracker, bar_tracker, exploration)
            print(f"  [ANALYZER] {analysis[:200]}...")

        # Run actor
        game_action, data, reasoning, parsed = run_actor(
            client, grid, state, config, analysis, avatar_tracker, bar_tracker, exploration
        )

        action_name = game_action.name
        label = ACTION_LABELS.get(action_name, action_name)
        coords = f" x={data['x']},y={data['y']}" if data else ""

        # Display step
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  Step {step_num + 1}/{config.max_actions}")
        print(sep)
        print(f"  Reasoning: {reasoning}")
        if parsed.get("expected_result"):
            print(f"  Expected: {parsed['expected_result']}")
        print(f"  Action: >>> {label}{coords}")
        if "memory" in parsed and isinstance(parsed["memory"], dict):
            print(f"\n  Memory:\n{format_memory(parsed['memory'])}")

        # Execute action
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
        bar_tracker.update(grid, step_num + 1)

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
        # Get health bar length for controller
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

        # Print harness info
        print(f"\n  Diff: {state.diff_text}")
        avatar_info = avatar_tracker.get_avatar_info()
        if avatar_info:
            print(f"  Avatar: {avatar_info}")
        bar_warn = bar_tracker.get_bar_warnings()
        if bar_warn:
            print(f"  Bar: {bar_warn}")
        explore_report = exploration.get_exploration_report()
        if explore_report:
            print(f"  Explore: {explore_report}")
        print(f"  State: {'NEW' if is_novel else 'REPEATED'} | No-progress: {state.no_progress_count}")

        # Record step
        record = StepRecord(
            action_num=step_num + 1,
            action=action_name,
            x=data.get("x") if data else None,
            y=data.get("y") if data else None,
            reasoning=reasoning,
            state=obs.state.name if obs.state else "UNKNOWN",
            levels_completed=obs.levels_completed or 0,
            diff_summary=state.diff_text,
        )
        state.steps.append(record)

        # Update display
        if display:
            status = obs.state.name if obs.state else "?"
            display.update(grid, f"Step {step_num + 1}: {label}{coords}  [{status}]")

        if config.save_frames:
            img = grid_to_image(grid, scale=4)
            img.save(frames_path / f"step_{step_num + 1:03d}_{label.lower()}.png")
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
            print(f"\n  🏆 WIN after {step_num + 1} steps! "
                  f"Levels completed: {obs.levels_completed}")
            break
        elif obs.state == GameState.GAME_OVER:
            print(f"\n  *** GAME OVER at step {step_num + 1} ***")

            death_msg = {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        "*** GAME OVER ***\n\n"
                        "REFLECT: What assumption was WRONG? What will you do DIFFERENTLY?\n"
                        "Update memory: increment deaths, add lesson, CORRECT wrong assumptions."
                    ),
                }],
            }
            state.messages.append(death_msg)

            obs = env.reset()
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
                state.prev_grid = None
                state.diff_text = "Level reset after GAME_OVER."
                state.frame_analysis = describe_frame(grid)
                state.no_progress_count = 0
                bar_tracker.update(grid, step_num + 1)
                if hasattr(obs, "available_actions") and obs.available_actions:
                    state.available_actions = list(obs.available_actions)
                if display:
                    display.update(grid, "GAME OVER — RESET")

                # Force re-analyze after death
                print("  [ANALYZER] Re-analyzing after death...")
                analysis = run_analyzer(client, grid, state, config, avatar_tracker, bar_tracker, exploration)

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
