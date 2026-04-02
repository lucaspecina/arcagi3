"""LLM Agent for ARC-AGI-3 using Azure AI Foundry."""

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
from arcengine import GameAction, GameState

from PIL import Image

from .grid_utils import (
    grid_to_base64,
    grid_to_image,
    grid_to_text_compact,
    image_diff,
    image_to_base64,
)

# Map action names to GameAction enum — actions are GENERIC, semantics are game-specific
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

# Human-readable labels for display only
ACTION_LABELS = {
    "RESET": "RESET",
    "ACTION1": "A1", "ACTION2": "A2", "ACTION3": "A3", "ACTION4": "A4",
    "ACTION5": "A5", "ACTION6": "A6(click)", "ACTION7": "A7",
}

SYSTEM_PROMPT = """\
You are playing an ARC-AGI-3 game — an abstract visual puzzle on a 64x64 grid \
with 16 colors. Each game has multiple levels.

THE CHALLENGE: You have NO instructions. You must discover through experimentation:
  1. CONTROLS — What does each action do? Actions are GENERIC (ACTION1-7). Their \
meaning changes per game. You must test each one to learn what it does.
  2. RULES — What are the mechanics? What objects exist? How do they interact?
  3. GOAL — What triggers level completion? You must infer it.

## What you see
A 64x64 colored grid (16 colors, values 0-15). Objects are patterns of colored \
pixels. You may control something on the grid — you must figure out what.

## Actions
Actions are ABSTRACT and GAME-SPECIFIC. Do NOT assume what they do.
- ACTION1 through ACTION5 — simple actions (no parameters). Could be movement, \
interaction, selection, rotation — anything. You must test to find out.
- ACTION6 x,y — positional action (click/place at grid coordinate 0-63).
- ACTION7 — typically undo, but verify.
- RESET — restart current level.

Each turn you will be told which actions are currently available. Only choose \
from those.

## How to think

FIRST TURN: Study the grid carefully. Identify distinct objects, colors, regions. \
Form a hypothesis. Then test ONE action to learn what it does.

EVERY TURN:
- OBSERVE: What changed? What stayed the same?
- LEARN: What did my action do? Update control knowledge.
- REASON: What rules/mechanics does this reveal?
- PLAN: What should I try next and why?

## Strategy
- Level 1 is a tutorial — it teaches basic mechanics. Pay close attention.
- Be SYSTEMATIC: test each available action once early on to map controls.
- Track what changed after each action — that's your primary learning signal.
- If stuck for several turns, RESET and try a different approach.
- Later levels add new mechanics on top of what you learned.
- EFFICIENCY MATTERS: your score penalizes wasted actions quadratically.

## Human observer
A human observer may provide comments or hints between turns.

## Persistent memory
You have a memory dict that persists across ALL turns (even when old messages \
are dropped). You MUST update it every turn. If you don't write it down, you \
WILL forget it.

## Response format
Respond with a JSON object:
{
  "reasoning": "OBSERVE what changed, LEARN from it, PLAN next steps",
  "action": "ACTION1|ACTION2|ACTION3|ACTION4|ACTION5|ACTION6|ACTION7|RESET",
  "x": 0,
  "y": 0,
  "memory": {
    "controls": {"ACTION1": "what it does", "ACTION2": "untested", ...},
    "rules": ["confirmed rule 1", "confirmed rule 2"],
    "goal": "current hypothesis about win condition",
    "map": "layout description: objects, colors, positions",
    "plan": "current strategy and next steps",
    "level": 1
  }
}

Memory rules:
- ALWAYS include and update memory every turn.
- "controls": map each action to observed effect. Mark untested as "untested".
- "rules": only confirmed mechanics (things you verified).
- "goal": best hypothesis, updated with evidence.
- "map": spatial layout, key objects and their positions.
- "plan": step-by-step strategy, updated when plan changes.
- "level": current level number (increment when you complete one).

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


@dataclass
class AgentState:
    """Mutable state of the agent during a game session."""

    steps: list[StepRecord] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    prev_grid: np.ndarray | None = None
    memory: str = ""
    human_feedback: str = ""
    available_actions: list[int] = field(default_factory=list)


def create_client(config: AgentConfig):
    """Create OpenAI client configured for Azure Foundry."""
    from openai import OpenAI

    if not config.base_url:
        raise ValueError(
            "Set AZURE_FOUNDRY_ENDPOINT env var or pass base_url. "
            "Example: https://your-resource.openai.azure.com/openai/v1/"
        )
    if not config.api_key:
        raise ValueError(
            "Set AZURE_INFERENCE_CREDENTIAL env var or pass api_key."
        )

    return OpenAI(base_url=config.base_url, api_key=config.api_key)


def print_raw_prompt(messages: list[dict]) -> None:
    """Print the raw prompt exactly as it would be sent to the model."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("RAW PROMPT TO MODEL")
    print(sep)
    for msg in messages:
        role = msg["role"].upper()
        print(f"\n{'─' * 35} {role} {'─' * 35}")
        content = msg["content"]
        if isinstance(content, str):
            print(content)
        elif isinstance(content, list):
            for part in content:
                if part["type"] == "text":
                    print(part["text"])
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:image"):
                        b64_data = url.split(",", 1)[1] if "," in url else url
                        size_kb = len(b64_data) * 3 // 4 // 1024
                        print(f"  [IMAGE: base64 PNG, ~{size_kb}KB]")
                    else:
                        print(f"  [IMAGE: {url}]")
    print(f"\n{sep}\n")


def prompt_human(prompt_text: str) -> str | None:
    """Prompt the human observer for input. Returns None on quit."""
    try:
        user_input = input(f"\n  > {prompt_text} (Enter to continue, q to quit): ")
        if user_input.strip().lower() == "q":
            return None
        return user_input.strip()
    except (KeyboardInterrupt, EOFError):
        return None


def build_user_message(
    grid: np.ndarray,
    prev_grid: np.ndarray | None,
    state: AgentState,
    config: AgentConfig,
) -> dict:
    """Build the user message with current observation."""
    content = []

    # Text description of state
    text_parts = []
    text_parts.append(f"Step {len(state.steps) + 1}/{config.max_actions}")

    if state.steps:
        last = state.steps[-1]
        text_parts.append(f"Last action: {last.action} -> {last.state}")
        text_parts.append(f"Levels completed: {last.levels_completed}")

    # Available actions from the environment
    if state.available_actions:
        action_names = []
        for aid in state.available_actions:
            name = f"ACTION{aid}"
            if aid == 0:
                name = "RESET"
            action_names.append(name)
        # Always include RESET as an option
        if "RESET" not in action_names:
            action_names.append("RESET")
        text_parts.append(f"Available actions: {', '.join(action_names)}")

    # Recent action history (last 5)
    if state.steps:
        recent = state.steps[-5:]
        history = " | ".join(
            f"{s.action}({'ok' if s.state == 'IN_PROGRESS' else s.state})"
            for s in recent
        )
        text_parts.append(f"Recent: {history}")

    # Agent memory/scratchpad — persists even when old messages are dropped
    if state.memory:
        text_parts.append(f"\n=== YOUR SCRATCHPAD (persists across turns) ===\n{state.memory}\n===")

    # Human observer feedback
    if state.human_feedback:
        text_parts.append(f"\n=== HUMAN OBSERVER ===\n{state.human_feedback}\n===")

    content.append({"type": "text", "text": "\n".join(text_parts)})

    # Vision: current frame image
    if config.use_vision:
        b64 = grid_to_base64(grid, scale=2)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
        })

        # Diff image if we have a previous frame
        if prev_grid is not None:
            diff_img = image_diff(prev_grid, grid, scale=2)
            diff_b64 = image_to_base64(diff_img)
            content.append({"type": "text", "text": "Changes since last step (bright = changed):"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{diff_b64}", "detail": "low"},
            })

    # Text: compact grid representation
    if config.use_text and not config.use_vision:
        text_grid = grid_to_text_compact(grid)
        content.append({
            "type": "text",
            "text": f"Grid (hex, 0-f = 16 colors):\n{text_grid}",
        })

    return {"role": "user", "content": content}


def parse_response(text: str) -> dict:
    """Extract JSON action from LLM response text, supporting nested objects."""
    # Try to find the outermost JSON object using balanced braces
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

    # Fallback: try parsing the whole response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: extract action keyword
    text_upper = text.upper()
    for action_name in ["RESET", "ACTION7", "ACTION6", "ACTION5", "ACTION4", "ACTION3", "ACTION2", "ACTION1"]:
        if action_name in text_upper:
            return {"action": action_name, "reasoning": text}

    return {"action": "ACTION1", "reasoning": f"Failed to parse: {text[:200]}"}


def choose_action(
    client,
    grid: np.ndarray,
    state: AgentState,
    config: AgentConfig,
) -> tuple[GameAction, dict | None, str]:
    """Ask the LLM to choose an action."""
    user_msg = build_user_message(grid, state.prev_grid, state, config)

    # Clear human feedback after it's been included in the message
    state.human_feedback = ""

    # Build messages: system + recent history + current
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add recent message history (keep it bounded)
    history_start = max(0, len(state.messages) - config.message_history_limit * 2)
    messages.extend(state.messages[history_start:])
    messages.append(user_msg)

    # Show raw prompt on first call
    if config.show_raw_prompt and len(state.steps) == 0:
        print_raw_prompt(messages)

    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_completion_tokens=1500,
    )

    reply_text = response.choices[0].message.content or ""
    parsed = parse_response(reply_text)

    # Store in message history
    state.messages.append(user_msg)
    state.messages.append({"role": "assistant", "content": reply_text})

    # Update agent memory if provided
    if "memory" in parsed:
        mem = parsed["memory"]
        if isinstance(mem, dict):
            state.memory = json.dumps(mem, indent=2)
        else:
            state.memory = str(mem)

    # Map to GameAction
    action_name = parsed.get("action", "INTERACT").upper()
    game_action = ACTION_MAP.get(action_name, GameAction.ACTION5)

    # Prepare data for CLICK
    data = None
    if game_action == GameAction.ACTION6:
        x = int(parsed.get("x", 32))
        y = int(parsed.get("y", 32))
        data = {"x": max(0, min(63, x)), "y": max(0, min(63, y))}

    reasoning = parsed.get("reasoning", reply_text[:200])
    return game_action, data, reasoning


def run_agent(env, config: AgentConfig | None = None) -> AgentState:
    """Run the LLM agent on an ARC-AGI-3 environment."""
    if config is None:
        config = AgentConfig()

    client = create_client(config)
    state = AgentState()
    saved_frames: list[Image.Image] = []

    # Set up frame saving
    if config.save_frames:
        from pathlib import Path
        frames_path = Path(config.frames_dir)
        frames_path.mkdir(parents=True, exist_ok=True)
        print(f"  Saving frames to: {frames_path.resolve()}")

    # Get initial observation
    obs = env.reset()
    if obs is None or not obs.frame:
        print("ERROR: No initial observation from environment")
        return state

    grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

    # Track available actions from the environment
    if hasattr(obs, "available_actions") and obs.available_actions:
        state.available_actions = list(obs.available_actions)

    # Save initial frame
    if config.save_frames:
        img = grid_to_image(grid, scale=4)
        img.save(frames_path / "step_000_initial.png")
        saved_frames.append(img.copy())

    # Interactive: prompt human before first action
    if config.step_mode:
        feedback = prompt_human("Comment/question before first action")
        if feedback is None:
            print("  Stopped by user.")
            return state
        if feedback:
            state.human_feedback = feedback

    for step_num in range(config.max_actions):
        # Ask LLM for action
        game_action, data, reasoning = choose_action(client, grid, state, config)

        action_name = game_action.name
        label = ACTION_LABELS.get(action_name, action_name)
        coords = f" x={data['x']},y={data['y']}" if data else ""
        print(f"\n  [{step_num + 1}/{config.max_actions}]")
        print(f"  Thinking: {reasoning}")
        print(f"  Action:   >>> {label}{coords}")

        # Execute action
        obs = env.step(game_action, data=data or {})

        if obs is None:
            print("  WARNING: No observation returned")
            continue

        # Record step
        record = StepRecord(
            action_num=step_num + 1,
            action=action_name,
            x=data.get("x") if data else None,
            y=data.get("y") if data else None,
            reasoning=reasoning,
            state=obs.state.name if obs.state else "UNKNOWN",
            levels_completed=obs.levels_completed or 0,
        )
        state.steps.append(record)

        # Update grid and available actions
        state.prev_grid = grid
        if obs.frame:
            grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
        if hasattr(obs, "available_actions") and obs.available_actions:
            state.available_actions = list(obs.available_actions)

        # Save frame
        if config.save_frames:
            img = grid_to_image(grid, scale=4)
            img.save(frames_path / f"step_{step_num + 1:03d}_{label.lower()}.png")
            saved_frames.append(img.copy())

        # Step mode: prompt human for feedback
        if config.step_mode:
            feedback = prompt_human("Comment/question")
            if feedback is None:
                print("  Stopped by user.")
                break
            if feedback:
                state.human_feedback = feedback

        # Delay for visual rendering
        if config.delay > 0:
            time.sleep(config.delay)

        # Check terminal states
        if obs.state == GameState.WIN:
            print(f"  WIN after {step_num + 1} steps! "
                  f"Levels completed: {obs.levels_completed}")
            break
        elif obs.state == GameState.GAME_OVER:
            print(f"  GAME OVER at step {step_num + 1}. Resetting...")
            obs = env.reset()
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
                state.prev_grid = None

    # Save GIF of all frames
    if config.save_frames and len(saved_frames) > 1:
        gif_path = frames_path / "replay.gif"
        saved_frames[0].save(
            gif_path,
            save_all=True,
            append_images=saved_frames[1:],
            duration=500,  # 500ms per frame
            loop=0,
        )
        print(f"  Replay GIF saved: {gif_path.resolve()}")

    return state
