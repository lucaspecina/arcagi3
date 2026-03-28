"""LLM Agent for ARC-AGI-3 using Azure AI Foundry."""

import json
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
from arcengine import GameAction, GameState

from PIL import Image

from .grid_utils import (
    grid_to_ansi,
    grid_to_base64,
    grid_to_image,
    grid_to_text_compact,
    image_diff,
    image_to_base64,
)

# Map human-readable action names to GameAction enum
ACTION_MAP = {
    "UP": GameAction.ACTION1,
    "DOWN": GameAction.ACTION2,
    "LEFT": GameAction.ACTION3,
    "RIGHT": GameAction.ACTION4,
    "INTERACT": GameAction.ACTION5,
    "CLICK": GameAction.ACTION6,
    "UNDO": GameAction.ACTION7,
    "RESET": GameAction.RESET,
    "ACTION1": GameAction.ACTION1,
    "ACTION2": GameAction.ACTION2,
    "ACTION3": GameAction.ACTION3,
    "ACTION4": GameAction.ACTION4,
    "ACTION5": GameAction.ACTION5,
    "ACTION6": GameAction.ACTION6,
    "ACTION7": GameAction.ACTION7,
}

SYSTEM_PROMPT = """\
You are playing an ARC-AGI-3 game. These are abstract visual puzzle environments on a \
64x64 grid with 16 colors. Each environment is a game with multiple levels.

THE CRITICAL CHALLENGE: You have NO instructions. The game tells you NOTHING.
You must discover THREE things through experimentation:
  1. CONTROLS — What does each action do? Which actions move things? Which interact?
  2. RULES — What are the mechanics? What objects exist? How do they interact?
  3. GOAL — What are you trying to achieve? What triggers level completion?

## What you see
A 64x64 colored grid. Objects are patterns of colored pixels. You control something \
on the grid (an avatar, a cursor, a block — you must figure out what).

## Available actions
- UP, DOWN, LEFT, RIGHT — directional movement/manipulation
- INTERACT — general interaction (like pressing a button, picking up, activating)
- CLICK x,y — click on a specific grid coordinate (0-63 range)
- UNDO — undo last action
- RESET — restart current level from scratch

## How to think

BEFORE your first action, study the initial image carefully:
- What distinct objects/shapes/regions do you see?
- What colors are used and what might they represent?
- What looks like it could be controllable vs static?
- What looks like it could be a goal/target?
- Form a hypothesis about what kind of game this might be.

ON EVERY TURN, your reasoning must address:
- OBSERVATION: What changed in the grid? What stayed the same?
- CONTROLS: What did my last action do? Update your understanding of controls.
- RULES: What rules/mechanics have I discovered? What's still unknown?
- GOAL: What do I think the objective is? Any new evidence?
- PLAN: What should I try next and why?

## Strategy
- Level 1 is ALWAYS a tutorial — it teaches you the basic mechanics. Pay close attention.
- Be SYSTEMATIC: test each action methodically, don't repeat the same thing.
- When something changes on the grid, that's a clue — analyze WHY it changed.
- Keep a mental map of what you've tried and what worked.
- If stuck for several turns, RESET and try a completely different approach.
- Later levels introduce new mechanics on top of what you learned.

## Your knowledge state (CRITICAL — this is your long-term memory)
You have a persistent knowledge dict that survives ALL turns, even when older messages \
are dropped. You MUST update it every turn with your latest discoveries.

If you don't write it down, you WILL forget it.

## Response format
Respond with a JSON object:
{
  "reasoning": "Your detailed thinking: OBSERVATION (what changed), analysis, and PLAN (next steps)",
  "action": "UP|DOWN|LEFT|RIGHT|INTERACT|CLICK|UNDO|RESET",
  "x": 0,
  "y": 0,
  "memory": {
    "controls": {"UP": "what it does", "DOWN": "what it does", ...},
    "rules": ["rule 1 discovered", "rule 2 discovered", ...],
    "goal": "current hypothesis about what wins the level",
    "map": "description of the layout and key objects/positions",
    "plan": "current strategy to win — what to try next and why"
  }
}

RULES for the memory dict:
- ALWAYS include it. Update it EVERY turn.
- "controls": map each action to what you observed it does. Mark unknown ones as "untested".
- "rules": list of confirmed game mechanics. Only add things you've verified.
- "goal": your best hypothesis about the win condition. Update as you learn more.
- "map": describe the spatial layout, objects, colors, and positions you've identified.
- "plan": your step-by-step strategy. Update when your plan changes.

The "x" and "y" fields are ONLY needed for CLICK actions.
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
    show_grid: bool = False

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
    for action_name in ["RESET", "INTERACT", "CLICK", "UP", "DOWN", "LEFT", "RIGHT", "UNDO"]:
        if action_name in text_upper:
            return {"action": action_name, "reasoning": text}

    return {"action": "INTERACT", "reasoning": f"Failed to parse: {text[:200]}"}


def choose_action(
    client,
    grid: np.ndarray,
    state: AgentState,
    config: AgentConfig,
) -> tuple[GameAction, dict | None, str]:
    """Ask the LLM to choose an action.

    Returns:
        (action, data_dict_or_None, reasoning)
    """
    user_msg = build_user_message(grid, state.prev_grid, state, config)

    # Build messages: system + recent history + current
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add recent message history (keep it bounded)
    history_start = max(0, len(state.messages) - config.message_history_limit * 2)
    messages.extend(state.messages[history_start:])
    messages.append(user_msg)

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
    """Run the LLM agent on an ARC-AGI-3 environment.

    Args:
        env: EnvironmentWrapper from arc_agi.Arcade.make()
        config: Agent configuration. Uses env vars if not provided.

    Returns:
        AgentState with full history.
    """
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

    # Show initial grid
    if config.show_grid:
        print("  Initial state:")
        print(grid_to_ansi(grid))

    # Save initial frame
    if config.save_frames:
        img = grid_to_image(grid, scale=4)
        img.save(frames_path / "step_000_initial.png")
        saved_frames.append(img.copy())

    for step_num in range(config.max_actions):
        # Ask LLM for action
        game_action, data, reasoning = choose_action(client, grid, state, config)

        action_name = game_action.name
        action_labels = {
            "ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT",
            "ACTION5": "INTERACT", "ACTION6": "CLICK", "ACTION7": "UNDO", "RESET": "RESET",
        }
        label = action_labels.get(action_name, action_name)
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

        # Update grid
        state.prev_grid = grid
        if obs.frame:
            grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

        # Show grid in terminal
        if config.show_grid:
            print(grid_to_ansi(grid))

        # Save frame
        if config.save_frames:
            img = grid_to_image(grid, scale=4)
            img.save(frames_path / f"step_{step_num + 1:03d}_{label.lower()}.png")
            saved_frames.append(img.copy())

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
