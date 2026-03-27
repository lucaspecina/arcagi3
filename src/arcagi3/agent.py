"""LLM Agent for ARC-AGI-3 using Azure AI Foundry."""

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
from arcengine import GameAction, GameState

from .grid_utils import grid_to_base64, grid_to_text_compact, image_diff, image_to_base64

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
You are an agent playing an abstract reasoning game on a 64x64 grid with 16 colors.

Your goal: figure out the rules of the environment, complete levels, and win the game.
There are NO written instructions — you must discover everything through interaction.

Each turn you receive:
- The current grid state (as an image and/or text)
- A diff showing what changed since last turn
- Your action history

You must respond with a JSON object:
{
  "reasoning": "Brief analysis: what changed, what you learned, what to try next",
  "action": "UP|DOWN|LEFT|RIGHT|INTERACT|CLICK|UNDO|RESET",
  "x": 0,
  "y": 0
}

The "x" and "y" fields are ONLY needed for CLICK actions (0-63 range).

Tips:
- Level 1 is always a tutorial — pay attention to what works
- Try systematic exploration first: move in each direction, interact with objects
- Track what causes changes in the grid — those are the mechanics
- If stuck, try RESET to restart the current level
- The game has multiple levels with increasing complexity
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

    # Agent memory/scratchpad
    if state.memory:
        text_parts.append(f"Your notes: {state.memory}")

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
    """Extract JSON action from LLM response text."""
    # Try to find JSON in the response
    # First try: find JSON block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

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
        max_completion_tokens=300,
    )

    reply_text = response.choices[0].message.content or ""
    parsed = parse_response(reply_text)

    # Store in message history
    state.messages.append(user_msg)
    state.messages.append({"role": "assistant", "content": reply_text})

    # Update agent memory if provided
    if "memory" in parsed:
        state.memory = parsed["memory"]

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

    # Get initial observation
    obs = env.reset()
    if obs is None or not obs.frame:
        print("ERROR: No initial observation from environment")
        return state

    grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

    for step_num in range(config.max_actions):
        # Ask LLM for action
        game_action, data, reasoning = choose_action(client, grid, state, config)

        action_name = game_action.name
        print(f"  Step {step_num + 1}: {action_name}"
              f"{f' ({data})' if data else ''}"
              f" — {reasoning[:80]}")

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

    return state
