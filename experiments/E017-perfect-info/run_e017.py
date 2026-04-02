"""E017: Perfect Information Diagnostic.

KEY QUESTION: If we give GPT-5.4 COMPLETE game instructions (what it controls,
what each action does, the goal, the strategy), can it solve the puzzle?

This is NOT for the final system. It's a diagnostic to determine:
- If it solves → the bottleneck is info extraction, harness approach is viable
- If it doesn't → LLM can't plan sequences, need fundamentally different approach

Setup: g50t with full instructions, NO reflector overhead, just actor with
perfect knowledge. 30 steps. Also test a simple movement game (ft09).
"""
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import arc_agi
from arcengine import GameAction, GameState
from openai import OpenAI
import os

from arcagi3.grid_utils import (
    compute_diff,
    grid_to_base64,
    grid_to_image,
    grid_hash,
)


# Perfect info prompt for g50t — everything we know from source code
G50T_PERFECT_PROMPT = """\
You are playing a TILE-SWAP PUZZLE game on a 64x64 grid.

WHAT YOU CONTROL:
- You control a blue tile/object.

ACTIONS:
- ACTION1: Move UP (swap your tile with the one above)
- ACTION2: Move DOWN (swap your tile with the one below)
- ACTION3: Move LEFT (swap your tile with the one to the left)
- ACTION4: Move RIGHT (swap your tile with the one to the right)
- ACTION5: SPECIAL — this records your path and rewinds to start, creating
  a clone that replays your moves. Use it when you've reached a checkpoint.
- RESET: Start over

HOW THE GAME WORKS:
- This is a recording puzzle (like Braid). You navigate to checkpoints in order.
- At each checkpoint, press ACTION5 to "record" your path.
- After recording, you rewind to start and a clone replays your moves.
- You then navigate to the NEXT checkpoint while your clone handles the previous.
- The puzzle is solved when all checkpoints are simultaneously covered.

RESOURCE BAR:
- There's a bar that shrinks every 2 moves. When it's gone, you lose.
- Be EFFICIENT. Don't waste moves.

STRATEGY:
1. Look at the board. Identify colored markers (checkpoints).
2. Navigate to the first/nearest checkpoint.
3. Press ACTION5 to record.
4. Navigate to the next checkpoint.
5. Repeat until solved.

RESPOND WITH JSON:
{"action": "ACTION1|ACTION2|ACTION3|ACTION4|ACTION5|RESET", "reasoning": "brief why"}
"""

# Perfect info prompt for ft09 — generic movement game
FT09_PERFECT_PROMPT = """\
You are playing a MOVEMENT PUZZLE game on a 64x64 grid.

WHAT YOU CONTROL:
- You control a colored object (likely the one that moves when you act).

ACTIONS (test each one to learn what they do):
- ACTION1-ACTION5: Likely movement or interaction
- RESET: Start over

HOW TO PLAY:
- Explore the environment by testing each action.
- Watch what changes after each action — objects moving, colors changing, bars.
- A shrinking bar = time limit. Be efficient.
- Look for a goal: reaching a target, collecting items, or matching patterns.

STRATEGY:
1. Test each action once to learn what they do.
2. Identify the goal from visual cues.
3. Execute a plan to reach the goal efficiently.

RESPOND WITH JSON:
{"action": "ACTION1|ACTION2|ACTION3|ACTION4|ACTION5|RESET", "reasoning": "brief why"}
"""

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


def parse_action(text: str) -> str:
    """Extract action name from LLM response."""
    try:
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        data = json.loads(text[start:i+1])
                        return data.get("action", "ACTION1").upper()
    except (json.JSONDecodeError, KeyError):
        pass
    # Fallback
    for name in ["RESET", "ACTION7", "ACTION6", "ACTION5", "ACTION4", "ACTION3", "ACTION2", "ACTION1"]:
        if name in text.upper():
            return name
    return "ACTION1"


def run_perfect_info(game_id: str, system_prompt: str, max_steps: int = 30):
    """Run a game with perfect information prompt — minimal harness."""
    client = OpenAI(
        base_url=os.environ["AZURE_FOUNDRY_BASE_URL"],
        api_key=os.environ["AZURE_INFERENCE_CREDENTIAL"],
    )
    model = os.environ.get("AZURE_MODEL", "gpt-5.4")

    arc = arc_agi.Arcade()
    env = arc.make(game_id, render_mode=None)
    obs = env.reset()
    grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

    # Track available actions
    avail = []
    if hasattr(obs, "available_actions") and obs.available_actions:
        avail = [f"ACTION{a}" if a != 0 else "RESET" for a in obs.available_actions]

    messages = [{"role": "system", "content": system_prompt}]
    prev_grid = None
    results = []

    frames_dir = Path(f"experiments/E017-perfect-info/frames/{game_id}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    img = grid_to_image(grid, scale=4)
    img.save(frames_dir / "step_000.png")

    for step in range(max_steps):
        # Build user message with current state
        parts = [f"Step {step+1}/{max_steps}"]
        if avail:
            parts.append(f"Available: {', '.join(avail)}")

        if prev_grid is not None:
            diff = compute_diff(prev_grid, grid)
            parts.append(f"Last action result: {diff['description']}")
        else:
            parts.append("This is the first frame. Observe and decide.")

        content = [{"type": "text", "text": "\n".join(parts)}]

        # Add image
        b64 = grid_to_base64(grid, scale=2)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })

        messages.append({"role": "user", "content": content})

        # Call LLM
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=300,
            timeout=120,
        )
        reply = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": reply})

        # Keep history manageable
        if len(messages) > 22:
            messages = [messages[0]] + messages[-20:]

        action_name = parse_action(reply)
        game_action = ACTION_MAP.get(action_name, GameAction.ACTION1)

        print(f"  Step {step+1}: {action_name} — {reply[:100]}")

        # Execute
        prev_grid = grid
        data = {}
        if game_action == GameAction.ACTION6:
            data = {"x": 32, "y": 32}
        obs = env.step(game_action, data=data)

        if obs and obs.frame:
            grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

        img = grid_to_image(grid, scale=4)
        img.save(frames_dir / f"step_{step+1:03d}.png")

        results.append({
            "step": step + 1,
            "action": action_name,
            "state": obs.state.name if obs and obs.state else "?",
            "levels": obs.levels_completed if obs else 0,
        })

        if obs and obs.state == GameState.WIN:
            print(f"  🏆 WIN at step {step+1}! Levels: {obs.levels_completed}")
            break
        elif obs and obs.state == GameState.GAME_OVER:
            print(f"  *** GAME OVER at step {step+1}")
            obs = env.reset()
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
                prev_grid = None

    levels = results[-1]["levels"] if results else 0
    deaths = sum(1 for r in results if r["state"] == "GAME_OVER")

    try:
        scorecard = arc.get_scorecard()
        score = scorecard.score if scorecard else 0
    except Exception:
        score = 0

    return {
        "game_id": game_id,
        "steps": len(results),
        "levels_completed": levels,
        "deaths": deaths,
        "score": score,
        "actions": results,
    }


def main():
    print("=== E017: Perfect Information Diagnostic ===\n")

    games = [
        ("g50t-5849a774", G50T_PERFECT_PROMPT, "g50t (full instructions)"),
        ("ft09-0d8bbf25", FT09_PERFECT_PROMPT, "ft09 (basic instructions)"),
    ]

    all_results = []
    for game_id, prompt, label in games:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        start = time.time()
        result = run_perfect_info(game_id, prompt, max_steps=30)
        result["elapsed"] = round(time.time() - start, 1)
        result["label"] = label
        all_results.append(result)
        print(f"\n  → {label}: {result['levels_completed']} levels, "
              f"{result['deaths']} deaths, {result['elapsed']:.0f}s, score={result['score']}")

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['label']}: levels={r['levels_completed']}, "
              f"deaths={r['deaths']}, score={r['score']}")

    out = Path("experiments/E017-perfect-info")
    with open(out / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
