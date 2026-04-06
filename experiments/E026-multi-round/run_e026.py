"""E026: Multi-Round Discovery — let the LLM explore and figure it out.

PHILOSOPHY: The LLM is the brain. We give it eyes (frames), hands (actions),
and memory (previous observations). It explores, hypothesizes, tests, and
discovers the game objective on its own. NO game-specific code.

Architecture:
  Round 1: Random exploration (0 LLM) → comprehensive analysis (1 LLM)
  Round 2+: Execute plan → observe results → reflect + replan (1 LLM each)
  Up to 5 rounds, ~60 total game actions, ~5-8 LLM calls

Key differences from E024/E025:
- GPT-5.4 (best model)
- Multi-round: the LLM SEES what happened and LEARNS from failures
- NO game-specific navigation, BFS, or hardcoded strategies
- The LLM decides EVERYTHING — what to do, how to interpret results
- Exploration is the LLM's job, not the harness's
"""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import arc_agi
from arcengine import GameAction, GameState
from openai import OpenAI

from arcagi3.grid_utils import (
    compute_diff,
    describe_frame,
    grid_to_base64,
    grid_to_image,
)

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


# ── Blind exploration (no LLM) ───────────────────────────────────────

def blind_explore(env, available_actions, n_steps=15):
    """Random exploration to gather initial data. Zero LLM calls."""
    obs = env.reset()
    grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
    initial_grid = grid.copy()

    actions = [f"ACTION{a}" for a in available_actions if a != 0]
    if not actions:
        actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]

    log = []
    frames = [grid.copy()]

    # Strategy: test each action once, then random
    sequence = list(actions)
    while len(sequence) < n_steps:
        sequence.append(random.choice(actions))
    sequence = sequence[:n_steps]

    for i, action_name in enumerate(sequence):
        prev_grid = grid.copy()
        game_action = ACTION_MAP.get(action_name, GameAction.ACTION1)

        data = {}
        if game_action == GameAction.ACTION6:
            data = {"x": random.randint(5, 58), "y": random.randint(5, 58)}

        obs = env.step(game_action, data=data)
        if obs and obs.frame:
            grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

        diff = compute_diff(prev_grid, grid)
        entry = {
            "step": i + 1,
            "action": action_name + (f" x={data['x']},y={data['y']}" if data else ""),
            "changed_cells": diff.get("changed_cells", 0),
            "description": diff["description"],
            "movements": diff.get("movements", []),
            "swaps": diff.get("swaps", []),
            "state": obs.state.name if obs and obs.state else "?",
            "levels": obs.levels_completed if obs else 0,
        }
        log.append(entry)
        frames.append(grid.copy())

        if obs and obs.state == GameState.GAME_OVER:
            obs = env.reset()
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
            frames.append(grid.copy())

        if obs and obs.state == GameState.WIN:
            break

    return {
        "log": log,
        "initial_grid": initial_grid,
        "final_grid": grid,
        "frames": frames,
        "initial_description": describe_frame(initial_grid),
        "total_steps": len(log),
        "deaths": sum(1 for e in log if e["state"] == "GAME_OVER"),
        "levels": log[-1]["levels"] if log else 0,
    }


# ── LLM interaction ──────────────────────────────────────────────────

INITIAL_ANALYSIS_PROMPT = """\
You are playing an ARC-AGI-3 game — a 64x64 visual puzzle with unknown rules.
You have NO instructions. You must figure out the game by observing what happens.

You just completed a blind exploration phase. You have:
1. The initial frame (image)
2. A frame from mid-exploration
3. The current frame (image)
4. A log of every action and what changed

Your task: Analyze ALL this data and figure out:
1. What do you control? What moves when you act?
2. What does each action do? Are effects position-dependent?
3. What's the game objective? What pattern suggests a goal?
4. What should you try next?

IMPORTANT: You may NOT know the goal yet. That's OK. Propose hypotheses
and a plan to TEST them. Your plan should be EXPLORATORY — try different
things to discover how the game works.

Respond with JSON:
{
  "observations": "what you observed from the exploration data",
  "controlled_object": "what you think you control, with evidence",
  "action_effects": {"ACTION1": "effect", "ACTION2": "effect", ...},
  "goal_hypotheses": [
    "hypothesis 1 — what the goal might be",
    "hypothesis 2 — alternative"
  ],
  "next_plan": ["ACTION1", "ACTION2", ...],
  "plan_reasoning": "why this plan will help discover or achieve the goal"
}

Give a plan of 15-20 actions. If you're unsure of the goal, make the plan
EXPLORATORY — try things you haven't tried yet. Be BOLD — try actions
in combinations you haven't tried, at positions you haven't been.
"""

REFLECTION_TEMPLATE = (
    "You are playing an ARC-AGI-3 game — a 64x64 visual puzzle with unknown rules.\n\n"
    "PREVIOUS ANALYSIS:\n{previous_analysis}\n\n"
    "EXECUTION RESULTS:\n"
    "You executed a plan. Here's what happened:\n{execution_log}\n\n"
    "Current frame is shown.\n\n"
    "REFLECT: What did you learn? Did your hypotheses hold up?\n"
    "IMPORTANT: If you achieved 0 levels and things didn't work, you need to\n"
    "try something FUNDAMENTALLY DIFFERENT, not just minor variations.\n\n"
    "Think about:\n"
    "- Did any action produce an unexpected result? That's a CLUE.\n"
    "- Did the game state change in ways you didn't predict?\n"
    "- Is there something in the image you haven't interacted with yet?\n"
    "- Could the goal be completely different from what you assumed?\n\n"
    'Respond with JSON:\n'
    '{{"reflection": "what you learned from the last round",\n'
    ' "updated_hypotheses": ["revised hypothesis 1", "revised hypothesis 2"],\n'
    ' "what_to_try_differently": "specific thing to change",\n'
    ' "next_plan": ["ACTION1", "ACTION2", ...],\n'
    ' "plan_reasoning": "why this new plan addresses what went wrong"}}\n\n'
    "Give a plan of 10-15 actions. Try something NEW."
)


def call_llm(client, model, system_prompt, text, frames, temperature=0.5):
    """Make one LLM call with text + frame images."""
    content = [{"type": "text", "text": text}]

    for i, (label, frame) in enumerate(frames):
        b64 = grid_to_base64(frame, scale=4)
        content.append({"type": "text", "text": f"\n--- {label} ---"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=3000,
        timeout=180,
    )
    return response.choices[0].message.content or ""


def parse_json(text):
    """Extract JSON from LLM response."""
    start = text.find("{")
    if start == -1:
        return {"next_plan": ["ACTION1"] * 10, "error": "no_json"}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    break
    return {"next_plan": ["ACTION1"] * 10, "error": "parse_failed"}


def format_log(log):
    """Format action log for LLM consumption."""
    lines = []
    for entry in log:
        line = f"Step {entry['step']}: {entry['action']}"
        line += f" → {entry['description'][:100]}"
        if entry.get("movements"):
            for m in entry["movements"][:2]:
                line += f"\n  move: {m['color_name']} ({m['from_x']},{m['from_y']})→({m['to_x']},{m['to_y']})"
        if entry.get("swaps"):
            for s in entry["swaps"][:2]:
                line += f"\n  SWAP: {s['color_a_name']} ↔ {s['color_b_name']}"
        if entry["state"] not in ("NOT_FINISHED", "?"):
            line += f"\n  *** {entry['state']} ***"
        lines.append(line)
    return "\n".join(lines)


# ── Execution ─────────────────────────────────────────────────────────

def execute_plan(env, plan, grid):
    """Execute a sequence of actions. Returns log and updated grid."""
    log = []
    obs = None

    for i, raw_action in enumerate(plan):
        action_name = raw_action.split()[0] if " " in raw_action else raw_action
        game_action = ACTION_MAP.get(action_name, GameAction.ACTION1)

        data = {}
        if game_action == GameAction.ACTION6:
            x_match = re.search(r'x=(\d+)', raw_action)
            y_match = re.search(r'y=(\d+)', raw_action)
            x = int(x_match.group(1)) if x_match else 32
            y = int(y_match.group(1)) if y_match else 32
            data = {"x": max(0, min(63, x)), "y": max(0, min(63, y))}

        prev_grid = grid.copy()
        obs = env.step(game_action, data=data)

        if obs and obs.frame:
            grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

        diff = compute_diff(prev_grid, grid)

        entry = {
            "step": i + 1,
            "action": raw_action,
            "changed_cells": diff.get("changed_cells", 0),
            "description": diff["description"],
            "movements": diff.get("movements", []),
            "swaps": diff.get("swaps", []),
            "state": obs.state.name if obs and obs.state else "?",
            "levels": obs.levels_completed if obs else 0,
        }
        log.append(entry)

        status = f"  [R] Step {i+1}: {raw_action} → {diff['description'][:60]}"
        print(status)

        if obs and obs.state == GameState.WIN:
            print(f"  [R] WIN! Levels: {obs.levels_completed}")
            break

        if obs and obs.state == GameState.GAME_OVER:
            print(f"  [R] GAME OVER")
            obs = env.reset()
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

    levels = log[-1]["levels"] if log else 0
    deaths = sum(1 for e in log if e["state"] == "GAME_OVER")
    return log, grid, levels, deaths


# ── Main pipeline ─────────────────────────────────────────────────────

def run_game(game_id, arc, model="gpt-5.4", max_rounds=5, save_dir=None):
    """Multi-round discovery loop."""
    client = OpenAI(
        base_url=os.environ["AZURE_FOUNDRY_BASE_URL"],
        api_key=os.environ["AZURE_INFERENCE_CREDENTIAL"],
    )

    env = arc.make(game_id, render_mode=None)
    obs = env.reset()
    avail = list(obs.available_actions) if hasattr(obs, "available_actions") and obs.available_actions else []
    grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

    api_calls = 0
    total_steps = 0
    total_deaths = 0
    best_levels = 0

    # Phase 1: Blind exploration
    print(f"\n  === ROUND 0: BLIND EXPLORATION ===")
    explore = blind_explore(env, avail, n_steps=15)
    total_steps += explore["total_steps"]
    total_deaths += explore["deaths"]
    grid = explore["final_grid"]

    if explore["levels"] > 0:
        return {"game_id": game_id, "levels": explore["levels"],
                "deaths": explore["deaths"], "api_calls": 0, "method": "blind_luck"}

    # Phase 2: Initial analysis
    print(f"\n  === ROUND 1: INITIAL ANALYSIS ===")
    text = (
        f"Available actions: {', '.join(f'ACTION{a}' for a in avail)}\n\n"
        f"=== INITIAL FRAME DESCRIPTION ===\n{explore['initial_description']}\n\n"
        f"=== EXPLORATION LOG ===\n{format_log(explore['log'])}\n"
    )

    n_frames = len(explore["frames"])
    frame_images = [
        ("Initial frame", explore["frames"][0]),
        ("Mid-exploration", explore["frames"][n_frames // 2]),
        ("Current frame", explore["frames"][-1]),
    ]

    print("  Calling LLM for initial analysis...")
    reply = call_llm(client, model, INITIAL_ANALYSIS_PROMPT, text, frame_images)
    analysis = parse_json(reply)
    api_calls += 1

    plan = analysis.get("next_plan", [])
    print(f"  Hypotheses: {analysis.get('goal_hypotheses', ['?'])}")
    print(f"  Plan ({len(plan)} actions): {plan[:8]}...")

    # Phase 3: Execute + reflect loop
    for round_num in range(2, max_rounds + 1):
        if not plan:
            break

        print(f"\n  === ROUND {round_num}: EXECUTE + REFLECT ===")

        # DON'T reset — let the game state carry forward between rounds
        # This preserves positional progress from previous rounds
        exec_log, grid, levels, deaths = execute_plan(env, plan, grid)
        total_steps += len(exec_log)
        total_deaths += deaths

        if levels > best_levels:
            best_levels = levels
        if levels > 0:
            print(f"  PROGRESS! {levels} levels completed!")
            # Keep going — try to get more levels
            # Don't break, let the loop continue

        # Reflect
        exec_text = format_log(exec_log)
        prev_analysis = json.dumps({
            k: analysis.get(k) for k in
            ["observations", "goal_hypotheses", "action_effects", "controlled_object"]
            if k in analysis
        }, indent=2)

        reflect_text = REFLECTION_TEMPLATE.format(
            previous_analysis=prev_analysis,
            execution_log=exec_text,
        )

        print("  Reflecting...")
        reply = call_llm(
            client, model, reflect_text, "",
            [("Current frame", grid)],
            temperature=0.6,  # Slightly higher for creative replanning
        )
        analysis = parse_json(reply)
        api_calls += 1

        plan = analysis.get("next_plan", [])
        print(f"  Reflection: {analysis.get('reflection', '?')[:120]}")
        print(f"  New plan ({len(plan)} actions): {plan[:8]}...")

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            img = grid_to_image(grid, scale=4)
            img.save(Path(save_dir) / f"round{round_num}_end.png")

    return {
        "game_id": game_id,
        "levels": best_levels,
        "deaths": total_deaths,
        "total_steps": total_steps,
        "api_calls": api_calls,
        "rounds": min(max_rounds, round_num if 'round_num' in dir() else 1),
        "final_analysis": analysis,
        "method": "multi-round-discovery",
    }


def main():
    games = ["g50t-5849a774", "ls20-9607627b"]
    model = "gpt-5.4"
    max_rounds = 5

    if len(sys.argv) > 1:
        games = [g for g in sys.argv[1:] if not g.startswith("--")]
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model = sys.argv[idx + 1]
    if "--rounds" in sys.argv:
        idx = sys.argv.index("--rounds")
        max_rounds = int(sys.argv[idx + 1])

    print(f"=== E026: Multi-Round Discovery ===")
    print(f"  Games: {games}")
    print(f"  Model: {model}")
    print(f"  Max rounds: {max_rounds}")
    print()

    arc = arc_agi.Arcade()
    all_results = []
    total_api = 0

    for game_id in games:
        print(f"\n{'=' * 70}")
        print(f"  GAME: {game_id}")
        print(f"{'=' * 70}")
        start = time.time()
        try:
            result = run_game(
                game_id, arc, model=model, max_rounds=max_rounds,
                save_dir=f"experiments/E026-multi-round/frames/{game_id}",
            )
            result["elapsed"] = round(time.time() - start, 1)
            result["model"] = model
            all_results.append(result)
            total_api += result.get("api_calls", 0)
            print(f"\n  -> {game_id}: {result['levels']} levels, "
                  f"{result['deaths']} deaths, {result['api_calls']} API calls, "
                  f"{result['elapsed']:.0f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results.append({
                "game_id": game_id, "levels": 0, "deaths": 0,
                "api_calls": 0, "error": str(e),
                "elapsed": round(time.time() - start, 1),
            })

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — E026 Multi-Round Discovery")
    print(f"{'=' * 70}")
    for r in all_results:
        gid = r.get("game_id", "?").split("-")[0]
        calls = r.get("api_calls", 0)
        print(f"  {gid}: levels={r['levels']}, deaths={r['deaths']}, "
              f"api_calls={calls}, time={r.get('elapsed', 0):.0f}s")
    print(f"\n  TOTAL API CALLS: {total_api}")
    print(f"  TOTAL LEVELS: {sum(r['levels'] for r in all_results)}")

    out = Path("experiments/E026-multi-round")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
