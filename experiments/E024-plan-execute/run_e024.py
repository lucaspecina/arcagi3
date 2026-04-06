"""E024: Plan-then-Execute — minimal LLM calls, maximum information.

APPROACH: Instead of calling the LLM every step (90+ calls for 30 steps),
we do:
  Phase 1: Random exploration without LLM (~15 steps, 0 API calls)
  Phase 2: ONE comprehensive LLM call with ALL exploration data + images
  Phase 3: Execute the plan, correct only on major failures (~3-5 calls)

Total API calls: ~5-8 instead of ~90.
Budget per run: ~$0.50 instead of ~$3.

The key insight: the LLM is GOOD at analyzing data all at once (E017 proved it).
It's BAD at incremental learning from single observations (E019-E023 proved that).
So give it ALL the data at once and let it reason comprehensively.
"""
import json
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import os
import arc_agi
from arcengine import GameAction, GameState
from openai import OpenAI

from arcagi3.grid_utils import (
    compute_diff,
    describe_frame,
    grid_to_base64,
    grid_to_image,
    grid_hash,
    COLOR_NAMES,
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

# ── Phase 1: Random Exploration (no LLM) ──────────────────────────

def random_explore(env, available_actions, n_steps=15, save_dir=None):
    """Execute random actions and collect comprehensive observation data."""
    obs = env.reset()
    grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
    initial_grid = grid.copy()

    actions = [f"ACTION{a}" for a in available_actions if a != 0]
    if not actions:
        actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]

    log = []
    frames = [grid.copy()]
    prev_grid = None

    # Strategy: test each action once, then random
    sequence = list(actions) + [random.choice(actions) for _ in range(n_steps - len(actions))]
    sequence = sequence[:n_steps]

    for i, action_name in enumerate(sequence):
        prev_grid = grid.copy()
        game_action = ACTION_MAP.get(action_name, GameAction.ACTION1)

        data = {}
        if game_action == GameAction.ACTION6:
            # For click games, click at random interesting positions
            data = {"x": random.randint(5, 58), "y": random.randint(5, 58)}

        obs = env.step(game_action, data=data)

        if obs and obs.frame:
            grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

        diff = compute_diff(prev_grid, grid)
        frames.append(grid.copy())

        entry = {
            "step": i + 1,
            "action": action_name,
            "data": data,
            "diff_description": diff["description"],
            "changed_cells": diff.get("changed_cells", 0),
            "movements": diff.get("movements", []),
            "swaps": diff.get("swaps", []),
            "state": obs.state.name if obs and obs.state else "?",
            "levels": obs.levels_completed if obs else 0,
        }
        log.append(entry)

        status = f"step {i+1}: {action_name}"
        if data:
            status += f" x={data.get('x')},y={data.get('y')}"
        status += f" → {diff['description'][:60]}"
        print(f"  [EXPLORE] {status}")

        if obs and obs.state == GameState.GAME_OVER:
            print(f"  [EXPLORE] GAME OVER at step {i+1}")
            obs = env.reset()
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
            frames.append(grid.copy())

        if obs and obs.state == GameState.WIN:
            print(f"  [EXPLORE] WIN at step {i+1}!")
            break

    # Save key frames
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for idx in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -1]:
            img = grid_to_image(frames[idx], scale=4)
            img.save(save_path / f"frame_{idx:03d}.png")

    return {
        "log": log,
        "initial_frame": initial_grid,
        "final_frame": grid,
        "key_frames": [frames[0], frames[len(frames)//2], frames[-1]],
        "initial_description": describe_frame(initial_grid),
        "final_description": describe_frame(grid),
        "total_steps": len(log),
        "deaths": sum(1 for e in log if e["state"] == "GAME_OVER"),
        "levels": log[-1]["levels"] if log else 0,
    }


# ── Phase 2: Comprehensive LLM Analysis ────────────────────────────

ANALYSIS_PROMPT = """\
You are analyzing exploration data from an ARC-AGI-3 game — a 64x64 visual puzzle.
You have been given:
1. The initial frame (image)
2. A mid-exploration frame
3. The final frame after exploration
4. A complete log of every action taken and what changed

Your job: analyze ALL this data at once and produce a COMPREHENSIVE model of the game.

Answer these questions:
1. WHAT DO I CONTROL? Which object moves when I act? What color/shape?
2. ACTION MAPPING: For each action, what direction does it move? Which actions have no effect? Are effects position-dependent?
3. GAME MECHANICS: Is this a swap game? Movement game? Click game? How do objects interact?
4. GOAL: What am I trying to achieve? Where should I navigate to?
5. RESOURCE MANAGEMENT: Is there a bar/counter? How fast does it deplete?
6. STRATEGY: Given what I've learned, what's the optimal sequence of actions?

IMPORTANT: Look at the ACTION EFFECT LOG carefully. If an action had different effects
at different times, explain WHY (position-dependent, cooldown, etc.).

Respond with JSON:
{
  "controlled_object": "description and current position",
  "action_map": {"ACTION1": "direction or effect", "ACTION2": "...", ...},
  "mechanics": "how the game works (swap, movement, click, etc.)",
  "goal_hypothesis": "what I think wins, with evidence",
  "resource_info": "bar status and depletion rate",
  "optimal_plan": ["ACTION3", "ACTION4", "ACTION3", ...],
  "plan_reasoning": "why this sequence should work",
  "confidence": "low|medium|high",
  "unknowns": ["what I still need to figure out"]
}

Make the plan SPECIFIC — give me an exact sequence of 15-20 actions.
"""


def analyze_exploration(client, model, explore_data):
    """Single comprehensive LLM call to analyze all exploration data."""
    # Build exploration summary text
    text_parts = [
        "=== EXPLORATION DATA ===",
        f"Total steps: {explore_data['total_steps']}",
        f"Deaths: {explore_data['deaths']}",
        f"Levels completed: {explore_data['levels']}",
        "",
        "=== INITIAL FRAME DESCRIPTION ===",
        explore_data["initial_description"],
        "",
        "=== ACTION EFFECT LOG ===",
    ]

    for entry in explore_data["log"]:
        text_parts.append(
            f"Step {entry['step']}: {entry['action']}"
            + (f" x={entry['data'].get('x')},y={entry['data'].get('y')}" if entry.get("data") else "")
            + f" → {entry['diff_description']}"
        )
        if entry["movements"]:
            for m in entry["movements"]:
                text_parts.append(
                    f"  movement: {m['color_name']} ({m['from_x']},{m['from_y']})→"
                    f"({m['to_x']},{m['to_y']}) delta=({m['dx']},{m['dy']})"
                )
        if entry["swaps"]:
            for s in entry["swaps"]:
                text_parts.append(
                    f"  SWAP: {s['color_a_name']} ↔ {s['color_b_name']}"
                )

    text_parts.extend([
        "",
        "=== FINAL FRAME DESCRIPTION ===",
        explore_data["final_description"],
    ])

    text = "\n".join(text_parts)

    # Build message with images
    content = [{"type": "text", "text": text}]

    for i, frame in enumerate(explore_data["key_frames"]):
        label = ["Initial", "Mid-exploration", "Final"][i] if i < 3 else f"Frame {i}"
        b64 = grid_to_base64(frame, scale=2)
        content.append({"type": "text", "text": f"\n--- {label} Frame ---"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })

    messages = [
        {"role": "system", "content": ANALYSIS_PROMPT},
        {"role": "user", "content": content},
    ]

    print("  [ANALYSIS] Calling LLM for comprehensive analysis...")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_completion_tokens=3000,
        timeout=180,
    )
    reply = response.choices[0].message.content or ""
    print(f"  [ANALYSIS] Got {len(reply)} chars")
    return reply


# ── Phase 3: Plan Execution ─────────────────────────────────────────

CORRECTION_TEMPLATE = (
    "You are executing a plan in an ARC-AGI-3 game. Something unexpected happened.\n\n"
    "Your original plan: {plan}\n"
    "Steps completed so far: {completed}\n"
    "Last action: {last_action}\n"
    "Expected result: {expected}\n"
    "Actual result: {actual}\n\n"
    "Current frame is shown. What should you do?\n"
    "- If the plan is still viable, continue with the next action.\n"
    "- If something went wrong, adjust the remaining plan.\n\n"
    'Respond with JSON: {{"assessment": "what went wrong", "continue_plan": true/false, '
    '"adjusted_plan": ["next actions..."], "reasoning": "why"}}'
)


def execute_plan(env, client, model, plan, analysis, max_steps=20, save_dir=None):
    """Execute the action plan, calling LLM only on failures."""
    # RESET to start fresh — plan is designed from initial state
    obs_data = env.step(GameAction.RESET, data={})
    if obs_data and obs_data.frame:
        grid = np.array(obs_data.frame[0]) if isinstance(obs_data.frame[0], list) else obs_data.frame[0]
    else:
        obs_data = env.reset()
        grid = np.array(obs_data.frame[0]) if isinstance(obs_data.frame[0], list) else obs_data.frame[0]
    print(f"  [EXEC] Reset to initial state before executing plan")

    results = []
    current_plan = list(plan)
    corrections = 0
    api_calls = 0

    for i in range(min(max_steps, len(current_plan) + 5)):
        if not current_plan:
            break

        raw_action = current_plan.pop(0)
        # Parse "ACTION6 x=19,y=58" format
        action_name = raw_action.split()[0] if " " in raw_action else raw_action
        game_action = ACTION_MAP.get(action_name, GameAction.ACTION1)

        data = {}
        if game_action == GameAction.ACTION6:
            # Extract coordinates from plan string
            import re
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
        had_movement = len(diff.get("movements", [])) > 0

        step_info = {
            "step": i + 1,
            "action": action_name,
            "diff": diff["description"],
            "had_movement": had_movement,
            "state": obs.state.name if obs and obs.state else "?",
            "levels": obs.levels_completed if obs else 0,
        }
        results.append(step_info)
        print(f"  [EXEC] Step {i+1}: {action_name} → {diff['description'][:60]}")

        if save_dir:
            img = grid_to_image(grid, scale=4)
            img.save(Path(save_dir) / f"exec_{i+1:03d}_{action_name.lower()}.png")

        # Check terminal
        if obs and obs.state == GameState.WIN:
            print(f"  [EXEC] WIN at step {i+1}! Levels: {obs.levels_completed}")
            break
        elif obs and obs.state == GameState.GAME_OVER:
            print(f"  [EXEC] GAME OVER at step {i+1}")
            obs = env.reset()
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
            # Ask LLM for correction
            if corrections < 3:  # Max 3 corrections
                corrections += 1
                api_calls += 1
                corrected_plan = get_correction(
                    client, model, grid, plan, results, action_name, diff["description"]
                )
                if corrected_plan:
                    current_plan = corrected_plan
                    print(f"  [EXEC] Plan corrected: {corrected_plan[:5]}...")

        # If action had no effect for 3+ consecutive steps, ask for correction
        if (len(results) >= 3 and
            all(not r["had_movement"] for r in results[-3:]) and
            corrections < 3):
            corrections += 1
            api_calls += 1
            corrected_plan = get_correction(
                client, model, grid, plan, results, action_name, diff["description"]
            )
            if corrected_plan:
                current_plan = corrected_plan
                print(f"  [EXEC] Plan corrected after stagnation: {corrected_plan[:5]}...")

    levels = results[-1]["levels"] if results else 0
    deaths = sum(1 for r in results if r["state"] == "GAME_OVER")
    return {
        "steps": len(results),
        "levels": levels,
        "deaths": deaths,
        "corrections": corrections,
        "api_calls_execution": api_calls,
        "final_grid": grid,
    }


def get_correction(client, model, grid, original_plan, history, last_action, actual):
    """Ask LLM for plan correction."""
    completed = [r["action"] for r in history]
    text = CORRECTION_TEMPLATE.format(
        plan=original_plan[:20],
        completed=completed[-10:],
        last_action=last_action,
        expected="movement toward target",
        actual=actual,
    )

    content = [{"type": "text", "text": text}]
    b64 = grid_to_base64(grid, scale=2)
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
    })

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.3,
            max_completion_tokens=1000,
            timeout=120,
        )
        reply = response.choices[0].message.content or ""
        # Parse JSON
        start = reply.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(reply)):
                if reply[i] == "{": depth += 1
                elif reply[i] == "}":
                    depth -= 1
                    if depth == 0:
                        data = json.loads(reply[start:i+1])
                        return data.get("adjusted_plan", [])
    except Exception as e:
        print(f"  [CORRECTION] Error: {e}")
    return None


def parse_analysis(text):
    """Extract JSON from analysis response."""
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    return {"optimal_plan": ["ACTION3"] * 10, "error": "parse_failed"}


# ── Main ────────────────────────────────────────────────────────────

def run_game(game_id, arc, save_dir=None):
    """Run the full plan-then-execute pipeline on one game."""
    client = OpenAI(
        base_url=os.environ["AZURE_FOUNDRY_BASE_URL"],
        api_key=os.environ["AZURE_INFERENCE_CREDENTIAL"],
    )
    model = os.environ.get("AZURE_MODEL", "gpt-5.4")

    env = arc.make(game_id, render_mode=None)
    obs = env.reset()
    avail = list(obs.available_actions) if hasattr(obs, "available_actions") and obs.available_actions else []

    api_calls = 0

    # Phase 1: Random exploration (0 API calls)
    print(f"\n  ═══ PHASE 1: RANDOM EXPLORATION (no LLM) ═══")
    explore_data = random_explore(env, avail, n_steps=12, save_dir=save_dir)
    print(f"  Exploration: {explore_data['total_steps']} steps, "
          f"{explore_data['deaths']} deaths, {explore_data['levels']} levels")

    if explore_data["levels"] > 0:
        return {"levels": explore_data["levels"], "deaths": explore_data["deaths"],
                "api_calls": 0, "method": "lucky_exploration"}

    # Phase 2: Comprehensive analysis (1 API call)
    print(f"\n  ═══ PHASE 2: COMPREHENSIVE ANALYSIS (1 LLM call) ═══")
    analysis_text = analyze_exploration(client, model, explore_data)
    analysis = parse_analysis(analysis_text)
    api_calls += 1

    plan = analysis.get("optimal_plan", [])
    print(f"  [ANALYSIS] Goal: {analysis.get('goal_hypothesis', '?')}")
    print(f"  [ANALYSIS] Mechanics: {analysis.get('mechanics', '?')}")
    print(f"  [ANALYSIS] Plan ({len(plan)} actions): {plan[:10]}...")
    print(f"  [ANALYSIS] Confidence: {analysis.get('confidence', '?')}")

    # Phase 3: Execute plan (0-3 API calls for corrections)
    print(f"\n  ═══ PHASE 3: EXECUTE PLAN ═══")
    exec_result = execute_plan(env, client, model, plan, analysis,
                               max_steps=18, save_dir=save_dir)
    api_calls += exec_result["api_calls_execution"]

    return {
        "game_id": game_id,
        "levels": exec_result["levels"],
        "deaths": exec_result["deaths"],
        "explore_steps": explore_data["total_steps"],
        "exec_steps": exec_result["steps"],
        "total_steps": explore_data["total_steps"] + exec_result["steps"],
        "api_calls": api_calls,
        "corrections": exec_result["corrections"],
        "analysis": analysis,
        "method": "plan-then-execute",
    }


def main():
    games = ["g50t-5849a774", "sb26-7fbdac44"]
    if len(sys.argv) > 1:
        games = sys.argv[1:]

    print(f"=== E024: Plan-then-Execute ===")
    print(f"  Games: {games}")
    print()

    arc = arc_agi.Arcade()
    all_results = []

    for game_id in games:
        print(f"\n{'=' * 70}")
        print(f"  GAME: {game_id}")
        print(f"{'=' * 70}")
        start = time.time()
        try:
            result = run_game(
                game_id, arc,
                save_dir=f"experiments/E024-plan-execute/frames/{game_id}"
            )
            result["elapsed"] = round(time.time() - start, 1)
            all_results.append(result)
            print(f"\n  → {game_id}: {result['levels']} levels, "
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
    print(f"  SUMMARY — E024 Plan-then-Execute")
    print(f"{'=' * 70}")
    total_api = 0
    for r in all_results:
        gid = r.get("game_id", "?").split("-")[0]
        calls = r.get("api_calls", 0)
        total_api += calls
        print(f"  {gid}: levels={r['levels']}, deaths={r['deaths']}, "
              f"api_calls={calls}, time={r.get('elapsed', 0):.0f}s")
    print(f"\n  TOTAL API CALLS: {total_api}")
    print(f"  TOTAL LEVELS: {sum(r['levels'] for r in all_results)}")

    out = Path("experiments/E024-plan-execute")
    with open(out / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
