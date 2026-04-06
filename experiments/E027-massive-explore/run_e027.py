"""E027: Massive Exploration — understand the games before reasoning.

INSIGHT: All previous experiments (E019-E026) got 0 levels across ALL games.
We've been asking the LLM to plan after just 15 random actions. That's not
enough data. The LLM hypothesizes well but can't find objectives.

NEW APPROACH:
  Phase 1 (FREE): 500 random actions per game, 0 LLM calls
    - Systematic: test each action 10x in a row, then pairs, then random
    - Track GAME_OVERs, level completions, pixel change patterns
    - Build comprehensive action-effect profiles
  Phase 2 (1 LLM call): Send ALL data to LLM for deep analysis
    - Full action effect map with statistics
    - Key frames at state transitions
    - Ask for a 100-action strategic plan
  Phase 3 (FREE): Execute the plan
  Phase 4 (1 LLM call if needed): Reflect + replan

Total: 500+ game actions, 1-2 LLM calls per game
100% generalizable — works identically on all 25 games.
"""
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
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


# ── Phase 1: Massive systematic exploration (0 LLM calls) ──────────

def massive_explore(env, available_actions, total_steps=500):
    """Thorough systematic exploration. Zero LLM calls.

    Strategy:
    1. Test each action 10x in a row (understand individual effects)
    2. Test each pair of actions 3x (understand interactions)
    3. Fill remaining with random exploration
    """
    obs = env.reset()
    grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
    initial_grid = grid.copy()

    actions = [f"ACTION{a}" for a in available_actions if a != 0]
    if not actions:
        actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]

    # Build systematic exploration sequence
    sequence = []

    # Phase 1a: Each action 10x in a row
    for act in actions:
        sequence.extend([act] * 10)

    # Phase 1b: Each pair 3x (AB AB AB)
    for a in actions:
        for b in actions:
            if a != b:
                for _ in range(3):
                    sequence.extend([a, b])

    # Phase 1c: Fill remaining with random
    while len(sequence) < total_steps:
        sequence.append(random.choice(actions))
    sequence = sequence[:total_steps]

    # Execute and track everything
    log = []
    frames = [grid.copy()]
    key_frames = [(0, "initial", grid.copy())]  # (step, reason, grid)

    # Statistics
    action_stats = defaultdict(lambda: {
        "total_uses": 0,
        "cells_changed": [],
        "caused_game_over": 0,
        "caused_level": 0,
        "no_effect": 0,
        "movement_dirs": [],
    })

    deaths = 0
    levels = 0
    max_levels = 0
    consecutive_no_change = 0

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
        changed = diff.get("changed_cells", 0)

        # Track statistics
        stats = action_stats[action_name]
        stats["total_uses"] += 1
        stats["cells_changed"].append(changed)

        if changed == 0:
            stats["no_effect"] += 1
            consecutive_no_change += 1
        else:
            consecutive_no_change = 0

        # Track movement directions
        for m in diff.get("movements", []):
            dx = m["to_x"] - m["from_x"]
            dy = m["to_y"] - m["from_y"]
            stats["movement_dirs"].append((dx, dy))

        entry = {
            "step": i + 1,
            "action": action_name + (f" x={data['x']},y={data['y']}" if data else ""),
            "changed_cells": changed,
            "description": diff["description"][:80],
            "movements": diff.get("movements", []),
            "state": obs.state.name if obs and obs.state else "?",
            "levels": obs.levels_completed if obs else 0,
        }
        log.append(entry)

        # Track state transitions
        if obs and obs.state == GameState.GAME_OVER:
            deaths += 1
            stats["caused_game_over"] += 1
            key_frames.append((i + 1, f"GAME_OVER after {action_name}", prev_grid.copy()))
            obs = env.reset()
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
            key_frames.append((i + 1, "after_reset", grid.copy()))

        if obs and obs.state == GameState.WIN:
            stats["caused_level"] += 1
            cur_levels = obs.levels_completed if obs else 0
            if cur_levels > max_levels:
                max_levels = cur_levels
                key_frames.append((i + 1, f"WIN level {cur_levels}", grid.copy()))

        # Save key frames at intervals
        if (i + 1) % 100 == 0:
            key_frames.append((i + 1, f"checkpoint_{i+1}", grid.copy()))

        # Print progress every 100 steps
        if (i + 1) % 100 == 0:
            print(f"    Step {i+1}/{total_steps}: {deaths} deaths, {max_levels} levels")

    # Compile action statistics summary
    action_summary = {}
    for act_name, stats in action_stats.items():
        uses = stats["total_uses"]
        changes = stats["cells_changed"]
        avg_change = sum(changes) / len(changes) if changes else 0

        # Determine dominant movement direction
        dirs = stats["movement_dirs"]
        if dirs:
            avg_dx = sum(d[0] for d in dirs) / len(dirs)
            avg_dy = sum(d[1] for d in dirs) / len(dirs)
            if abs(avg_dx) > abs(avg_dy):
                dom_dir = "RIGHT" if avg_dx > 0 else "LEFT"
            elif abs(avg_dy) > 0:
                dom_dir = "DOWN" if avg_dy > 0 else "UP"
            else:
                dom_dir = "NONE"
        else:
            dom_dir = "NONE"
            avg_dx, avg_dy = 0, 0

        action_summary[act_name] = {
            "uses": uses,
            "avg_cells_changed": round(avg_change, 1),
            "no_effect_pct": round(100 * stats["no_effect"] / max(uses, 1)),
            "game_over_pct": round(100 * stats["caused_game_over"] / max(uses, 1)),
            "levels_caused": stats["caused_level"],
            "dominant_direction": dom_dir,
            "avg_movement": (round(avg_dx, 1), round(avg_dy, 1)),
        }

    return {
        "log": log,
        "initial_grid": initial_grid,
        "final_grid": grid,
        "key_frames": key_frames,
        "action_summary": action_summary,
        "initial_description": describe_frame(initial_grid),
        "total_steps": len(log),
        "deaths": deaths,
        "levels": max_levels,
        "available_actions": actions,
    }


# ── Phase 2: LLM analysis (1 call) ─────────────────────────────────

ANALYSIS_PROMPT = """\
You are playing an ARC-AGI-3 game — a 64x64 visual puzzle with unknown rules.
You have NO instructions. You must figure out the game by observing.

You just completed EXTENSIVE exploration: {total_steps} actions total.
Here's everything you know:

=== ACTION EFFECT SUMMARY ===
{action_summary}

=== INITIAL FRAME DESCRIPTION ===
{initial_description}

=== KEY EVENTS (deaths, level completions, big changes) ===
{key_events}

=== SAMPLE EXPLORATION LOG (most informative steps) ===
{sample_log}

Images show: initial frame, post-exploration frame, and frames at key state changes.

YOUR TASK:
1. What do you control? What does each action do?
2. What KILLED you (if anything)? What should you AVOID?
3. What completed levels (if anything)? What should you REPEAT?
4. What's the game objective? Be specific — what pattern do you see?
5. Create a DETAILED 100-action plan to achieve the objective.

The plan should be AGGRESSIVE — you have extensive data, now ACT on it.
If you haven't completed any levels yet, try BOLD combinations.

Respond with JSON:
{{
  "game_understanding": "comprehensive description of what you learned",
  "controlled_object": "what you control",
  "action_effects": {{"ACTION1": "effect with evidence", ...}},
  "dangers": "what causes GAME_OVER",
  "goal_hypothesis": "your best guess at the objective",
  "strategy": "high-level approach",
  "plan": ["ACTION1", "ACTION2", ...],
  "plan_reasoning": "why this specific sequence should work"
}}

Give a plan of 80-100 actions. Be PRECISE and STRATEGIC, not random.
"""

REFLECT_PROMPT = """\
You are playing an ARC-AGI-3 game. Your previous plan was executed.

PREVIOUS UNDERSTANDING:
{previous_understanding}

EXECUTION RESULTS:
{execution_log}

Levels completed: {levels}. Deaths: {deaths}.

Images show the current game state.

What went wrong? What should you try completely differently?
Give a NEW 80-100 action plan.

Respond with JSON:
{{
  "reflection": "what you learned",
  "revised_hypothesis": "updated goal hypothesis",
  "plan": ["ACTION1", "ACTION2", ...],
  "plan_reasoning": "why this new approach"
}}
"""


def call_llm(client, model, prompt, frames, temperature=0.5):
    """Make one LLM call with text + frame images."""
    content = [{"type": "text", "text": prompt}]

    for label, frame in frames:
        b64 = grid_to_base64(frame, scale=4)
        content.append({"type": "text", "text": f"\n--- {label} ---"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })

    messages = [{"role": "user", "content": content}]

    kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": 4000,
        "timeout": 180,
    }
    # Some models don't support temperature != 1
    if model in ("gpt-5.4", "gpt-5.4-pro"):
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def parse_json(text):
    """Extract JSON from LLM response."""
    start = text.find("{")
    if start == -1:
        return {"plan": ["ACTION1"] * 50, "error": "no_json"}
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
    return {"plan": ["ACTION1"] * 50, "error": "parse_failed"}


def format_key_events(log):
    """Extract the most interesting events from the log."""
    events = []
    for entry in log:
        if entry["state"] in ("GAME_OVER", "WIN"):
            events.append(
                f"Step {entry['step']}: {entry['action']} → {entry['state']} "
                f"({entry['description'][:60]})"
            )
        elif entry["changed_cells"] > 20:
            events.append(
                f"Step {entry['step']}: {entry['action']} → BIG CHANGE "
                f"({entry['changed_cells']} cells: {entry['description'][:60]})"
            )
    if not events:
        events.append("No deaths, no level completions, no major changes.")
    return "\n".join(events[:50])


def format_sample_log(log, max_entries=60):
    """Select the most informative log entries."""
    # Include: first uses of each action, state changes, big changes, and some random
    selected = set()

    # First occurrence of each action
    seen_actions = set()
    for i, entry in enumerate(log):
        act = entry["action"].split()[0]
        if act not in seen_actions:
            seen_actions.add(act)
            selected.add(i)

    # State transitions
    for i, entry in enumerate(log):
        if entry["state"] not in ("NOT_FINISHED", "?"):
            selected.add(i)

    # Biggest changes
    by_change = sorted(range(len(log)), key=lambda i: log[i]["changed_cells"], reverse=True)
    for i in by_change[:15]:
        selected.add(i)

    # Some from each 100-step chunk
    for chunk_start in range(0, len(log), 100):
        chunk = list(range(chunk_start, min(chunk_start + 100, len(log))))
        for idx in random.sample(chunk, min(5, len(chunk))):
            selected.add(idx)

    selected = sorted(selected)[:max_entries]

    lines = []
    for i in selected:
        entry = log[i]
        line = f"Step {entry['step']}: {entry['action']} → {entry['description'][:80]}"
        if entry.get("movements"):
            for m in entry["movements"][:2]:
                line += f"\n  move: {m['color_name']} ({m['from_x']},{m['from_y']})→({m['to_x']},{m['to_y']})"
        if entry["state"] not in ("NOT_FINISHED", "?"):
            line += f"\n  *** {entry['state']} ***"
        lines.append(line)
    return "\n".join(lines)


# ── Phase 3: Execute plan ──────────────────────────────────────────

def execute_plan(env, plan, grid):
    """Execute a sequence of actions."""
    log = []

    for i, raw_action in enumerate(plan):
        action_name = raw_action.split()[0] if " " in raw_action else raw_action
        if action_name not in ACTION_MAP:
            continue
        game_action = ACTION_MAP[action_name]

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
            "description": diff["description"][:80],
            "movements": diff.get("movements", []),
            "state": obs.state.name if obs and obs.state else "?",
            "levels": obs.levels_completed if obs else 0,
        }
        log.append(entry)

        if (i + 1) % 20 == 0:
            print(f"    Execute step {i+1}/{len(plan)}: {entry['description'][:50]}")

        if obs and obs.state == GameState.WIN:
            print(f"    WIN at step {i+1}! Levels: {obs.levels_completed}")
            break

        if obs and obs.state == GameState.GAME_OVER:
            print(f"    GAME OVER at step {i+1}")
            obs = env.reset()
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

    levels = log[-1]["levels"] if log else 0
    deaths = sum(1 for e in log if e["state"] == "GAME_OVER")
    return log, grid, levels, deaths


# ── Main pipeline ─────────────────────────────────────────────────────

def run_game(game_id, arc, model="gpt-5.4", save_dir=None):
    """Full pipeline: massive explore → LLM analysis → execute → reflect."""
    client = OpenAI(
        base_url=os.environ["AZURE_FOUNDRY_BASE_URL"],
        api_key=os.environ["AZURE_INFERENCE_CREDENTIAL"],
    )

    env = arc.make(game_id, render_mode=None)
    obs = env.reset()
    avail = list(obs.available_actions) if hasattr(obs, "available_actions") and obs.available_actions else []

    api_calls = 0
    best_levels = 0

    # ── Phase 1: Massive exploration ──
    print(f"\n  Phase 1: Massive systematic exploration (500 steps, 0 LLM calls)")
    explore = massive_explore(env, avail, total_steps=500)
    grid = explore["final_grid"]

    print(f"  Exploration complete: {explore['deaths']} deaths, {explore['levels']} levels")
    print(f"  Action summary:")
    for act, stats in sorted(explore["action_summary"].items()):
        print(f"    {act}: avg_change={stats['avg_cells_changed']}, "
              f"no_effect={stats['no_effect_pct']}%, "
              f"game_over={stats['game_over_pct']}%, "
              f"dir={stats['dominant_direction']}")

    if explore["levels"] > 0:
        print(f"  BLIND LUCK: {explore['levels']} levels from random play!")
        best_levels = explore["levels"]
        # Still continue — try to get more

    # ── Phase 2: LLM analysis ──
    print(f"\n  Phase 2: LLM analysis (1 call)")

    action_summary_text = json.dumps(explore["action_summary"], indent=2)
    key_events_text = format_key_events(explore["log"])
    sample_log_text = format_sample_log(explore["log"])

    prompt = ANALYSIS_PROMPT.format(
        total_steps=explore["total_steps"],
        action_summary=action_summary_text,
        initial_description=explore["initial_description"],
        key_events=key_events_text,
        sample_log=sample_log_text,
    )

    # Select frames to show the LLM
    frames_to_show = []
    for step, reason, frame in explore["key_frames"]:
        if len(frames_to_show) < 4:  # Max 4 frames to save tokens
            frames_to_show.append((f"{reason} (step {step})", frame))

    # Always include initial and final
    if len(frames_to_show) == 0:
        frames_to_show = [
            ("Initial", explore["initial_grid"]),
            ("Final", explore["final_grid"]),
        ]

    print(f"  Calling LLM...")
    reply = call_llm(client, model, prompt, frames_to_show)
    analysis = parse_json(reply)
    api_calls += 1

    print(f"  Goal hypothesis: {analysis.get('goal_hypothesis', '?')[:120]}")
    print(f"  Strategy: {analysis.get('strategy', '?')[:120]}")
    plan = analysis.get("plan", [])
    print(f"  Plan: {len(plan)} actions")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "analysis.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)

    # ── Phase 3: Execute plan ──
    if plan:
        print(f"\n  Phase 3: Executing {len(plan)}-action plan")
        # Reset to start fresh with the plan
        obs = env.reset()
        if obs and obs.frame:
            grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

        exec_log, grid, levels, deaths = execute_plan(env, plan, grid)
        if levels > best_levels:
            best_levels = levels

        print(f"  Plan result: {levels} levels, {deaths} deaths")

        # ── Phase 4: Reflect if needed ──
        if best_levels == 0 and len(plan) > 0:
            print(f"\n  Phase 4: Reflection (1 call)")

            exec_text = "\n".join(
                f"Step {e['step']}: {e['action']} → {e['description'][:60]}"
                + (f" [{e['state']}]" if e["state"] not in ("NOT_FINISHED", "?") else "")
                for e in exec_log
            )

            prev_understanding = json.dumps({
                k: analysis.get(k) for k in
                ["game_understanding", "goal_hypothesis", "action_effects", "strategy"]
                if k in analysis
            }, indent=2)

            reflect_prompt = REFLECT_PROMPT.format(
                previous_understanding=prev_understanding,
                execution_log=exec_text[:3000],
                levels=levels,
                deaths=deaths,
            )

            print(f"  Calling LLM for reflection...")
            reply = call_llm(client, model, reflect_prompt, [("Current frame", grid)])
            analysis2 = parse_json(reply)
            api_calls += 1

            plan2 = analysis2.get("plan", [])
            print(f"  Reflection: {analysis2.get('reflection', '?')[:120]}")
            print(f"  New plan: {len(plan2)} actions")

            if plan2:
                print(f"\n  Phase 5: Executing revised plan")
                obs = env.reset()
                if obs and obs.frame:
                    grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

                exec_log2, grid, levels2, deaths2 = execute_plan(env, plan2, grid)
                if levels2 > best_levels:
                    best_levels = levels2
                print(f"  Revised plan result: {levels2} levels, {deaths2} deaths")

    return {
        "game_id": game_id,
        "levels": best_levels,
        "explore_deaths": explore["deaths"],
        "explore_levels": explore["levels"],
        "api_calls": api_calls,
        "total_explore_steps": explore["total_steps"],
        "action_summary": explore["action_summary"],
        "final_analysis": analysis,
        "method": "massive-explore",
    }


def run_random_baseline(arc, games, steps_per_game=500):
    """Pure random baseline — zero LLM calls.

    Just smash random actions and see if any game is solvable by chance.
    This is the lowest possible bar.
    """
    print(f"\n{'='*70}")
    print(f"  RANDOM BASELINE — {steps_per_game} random actions, 0 LLM calls")
    print(f"{'='*70}\n")

    results = []
    for game_id in games:
        env = arc.make(game_id, render_mode=None)
        obs = env.reset()
        avail = list(obs.available_actions) if hasattr(obs, "available_actions") and obs.available_actions else []
        actions = [f"ACTION{a}" for a in avail if a != 0]
        if not actions:
            actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]

        grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
        deaths = 0
        max_levels = 0

        for step in range(steps_per_game):
            action_name = random.choice(actions)
            game_action = ACTION_MAP.get(action_name, GameAction.ACTION1)
            data = {}
            if game_action == GameAction.ACTION6:
                data = {"x": random.randint(5, 58), "y": random.randint(5, 58)}

            obs = env.step(game_action, data=data)
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

            if obs and obs.state == GameState.GAME_OVER:
                deaths += 1
                obs = env.reset()
                if obs and obs.frame:
                    grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]

            if obs and obs.state == GameState.WIN:
                cur_levels = obs.levels_completed if obs else 0
                if cur_levels > max_levels:
                    max_levels = cur_levels

        gid = game_id.split("-")[0]
        n_actions = len(actions)
        print(f"  {gid}: levels={max_levels}, deaths={deaths}, "
              f"actions={n_actions}, steps={steps_per_game}")

        results.append({
            "game_id": game_id,
            "levels": max_levels,
            "deaths": deaths,
            "num_actions": n_actions,
            "steps": steps_per_game,
        })

    total_levels = sum(r["levels"] for r in results)
    total_deaths = sum(r["deaths"] for r in results)
    print(f"\n  TOTAL: {total_levels} levels, {total_deaths} deaths across {len(games)} games")
    return results


def main():
    # Default: run random baseline on ALL games first, then LLM on promising ones
    mode = "baseline"  # "baseline" or "full"
    model = "gpt-5.4"

    if "--full" in sys.argv:
        mode = "full"
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model = sys.argv[idx + 1]

    # Game list from CLI or all games
    cli_games = [g for g in sys.argv[1:] if not g.startswith("--")]

    print(f"=== E027: Massive Exploration ===")
    print(f"  Mode: {mode}")
    print(f"  Model: {model}")
    print()

    arc = arc_agi.Arcade()

    if not cli_games:
        # Get all game IDs from the arcade
        envs = arc.get_environments()
        seen = set()
        all_games = []
        for e in envs:
            gid = e.game_id if hasattr(e, "game_id") else str(e)
            if gid not in seen:
                seen.add(gid)
                all_games.append(gid)
        cli_games = all_games
        print(f"  Found {len(cli_games)} unique games")

    if mode == "baseline":
        # Pure random baseline — costs $0
        results = run_random_baseline(arc, cli_games, steps_per_game=500)

        out = Path("experiments/E027-massive-explore")
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "baseline_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Identify games with most deaths (most interactive / most learnable)
        by_deaths = sorted(results, key=lambda r: r["deaths"], reverse=True)
        print(f"\n  Games ranked by interactivity (deaths):")
        for r in by_deaths[:10]:
            gid = r["game_id"].split("-")[0]
            print(f"    {gid}: {r['deaths']} deaths, {r['levels']} levels")

        # If any games completed levels, those are our targets
        with_levels = [r for r in results if r["levels"] > 0]
        if with_levels:
            print(f"\n  GAMES WITH LEVEL COMPLETIONS:")
            for r in with_levels:
                gid = r["game_id"].split("-")[0]
                print(f"    {gid}: {r['levels']} levels!")

    elif mode == "full":
        all_results = []
        total_api = 0

        for game_id in cli_games:
            gid = game_id.split("-")[0]
            print(f"\n{'='*70}")
            print(f"  GAME: {game_id}")
            print(f"{'='*70}")
            start = time.time()
            try:
                result = run_game(
                    game_id, arc, model=model,
                    save_dir=f"experiments/E027-massive-explore/frames/{gid}",
                )
                result["elapsed"] = round(time.time() - start, 1)
                result["model"] = model
                all_results.append(result)
                total_api += result.get("api_calls", 0)
                print(f"\n  -> {gid}: {result['levels']} levels, "
                      f"{result['api_calls']} API calls, {result['elapsed']:.0f}s")
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results.append({
                    "game_id": game_id, "levels": 0,
                    "api_calls": 0, "error": str(e),
                    "elapsed": round(time.time() - start, 1),
                })

        print(f"\n{'='*70}")
        print(f"  SUMMARY — E027 Massive Exploration")
        print(f"{'='*70}")
        for r in all_results:
            gid = r.get("game_id", "?").split("-")[0]
            print(f"  {gid}: levels={r['levels']}, api_calls={r.get('api_calls', 0)}, "
                  f"time={r.get('elapsed', 0):.0f}s")
        print(f"\n  TOTAL API CALLS: {total_api}")
        print(f"  TOTAL LEVELS: {sum(r['levels'] for r in all_results)}")

        out = Path("experiments/E027-massive-explore")
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
