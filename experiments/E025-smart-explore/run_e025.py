"""E025: Smart Exploration + Plan-Execute with gpt-5.3-chat.

KEY IMPROVEMENTS over E024:
1. STRUCTURED exploration: test each action 3x, track avatar trajectory
2. CHEAPER model: gpt-5.3-chat (budget nearly exhausted for gpt-5.4)
3. MULTI-ROUND: if first plan fails, explore more + replan (up to 2 rounds)
4. GAME-TYPE ADAPTIVE: movement games vs click games get different exploration
5. BETTER ANALYSIS PROMPT: include avatar trajectory, map landmarks, clear action map

API CALL ESTIMATE: ~3-6 per game (1 analysis + 0-2 corrections + possible replan)
Test: ls20, ft09, m0r0 (3 different game types)
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


# ── Phase 1: Structured Exploration ──────────────────────────────────

def find_avatar(prev_grid, curr_grid, movements):
    """Identify the avatar from movement data — the object the player controls."""
    if not movements:
        return None
    # Avatar is usually the smallest moving object, or the one that moves consistently
    # Return the first movement's color and position
    m = movements[0]
    return {"color": m["color_name"], "x": m["to_x"], "y": m["to_y"]}


def structured_explore(env, available_actions, n_steps=20, save_dir=None):
    """Maze-aware exploration: probe walls, map corridors, track avatar."""
    obs = env.reset()
    grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
    initial_grid = grid.copy()

    actions = [f"ACTION{a}" for a in available_actions if a != 0]
    if not actions:
        actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]

    is_click_game = "ACTION6" in actions and len(actions) <= 2

    log = []
    frames = [grid.copy()]
    avatar_trail = []  # All avatar positions
    action_effects = {}  # action -> list of (dx, dy)
    walls = []  # List of (x, y, action) = wall hit at position trying action
    avatar_pos = None  # Current known avatar position
    step_count = 0

    def do_step(action_name, data=None):
        nonlocal grid, avatar_pos, step_count
        data = data or {}
        prev_grid = grid.copy()
        game_action = ACTION_MAP.get(action_name, GameAction.ACTION1)
        o = env.step(game_action, data=data)

        if o and o.frame:
            grid = np.array(o.frame[0]) if isinstance(o.frame[0], list) else o.frame[0]

        diff = compute_diff(prev_grid, grid)
        movements = diff.get("movements", [])
        step_count += 1

        moved = False
        if movements:
            m = movements[0]
            avatar_pos = (m["to_x"], m["to_y"])
            avatar_trail.append({"step": step_count, "action": action_name,
                                 "x": m["to_x"], "y": m["to_y"], "color": m["color_name"]})
            if action_name not in action_effects:
                action_effects[action_name] = []
            action_effects[action_name].append({"dx": m["dx"], "dy": m["dy"], "color": m["color_name"]})
            moved = True
        elif avatar_pos and diff.get("changed_cells", 0) <= 5:
            # Wall hit — avatar didn't move but resource bar changed
            walls.append({"x": avatar_pos[0], "y": avatar_pos[1], "action": action_name})

        entry = {
            "step": step_count, "action": action_name, "data": data,
            "diff_description": diff["description"],
            "changed_cells": diff.get("changed_cells", 0),
            "movements": movements, "swaps": diff.get("swaps", []),
            "state": o.state.name if o and o.state else "?",
            "levels": o.levels_completed if o else 0, "moved": moved,
        }
        log.append(entry)
        frames.append(grid.copy())

        status = f"step {step_count}: {action_name}"
        if data:
            status += f" x={data.get('x')},y={data.get('y')}"
        status += " → MOVED" if moved else " → BLOCKED"
        if avatar_pos:
            status += f" @({avatar_pos[0]},{avatar_pos[1]})"
        print(f"  [EXPLORE] {status}")

        if o and o.state == GameState.GAME_OVER:
            print(f"  [EXPLORE] GAME OVER at step {step_count}")
            o2 = env.reset()
            if o2 and o2.frame:
                grid = np.array(o2.frame[0]) if isinstance(o2.frame[0], list) else o2.frame[0]
            avatar_pos = None
            frames.append(grid.copy())

        return o, moved

    if is_click_game:
        # Click game: click on each detected object
        sequence = _click_exploration_sequence(initial_grid, n_steps)
        for action_name, data in sequence:
            o, _ = do_step(action_name, data)
            if o and o.state == GameState.WIN:
                break
    else:
        # MOVEMENT GAME: Intelligent maze probing
        # Phase A: Determine action map (test each action once)
        for act in actions:
            do_step(act)

        # Phase B: Probe maze — go each direction until wall, then try perpendicular
        directions = list(actions)  # e.g. [ACTION1..ACTION4]
        remaining = n_steps - len(actions)

        for probe_dir in directions:
            if remaining <= 0:
                break
            # Move in this direction until blocked
            for _ in range(4):
                if remaining <= 0:
                    break
                o, moved = do_step(probe_dir)
                remaining -= 1
                if not moved:
                    break  # Hit wall, try next direction
                if o and o.state in (GameState.WIN, GameState.GAME_OVER):
                    break

        # Phase C: Navigate back through different paths
        # Try to find paths the direct probing missed
        while remaining > 0:
            # Pick a random direction
            act = random.choice(directions)
            o, moved = do_step(act)
            remaining -= 1
            if o and o.state == GameState.WIN:
                break

    # Build action summary
    action_summary = {}
    for act, effects in action_effects.items():
        if not effects:
            action_summary[act] = "NO EFFECT"
        else:
            dxs = [e["dx"] for e in effects]
            dys = [e["dy"] for e in effects]
            avg_dx = sum(dxs) / len(dxs)
            avg_dy = sum(dys) / len(dys)
            color = effects[0]["color"]
            # Determine direction label
            dir_label = ""
            if abs(avg_dx) > abs(avg_dy):
                dir_label = "RIGHT" if avg_dx > 0 else "LEFT"
            elif abs(avg_dy) > abs(avg_dx):
                dir_label = "DOWN" if avg_dy > 0 else "UP"
            action_summary[act] = (
                f"{dir_label}: moves {color} ({avg_dx:+.0f},{avg_dy:+.0f}) per step, "
                f"{len(effects)} observations"
            )

    # For actions without effects
    for act in actions:
        if act not in action_summary:
            action_summary[act] = "NO EFFECT (no avatar movement detected)"

    # Build wall summary
    wall_summary = []
    for w in walls:
        dir_name = action_summary.get(w["action"], w["action"]).split(":")[0]
        wall_summary.append(f"WALL at ({w['x']},{w['y']}) going {dir_name} ({w['action']})")

    # Save key frames
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames):
            if idx % max(1, len(frames) // 8) == 0 or idx == len(frames) - 1:
                img = grid_to_image(frame, scale=4)
                img.save(save_path / f"explore_{idx:03d}.png")

    return {
        "log": log,
        "initial_frame": initial_grid,
        "final_frame": grid,
        "key_frames": [frames[0], frames[len(frames)//2], frames[-1]],
        "initial_description": describe_frame(initial_grid),
        "final_description": describe_frame(grid),
        "total_steps": step_count,
        "deaths": sum(1 for e in log if e["state"] == "GAME_OVER"),
        "levels": log[-1]["levels"] if log else 0,
        "avatar_trail": avatar_trail,
        "action_summary": action_summary,
        "action_effects": action_effects,
        "walls": walls,
        "wall_summary": wall_summary,
        "is_click_game": is_click_game,
    }


def _movement_exploration_sequence(actions, n_steps):
    """Build exploration sequence for movement games."""
    sequence = []

    # Phase A: test each action 3 times
    for act in actions:
        for _ in range(3):
            sequence.append((act, {}))

    # Phase B: explore by alternating — try to cover the map
    # Go in one direction several times, then another
    remaining = n_steps - len(sequence)
    if remaining > 0:
        # Zigzag: go one way 3x, then perpendicular 2x, repeat
        for _ in range(remaining):
            sequence.append((random.choice(actions), {}))

    return sequence[:n_steps]


def _click_exploration_sequence(grid, n_steps):
    """Build exploration sequence for click games — click on actual objects."""
    # Find all non-background object centers
    unique_colors = np.unique(grid)
    bg_color = 0  # Usually black/near-black

    targets = []
    for color in unique_colors:
        if color == bg_color or color == 15:  # Skip bg and near-black
            continue
        positions = np.argwhere(grid == color)
        if len(positions) > 0 and len(positions) < 500:  # Not background
            center_y = int(np.mean(positions[:, 0]))
            center_x = int(np.mean(positions[:, 1]))
            targets.append({"x": center_x, "y": center_y, "color": int(color)})

    sequence = []

    # Click on each detected object
    for t in targets[:n_steps // 2]:
        sequence.append(("ACTION6", {"x": t["x"], "y": t["y"]}))

    # Then try ACTION7 after some clicks
    if len(sequence) > 2:
        sequence.insert(len(sequence) // 2, ("ACTION7", {}))
        sequence.append(("ACTION7", {}))

    # Fill remaining with random clicks
    while len(sequence) < n_steps:
        t = random.choice(targets) if targets else {"x": 32, "y": 32}
        sequence.append(("ACTION6", {"x": t["x"], "y": t["y"]}))

    return sequence[:n_steps]


# ── Phase 2: LLM Analysis ────────────────────────────────────────────

ANALYSIS_PROMPT = """\
You are analyzing exploration data from an ARC-AGI-3 game — a 64x64 visual puzzle.

You receive:
1. Three frames (initial, mid, final) as images
2. An AUTHORITATIVE action map (measured from actual pixel data — trust it completely)
3. Avatar trajectory showing where the controlled object moved
4. WALL DATA: positions where movement was blocked (the game has walls/corridors!)
5. A complete action log

Your job: Figure out the GOAL and produce an OPTIMAL ACTION PLAN that navigates
AROUND walls. This is likely a MAZE or room-based game.

CRITICAL RULES:
- The action map is AUTHORITATIVE. Do NOT override it.
- The game has WALLS. If exploration shows the avatar was blocked at position (X,Y)
  going UP, you MUST plan a path AROUND that wall.
- Look at the images carefully to see the maze/room structure. Corridors are
  typically a different shade from walls.
- Count steps precisely: if each step moves 5 pixels and you need to go 35 pixels,
  that's exactly 7 steps.

Focus on:
1. GOAL: What's the target? Look for distinct colored objects, especially ones
   that are isolated or in a special chamber.
2. MAZE STRUCTURE: From the images, identify corridors, rooms, and openings.
3. PATH PLANNING: Plan a path that follows corridors and avoids walls.
4. PLAN: Give me an EXACT sequence of actions. Be generous — 20-35 actions.

Respond with JSON:
{
  "goal_hypothesis": "what I think wins, with specific evidence",
  "target_position": {"x": N, "y": N},
  "current_position": {"x": N, "y": N},
  "maze_description": "describe the corridors and rooms you see in the image",
  "mechanics": "how the game works",
  "optimal_plan": ["ACTION1", "ACTION3", "ACTION1", ...],
  "plan_reasoning": "step-by-step navigation through corridors, with wall avoidance",
  "confidence": "low|medium|high"
}

Make the plan LONG ENOUGH. Better to overshoot than fall short.
"""


def analyze_exploration(client, model, explore_data):
    """Single comprehensive LLM call with all exploration data."""
    text_parts = [
        "=== EXPLORATION DATA ===",
        f"Total steps: {explore_data['total_steps']}",
        f"Deaths: {explore_data['deaths']}",
        f"Levels completed: {explore_data['levels']}",
        f"Game type: {'CLICK' if explore_data['is_click_game'] else 'MOVEMENT'}",
        "",
        "=== AUTHORITATIVE ACTION MAP (measured from pixel data) ===",
    ]

    for act, desc in explore_data["action_summary"].items():
        text_parts.append(f"  {act}: {desc}")

    if explore_data["avatar_trail"]:
        text_parts.extend([
            "",
            "=== AVATAR TRAJECTORY ===",
        ])
        for t in explore_data["avatar_trail"]:
            text_parts.append(
                f"  Step {t['step']}: {t['action']} → avatar ({t['color']}) at ({t['x']},{t['y']})"
            )

    if explore_data.get("wall_summary"):
        text_parts.extend([
            "",
            "=== WALLS / BLOCKED DIRECTIONS ===",
            "These are positions where movement was BLOCKED (avatar could not move):",
        ])
        for w in explore_data["wall_summary"]:
            text_parts.append(f"  {w}")

    text_parts.extend([
        "",
        "=== INITIAL FRAME DESCRIPTION ===",
        explore_data["initial_description"],
        "",
        "=== COMPLETE ACTION LOG ===",
    ])

    for entry in explore_data["log"]:
        line = f"Step {entry['step']}: {entry['action']}"
        if entry.get("data"):
            line += f" x={entry['data'].get('x')},y={entry['data'].get('y')}"
        line += f" → {entry['diff_description']}"
        text_parts.append(line)
        for m in entry["movements"]:
            text_parts.append(
                f"  movement: {m['color_name']} ({m['from_x']},{m['from_y']})→"
                f"({m['to_x']},{m['to_y']}) delta=({m['dx']},{m['dy']})"
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
        label = ["Initial", "Mid-exploration", "Final"][i]
        b64 = grid_to_base64(frame, scale=3)  # Slightly larger for better visibility
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
    kwargs = dict(
        model=model,
        messages=messages,
        max_completion_tokens=3000,
        timeout=180,
    )
    # Some models don't support temperature != 1
    if "5.4" in model or "claude" in model:
        kwargs["temperature"] = 0.3
    response = client.chat.completions.create(**kwargs)
    reply = response.choices[0].message.content or ""
    print(f"  [ANALYSIS] Got {len(reply)} chars")
    return reply


# ── Phase 3: Execute Plan ─────────────────────────────────────────────

CORRECTION_TEMPLATE = (
    "You are executing a plan in an ARC-AGI-3 game. Something went wrong.\n\n"
    "Original plan: {plan}\n"
    "Completed so far: {completed}\n"
    "Last action: {last_action}\n"
    "Result: {actual}\n"
    "Deaths so far: {deaths}\n\n"
    "Look at the current frame. Reassess: where is the avatar? Where should it go?\n"
    "Give me a new action sequence to reach the goal.\n\n"
    'Respond with JSON: {{"assessment": "what happened", '
    '"adjusted_plan": ["ACTION1", "ACTION2", ...], '
    '"reasoning": "why this new plan works"}}'
)


def find_avatar_pos(prev_grid, curr_grid, avatar_color=None):
    """Find avatar position by tracking a specific color's movement.
    Returns (x, y, color) or None if no movement detected for that color."""
    diff = compute_diff(prev_grid, curr_grid)
    movements = diff.get("movements", [])
    if not movements:
        return None

    if avatar_color is not None:
        # Look for specific color
        for m in movements:
            if m["color_name"] == avatar_color:
                return (m["to_x"], m["to_y"], m["color_name"])
        # If exact color not found, check if it's close
        # Fall through to first movement as backup

    # Return first movement (avatar detection)
    m = movements[0]
    return (m["to_x"], m["to_y"], m["color_name"])


def greedy_navigate(env, target_x, target_y, action_map, max_steps=50, save_dir=None):
    """DFS-style greedy navigation toward target. NO LLM calls.

    Uses visited-set to avoid oscillation. When all preferred directions
    lead to visited positions, tries any unvisited direction. Tracks
    the avatar by its color (not just first movement).
    """
    obs = env.step(GameAction.RESET, data={})
    if obs and obs.frame:
        grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
    else:
        obs = env.reset()
        grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
    print(f"  [NAV] Reset to initial state, target=({target_x},{target_y})")

    # Build direction -> action mapping
    dir_to_action = {}
    for act, desc in action_map.items():
        if "UP" in desc:
            dir_to_action["UP"] = act
        elif "DOWN" in desc:
            dir_to_action["DOWN"] = act
        elif "LEFT" in desc:
            dir_to_action["LEFT"] = act
        elif "RIGHT" in desc:
            dir_to_action["RIGHT"] = act

    results = []
    avatar_pos = None
    avatar_color = None  # Track specific color
    visited = set()
    total_deaths = 0
    step_num = 0

    def try_action(action_name):
        nonlocal grid, obs, avatar_pos, avatar_color, step_num
        game_action = ACTION_MAP.get(action_name, GameAction.ACTION1)
        prev_grid = grid.copy()
        obs = env.step(game_action, data={})
        step_num += 1
        if obs and obs.frame:
            grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
        result = find_avatar_pos(prev_grid, grid, avatar_color)
        if result:
            x, y, color = result
            avatar_pos = (x, y)
            if avatar_color is None:
                avatar_color = color
                print(f"  [NAV] Avatar detected: {color}")
            return True
        return False

    def distance(x, y):
        return abs(x - target_x) + abs(y - target_y)

    # First move: find avatar
    for act in action_map:
        if try_action(act):
            dist = distance(*avatar_pos)
            results.append({"step": step_num, "action": act, "had_movement": True,
                            "state": obs.state.name if obs else "?",
                            "levels": obs.levels_completed if obs else 0})
            print(f"  [NAV] Step {step_num}: {act} → @{avatar_pos} dist={dist} ✓")
            break

    if avatar_pos is None:
        print(f"  [NAV] Could not find avatar")
        return {"steps": 0, "levels": 0, "deaths": 0, "corrections": 0,
                "api_calls_execution": 0, "final_grid": grid}

    stuck_count = 0

    while step_num < max_steps:
        visited.add(avatar_pos)
        dx = target_x - avatar_pos[0]
        dy = target_y - avatar_pos[1]

        # Build preference order: toward target first, then unvisited
        preferred = []
        if abs(dx) >= abs(dy):
            if dx > 0 and "RIGHT" in dir_to_action:
                preferred.append(("RIGHT", dir_to_action["RIGHT"]))
            if dx < 0 and "LEFT" in dir_to_action:
                preferred.append(("LEFT", dir_to_action["LEFT"]))
            if dy < 0 and "UP" in dir_to_action:
                preferred.append(("UP", dir_to_action["UP"]))
            if dy > 0 and "DOWN" in dir_to_action:
                preferred.append(("DOWN", dir_to_action["DOWN"]))
        else:
            if dy < 0 and "UP" in dir_to_action:
                preferred.append(("UP", dir_to_action["UP"]))
            if dy > 0 and "DOWN" in dir_to_action:
                preferred.append(("DOWN", dir_to_action["DOWN"]))
            if dx > 0 and "RIGHT" in dir_to_action:
                preferred.append(("RIGHT", dir_to_action["RIGHT"]))
            if dx < 0 and "LEFT" in dir_to_action:
                preferred.append(("LEFT", dir_to_action["LEFT"]))

        # Add opposite directions as fallbacks
        for dir_name, act in dir_to_action.items():
            if not any(a == act for _, a in preferred):
                preferred.append((dir_name, act))

        # Try each direction, prefer unvisited destinations
        old_pos = avatar_pos
        moved = False
        chosen = "?"

        # Build opposite action map for undoing moves
        opposite = {}
        for dn, ac in dir_to_action.items():
            opp_name = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}.get(dn)
            if opp_name and opp_name in dir_to_action:
                opposite[ac] = dir_to_action[opp_name]

        best_visited_act = None  # Fallback: best move to a visited position
        best_visited_dist = 999

        for dir_name, act in preferred:
            if try_action(act):
                if avatar_pos not in visited:
                    # Great — unvisited destination
                    moved = True
                    chosen = act
                    break
                else:
                    # Moved to visited position — remember as fallback
                    new_dist = distance(*avatar_pos)
                    if best_visited_act is None or new_dist < best_visited_dist:
                        best_visited_act = act
                        best_visited_dist = new_dist
                    # Undo this move to try other directions
                    if act in opposite:
                        try_action(opposite[act])
                    continue
            # Action was blocked, try next

        if not moved and best_visited_act:
            # All unvisited directions blocked/unavailable, go to best visited
            try_action(best_visited_act)
            moved = True
            chosen = best_visited_act

        dist = distance(*avatar_pos) if avatar_pos else 999
        results.append({
            "step": step_num, "action": chosen,
            "had_movement": moved,
            "state": obs.state.name if obs else "?",
            "levels": obs.levels_completed if obs else 0,
        })

        if moved:
            if avatar_pos == old_pos:
                stuck_count += 1
                print(f"  [NAV] Step {step_num}: {chosen} → SAME POS @{avatar_pos} ✗")
            else:
                stuck_count = 0
                print(f"  [NAV] Step {step_num}: {chosen} → @{avatar_pos} dist={dist} ✓")
        else:
            stuck_count += 1
            print(f"  [NAV] Step {step_num}: BLOCKED @{avatar_pos} ✗")

        if save_dir and moved:
            img = grid_to_image(grid, scale=4)
            img.save(Path(save_dir) / f"nav_{step_num:03d}.png")

        if obs and obs.state == GameState.WIN:
            print(f"  [NAV] WIN at step {step_num}! Levels: {obs.levels_completed}")
            break

        if obs and obs.state == GameState.GAME_OVER:
            total_deaths += 1
            print(f"  [NAV] GAME OVER at step {step_num}")
            obs = env.reset()
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
            avatar_pos = None
            avatar_color = None
            visited.clear()
            # Re-find avatar
            for act in action_map:
                if try_action(act):
                    break

        if dist < 3:
            print(f"  [NAV] Near target @{avatar_pos}!")

        # If truly stuck for too long, give up this navigation
        if stuck_count > 8:
            print(f"  [NAV] Giving up — stuck for {stuck_count} steps")
            break

    levels = results[-1]["levels"] if results else 0
    return {
        "steps": len(results),
        "levels": levels,
        "deaths": total_deaths,
        "corrections": 0,
        "api_calls_execution": 0,
        "final_grid": grid,
    }


def execute_plan(env, client, model, plan, action_map=None, max_steps=30, save_dir=None):
    """Execute LLM plan directly (fallback for click games, etc.)."""
    obs = env.step(GameAction.RESET, data={})
    if obs and obs.frame:
        grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
    else:
        obs = env.reset()
        grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
    print(f"  [EXEC] Reset to initial state")

    results = []
    current_plan = list(plan)
    corrections = 0
    api_calls = 0
    total_deaths = 0

    for i in range(max_steps):
        if not current_plan:
            break

        raw_action = current_plan.pop(0)
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
        had_movement = len(diff.get("movements", [])) > 0

        step_info = {
            "step": i + 1, "action": raw_action,
            "diff": diff["description"], "had_movement": had_movement,
            "state": obs.state.name if obs and obs.state else "?",
            "levels": obs.levels_completed if obs else 0,
        }
        results.append(step_info)
        print(f"  [EXEC] Step {i+1}: {raw_action} → {diff['description'][:50]}"
              + (" ✓" if had_movement else " ✗"))

        if obs and obs.state == GameState.WIN:
            print(f"  [EXEC] WIN at step {i+1}! Levels: {obs.levels_completed}")
            break

        if obs and obs.state == GameState.GAME_OVER:
            total_deaths += 1
            obs = env.reset()
            if obs and obs.frame:
                grid = np.array(obs.frame[0]) if isinstance(obs.frame[0], list) else obs.frame[0]
            if corrections < 2:
                corrections += 1
                api_calls += 1
                new_plan = get_correction(client, model, grid, plan, results,
                                          raw_action, diff["description"], total_deaths)
                if new_plan:
                    current_plan = new_plan

    levels = results[-1]["levels"] if results else 0
    return {
        "steps": len(results),
        "levels": levels,
        "deaths": total_deaths,
        "corrections": corrections,
        "api_calls_execution": api_calls,
        "final_grid": grid,
    }


def get_correction(client, model, grid, original_plan, history, last_action, actual, deaths):
    """Ask LLM for plan correction."""
    completed = [r["action"] for r in history]
    text = CORRECTION_TEMPLATE.format(
        plan=original_plan[:25],
        completed=completed[-10:],
        last_action=last_action,
        actual=actual,
        deaths=deaths,
    )

    content = [{"type": "text", "text": text}]
    b64 = grid_to_base64(grid, scale=3)
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
    })

    try:
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=1500,
            timeout=120,
        )
        if "5.4" in model or "claude" in model:
            kwargs["temperature"] = 0.3
        response = client.chat.completions.create(**kwargs)
        reply = response.choices[0].message.content or ""
        parsed = parse_json(reply)
        return parsed.get("adjusted_plan", [])
    except Exception as e:
        print(f"  [CORRECTION] Error: {e}")
    return None


def parse_json(text):
    """Extract JSON from LLM response."""
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
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    return {"optimal_plan": ["ACTION1"] * 10, "error": "parse_failed"}


# ── Main Pipeline ─────────────────────────────────────────────────────

def run_game(game_id, arc, model="gpt-5.3-chat", save_dir=None):
    """Run full pipeline on one game."""
    client = OpenAI(
        base_url=os.environ["AZURE_FOUNDRY_BASE_URL"],
        api_key=os.environ["AZURE_INFERENCE_CREDENTIAL"],
    )

    env = arc.make(game_id, render_mode=None)
    obs = env.reset()
    avail = list(obs.available_actions) if hasattr(obs, "available_actions") and obs.available_actions else []

    api_calls = 0

    # Phase 1: Structured exploration (0 API calls)
    print(f"\n  === PHASE 1: STRUCTURED EXPLORATION (0 LLM calls) ===")
    explore_data = structured_explore(env, avail, n_steps=20, save_dir=save_dir)
    print(f"  Exploration: {explore_data['total_steps']} steps, "
          f"{explore_data['deaths']} deaths, {explore_data['levels']} levels")
    print(f"  Action map:")
    for act, desc in explore_data["action_summary"].items():
        print(f"    {act}: {desc}")

    if explore_data["levels"] > 0:
        return {"game_id": game_id, "levels": explore_data["levels"],
                "deaths": explore_data["deaths"], "api_calls": 0,
                "method": "lucky_exploration"}

    # Phase 2: Analysis (1 API call)
    print(f"\n  === PHASE 2: LLM ANALYSIS (1 call, model={model}) ===")
    analysis_text = analyze_exploration(client, model, explore_data)
    analysis = parse_json(analysis_text)
    api_calls += 1

    plan = analysis.get("optimal_plan", [])
    print(f"  Goal: {analysis.get('goal_hypothesis', '?')[:100]}")
    print(f"  Mechanics: {analysis.get('mechanics', '?')[:80]}")
    print(f"  Plan ({len(plan)} steps): {plan[:8]}...")
    print(f"  Confidence: {analysis.get('confidence', '?')}")

    # Phase 3: Execute the plan directly (LLM plan is our best bet)
    print(f"\n  === PHASE 3: EXECUTE PLAN ({len(plan)} actions) ===")
    exec_result = execute_plan(env, client, model, plan,
                               action_map=explore_data["action_summary"],
                               max_steps=35, save_dir=save_dir)
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
        "analysis_raw": analysis_text[:500],
        "method": "plan-then-execute",
    }


def main():
    games = ["ls20-9607627b", "ft09-0d8bbf25", "m0r0-dadda488"]
    model = "gpt-5.3-chat"

    if len(sys.argv) > 1:
        games = [g for g in sys.argv[1:] if not g.startswith("--")]
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model = sys.argv[idx + 1]

    print(f"=== E025: Smart Exploration + Plan-Execute ===")
    print(f"  Games: {games}")
    print(f"  Model: {model}")
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
            result = run_game(game_id, arc, model=model,
                              save_dir=f"experiments/E025-smart-explore/frames/{game_id}")
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
    print(f"  SUMMARY — E025 Smart Explore + Plan-Execute")
    print(f"{'=' * 70}")
    for r in all_results:
        gid = r.get("game_id", "?").split("-")[0]
        calls = r.get("api_calls", 0)
        print(f"  {gid}: levels={r['levels']}, deaths={r['deaths']}, "
              f"api_calls={calls}, time={r.get('elapsed', 0):.0f}s")
    print(f"\n  TOTAL API CALLS: {total_api}")
    print(f"  TOTAL LEVELS: {sum(r['levels'] for r in all_results)}")

    out = Path("experiments/E025-smart-explore")
    with open(out / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
