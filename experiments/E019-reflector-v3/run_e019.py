"""E019: Reflector v3 — swap detection + stagnation enforcement.

Improvements over E015:
- Diff now detects SWAPS (two objects exchanging positions)
- Harness forces goal change after 5 stagnant steps
- Structured action history table
- Enumerated belief review with forced justification

3 games × 1 run each (budget-conscious). 30 steps.
"""
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import arc_agi

from arcagi3.agent import AgentConfig, run_agent


GAMES = ["g50t-5849a774", "vc33-9851e02b", "m0r0-dadda488"]


def run_one(game_id: str, run_id: int, arc: arc_agi.Arcade) -> dict:
    env = arc.make(game_id, render_mode=None)
    config = AgentConfig(
        max_actions=30,
        use_vision=True,
        use_text=False,
        temperature=0.4,
        save_frames=True,
        frames_dir=f"experiments/E019-reflector-v3/frames/{game_id}_run{run_id}",
        analyze_every=5,
    )
    start = time.time()
    state = run_agent(env, config)
    elapsed = time.time() - start

    deaths = sum(1 for s in state.steps if s.state == "GAME_OVER")
    levels = state.steps[-1].levels_completed if state.steps else 0

    return {
        "game_id": game_id,
        "run_id": run_id,
        "steps": len(state.steps),
        "deaths": deaths,
        "levels_completed": levels,
        "elapsed": round(elapsed, 1),
        "final_beliefs": state.memory,
    }


def main():
    games = GAMES
    if len(sys.argv) > 1:
        games = sys.argv[1:]

    n_runs = 1
    print(f"=== E019: Reflector v3 (swap detection + stagnation fix) ===")
    print(f"  Games: {games}")
    print(f"  Runs per game: {n_runs}")
    print()

    arc = arc_agi.Arcade()
    all_results = []

    for game_id in games:
        for run_id in range(1, n_runs + 1):
            print(f"\n{'=' * 70}")
            print(f"  GAME: {game_id} | RUN: {run_id}/{n_runs}")
            print(f"{'=' * 70}")
            try:
                result = run_one(game_id, run_id, arc)
                all_results.append(result)
                print(f"\n  → {game_id} run {run_id}: {result['levels_completed']} levels, "
                      f"{result['deaths']} deaths, {result['elapsed']:.0f}s")
            except Exception as e:
                print(f"\n  ✗ {game_id} run {run_id} FAILED: {e}")
                all_results.append({
                    "game_id": game_id, "run_id": run_id,
                    "steps": 0, "deaths": 0, "levels_completed": 0,
                    "elapsed": 0, "error": str(e),
                })

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — E019 Reflector v3")
    print(f"{'=' * 70}")
    for r in all_results:
        gid = r["game_id"].split("-")[0]
        print(f"  {gid}: levels={r['levels_completed']}, deaths={r['deaths']}, "
              f"time={r.get('elapsed', 0):.0f}s")

    total_levels = sum(r["levels_completed"] for r in all_results)
    print(f"\n  TOTAL: {total_levels} levels completed")

    out = Path("experiments/E019-reflector-v3")
    with open(out / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
