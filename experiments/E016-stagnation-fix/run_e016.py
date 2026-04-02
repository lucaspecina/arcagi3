"""E016: Stagnation fix — harness-enforced goal changes.

Key improvement: if the LLM won't change its frozen goal after 5+ stale
steps, the harness injects a STAGNATION ALERT directly.

Also: if the reflector output still keeps the same goal after stagnation
alert, the harness forces a goal reset by clearing the goal belief.

Same games for comparison: g50t, ft09, vc33 × 2 runs.
"""
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import arc_agi

from arcagi3.agent import AgentConfig, run_agent


GAMES = ["g50t-5849a774", "ft09-0d8bbf25", "vc33-9851e02b"]


def run_one(game_id: str, run_id: int, arc: arc_agi.Arcade) -> dict:
    env = arc.make(game_id, render_mode=None)
    config = AgentConfig(
        max_actions=30,
        use_vision=True,
        use_text=False,
        temperature=0.5,
        save_frames=True,
        frames_dir=f"experiments/E016-stagnation-fix/frames/{game_id}_run{run_id}",
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

    n_runs = 2
    print(f"=== E016: Stagnation Fix ===")
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
    print(f"  SUMMARY — E016 Stagnation Fix")
    print(f"{'=' * 70}")
    print(f"  {'Game':<20} {'Run':>4} {'Levels':>7} {'Deaths':>7} {'Steps':>6} {'Time':>6}")
    print(f"  {'-'*20} {'-'*4} {'-'*7} {'-'*7} {'-'*6} {'-'*6}")
    for r in all_results:
        gid = r["game_id"].split("-")[0]
        print(f"  {gid:<20} {r['run_id']:>4} {r['levels_completed']:>7} "
              f"{r['deaths']:>7} {r['steps']:>6} {r.get('elapsed', 0):>5.0f}s")

    total_levels = sum(r["levels_completed"] for r in all_results)
    total_deaths = sum(r["deaths"] for r in all_results)
    print(f"\n  TOTAL: {total_levels} levels completed, {total_deaths} deaths")

    out = Path("experiments/E016-stagnation-fix")
    with open(out / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    try:
        scorecard = arc.get_scorecard()
        if scorecard:
            print(f"\n  ARC Score: {scorecard.score}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
