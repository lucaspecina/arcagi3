"""E023: Multi-Run Learning — second attempt with prior beliefs.

KEY HYPOTHESIS: If we inject beliefs from a previous exploratory run,
the LLM can skip the discovery phase and go straight to execution.

Run 1: Use E021 final beliefs as prior knowledge
Run 2: Fresh run for comparison (systematic_explore=True)

Test on g50t and sb26. 30 steps each.

API CALL ESTIMATE: ~90 calls per 30-step run (analyzer + actor + reflector).
2 runs × 2 games = 4 runs × ~90 = ~360 API calls total.
"""
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import arc_agi

from arcagi3.agent import AgentConfig, run_agent


# Prior beliefs from E021 g50t run (corrected RESET, detected ACTION5)
G50T_PRIOR_BELIEFS = json.dumps({
    "controls": {
        "ACTION1": "No visible effect observed (may be position-dependent)",
        "ACTION2": "No avatar movement (may shrink resource bar)",
        "ACTION3": "Swaps avatar with adjacent object DOWNWARD (confirmed from exploration)",
        "ACTION4": "Swaps avatar with adjacent object (direction unclear, position-dependent)",
        "ACTION5": "May move avatar (observed once, needs more testing)",
        "RESET": "RESTARTS the level — returns everything to initial state. NOT a movement action."
    },
    "rules": [
        "Actions swap avatar with adjacent objects — only work when something is adjacent",
        "Resource bar shrinks with most actions — be efficient",
        "RESET restarts the level, don't use it as movement"
    ],
    "goal": "Navigate blue object toward the red object at center (30,31) (confidence: medium)",
    "objects": [
        "Controlled blue object (avatar)",
        "Red target object at center (~30,31)",
        "Black objects that can be swapped with"
    ],
    "dangers": [
        "Resource bar depletes with each action — game over when empty"
    ],
    "unknowns": [
        "Exact direction mapping for each action",
        "What happens when reaching the red object",
        "Purpose of ACTION5 (might be special — test at target)"
    ],
    "failed_approaches": [
        "Random exploration without direction — wastes resource bar",
        "Using RESET as movement — it restarts the level"
    ]
})


def run_one(game_id: str, run_id: int, arc: arc_agi.Arcade,
            prior: str = "", label: str = "") -> dict:
    env = arc.make(game_id, render_mode=None)
    config = AgentConfig(
        max_actions=30,
        use_vision=True,
        use_text=False,
        temperature=0.3,  # Lower temp for more focused execution
        save_frames=True,
        frames_dir=f"experiments/E023-multi-run/frames/{game_id}_{label}_run{run_id}",
        analyze_every=5,
        systematic_explore=True,
        prior_beliefs=prior,
    )
    start = time.time()
    state = run_agent(env, config)
    elapsed = time.time() - start

    deaths = sum(1 for s in state.steps if s.state == "GAME_OVER")
    levels = state.steps[-1].levels_completed if state.steps else 0

    return {
        "game_id": game_id,
        "run_id": run_id,
        "label": label,
        "steps": len(state.steps),
        "deaths": deaths,
        "levels_completed": levels,
        "elapsed": round(elapsed, 1),
        "final_beliefs": state.memory,
    }


def main():
    games_config = [
        ("g50t-5849a774", G50T_PRIOR_BELIEFS, "with-prior"),
        ("g50t-5849a774", "", "fresh"),
    ]

    if len(sys.argv) > 1:
        # Allow custom game override
        gid = sys.argv[1]
        games_config = [(gid, "", "fresh")]

    print(f"=== E023: Multi-Run Learning ===")
    print(f"  Runs: {len(games_config)}")
    print()

    arc = arc_agi.Arcade()
    all_results = []

    for game_id, prior, label in games_config:
        print(f"\n{'=' * 70}")
        print(f"  GAME: {game_id} | MODE: {label}")
        print(f"{'=' * 70}")
        try:
            result = run_one(game_id, 1, arc, prior=prior, label=label)
            all_results.append(result)
            print(f"\n  -> {game_id} ({label}): {result['levels_completed']} levels, "
                  f"{result['deaths']} deaths, {result['elapsed']:.0f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n  X {game_id} ({label}) FAILED: {e}")
            all_results.append({
                "game_id": game_id, "label": label,
                "steps": 0, "deaths": 0, "levels_completed": 0,
                "elapsed": 0, "error": str(e),
            })

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — E023 Multi-Run Learning")
    print(f"{'=' * 70}")
    for r in all_results:
        gid = r["game_id"].split("-")[0]
        lbl = r.get("label", "?")
        print(f"  {gid} ({lbl}): levels={r['levels_completed']}, deaths={r['deaths']}, "
              f"time={r.get('elapsed', 0):.0f}s")

    total_levels = sum(r["levels_completed"] for r in all_results)
    print(f"\n  TOTAL: {total_levels} levels completed")

    out = Path("experiments/E023-multi-run")
    with open(out / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
