"""E001: Baseline capability test — what can GPT-5.4 do out of the box?

Runs the current agent on a few games with limited actions and captures
detailed results for analysis.
"""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import arc_agi
import numpy as np

from arcagi3.agent import AgentConfig, AgentState, run_agent


def run_single_game(game_id: str, max_actions: int = 30) -> dict:
    """Run agent on a single game and return structured results."""
    arc = arc_agi.Arcade()
    env = arc.make(game_id, render_mode=None)

    if env is None:
        return {"game_id": game_id, "error": "Could not create environment"}

    config = AgentConfig(
        max_actions=max_actions,
        use_vision=True,
        use_text=False,
        temperature=0.7,
        save_frames=True,
        frames_dir=f"experiments/E001-baseline/frames/{game_id}",
    )

    start_time = time.time()
    state = run_agent(env, config)
    elapsed = time.time() - start_time

    # Collect results
    result = {
        "game_id": game_id,
        "total_steps": len(state.steps),
        "elapsed_seconds": round(elapsed, 1),
        "final_state": state.steps[-1].state if state.steps else "NO_STEPS",
        "levels_completed": state.steps[-1].levels_completed if state.steps else 0,
        "final_memory": state.memory,
        "actions_taken": [
            {
                "step": s.action_num,
                "action": s.action,
                "x": s.x,
                "y": s.y,
                "state": s.state,
                "levels_completed": s.levels_completed,
                "reasoning": s.reasoning[:300],
            }
            for s in state.steps
        ],
        "deaths": sum(1 for s in state.steps if s.state == "GAME_OVER"),
        "unique_actions": list(set(s.action for s in state.steps)),
    }

    # Get scorecard
    try:
        scorecard = arc.get_scorecard()
        if scorecard:
            result["score"] = scorecard.score
    except Exception:
        pass

    return result


def main():
    # Games to test — start with a few
    games = ["ls20"]  # Add more after first test
    max_actions = 30  # Keep it small for baseline

    if len(sys.argv) > 1:
        games = sys.argv[1:]

    print(f"=== E001: Baseline Capability Test ===")
    print(f"Games: {games}")
    print(f"Max actions per game: {max_actions}")
    print()

    results = []
    for game_id in games:
        print(f"\n{'='*60}")
        print(f"  Running: {game_id}")
        print(f"{'='*60}")

        result = run_single_game(game_id, max_actions)
        results.append(result)

        # Print summary
        print(f"\n  --- RESULT for {game_id} ---")
        print(f"  Steps: {result['total_steps']}")
        print(f"  Deaths: {result['deaths']}")
        print(f"  Levels completed: {result['levels_completed']}")
        print(f"  Final state: {result['final_state']}")
        print(f"  Unique actions: {result['unique_actions']}")
        print(f"  Time: {result['elapsed_seconds']}s")
        if "score" in result:
            print(f"  Score: {result['score']}")

    # Save results
    output_path = Path("experiments/E001-baseline/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save detailed analysis
    analysis_path = Path("experiments/E001-baseline/analysis.md")
    with open(analysis_path, "w") as f:
        f.write("# E001: Baseline Capability Test — Analysis\n\n")
        for r in results:
            f.write(f"## Game: {r['game_id']}\n\n")
            f.write(f"- Steps: {r['total_steps']}\n")
            f.write(f"- Deaths: {r['deaths']}\n")
            f.write(f"- Levels completed: {r['levels_completed']}\n")
            f.write(f"- Final state: {r['final_state']}\n")
            f.write(f"- Unique actions: {r['unique_actions']}\n\n")
            f.write("### Action sequence\n\n")
            for a in r["actions_taken"]:
                coords = f" ({a['x']},{a['y']})" if a["x"] is not None else ""
                f.write(f"- Step {a['step']}: {a['action']}{coords} → {a['state']}\n")
                f.write(f"  > {a['reasoning'][:200]}\n\n")
            f.write("### Final memory\n\n")
            f.write(f"```json\n{r['final_memory']}\n```\n\n")
    print(f"Analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
