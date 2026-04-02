"""E003: Analyzer-Actor split test."""
import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import arc_agi
from arcagi3.agent import AgentConfig, run_agent


def main():
    games = ["ls20-9607627b"]
    if len(sys.argv) > 1:
        games = sys.argv[1:]

    print("=== E003: Analyzer-Actor Split ===")
    for game_id in games:
        print(f"\n{'='*60}")
        print(f"  Running: {game_id}")
        print(f"{'='*60}")

        arc = arc_agi.Arcade()
        env = arc.make(game_id, render_mode=None)

        config = AgentConfig(
            max_actions=50,
            use_vision=True,
            use_text=False,
            temperature=0.7,
            save_frames=True,
            frames_dir=f"experiments/E003-analyzer-actor/frames/{game_id}",
            analyze_every=3,
        )

        start = time.time()
        state = run_agent(env, config)
        elapsed = time.time() - start

        print(f"\n--- RESULT for {game_id} ---")
        print(f"Steps: {len(state.steps)}")
        deaths = sum(1 for s in state.steps if s.state == "GAME_OVER")
        print(f"Deaths: {deaths}")
        levels = state.steps[-1].levels_completed if state.steps else 0
        print(f"Levels completed: {levels}")
        print(f"Final state: {state.steps[-1].state if state.steps else 'N/A'}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Analyses run: {len(state.analysis_history)}")

        # Save results
        out = Path("experiments/E003-analyzer-actor")
        result = {
            "game_id": game_id,
            "steps": len(state.steps),
            "deaths": deaths,
            "levels_completed": levels,
            "elapsed": round(elapsed, 1),
            "analyses": len(state.analysis_history),
            "final_memory": state.memory,
            "last_analysis": state.last_analysis,
            "actions": [
                {"step": s.action_num, "action": s.action, "state": s.state,
                 "reasoning": s.reasoning[:200], "diff": s.diff_summary[:200]}
                for s in state.steps
            ],
        }
        with open(out / "results.json", "w") as f:
            json.dump(result, f, indent=2)

        try:
            scorecard = arc.get_scorecard()
            if scorecard:
                print(f"Score: {scorecard.score}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
