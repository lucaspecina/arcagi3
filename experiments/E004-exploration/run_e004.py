"""E004: Exploration controller test."""
import json, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import arc_agi
from arcagi3.agent import AgentConfig, run_agent

def main():
    games = ["ls20-9607627b"]
    if len(sys.argv) > 1:
        games = sys.argv[1:]

    print("=== E004: Exploration Controller ===")
    for game_id in games:
        print(f"\n{'='*60}\n  Running: {game_id}\n{'='*60}")
        arc = arc_agi.Arcade()
        env = arc.make(game_id, render_mode=None)
        config = AgentConfig(
            max_actions=50, use_vision=True, use_text=False,
            temperature=0.7, save_frames=True,
            frames_dir=f"experiments/E004-exploration/frames/{game_id}",
            analyze_every=3,
        )
        start = time.time()
        state = run_agent(env, config)
        elapsed = time.time() - start

        deaths = sum(1 for s in state.steps if s.state == "GAME_OVER")
        levels = state.steps[-1].levels_completed if state.steps else 0
        print(f"\n--- RESULT ---")
        print(f"Steps: {len(state.steps)}, Deaths: {deaths}, Levels: {levels}")
        print(f"Time: {elapsed:.1f}s, Analyses: {len(state.analysis_history)}")

        out = Path("experiments/E004-exploration")
        result = {
            "game_id": game_id, "steps": len(state.steps), "deaths": deaths,
            "levels_completed": levels, "elapsed": round(elapsed, 1),
            "analyses": len(state.analysis_history),
            "final_memory": state.memory, "last_analysis": state.last_analysis,
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
