"""Entry point for running the ARC-AGI-3 LLM agent."""

import argparse
import sys

from dotenv import load_dotenv


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run ARC-AGI-3 LLM agent")
    parser.add_argument("--game", default="ls20", help="Game ID to play (default: ls20)")
    parser.add_argument("--model", default="", help="LLM model name (default: from env)")
    parser.add_argument("--max-actions", type=int, default=80, help="Max actions per game")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision, use text only")
    parser.add_argument("--render", default="terminal", help="Render mode: terminal, human, or none")
    parser.add_argument("--window", action="store_true", help="Render in a separate window (shortcut for --render human)")
    parser.add_argument("--list-games", action="store_true", help="List available games and exit")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between steps in seconds")
    parser.add_argument("--step", action="store_true", help="Pause after each step, type comments/questions for the model")
    parser.add_argument("--raw", action="store_true", help="Show raw prompt sent to model before first action")
    parser.add_argument("--save-frames", action="store_true", help="Save each frame as PNG + final GIF replay")
    parser.add_argument("--frames-dir", default="frames", help="Directory to save frames (default: frames/)")
    args = parser.parse_args()

    import arc_agi

    from .agent import AgentConfig, run_agent

    # Create arcade
    arc = arc_agi.Arcade()

    # List games mode
    if args.list_games:
        envs = arc.get_environments()
        print(f"Available games ({len(envs)}):")
        for e in envs:
            print(f"  {e.game_id}: {getattr(e, 'title', '?')}")
        return

    # Window mode overrides render
    if args.window:
        args.render = "human"

    # Configure agent
    config = AgentConfig(
        model=args.model or "",
        max_actions=args.max_actions,
        use_vision=not args.no_vision,
        use_text=args.no_vision,
        temperature=args.temperature,
        delay=args.delay,
        step_mode=args.step,
        show_raw_prompt=args.raw,
        save_frames=args.save_frames,
        frames_dir=args.frames_dir,
    )

    # Create environment
    render_mode = None if args.render == "none" else args.render
    print(f"Creating environment: {args.game} (render={render_mode})")
    env = arc.make(args.game, render_mode=render_mode)

    if env is None:
        print(f"ERROR: Could not create environment '{args.game}'")
        sys.exit(1)

    print(f"Game: {env.info.game_id if env.info else args.game}")
    print(f"Model: {config.model or '(from env)'}")
    print(f"Max actions: {config.max_actions}")
    print(f"Vision: {config.use_vision}")
    print("---")

    # Run agent
    state = run_agent(env, config)

    # Print summary
    print("---")
    print(f"Total steps: {len(state.steps)}")
    if state.steps:
        last = state.steps[-1]
        print(f"Final state: {last.state}")
        print(f"Levels completed: {last.levels_completed}")

    # Get scorecard
    scorecard = arc.get_scorecard()
    if scorecard:
        print(f"Score: {scorecard.score}")

    return state


if __name__ == "__main__":
    main()
