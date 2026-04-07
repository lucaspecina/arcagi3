"""Bench runner — chained runs with belief transfer, parallel across games.

Usage:
    python -m arcagi3.bench --games ls20 --runs 3 --max-actions 30
    python -m arcagi3.bench --games ls20,g50t --runs 3 --model gpt-5.4-mini --judge
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass
class RunResult:
    """Result of a single agent run."""

    game_id: str
    run_num: int
    steps: int = 0
    levels_completed: int = 0
    score: float = 0.0
    final_state: str = ""
    beliefs: str = "{}"
    elapsed_secs: float = 0.0
    error: str = ""


@dataclass
class ChainResult:
    """Result of a chain of runs for one game."""

    game_id: str
    runs: list[RunResult] = field(default_factory=list)
    judge_score: float = 0.0
    judge_summary: str = ""
    total_elapsed: float = 0.0


@dataclass
class BenchResult:
    """Result of a full benchmark (all games)."""

    chains: list[ChainResult] = field(default_factory=list)
    metric: float = 0.0
    wall_time: float = 0.0


def run_single_game(
    game_id: str,
    model: str,
    max_actions: int,
    runs_per_game: int,
    temperature: float,
    use_judge: bool,
    judge_model: str,
    verbose: bool,
) -> ChainResult:
    """Run a chain of N runs for one game with belief transfer."""
    import arc_agi

    from .agent import AgentConfig, run_agent
    from .judge import load_golden, print_judge_result, run_judge

    chain = ChainResult(game_id=game_id)
    prior_beliefs = ""
    beliefs_per_run: list[str] = []
    last_steps = []  # StepRecords from the most recent run
    t0 = time.time()

    for run_num in range(1, runs_per_game + 1):
        prefix = f"[{game_id} run {run_num}/{runs_per_game}]"
        print(f"\n{'=' * 60}")
        print(f"  {prefix} Starting...")
        print(f"{'=' * 60}")

        # Create fresh arcade and environment for each run
        arc = arc_agi.Arcade()
        env = arc.make(game_id, render_mode=None)
        if env is None:
            result = RunResult(
                game_id=game_id, run_num=run_num, error=f"Could not create env '{game_id}'"
            )
            chain.runs.append(result)
            print(f"  {prefix} ERROR: {result.error}")
            continue

        config = AgentConfig(
            model=model,
            max_actions=max_actions,
            temperature=temperature,
            use_vision=True,
            prior_beliefs=prior_beliefs,
            show_window=False,
            step_mode=False,
            save_frames=False,
        )

        run_t0 = time.time()
        try:
            state = run_agent(env, config)
        except Exception as e:
            result = RunResult(
                game_id=game_id, run_num=run_num,
                elapsed_secs=time.time() - run_t0, error=str(e),
            )
            chain.runs.append(result)
            print(f"  {prefix} CRASHED: {e}")
            continue

        elapsed = time.time() - run_t0

        # Extract results
        scorecard = arc.get_scorecard()
        score = scorecard.score if scorecard else 0.0
        levels = 0
        final_state = ""
        if state.steps:
            last = state.steps[-1]
            levels = last.levels_completed
            final_state = last.state

        result = RunResult(
            game_id=game_id,
            run_num=run_num,
            steps=len(state.steps),
            levels_completed=levels,
            score=score,
            final_state=final_state,
            beliefs=state.memory or "{}",
            elapsed_secs=elapsed,
        )
        chain.runs.append(result)

        # Transfer beliefs to next run and track progression
        prior_beliefs = state.memory or ""
        beliefs_per_run.append(state.memory or "{}")
        last_steps = state.steps

        print(f"\n  {prefix} Done in {elapsed:.1f}s — "
              f"steps={len(state.steps)}, levels={levels}, score={score:.1f}")

    chain.total_elapsed = time.time() - t0

    # Run judge on the final run's beliefs (most mature understanding)
    if use_judge and chain.runs:
        last_run = chain.runs[-1]
        if not last_run.error:
            golden = load_golden(game_id)
            if golden:
                print(f"\n  [{game_id}] Running judge on final beliefs...")
                judge_result = run_judge(
                    game_id=game_id,
                    beliefs_json=last_run.beliefs,
                    steps=last_steps,
                    model=judge_model,
                    beliefs_per_run=beliefs_per_run,
                )
                chain.judge_score = judge_result.final_score
                chain.judge_summary = judge_result.summary
                if verbose:
                    print_judge_result(judge_result)
                else:
                    print(f"  [{game_id}] Judge score: {chain.judge_score}/100 — {chain.judge_summary[:100]}")
            else:
                print(f"  [{game_id}] No golden thinking found, skipping judge")

    return chain


def compute_metric(chains: list[ChainResult]) -> float:
    """Compute the combined metric across all game chains.

    Priority: real score > levels completed > judge score.
    """
    game_metrics = []

    for chain in chains:
        # Best score across all runs in the chain
        best_score = max((r.score for r in chain.runs if not r.error), default=0.0)
        best_levels = max((r.levels_completed for r in chain.runs if not r.error), default=0)

        if best_score > 0:
            game_metrics.append(best_score)
        elif best_levels > 0:
            game_metrics.append(best_levels * 100)
        else:
            game_metrics.append(chain.judge_score)

    if not game_metrics:
        return 0.0
    return sum(game_metrics) / len(game_metrics)


def print_bench_summary(result: BenchResult) -> None:
    """Print a summary of the benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"  BENCH SUMMARY")
    print(f"{'=' * 60}")

    for chain in result.chains:
        print(f"\n  [{chain.game_id}] ({chain.total_elapsed:.1f}s total)")
        for r in chain.runs:
            status = "ERR" if r.error else "OK"
            print(f"    run {r.run_num}: {r.elapsed_secs:.1f}s | "
                  f"steps={r.steps} levels={r.levels_completed} "
                  f"score={r.score:.1f} [{status}]")
        if chain.judge_score > 0 or chain.judge_summary:
            print(f"    judge: {chain.judge_score}/100 — {chain.judge_summary[:80]}")

    print(f"\n  COMBINED METRIC: {result.metric:.1f}")
    print(f"  WALL TIME: {result.wall_time:.1f}s")
    print(f"{'=' * 60}")


def main():
    load_dotenv()

    # Handle Unicode output safely on Windows (cp1252 default)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Bench runner — chained runs with belief transfer")
    parser.add_argument("--games", default="ls20", help="Comma-separated game IDs (default: ls20)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per game in chain (default: 3)")
    parser.add_argument("--max-actions", type=int, default=30, help="Max actions per run (default: 30)")
    parser.add_argument("--model", default="gpt-5.4-mini", help="LLM model (default: gpt-5.4-mini)")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--judge", action="store_true", help="Run LLM judge after each chain")
    parser.add_argument("--judge-model", default="gpt-5.4-mini", help="Model for the judge")
    parser.add_argument("--no-parallel", action="store_true", help="Run games sequentially instead of parallel")
    parser.add_argument("--verbose", action="store_true", help="Show full judge details")
    args = parser.parse_args()

    game_ids = [g.strip() for g in args.games.split(",")]
    print(f"Bench: games={game_ids}, runs={args.runs}, max_actions={args.max_actions}, "
          f"model={args.model}, judge={args.judge}")

    bench = BenchResult()
    wall_t0 = time.time()

    if len(game_ids) > 1 and not args.no_parallel:
        # Parallel across games
        print(f"\nRunning {len(game_ids)} games in parallel...")
        with ThreadPoolExecutor(max_workers=len(game_ids)) as executor:
            futures = {
                executor.submit(
                    run_single_game,
                    game_id=gid,
                    model=args.model,
                    max_actions=args.max_actions,
                    runs_per_game=args.runs,
                    temperature=args.temperature,
                    use_judge=args.judge,
                    judge_model=args.judge_model,
                    verbose=args.verbose,
                ): gid
                for gid in game_ids
            }
            for future in as_completed(futures):
                gid = futures[future]
                try:
                    chain = future.result()
                    bench.chains.append(chain)
                except Exception as e:
                    print(f"  [{gid}] FAILED: {e}")
                    bench.chains.append(ChainResult(game_id=gid))
    else:
        # Sequential
        for gid in game_ids:
            chain = run_single_game(
                game_id=gid,
                model=args.model,
                max_actions=args.max_actions,
                runs_per_game=args.runs,
                temperature=args.temperature,
                use_judge=args.judge,
                judge_model=args.judge_model,
                verbose=args.verbose,
            )
            bench.chains.append(chain)

    bench.wall_time = time.time() - wall_t0
    bench.metric = compute_metric(bench.chains)

    # Sort chains by game_id for consistent output
    bench.chains.sort(key=lambda c: c.game_id)

    print_bench_summary(bench)
    return bench


if __name__ == "__main__":
    main()
