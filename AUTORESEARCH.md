# Autoresearch

## Status: OFF

## What is this
When Status is ON, Claude Code operates as an autonomous research agent
that iterates on the harness to improve agent performance. When OFF,
normal development mode — human drives everything.

**Inspired by karpathy/autoresearch.** The LLM agent IS the orchestrator.
This file IS the program.

---

## Setup protocol (one-time, at start of session)

1. Read this file completely.
2. Read CLAUDE.md, CURRENT_STATE.md, and the current agent code:
   - `src/arcagi3/agent.py` (prompts, loop logic)
   - `src/arcagi3/bench.py` (bench runner)
   - `src/arcagi3/judge.py` (evaluation)
3. Read `results.tsv` if it exists — get `best_metric` from history.
4. Create branch: `autoresearch/<topic>-<YYYY-MM-DD>` from current main.
5. **Only if results.tsv does NOT exist**: run a baseline bench and log it.
   Otherwise, the last best metric from results.tsv IS the baseline.
6. Confirm with human: "Best metric = X. Starting autoresearch loop. Interrupt me anytime."

---

## The experiment loop

**NEVER STOP.** Once the loop begins, do NOT pause to ask the human
if you should continue. The loop runs until the human interrupts, context
runs out, or a stop condition is met.

```
LOOP FOREVER:
  1. THINK — look at results.tsv, current code, what worked/failed.
     Pick ONE idea to try. Ideas should target the weakest tier
     from the judge (if Tier 1 fails, fix perception before goal ID).

  2. EDIT — modify ONLY allowed files (see Scope below).
     Make ONE focused change. Small, testable, reversible.

  3. COMMIT — git add + git commit with descriptive message.

  4. RUN — execute bench (15 min timeout, kill if exceeds):
     timeout 900 python -m arcagi3.bench \
       --games ls20 --runs 3 --max-actions 30 \
       --model gpt-5.4-mini --judge
     If timeout fires, treat as crash.

  5. PARSE — extract the COMBINED METRIC from bench output.

  6. DECIDE:
     - If metric IMPROVED (strictly greater than best): KEEP.
       Update best_metric. Log as "keep" in results.tsv.
     - If metric EQUAL or WORSE: DISCARD.
       Run: git reset HEAD~1 --hard
       Log as "discard" in results.tsv.
     - If bench CRASHED: read the error. If trivial fix, fix and re-run.
       If fundamentally broken, revert and log as "crash" in results.tsv.

  7. LOG — append one line to results.tsv (see format below).

  8. REPEAT — go to step 1.
```

---

## Scope — what you can modify

**ALLOWED (agent harness):**
- `src/arcagi3/agent.py` — prompts (ANALYZER, REFLECTOR, ACTOR), loop logic,
  context building, belief management, stagnation handling
- `src/arcagi3/exploration.py` — exploration controller logic
- `src/arcagi3/trackers.py` — avatar/bar tracker logic
- `src/arcagi3/grid_utils.py` — grid processing, diff computation

**READ-ONLY (evaluation infrastructure):**
- `src/arcagi3/bench.py` — do NOT modify the benchmark runner
- `src/arcagi3/judge.py` — do NOT modify the judge
- `src/arcagi3/run.py` — do NOT modify the CLI
- `golden/*.md` — do NOT modify golden thinking

**NEVER MODIFY:**
- `PROJECT.md`, `CURRENT_STATE.md` — not on autoresearch branches
- Game source code, arc-agi library code
- `pyproject.toml` — no new dependencies

**THE RULE:** If the question "does this work for ALL 25 games?" is NO,
do NOT do it. Zero game-specific code. The harness must be 100% general.

---

## results.tsv format

Tab-separated, NOT committed to git (add to .gitignore).

```
commit	metric	judge_ls20	judge_g50t	levels	status	description
abc1234	20.0	20	-	0	baseline	Initial baseline
def5678	25.0	25	-	0	keep	Improved analyzer prompt: force spatial map
ghi9012	18.0	18	-	0	discard	Removed exploration phase — worse perception
jkl3456	0.0	0	-	0	crash	Syntax error in reflector prompt
```

Status values: `baseline`, `keep`, `discard`, `crash`.

---

## Config

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | gpt-5.4-mini | 5x faster than gpt-5.4, good quality |
| Games | ls20 (+ g50t when golden ready) | Test games for validation |
| Runs per game | 3 | Chain with belief transfer |
| Max actions per run | 30 | Enough to reach "+" in ls20 |
| Temperature | 0.7 | Default, good exploration/exploitation balance |
| Judge model | gpt-5.4-mini | Consistent, fast |
| Time budget per experiment | 15 min | `timeout 900`, kill and log as crash |
| Experiments per session | ~5-8 | Limited by context window |

---

## Strategy guidance

**What to optimize:** The judge score is tiered. Work bottom-up:
1. First get Tier 1 solid (perception) — if the agent can't see, nothing else matters
2. Then Tier 2 (mechanics discovery) — can it figure out what things do?
3. Then Tier 3 (goal identification) — does it understand what to achieve?
4. Then Tier 4 (execution) — can it actually do it?

**Ideas to try (not exhaustive):**
- Prompt engineering: better analyzer questions, more structured reflector
- Context management: what info to include/exclude per step
- Exploration strategy: how to allocate early actions
- Belief format: how to structure and update the memory JSON
- Stagnation recovery: what to do when stuck
- Multi-run learning: how to transfer knowledge between runs effectively

**What NOT to try:**
- Game-specific anything
- Adding new LLM calls (expensive, slow down the loop)
- Major architecture changes (keep changes small and testable)
- Changing the evaluation (bench, judge, golden)

---

## Stop conditions

Stop the loop when ANY of these is true:
- **Human interrupt** — user says stop
- **Context exhausted** — approaching context window limit
- **Breakthrough** — agent completes level 1 (levels_completed >= 1, score > 0)
- **Plateau** — 5+ consecutive discards with no improvement
- **Budget** — check with `bash scripts/check_budget.sh`

On stop: commit results.tsv, update CHANGELOG.md, summarize findings.

---

## Rules

- **ONE change per experiment.** Do not bundle multiple ideas.
- **SMALL changes.** A prompt tweak, a context adjustment, a logic fix.
- **COMMIT BEFORE RUNNING.** So git reset works cleanly.
- **TRUST THE JUDGE.** The metric is the arbiter. Don't argue with it.
- **LOG EVERYTHING.** Every experiment gets a results.tsv line.
- **PIVOT FAST.** 2-3 discards on the same approach = try something different.
- **BUILD, DON'T THEORIZE.** Implement and test, don't analyze forever.
- **GENERALIZABLE ONLY.** Would this work on a game you've never seen? If no, don't.
