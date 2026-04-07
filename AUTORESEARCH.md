# Autoresearch

## Status: OFF (paused 2026-04-07 morning)

**Last session:** 2026-04-06 night, single-brain mode. 5 iterations,
~8 hours of compute, 0 real improvements. See "Anti-patterns from past
sessions" below — the session got stuck in a prompt-tweak loop and
never tried a structural change. Read those before turning Status back ON.

**Verified baseline:** mean=21.25 (n=4) on ls20, gpt-5.4-mini,
30 actions × 3-run chain + judge. Iter 2 (e134f73, ANALYZER INTERACT
BEFORE NAVIGATE) is in main with weak +3.75 (n=3), borderline-significant.

## What is this
When Status is ON, Claude Code operates as an autonomous research agent
that iterates on the harness to improve agent performance. When OFF,
normal development mode — human drives everything.

**Inspired by karpathy/autoresearch.** The LLM agent IS the orchestrator.
This file IS the program.

---

## READ FIRST — Anti-patterns from past sessions

Each session that wastes a night gets a row here. Read before you start.

### 1. The prompt-tweak loop trap (2026-04-06 night)
**Symptom:** 5 iterations in a row, all of them are "add another paragraph
to ANALYZER/REFLECTOR/ACTOR prompt", net improvement ~0.

**Cause:** It feels like progress because you're committing and benching
every ~40 min. But the search space "adjective in a system prompt" is huge,
the noise is high (~6 std on a 100-point scale), and mini models often
ignore added text. You end up A/B testing English wording.

**Rule:** If your last 3 iterations were *all* prompt edits, your 4th
iteration MUST be something structurally different — see "Architectural
moves to consider" below. No exceptions, even if a prompt edit "feels
clearly better".

### 2. Single-sample interpretation (2026-04-06)
**Symptom:** Iter 1 sample = 25, baseline single sample = 30. "Iter 1 hurt!"
Then variance run gives baseline = 15, iter 1 replica = 15. Both are noise.

**Cause:** Judge std on ls20 with gpt-5.4-mini is ~6 points. n=1 tells you
nothing. n=2 barely separates +12. n=3 is the minimum to claim anything.

**Rule:** Every variance baseline gets ≥3 samples. Every iter result that
crosses a decision threshold gets ≥2 samples. If your decision changes
between sample 1 and sample 2, you don't have signal — wait for sample 3.

### 3. Silent timeout deaths (2026-04-06)
**Symptom:** Bench logs go stale at chain 3 step 7-9. No error, no traceback.
Looks like the python "froze". Spent ~1.5 hours debugging.

**Cause:** Shell `timeout 1800` (30 min) sends SIGTERM at 30 min, but a
3-run chain takes 38-42 min. Python dies mid-write to stdout, log just
stops.

**Rule:** Use `timeout 3600 python -u` always (60 min hard cap, unbuffered
stdout). When a log goes stale, FIRST check `tasklist | grep python.exe`
and the file mtime — don't grep "Done in" and assume the bench is alive.

### 4. Trust tracker output without inspecting it (2026-04-06)
**Symptom:** BarTracker reported 5 phantom "RESOURCE BAR" warnings every
step (interior maze walls misclassified as bars). Polluted context for
every run, every session, for months.

**Rule:** Before tweaking prompts, read one full run log end to end and
ask: "is the harness actually feeding the LLM clean data?" If 50% of
the tracker output is noise, fix that first — it's free signal.

---

## Architectural moves to consider BEFORE another prompt-tweak iter

These are NOT prompt edits. They are structurally different attacks on
the meta-cognition harness. Work through this list before reverting to
prompt edits.

- **Multi-hypothesis parallel reasoning.** Instead of a single belief
  state per run, maintain N=3 parallel "lines of play" (different goal
  hypotheses) for the first M steps. Each hypothesis picks its own
  actions. Then a critic step compares evidence and votes on which
  hypothesis to commit to.
- **Critic / debate module.** Add a third LLM call between reflector and
  actor whose only job is to challenge the reflector: "your top hypothesis
  is X — what is the strongest evidence AGAINST X right now?" Forces
  refutation, not just confirmation.
- **Replay learning.** After a chain finishes, run a "post-mortem" LLM
  call over the full action log + diffs and extract structured lessons
  ("when you saw Y you should have done Z"). Feed those into the next
  chain's prior.
- **Tool calling.** Give the LLM real tools: "compare frame A and frame B",
  "highlight pixels of color C", "list all isolated objects of size <10".
  Currently the LLM has to derive everything from raw text dumps.
- **Region-based observation.** Instead of dumping pixel diffs, segment
  the grid into semantic regions (HUD top, HUD bottom, playable area,
  corners) and report changes per region with structured fields.
- **Belief schema redesign.** The current memory is a flat dict of strings.
  Try a structured belief graph: nodes = entities, edges = causal
  hypotheses, with confidence scores and evidence pointers.
- **Stagnation-driven exploration.** When `no_progress_count >= 3`,
  switch the actor into a fundamentally different mode (random walk,
  ACTION6 grid sweep, or LLM-with-temperature=1.5) until something
  changes.
- **Self-consistency over a single step.** Sample the actor 3 times with
  high temperature and majority-vote the action.
- **Multi-model orchestration.** Use a stronger model (gpt-5.4 not mini)
  for the analyzer ONCE per run, and the cheap model for everything else.
  Tests whether the bottleneck is perception capacity or harness wiring.

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

  4. RUN — execute bench (60 min timeout, kill if exceeds):
     timeout 3600 python -u -m arcagi3.bench \
       --games ls20 --runs 3 --max-actions 30 \
       --model gpt-5.4-mini --judge
     A 3-run chain takes ~38-42 min wall time. Use 30 min timeout = killed
     mid chain 3. The -u flag forces unbuffered stdout for live log tails.
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
| Time budget per experiment | 60 min | `timeout 3600`, kill and log as crash |
| Experiments per session | ~5-8 | Limited by context window |

**Wall time reference**: baseline bench (3 runs × 30 actions + judge) takes
~38-42 min on gpt-5.4-mini (chain1 ~13 min, chain2 ~14-15 min, chain3 ~13-14 min,
judge ~30s). Budget 60 min hard timeout. The 30-min default will kill chain 3.

---

## Known gotchas (MUST READ before editing)

These are lessons from past broken runs. Ignoring them wastes hours.

1. **max_completion_tokens must be ≥ 8000 for reasoning models.**
   `gpt-5.4-mini` consumes internal reasoning tokens that count against
   the limit but don't appear in `message.content`. With 3000 tokens,
   the reflector returned empty content (finish_reason=length, len=0)
   starting at step 6 and beliefs froze forever. Current analyzer and
   reflector calls are set to 8000. If you lower this, verify you still
   get `finish_reason=stop` at every step.

2. **Always check finish_reason on LLM responses.** agent.py now warns
   when finish_reason != "stop" and when parse_response fails to find
   `updated_beliefs`. Do not silence these warnings — they catch bugs.

3. **Windows stdout is cp1252.** LLMs occasionally emit characters
   outside cp1252 (`≈`, `✓`, curly quotes) and crash print(). run.py
   and bench.py reconfigure stdout/stderr to utf-8 with errors='replace'
   at startup. If you add a new entry point, do the same.

4. **Parallel benches DO work, but only up to ~4.** Verified
   2026-04-06: 2 parallel benches on ls20 with gpt-5.4-mini run reliably
   without 401s or rate-limit drops. Earlier note about "never parallel"
   came from a different config and was disproved. Cap is around 4 — do
   not push it without re-verifying.

5. **Bench wall time is 38-42 min, not 30.** Use `timeout 3600 python -u
   -m arcagi3.bench …`. The previous default `timeout 1800` killed chain
   3 silently (process gone, log frozen mid-line, no traceback). When a
   log goes stale, check `tasklist | grep python.exe` AND the file mtime
   before assuming the bench is alive.

6. **Verify trackers before tweaking prompts.** The BarTracker shipped
   with a bug that emitted 5 phantom "RESOURCE BAR" warnings per step
   (interior maze walls treated as bars). Read one full run log end to
   end before iterating on prompts — you may discover the LLM was being
   fed garbage.

7. **results.tsv is now tracked in git** (as of 2026-04-07). Commit it
   alongside any experiment branch so the raw per-sample data survives
   alongside the narrative in research notes. Bench logs under `logs/`
   are still gitignored — too big, summarized in post-mortem notes.

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
