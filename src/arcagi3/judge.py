"""LLM Judge Oracle — evaluates agent understanding against golden thinking.

Used during autoresearch to measure progress when scorecard = 0.
The judge reads the agent's beliefs and action history, compares against
a golden thinking file that describes the correct understanding, and
scores each understanding milestone.

This is evaluation-only code. It does NOT affect the agent.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent.parent.parent / "golden"

JUDGE_PROMPT = """\
You are an evaluation judge for an ARC-AGI-3 agent. Your job is to assess
how well the agent UNDERSTOOD the game, based on its beliefs, reasoning,
and action history.

You will receive:
1. A "golden thinking" document describing the correct understanding of the game
2. The agent's beliefs after EACH run (showing learning progression)
3. The agent's action history from the final run
4. A list of understanding milestones to check

For EACH milestone, you must:
- Decide: ACHIEVED or NOT ACHIEVED
- Cite specific evidence from the agent's output that supports your decision
- Be STRICT: the agent must clearly demonstrate the understanding, not just
  vaguely gesture at it. Partial credit is NOT allowed per milestone.
- When evaluating, consider the PROGRESSION across runs — an agent that
  learned something new between runs shows real understanding.

Also check for ANTI-PATTERNS listed in the golden thinking. Each confirmed
anti-pattern is a -5 point penalty.

Respond with JSON:
{
  "milestones": [
    {"id": "M1", "achieved": true, "evidence": "agent beliefs say 'blue+orange square is avatar'"},
    {"id": "M2", "achieved": false, "evidence": "controls map ACTION1 to 'left' which is wrong"}
  ],
  "anti_patterns": [
    {"pattern": "thinks + is a wall", "confirmed": false, "evidence": "agent navigated to + deliberately"}
  ],
  "tier_scores": {
    "tier1_perception": 15,
    "tier2_mechanics": 20,
    "tier3_goal": 0,
    "tier4_execution": 0
  },
  "penalties": 0,
  "raw_score": 35,
  "final_score": 35,
  "summary": "Agent identified avatar and movement correctly, discovered + interaction, but never connected it to the goal."
}
"""


@dataclass
class JudgeResult:
    """Result from the LLM judge evaluation."""

    game_id: str
    milestones: list[dict] = field(default_factory=list)
    anti_patterns: list[dict] = field(default_factory=list)
    tier_scores: dict = field(default_factory=dict)
    penalties: int = 0
    raw_score: float = 0.0
    final_score: float = 0.0
    summary: str = ""


def load_golden(game_id: str) -> str | None:
    """Load golden thinking for a game. Returns None if not found."""
    path = GOLDEN_DIR / f"{game_id}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def format_action_history(steps: list) -> str:
    """Format agent step records into a readable action history."""
    lines = []
    for s in steps:
        parts = [f"Step {s.action_num}: {s.action}"]
        if s.x is not None and s.y is not None:
            parts.append(f"({s.x},{s.y})")
        if s.reasoning:
            parts.append(f"— {s.reasoning[:120]}")
        if s.diff_summary:
            parts.append(f"[{s.diff_summary[:80]}]")
        state_info = []
        if s.state:
            state_info.append(s.state)
        if s.levels_completed:
            state_info.append(f"levels={s.levels_completed}")
        if state_info:
            parts.append(f"({', '.join(state_info)})")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def format_belief_progression(beliefs_per_run: list[str]) -> str:
    """Format beliefs from each run to show learning progression."""
    parts = []
    for i, beliefs_json in enumerate(beliefs_per_run, 1):
        parts.append(f"### Run {i} beliefs")
        try:
            beliefs = json.loads(beliefs_json)
            # Compact summary: goal + controls + key discoveries
            summary = []
            if beliefs.get("goal"):
                summary.append(f"Goal: {beliefs['goal']}")
            if beliefs.get("controls"):
                ctrl = beliefs["controls"]
                if isinstance(ctrl, dict):
                    ctrl_short = {k: v[:60] if isinstance(v, str) else v for k, v in ctrl.items()}
                    summary.append(f"Controls: {json.dumps(ctrl_short)}")
            if beliefs.get("rules"):
                summary.append(f"Rules: {beliefs['rules']}")
            if beliefs.get("causal_model"):
                summary.append(f"Causal model: {beliefs['causal_model']}")
            if beliefs.get("objects"):
                summary.append(f"Objects: {beliefs['objects']}")
            if beliefs.get("dangers"):
                summary.append(f"Dangers: {beliefs['dangers']}")
            if beliefs.get("unknowns"):
                summary.append(f"Unknowns: {beliefs['unknowns']}")
            if beliefs.get("failed_approaches"):
                summary.append(f"Failed: {beliefs['failed_approaches']}")
            parts.append("\n".join(summary) if summary else beliefs_json)
        except (json.JSONDecodeError, TypeError):
            parts.append(beliefs_json[:500] if beliefs_json else "(empty)")
        parts.append("")
    return "\n".join(parts)


def run_judge(
    game_id: str,
    beliefs_json: str,
    steps: list,
    model: str = "gpt-5.4",
    beliefs_per_run: list[str] | None = None,
) -> JudgeResult:
    """Run the LLM judge on agent output.

    Args:
        game_id: Game identifier (e.g. "ls20")
        beliefs_json: Agent's final beliefs as JSON string
        steps: List of StepRecord from the agent run (last run)
        model: LLM model for the judge
        beliefs_per_run: Beliefs from each run in the chain (for progression)

    Returns:
        JudgeResult with score and milestone details
    """
    from openai import OpenAI

    golden = load_golden(game_id)
    if golden is None:
        print(f"  [JUDGE] No golden thinking found for {game_id}, skipping")
        return JudgeResult(game_id=game_id, summary="No golden thinking available")

    action_history = format_action_history(steps)

    # Build belief section
    if beliefs_per_run and len(beliefs_per_run) > 1:
        belief_section = f"""## Agent's Belief Progression ({len(beliefs_per_run)} runs)

{format_belief_progression(beliefs_per_run)}

## Agent's Final Beliefs (complete)

```json
{beliefs_json}
```"""
    else:
        belief_section = f"""## Agent's Final Beliefs

```json
{beliefs_json}
```"""

    user_message = f"""## Golden Thinking (correct understanding)

{golden}

{belief_section}

## Agent's Action History — final run ({len(steps)} steps)

{action_history}

---

Evaluate each milestone from the golden thinking. Be strict but fair.
Consider the belief PROGRESSION across runs — learning over time counts.
"""

    client = OpenAI(
        base_url=os.environ.get("AZURE_FOUNDRY_BASE_URL", ""),
        api_key=os.environ.get("AZURE_INFERENCE_CREDENTIAL", ""),
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return JudgeResult(game_id=game_id, summary=f"Judge JSON parse error: {raw[:200]}")

    return JudgeResult(
        game_id=game_id,
        milestones=parsed.get("milestones", []),
        anti_patterns=parsed.get("anti_patterns", []),
        tier_scores=parsed.get("tier_scores", {}),
        penalties=parsed.get("penalties", 0),
        raw_score=parsed.get("raw_score", 0.0),
        final_score=parsed.get("final_score", 0.0),
        summary=parsed.get("summary", ""),
    )


def print_judge_result(result: JudgeResult) -> None:
    """Pretty-print judge evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"  JUDGE EVALUATION — {result.game_id}")
    print(f"{'=' * 60}")

    # Milestones
    achieved = [m for m in result.milestones if m.get("achieved")]
    missed = [m for m in result.milestones if not m.get("achieved")]

    if achieved:
        print(f"\n  ACHIEVED ({len(achieved)}):")
        for m in achieved:
            print(f"    [x] {m['id']}: {m.get('evidence', '')[:80]}")

    if missed:
        print(f"\n  NOT ACHIEVED ({len(missed)}):")
        for m in missed:
            print(f"    [ ] {m['id']}: {m.get('evidence', '')[:80]}")

    # Anti-patterns
    confirmed_ap = [a for a in result.anti_patterns if a.get("confirmed")]
    if confirmed_ap:
        print(f"\n  ANTI-PATTERNS DETECTED ({len(confirmed_ap)}):")
        for a in confirmed_ap:
            print(f"    ! {a['pattern']}: {a.get('evidence', '')[:80]}")

    # Scores
    if result.tier_scores:
        print(f"\n  TIER SCORES:")
        for tier, score in result.tier_scores.items():
            print(f"    {tier}: {score}")

    if result.penalties:
        print(f"  Penalties: -{result.penalties}")

    print(f"\n  FINAL SCORE: {result.final_score}/100")
    print(f"  {result.summary}")
    print(f"{'=' * 60}")
