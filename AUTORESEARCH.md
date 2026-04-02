# Autoresearch

## Status: ON

## Session: 2026-04-02 — Capability Testing & Harness Iteration

### Objective
Validar la hipotesis core: los LLMs (GPT-5.4) YA tienen las capacidades
cognitivas para ARC-AGI-3 pero necesitan un harness inteligente para extraerlas.
Iterar sobre el harness hasta encontrar un approach que mueva la aguja.

### Workflow (MUST FOLLOW)
```
research/analysis → reasoning → debate (con Codex) → hypothesis
→ implementation → testing (run agent) → analysis
→ reasoning → debate → pivot or iterate
→ loop until breakthrough or budget exhausted
```

### Rules
- **NEVER STOP** — seguir iterando toda la noche
- **PIVOT FAST** — si una idea no muestra progreso en 2-3 tests, pivotear
- **USE CODEX** — codex-reply para debate en cada ciclo. Es trabajo de los dos.
- **BUDGET** — max $30 USD con GPT-5.4. No usar otros modelos pagos.
- **TRACK EVERYTHING** — cada ciclo queda en I-003, cada experimento queda logueado
- **COMMIT REGULARLY** — commit + push despues de cada milestone significativo

### Budget tracking
- Total budget: $70 USD shared resource (amalia-resource)
- Spent (as of 2026-04-02 01:20 UTC): $4.19 (6%)
- Today's estimate: ~$2.60
- Check: `bash scripts/check_budget.sh`
- Model: gpt-5.4 via Azure Foundry
- Rule: check budget every ~5 experiments. Warn at $60, stop at $70.
- Output tokens are 4x more expensive than input — keep responses SHORT

### Pivot log
| Hora | Idea | Resultado | Decision |
|------|------|-----------|----------|
| (se llena durante el run) | | | |

### Stop conditions
- Budget agotado ($30)
- Breakthrough claro (approach que resuelve >50% de tareas testeadas)
- Todas las ideas razonables exploradas sin progreso
