---
id: 4
title: Autoresearch multi-worker estilo CORAL (3 claude headless en paralelo)
status: planned
type: infrastructure
created: 2026-04-06
related: [I-003]
---

# I-004: Autoresearch multi-worker estilo CORAL

## Status
- **Estado:** Diseñado, NO implementado. Esperando validación del approach single-brain (esta noche) antes de invertir.
- **Hipotesis:** 3 instancias `claude -p` headless en worktrees aislados, cada una con scope distinto, comparten estado via `shared/` symlinkeado, producen diversidad real de exploración (vs single-brain donde mi sesgo determina la búsqueda).
- **Por que esperar:** Riesgos abiertos (claude -p con loops largos no probado, ARC-AGI-3 401 con paralelismo sin diagnosticar, costo de 3 Opus en paralelo sin estimar). Validar primero que single-brain funciona, después escalar.

## OBJETIVO
Construir una capa de autoresearch multi-agent inspirada en CORAL
(github.com/Human-Agent-Society/CORAL) y karpathy/autoresearch, donde
3 workers Claude Code corren en paralelo, cada uno explorando una capa
distinta del harness, compartiendo findings via filesystem.

## Por qué esto y no single-brain
- **Single-brain** (yo, Claude Code): el cerebro es uno solo, mi sesgo cognitivo
  determina qué ideas se prueban. El paralelismo solo está en los benches
  (cómputo), no en la generación de hipótesis. Es **1 autoresearch acelerado**,
  no multi-agent.
- **Multi-worker estilo CORAL**: cada worker es un LLM call con contexto
  independiente. Diversidad de exploración EMERGE — nadie le dice qué probar,
  pero al ver `shared/notes/` y tener scopes distintos, naturalmente cada uno
  explora un espacio diferente. Es lo más cercano a CORAL real con tooling
  actual.

## Inspiración: CORAL + karpathy/autoresearch

### CORAL (multi-agent autonomous research infrastructure)
- N agentes en git worktrees aislados
- `.coral/public/` symlinkeado a cada worktree (attempts, notes, skills)
- `coral eval` = stage + commit + grade en un comando
- Manager con heartbeat prompts ("reflect", "consolidate")
- `max_turns` por agente, después reboot
- CLI: `coral start | stop | status | log | ui | resume`
- Stop: `coral stop` halt all, `max_turns` self-stop, work persistido en git

### karpathy/autoresearch (minimalista)
- Sin auto-stop, corre forever
- Control via `program.md` editado por humano
- Time budget DURO por experimento (5 min training)
- Pattern: "wake up in the morning to a log of experiments"

### Lo que copiamos de cada uno
| Mecanismo | Inspirado en | Cómo |
|---|---|---|
| Time budget hard por bench | karpathy (5 min) | `timeout 2400` ya existe |
| Max iterations por worker | CORAL (max_turns) | `MAX_ITERATIONS=6` chequeado por worker |
| Stop graceful via archivo | karpathy (program.md) | `shared/STOP` chequeado entre iteraciones |
| Stop forzado | CORAL (coral stop) | `scripts/kill_workers.sh` con pkill |
| Persistencia git | ambos | git commits + worktrees + shared/results.tsv |
| Resume | CORAL (coral resume) | re-correr spawn script, lee results.tsv para baseline |
| Shared state symlinkeado | CORAL (.coral/public/) | `shared/` symlinkeado a cada worktree |
| Eval wrapper | CORAL (coral eval) | `scripts/eval.sh` (stage+commit+bench+log) |

### Lo que NO copiamos
| CORAL tiene | Por qué no |
|---|---|
| Web dashboard (puerto 8420) | overkill para 3 workers, tail logs alcanza |
| Heartbeat interrupts mid-run | requiere IPC, complejidad alta vs ganancia baja |
| Reboot automático en max_turns | requiere loop bash padre, postponer a v2 |
| LiteLLM proxy | ya tenemos Azure Foundry directo |
| Grader plug-in abstracto | judge.py + bench.py ya funcionan |

## Arquitectura

### Estructura de archivos
```
arcagi3/                                # main repo
├── AUTORESEARCH.md                     # ya existe — protocolo común a todos
├── scripts/
│   ├── spawn_autoresearch.sh           # NUEVO — manager: crea worktrees + lanza 3 workers
│   ├── kill_workers.sh                 # NUEVO — kill duro de emergencia
│   ├── eval.sh                         # NUEVO — wrapper "stage+commit+bench+log"
│   └── worker_prompt.md                # NUEVO — prompt completo de cada worker
└── src/arcagi3/                        # ya existe

../shared/                              # NUEVO, fuera de git
├── STOP                                # archivo kill switch (touch para frenar)
├── results.tsv                         # leaderboard global, todos appendean (flock)
├── results.tsv.lock
├── api.lock                            # serializa llamadas ARC-AGI-3 (mitiga 401)
├── scope/
│   ├── perception.md                   # qué archivos puede tocar perception
│   ├── mechanics.md                    # qué archivos puede tocar mechanics
│   └── goal.md                         # qué archivos puede tocar goal
├── notes/                              # findings cross-worker (timestamped)
├── logs/                               # stdout/stderr de cada worker + manager
└── done/                               # marker de fin con summary JSON

../arcagi3-perception/                  # git worktree, branch autoresearch/perception-<date>
../arcagi3-mechanics/                   # git worktree
../arcagi3-goal/                        # git worktree
```

Cada worktree tiene `shared → ../shared` (symlink) y `.env` copiado.

### Componentes

**1. `scripts/spawn_autoresearch.sh` (manager)**
- Crea `../shared/` con subcarpetas
- Borra `shared/STOP` (limpia estado previo)
- Crea 3 git worktrees con branches `autoresearch/<scope>-<fecha>`
- Symlinkea shared/ y copia .env
- Spawna 3 procesos `claude -p` en background con `--dangerously-skip-permissions --model claude-opus-4-6`
- Hace `wait` y reporta cuando todos terminan

**2. `scripts/worker_prompt.md` (cerebro de cada worker)**
- Identidad: "worker autónomo en worktree X, scope Y, parte de 3 workers paralelos"
- Read first: CLAUDE.md → AUTORESEARCH.md → shared/scope/<scope>.md → shared/results.tsv → shared/notes/*.md
- Loop: think → edit (dentro de scope) → eval.sh → keep/discard → notes (opcional) → repeat
- Stop checks ANTES de cada iteración:
  - `shared/STOP` existe? → escribir done/<scope>.json, exit
  - `MAX_ITERATIONS=6` alcanzado? → exit
  - 5 discards seguidos? → exit
- Al terminar: escribir `shared/done/<scope>.json` con summary

**3. `shared/scope/<scope>.md` (constraints)**
Define qué archivos puede modificar cada worker. Fuerza diversidad sin
asignación explícita de ideas. Tres scopes:
- **perception**: ANALYZER prompt, grid_utils.py, trackers.py → Tier 1
- **mechanics**: REFLECTOR prompt, belief schema → Tier 2
- **goal**: ACTOR prompt, exploration.py, stagnation logic → Tier 3 (donde más ganamos, 0/75)

**4. `scripts/eval.sh` (wrapper estilo coral eval)**
```bash
#!/usr/bin/env bash
set -e
MSG="$1"
SCOPE=$(basename "$(pwd)" | sed 's/arcagi3-//')
git add -A
git commit -m "$MSG" || { echo "Nothing to commit"; exit 1; }
COMMIT=$(git rev-parse --short HEAD)
LOG=$(mktemp)
flock shared/api.lock -c "timeout 2400 python -m arcagi3.bench \
  --games ls20 --runs 3 --max-actions 30 \
  --model gpt-5.4-mini --judge --no-parallel" 2>&1 | tee "$LOG"
METRIC=$(grep -oP "COMBINED METRIC: \K[0-9.]+" "$LOG" | tail -1)
JUDGE=$(grep -oP "Judge score: \K[0-9.]+" "$LOG" | tail -1)
flock shared/results.tsv.lock -c "echo -e '$COMMIT\t$METRIC\t$JUDGE\t-\t0\t$SCOPE\t$MSG' >> shared/results.tsv"
BEST=$(awk -F'\t' 'NR>1 {print $2}' shared/results.tsv | sort -gr | head -1)
if (( $(echo "$METRIC > $BEST" | bc -l) )); then
  echo "KEEP: $METRIC > $BEST"
  exit 0
else
  echo "DISCARD: $METRIC <= $BEST"
  git reset HEAD~1 --hard
  exit 1
fi
```

`flock` resuelve dos cosas: race en results.tsv Y serialización ARC-AGI-3 (mitiga 401).

**5. `scripts/kill_workers.sh` (botón rojo)**
```bash
#!/usr/bin/env bash
touch shared/STOP                       # graceful primero
sleep 60
pkill -f "claude.*--dangerously-skip-permissions" 2>/dev/null
echo "Workers killed."
```

## Stops (3 mecanismos en cascada)

| Trigger | Quién | Qué pasa |
|---|---|---|
| MAX_ITERATIONS=6 | worker | done/<scope>.json + exit limpio |
| 5 discards seguidos | worker | done/<scope>.json (razón: plateau) + exit |
| `touch shared/STOP` | humano | worker chequea, exit limpio |
| `timeout 14400` (4h) | spawn script | proceso muere |
| `kill_workers.sh` | humano emergencia | graceful 1 min + pkill duro |

## Cómo se decide ganador

Cuando los 3 terminan:
1. Yo (orchestrator humano-asistido) leo `shared/results.tsv` y los 3 `done/*.json`
2. Identifico qué scope produjo la mejora más grande sobre baseline
3. **Verifico que NO sea ruido**: re-corro el bench del commit ganador 1-2 veces
4. Si replica → mergeo esa branch a main
5. Otras 2 branches: archivar o mantener para iterar
6. Cleanup: borrar worktrees, mantener `shared/results.tsv` y `notes/` como histórico

## Riesgos abiertos (DEBEN validarse antes de full run)

| # | Riesgo | Mitigación / cómo validar |
|---|---|---|
| 1 | `claude -p` con loops largos (1-2 h) nunca probado | Test minimal: `claude -p "leé CLAUDE.md y dame 3 líneas"` (10 min) |
| 2 | `--dangerously-skip-permissions` en Windows | Validar en test minimal |
| 3 | Costo Opus 4.6 × 3 procesos × 2h sin estimar | Estimar tokens del worker_prompt × iteraciones × 3 antes de lanzar |
| 4 | Symlinks Windows requieren Developer Mode | Plan B: copiar shared/ + script merge final |
| 5 | ARC-AGI-3 API 401 con paralelismo | Mitigación ya en diseño: `flock shared/api.lock` serializa llamadas |
| 6 | Race conditions en `shared/results.tsv` | Mitigación ya en diseño: `flock shared/results.tsv.lock` |
| 7 | Goodhart sobre el judge | Verificar que `levels_completed` no quede en 0 mientras judge sube |
| 8 | Variance del baseline desconocida | Pre-requisito: medir desvío del baseline antes (parte del trabajo single-brain) |
| 9 | Sesgo de selección al elegir ganador | Replica obligatoria del commit ganador antes de mergear |
| 10 | Cleanup olvidado deja worktrees colgados | Script `cleanup_round.sh` obligatorio |

## Plan de validación incremental (NO saltarse pasos)

1. **Test minimal `claude -p`** (10 min) — confirmar headless en Windows, devuelve, no se cuelga
2. **Crear shared/, scopes, eval.sh** (30 min)
3. **Test 1 worker solo, MAX_ITERATIONS=2** (1.5 h) — validar loop, locks, results.tsv, done.json
4. **Test 3 workers paralelo, MAX_ITERATIONS=2** (1.5 h) — validar locks concurrentes, ARC-AGI-3 serialización, terminación limpia
5. **Run real, MAX_ITERATIONS=6** (~10 h wall time)
6. **Decisión ganador, replicar, mergear**

## Pre-requisitos (lo que tiene que existir antes)

- [ ] Single-brain autoresearch funcionando (esta noche, sin worktrees)
- [ ] Varianza del baseline medida (3-5 runs del commit baseline para conocer std del judge)
- [ ] Al menos 1 mejora sobre baseline confirmada en single-brain (señal de que el harness es mejorable)
- [ ] Estimación de costo por iteración (tokens × precio × iteraciones × workers)

## Conexión con CORAL real (referencia)

Si en algún momento queremos usar CORAL directamente en vez de adaptarlo:
- Repo: github.com/Human-Agent-Society/CORAL
- Comando típico: `uv run coral start -c task.yaml`
- task.yaml define: task description, grader, agents (count, runtime, model, max_turns), workspace
- Grader: subclass de `TaskGrader` con `evaluate() → float | ScoreBundle`
- Posible plan: empaquetar nuestro bench+judge como un grader CORAL y dejar
  que CORAL maneje toda la orquestación. Más simple pero menos flexible.

## Decisión pendiente
Cuando retomemos este issue (después de validar single-brain esta noche):
1. Implementar adaptación propia (control total, ~1 día de setup)
2. Empaquetar como grader CORAL y usar su CLI (menos control, ~2-3 h setup)

## Log

### 2026-04-06 · DISEÑO — Autoresearch multi-worker inspirado en CORAL
- Investigado: CORAL (Human-Agent-Society) y karpathy/autoresearch
- Confirmado: lo más fiel a CORAL con tooling actual es spawnar `claude -p` headless con worktrees
- Decidido: postergar implementación hasta validar approach single-brain primero
- Anotado el diseño completo en este issue para retomar después
