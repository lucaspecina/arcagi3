# Current State

## Que es esto?
Un proyecto para **ganar ARC Prize 2026**. Construimos un harness de meta-cognicion
que potencia GPT-5.4 para resolver puzzles interactivos ARC-AGI-3 (grillas 64x64,
16 colores, turn-based). El harness extrae las capacidades latentes del LLM via
buenas preguntas, memoria, abstracciones, multi-agente, y tools.

**El LLM es el cerebro. El harness lo potencia. Todo generalizable a todos los juegos.**

## Que funciona hoy

### Agente LLM — Analyzer/Reflector/Actor loop (`src/arcagi3/agent.py`)
Un solo LLM con 3 prompts distintos por step:
- **Analyzer** (cada N steps): percepcion completa de la escena, clasificacion de objetos
- **Actor** (cada step): elige accion basado en beliefs validadas
- **Reflector** (cada step, post-accion): meta-cognicion forzada, revisa CADA belief (KEEP/CHANGE/DROP)
- **Trackers deterministicos** (sin LLM): AvatarTracker, BarTracker, ExplorationController
- **Fase 0**: exploracion sistematica sin LLM (prueba cada accion desde posicion inicial)
- Belief transfer entre runs via `--prior-beliefs`
- Stagnation detection con forced goal change (5+ steps sin progreso)

### Bench runner (`src/arcagi3/bench.py`)
- Cadena de N runs con belief transfer automatico (run1 → beliefs → run2 → ...)
- Paralelizacion por juego (ls20 y g50t corren en threads separados)
- Metrica compuesta: scorecard real > levels completed > judge score
- `python -m arcagi3.bench --games ls20 --runs 3 --max-actions 30 --judge`

### LLM Judge Oracle (`src/arcagi3/judge.py`)
- Evalua entendimiento del agente contra golden thinking (`golden/<game>.md`)
- Recibe: progresion de beliefs (cada run) + action history del ultimo run
- Milestones en 4 tiers: percepcion (0-20) → mecanicas (20-50) → goal (50-75) → ejecucion (75-100)
- Anti-patterns con penalidades
- Modelo: gpt-5.4-mini (consistente, bajo costo)

### Golden thinking (`golden/`)
- `ls20.md` — completo (controls, mecanicas, milestones, anti-patterns)
- `g50t.md` — template vacio (pendiente)

### Utilidades de grilla (`src/arcagi3/grid_utils.py`)
- `grid_to_image()` — grilla 64x64 a PIL Image
- `grid_to_base64()` — grilla a base64 PNG (para enviar al LLM)
- `grid_to_text_compact()` — grilla a texto hex (1 char por celda)
- `compute_diff()` — diff entre dos grillas con deteccion de movimientos

### CLI (`src/arcagi3/run.py`)
- `python -m arcagi3.run --game ls20` — correr agente
- `python -m arcagi3.run --game ls20 --step --window` — interactivo + ventana visual
- `python -m arcagi3.run --game ls20 --judge` — con evaluacion LLM judge
- `python -m arcagi3.run --list-games` — listar juegos

## Modelos
- **Desarrollo/runs serios**: gpt-5.4
- **Autoresearch (iteracion rapida)**: gpt-5.4-mini (36s/5 actions, 5x mas rapido)
- **Judge**: gpt-5.4-mini

## Ultimo benchmark (2026-04-06)
- 3 runs encadenados, ls20, 15 actions, gpt-5.4-mini
- Judge score: 20/100 (Tier 1 completo: avatar, controles, "+", barra)
- No llego a descubrir mecanica del "+" ni goal de matching
- Wall time: ~12 min para cadena completa + judge

## Que NO funciona todavia
- **Golden thinking g50t** — pendiente de llenar
- **Autoresearch loop autonomo** — bench existe pero falta el loop keep/discard + results.tsv
- **Sin tests** — no hay pytest todavia
- **15 acciones insuficientes** — necesita 30+ para llegar al "+" en ls20
