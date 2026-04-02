# ARC-AGI 3 — Claude Code Configuration

## START HERE — Read these docs first
1. **PROJECT.md** — Estrella polar: vision y proposito
2. **CURRENT_STATE.md** — Que hace el sistema HOY (friendly)
3. **TODO.md** — Board operativo (que sigue)
4. **CHANGELOG.md** — Historial de cambios
5. **research/README.md** — Indice de investigacion y preguntas abiertas

## LA PREGUNTA
> Como diseñamos un sistema que orqueste LLMs frontier para resolver
> tareas ARC-AGI-3 de manera competitiva, y como iteramos rapido
> sobre approaches hasta encontrar uno ganador?
>
> Aplicar al evaluar, disenar, priorizar, o revisar.

## Where to find what
| Necesito... | Ir a... |
|---|---|
| Entender la vision y principios | PROJECT.md |
| Ver que funciona HOY | CURRENT_STATE.md |
| Saber que sigue | TODO.md |
| Ver historial de cambios | CHANGELOG.md |
| Ver trabajo en progreso | issues/ (issues activos) |
| Encontrar findings de research | research/notes/ y research/synthesis/ |
| Como trabajar en este proyecto | Este archivo (CLAUDE.md) |
| Config de autoresearch | AUTORESEARCH.md |

## Project overview
Proyecto para **ganar ARC Prize 2026 en ARC-AGI-3**. Harness de META-COGNICION que
potencia GPT-5.4 para resolver entornos interactivos ARC-AGI-3 de forma GENERALIZABLE.

**OBJETIVO CENTRAL**: Construir scaffold que EXTRAIGA las capacidades latentes del LLM
via buenas preguntas, memoria/abstracciones, multi-agente (critic, sintesis, debate),
tools, y multi-run learning. El LLM es el CEREBRO — el harness lo potencia.

**PROHIBIDO**: Codigo game-specific, leer source code de juegos, hardcodear mecanicas,
reemplazar al LLM con solvers programaticos. Todo debe funcionar en TODOS los juegos.

## Environment setup
```bash
pip install -e ".[dev]"    # instalar proyecto + deps de desarrollo
```
**API keys ya configuradas en `.env`** — no hace falta setup de credenciales.

## Tech stack
- **Python 3.11+** — lenguaje principal
- **openai SDK** — cliente para Azure AI Foundry (patron v1 API, NO AzureOpenAI)
- **Azure AI Foundry** — plataforma LLM (DeepSeek, GPT, Llama, etc.)
- **arc-agi** — toolkit oficial para entornos ARC-AGI-3
- **Pillow + numpy** — manipulacion de grillas e imagenes
- **pytest + ruff** — testing y linting

## External services
- **Azure AI Foundry**: `OpenAI(base_url=AZURE_FOUNDRY_BASE_URL, api_key=AZURE_INFERENCE_CREDENTIAL)`
- **Budget**: $70 USD total for gpt-5.4. Check with `bash scripts/check_budget.sh`
- **Pricing**: Input ~$3/1M, Output ~$12/1M (output 4x more expensive!)
- **Models**: gpt-5.4, gpt-5.4-pro, gpt-5.3-chat, gpt-5.2-chat, gpt-5.2-codex, claude-opus-4-6, claude-sonnet-4-6
- Env vars: `AZURE_INFERENCE_CREDENTIAL`, `AZURE_FOUNDRY_BASE_URL`, `AZURE_MODEL`

## Project structure
```
PROJECT.md           # Vision, LA PREGUNTA, WHY
CLAUDE.md            # Este archivo — config para Claude Code
TODO.md              # Board operativo (NOW/NEXT/BLOCKED/LATER/DONE)
CURRENT_STATE.md     # Que existe hoy (friendly)
CHANGELOG.md         # Historial con refs I-NNN
AUTORESEARCH.md      # Config autoresearch (ON/OFF)
issues/              # Issue tracking local (I-NNN-slug.md)
experiments/         # Experimentos con manifest.yaml (ENNN-slug/)
research/            # Investigacion
  README.md          # Indice de research lines
  notes/             # Exploraciones y dumps pesados
  synthesis/         # Conclusiones consolidadas
  archive/           # Docs superados
src/arcagi3/         # Codigo fuente
  agent.py           # Agente LLM (Azure Foundry)
  grid_utils.py      # Conversiones grid->image/text
  run.py             # CLI entry point
.claude/skills/      # Skills del proyecto (/test, /status, /review)
pyproject.toml       # Dependencias y config
.env.example         # Template de env vars
```

## Code conventions
- Comunicacion: **Espanol** (siempre)
- Codigo y comments: English
- Type hints en funciones publicas
- Docstrings solo donde la logica no es obvia

## Commands
```bash
# Run agent
python -m arcagi3.run --game ls20
python -m arcagi3.run --game ls20 --no-vision
python -m arcagi3.run --list-games

# Tests
pytest
pytest -x -v

# Lint
ruff check .
ruff format .
```

## Quality assurance
- **Level 1 (Tests)**: pytest + ruff pre-commit — cada funcion tiene test
- **Level 2 (System)**: Correr agente en subset de tareas ARC-AGI con API real (no mocks)
- **Level 3 (External)**: Submission a Kaggle/ARC Prize leaderboard

## Issue tracking
- Issues en `issues/I-NNN-slug.md` con Status header + Log
- Cross-ref con `I-NNN` en commits, codigo, docs, TODO.md, CHANGELOG.md
- Experiments en `experiments/ENNN-slug/` con manifest.yaml
- TODO.md = board operativo (NOW/NEXT/BLOCKED/LATER)
- Ver dev-workflow skills para protocolo completo

## Commit workflow — MANDATORY
```
1. VALIDATE   — tests + lint
2. REVIEW     — Codex review si MCP disponible (ver /codex-collab)
3. PRESENT    — explicar en espanol, ESPERAR aprobacion
4. DOCS       — actualizar docs afectados (trigger table)
5. COMMIT     — con Co-Authored-By y refs I-NNN
```
**NUNCA commitear sin aprobacion explicita del usuario.**

## Autoresearch
- Config en AUTORESEARCH.md (ON/OFF + config del run)
- Branch: `autoresearch/<topic>-<date>` desde base explicita
- Commits + pushes en branch de autoresearch
- Status header en issues = memoria persistente
- Stop conditions obligatorias
- NO modificar PROJECT.md ni CURRENT_STATE.md en branches de autoresearch

## Document maintenance — trigger table
| Que cambio | Documentos a actualizar |
|---|---|
| Empezo trabajo en issue | `issues/I-NNN.md` Status → active. `TODO.md` mover a NOW. |
| Completo paso significativo | `issues/I-NNN.md` update Status + Log entry. |
| Corrio experimento | `experiments/ENNN/manifest.yaml` crear. `issues/I-NNN.md` add EXP. |
| Cerro issue | `issues/I-NNN.md` Conclusion + Status. `TODO.md` → DONE. `CHANGELOG.md`. |
| Completo tarea | `TODO.md` mark done. `CHANGELOG.md` con ref I-NNN. |
| Agrego/removio archivo o modulo | `CLAUDE.md` project structure. `CURRENT_STATE.md`. |
| Cambio API signature | `CURRENT_STATE.md`. |
| Cambio test count | `CURRENT_STATE.md` test section. |
| Agrego dependencia | `pyproject.toml` AND `CLAUDE.md` tech stack. |
| Cambio convencion | `CLAUDE.md` actualizar inmediatamente. |
| Cambio scope o vision | `PROJECT.md` primero, propagar a `CLAUDE.md` y `TODO.md`. |
| Research profundo | `research/notes/` + ref desde `issues/I-NNN.md`. |
| Conclusion de research | `research/synthesis/` + update issue. |
| Research → decision | Promover a `PROJECT.md`. Update issue. |
| Cambio project skills | Verificar seccion skills/commands de `CLAUDE.md`. |

## Cleanup and maintenance
- "Actualizar" = el ecosistema COMPLETO: docs, skills, memorias, scripts, configs
- Si un cambio hace codigo/tests/scripts obsoletos → **ELIMINARLOS** (git tiene historia)
- Si un doc referencia algo que ya no existe → **ARREGLAR la referencia**
- Despues de milestones grandes: cleanup pass (refs stale, dead code, orphaned files)
- Al cerrar issues: evaluar si research/notes/ puede moverse a research/archive/

## Git conventions
- Branch: `main` para ahora (proyecto de research, una persona)
- Commits: imperative mood, en ingles, concisos, con ref I-NNN cuando aplica
- NUNCA force push, NUNCA commit sin aprobacion
