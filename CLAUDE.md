# ARC-AGI 3 — Claude Code Configuration

## START HERE — Read these docs first
1. **PROJECT.md** — Estrella polar: vision y LA PREGUNTA
2. **CURRENT_STATE.md** — Que hace el sistema HOY (friendly)
3. **TODO.md** — Board operativo (NOW/NEXT/BLOCKED/LATER)
4. **CHANGELOG.md** — Historial de cambios
5. **research/README.md** — Indice de investigacion y preguntas abiertas

## LA PREGUNTA — the guiding question
> Cual es el approach mas efectivo para resolver tareas ARC-AGI usando LLMs
> como motor de razonamiento, y como lo validamos rapido?

## Where to find what
| I need to... | Go to... |
|---|---|
| Understand the vision and principles | PROJECT.md |
| See what works TODAY | CURRENT_STATE.md |
| Know what is next | TODO.md |
| See change history | CHANGELOG.md |
| See work in progress | issues/ (active issues) |
| Find research findings | research/notes/ and research/synthesis/ |
| Configure autoresearch | AUTORESEARCH.md |
| How to work on this project | This file (CLAUDE.md) |

## Project overview
Investigacion para ARC Prize 2026. Pipeline de inferencia LLM para tareas ARC-AGI.
Foco inicial en ARC-AGI-2 (estatico), research paralelo en ARC-AGI-3 (interactivo).
Principio: empezar simple, medir todo, iterar rapido.

## Environment setup
```bash
pip install -e ".[dev]"    # instalar proyecto + deps de desarrollo
cp .env.example .env       # configurar API keys
```

## Tech stack
- **Python 3.11+** — lenguaje principal
- **openai SDK** — cliente para Azure AI Foundry (patron v1 API, NO AzureOpenAI)
- **Azure AI Foundry** — plataforma LLM (DeepSeek, GPT, Llama, etc.)
- **pytest** — testing

## External services
- **Azure AI Foundry**: `OpenAI(base_url=AZURE_FOUNDRY_BASE_URL, api_key=AZURE_INFERENCE_CREDENTIAL)`
- Env vars: `AZURE_INFERENCE_CREDENTIAL`, `AZURE_FOUNDRY_BASE_URL`, `AZURE_MODEL`

## Project structure
```
PROJECT.md           # Vision, LA PREGUNTA, WHY
CLAUDE.md            # Este archivo — config para Claude Code
TODO.md              # Board operativo (NOW/NEXT/BLOCKED/LATER)
CURRENT_STATE.md     # Estado actual del sistema
CHANGELOG.md         # Historial
AUTORESEARCH.md      # Modo autonomo (ON/OFF)
issues/              # Tracking local (I-NNN)
experiments/         # Experimentos formales (ENNN)
research/            # Investigacion
  README.md          # Indice de research lines
  notes/             # Exploraciones y notas
  synthesis/         # Conclusiones consolidadas
  archive/           # Docs viejos/superados
src/arcagi3/         # Codigo fuente
  agent.py           # Agente LLM (Azure Foundry)
  grid_utils.py      # Conversiones grid->image/text
  run.py             # CLI entry point
pyproject.toml       # Dependencias y config
.env.example         # Template de env vars
.claude/skills/      # Skills del proyecto
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
python -m arcagi3.run --game ls20 --no-vision    # text-only mode
python -m arcagi3.run --list-games                # listar juegos

# Tests
pytest
pytest -x -v

# Lint
ruff check .
ruff format .
```

## Quality assurance
- **Level 1 (Tests)**: pytest pre-commit — cada funcion tiene test
- **Level 2 (System)**: Correr pipeline completo en subset de tareas ARC-AGI
- **Level 3 (External)**: Submission a Kaggle leaderboard

## Issue tracking
- Issues en `issues/I-NNN-slug.md` con Status header + Log
- Cross-ref con `I-NNN` en commits, codigo, docs, otros issues
- Experiments en `experiments/ENNN-slug/` con manifest.yaml
- TODO.md = board operativo (NOW/NEXT/BLOCKED/LATER)
- Ver dev-workflow skills para protocolo completo

## Commit workflow — MANDATORY
1. Tests + Validation (Level 1 minimo)
2. Codex review (si MCP disponible)
3. Presentar al usuario en espanol — SIEMPRE, ESPERAR aprobacion
4. Actualizar docs (trigger table) + Commit con Co-Authored-By e I-NNN refs
5. Sugerir next steps

**NUNCA commitear sin aprobacion explicita del usuario.**

## Autoresearch
- Config en AUTORESEARCH.md (ON/OFF + config del run)
- Branch: `autoresearch/<topic>-<date>` desde base explicita
- Commits + pushes en branch de autoresearch
- Status header en issues = memoria persistente
- Stop conditions obligatorias
- Ver dev-workflow/autoresearch.md para protocolo completo

## Document maintenance — trigger table
| What changed | Documents to update |
|---|---|
| Started working on an issue | `issues/I-NNN.md` Status → active. `TODO.md` move to NOW. |
| Completed a significant step | `issues/I-NNN.md` update Status header + add Log entry. |
| Completed a task | `TODO.md` mark done. `CHANGELOG.md` add entry with I-NNN ref. |
| Ran an experiment | `experiments/ENNN/manifest.yaml` create. `issues/I-NNN.md` add EXP entry. |
| Closed an issue | `issues/I-NNN.md` Conclusion + Status. `TODO.md` move to DONE. `CHANGELOG.md` if code changed. Evaluate `research/notes/` → `research/archive/`. |
| Added/removed a file or module | `CLAUDE.md` project structure. `CURRENT_STATE.md` modules. |
| Changed an API signature | `CURRENT_STATE.md` Key APIs section. |
| Changed test count | `CURRENT_STATE.md` test coverage section. |
| Added a dependency | `pyproject.toml` AND `CLAUDE.md` tech stack. |
| Changed a convention | `CLAUDE.md` update immediately. |
| Changed scope or vision | `PROJECT.md` first, propagate to `CLAUDE.md` and `TODO.md`. |
| Deep research done | `research/notes/` + ref from `issues/I-NNN.md`. |
| Research conclusion reached | `research/synthesis/` + close or update issue. |
| Research becomes decision | Promote to `PROJECT.md`. Update issue status. |
| New issue created | `issues/I-NNN.md` create. `TODO.md` add to appropriate section. |
| Renamed/removed a function or module | Search ALL docs, skills, memories for references → update or remove. |
| Abandoned an issue | `issues/I-NNN.md` document why. `TODO.md` remove or note abandoned. |
| Changed project skills | Verify `CLAUDE.md` skills/commands section still accurate. |

## Cleanup and maintenance
- "Updating" means the FULL ecosystem: docs, skills, memories, scripts, configs
- If a change makes code/tests/scripts obsolete → DELETE THEM (git has history)
- If a doc references something that no longer exists → FIX the reference
- After major milestones: cleanup pass (stale refs, dead code, orphaned files)
- When closing issues: evaluate if research/notes/ can move to research/archive/

## Worktrees
When running multiple Claude Code sessions on the same repo, each session
MUST work in its own git worktree. Main session consolidates doc changes.

## Git conventions
- Branch: `main` para ahora (proyecto de research, una persona)
- Commits: imperative mood, en ingles, concisos
- NUNCA force push, NUNCA commit sin aprobacion
