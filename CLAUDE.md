# ARC-AGI 3 — Claude Code Configuration

## START HERE — Read these docs first
1. **PROJECT.md** — Vision y proposito (THE north star)
2. **CURRENT_STATE.md** — Que hace el sistema HOY
3. **TODO.md** — Tareas (que esta hecho/pendiente)
4. **CHANGELOG.md** — Historial de cambios
5. **research/README.md** — Indice de investigacion y preguntas abiertas

## LA PREGUNTA — the guiding question
> Cual es el approach mas efectivo para resolver tareas ARC-AGI usando LLMs
> como motor de razonamiento, y como lo validamos rapido?

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
TODO.md              # Tareas
CURRENT_STATE.md     # Estado actual del sistema
CHANGELOG.md         # Historial
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

## Commit workflow — MANDATORY
1. Tests + Validation (Level 1 minimo)
2. Codex review (si MCP disponible)
3. Presentar al usuario en espanol — SIEMPRE, ESPERAR aprobacion
4. Actualizar docs + Commit
5. Sugerir next steps

**NUNCA commitear sin aprobacion explicita del usuario.**

## Document maintenance — trigger table
| What changed | Documents to update |
|---|---|
| Completed a task | `TODO.md` mark [x]. `CHANGELOG.md` add entry. `CURRENT_STATE.md` if capabilities changed. |
| Added/removed a file or module | `CLAUDE.md` project structure. `CURRENT_STATE.md` modules section. |
| Changed an API signature | `CURRENT_STATE.md` Key APIs section |
| Changed test count | `CURRENT_STATE.md` test coverage section |
| Added a dependency | `pyproject.toml` AND `CLAUDE.md` tech stack |
| Changed a convention | `CLAUDE.md` update immediately |
| Changed scope or vision | `PROJECT.md` first, then propagate to `CLAUDE.md` and `TODO.md` |
| New exploration/debate | `research/notes/` + update `research/README.md` index |
| Consolidated research finding | `research/synthesis/` + update `research/README.md` index |

## Git conventions
- Branch: `main` para ahora (proyecto de research, una persona)
- Commits: imperative mood, en ingles, concisos
- NUNCA force push, NUNCA commit sin aprobacion
