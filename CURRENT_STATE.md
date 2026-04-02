# Current State

## Que es esto?
Un proyecto para **ganar ARC Prize 2026**. Construimos un harness de meta-cognicion
que potencia GPT-5.4 para resolver puzzles interactivos ARC-AGI-3 (grillas 64x64,
16 colores, turn-based). El harness extrae las capacidades latentes del LLM via
buenas preguntas, memoria, abstracciones, multi-agente, y tools.

**El LLM es el cerebro. El harness lo potencia. Todo generalizable a todos los juegos.**

## Que funciona hoy

### Agente LLM para ARC-AGI-3 (`src/arcagi3/agent.py`)
- Loop de decision: observa grilla → razona → elige accion → ejecuta
- Dos modos: **vision** (grilla como imagen PNG) y **text** (grilla como hex compacto)
- Acciones genéricas (ACTION1-7) — el agente descubre qué hace cada una
- Filtra acciones disponibles del environment (`available_actions` del frame)
- Memoria persistente (dict JSON) que sobrevive entre turnos
- Message history completo para contexto entre turnos
- Configuracion flexible via `AgentConfig` (modelo, max actions, vision on/off)

### Utilidades de grilla (`src/arcagi3/grid_utils.py`)
- `grid_to_image()` — convierte grilla 64x64 a PIL Image con colores ARC
- `grid_to_base64()` — grilla a base64 PNG (para enviar al LLM)
- `grid_to_text_compact()` — grilla a texto hex (1 char por celda, mas barato en tokens)
- `image_diff()` — imagen que muestra diferencias entre dos grillas

### CLI (`src/arcagi3/run.py`)
- `python -m arcagi3.run --game ls20` — correr agente en un juego
- `python -m arcagi3.run --game ls20 --no-vision` — modo text-only
- `python -m arcagi3.run --list-games` — listar juegos disponibles

### Research (`research/notes/`)
- Overview de ARC-AGI-2 vs ARC-AGI-3, approaches conocidos, estado del arte
- Documentacion completa del toolkit `arc-agi` (API, agents, benchmarking)
- Como cargar datasets ARC-AGI-1/2 en Python

## Que NO funciona todavia
- **No testeado con API keys reales** — necesita credenciales Azure Foundry + ARC API key
- **Sin tests** — no hay pytest todavia
- **Parse de respuesta LLM basico** — regex JSON, fragil
- **Sin retry logic** — errores de API no se reintentan
- **Sin checkpointing** — si se interrumpe, se pierde todo el estado
- **Sin truncation de historial** — message history crece sin limite
- **Sin pipeline ARC-AGI-2** — solo soporta ARC-AGI-3 (interactivo)

## Como probarlo
```bash
# Setup
pip install -e ".[dev]"
cp .env.example .env
# Editar .env con tus credenciales Azure Foundry + ARC API key

# Correr
python -m arcagi3.run --game ls20              # vision mode
python -m arcagi3.run --game ls20 --no-vision  # text-only mode
python -m arcagi3.run --list-games             # ver juegos disponibles
```
