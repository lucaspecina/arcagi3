# Current State

## Que es esto?
Un proyecto de investigacion para ARC Prize 2026. Tenemos un agente LLM que
interactua con entornos ARC-AGI-3 (puzzles interactivos turn-based en grillas
64x64 con 16 colores). El agente usa Azure AI Foundry para inferencia y el
toolkit oficial `arc-agi` para conectarse a los entornos.

La idea es explorar si los LLMs pueden razonar sobre estos puzzles, empezando
simple y midiendo todo.

## Que funciona hoy

### Agente LLM para ARC-AGI-3 (`src/arcagi3/agent.py`)
- Loop de decision: observa grilla → razona → elige accion → ejecuta
- Dos modos: **vision** (grilla como imagen PNG) y **text** (grilla como hex compacto)
- Message history completo para contexto entre turnos
- Prompt estructurado con system message + observaciones + historial
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
