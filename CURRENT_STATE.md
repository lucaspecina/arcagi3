# Current State

## Overview
Pipeline basico de agente LLM para ARC-AGI-3 implementado. Usa Azure AI Foundry
para inferencia y el toolkit oficial `arc-agi` para interactuar con los entornos.
Todavia no se ha testeado (requiere API keys configuradas).

## Modules / Components
| Module | Purpose | Status |
|--------|---------|--------|
| `src/arcagi3/agent.py` | Agente LLM — choose_action loop con vision+text | Implementado |
| `src/arcagi3/grid_utils.py` | Grid->image, grid->text, image diff | Implementado |
| `src/arcagi3/run.py` | CLI entry point | Implementado |
| `research/` | Investigacion y notas | Research inicial completo |

## Key APIs
- `run_agent(env, config)` — corre el agente en un environment, retorna `AgentState`
- `AgentConfig` — configuracion: model, base_url, api_key, max_actions, use_vision, etc.
- `grid_to_image(grid, scale)` — convierte grilla 64x64 a PIL Image
- `grid_to_base64(grid, scale)` — grilla a base64 PNG
- `grid_to_text_compact(grid)` — grilla a texto hex (1 char por celda)
- `image_diff(prev, curr, scale)` — imagen diff entre dos grillas

## Test coverage
Sin tests todavia.

## Known limitations
- No se ha testeado con API keys reales
- Parse de respuesta LLM es basico (regex JSON)
- No hay retry logic para errores de API
- No hay checkpointing (se pierde estado si se interrumpe)
- Message history fija (no hay truncation inteligente)
- Falta integracion con ARC-AGI-2 (estatico)
