# ARC-AGI-3 Toolkit — Como cargar y usar

Fecha: 2026-03-27

## Instalacion

```bash
pip install arc-agi   # v0.9.6 — incluye arcengine como dependencia
# o
uv add arc-agi
```

Dependencias auto-instaladas: `arcengine>=0.9.3`, `flask`, `matplotlib`, `pydantic`, `python-dotenv`, `pillow`, `requests`

## API Key

Registrarse en https://three.arcprize.org para obtener `ARC_API_KEY`.
Sin key se usa una anonima con acceso limitado.

```bash
export ARC_API_KEY="your-api-key-here"
# o en .env
```

## Core API

### Imports
```python
import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState, FrameDataRaw
```

### Arcade (entry point)
```python
arc = arc_agi.Arcade(
    arc_api_key="",                          # o ARC_API_KEY env var
    operation_mode=OperationMode.NORMAL,     # NORMAL | ONLINE | OFFLINE | COMPETITION
)
```

Metodos principales:
- `arc.make(game_id, seed=0, render_mode=None)` -> `EnvironmentWrapper`
- `arc.get_environments()` -> lista de juegos disponibles
- `arc.create_scorecard()` -> scorecard ID
- `arc.get_scorecard()` -> `EnvironmentScorecard`
- `arc.close_scorecard()` -> scorecard final

### EnvironmentWrapper
```python
env = arc.make("ls20", render_mode="terminal")
env.observation_space   # FrameDataRaw | None — ultima observacion
env.action_space        # list[GameAction] — acciones disponibles
env.info                # EnvironmentInfo — metadata del juego

obs = env.reset()       # reiniciar, obtener observacion inicial
obs = env.step(action, data={}, reasoning={})  # ejecutar accion
```

### GameAction
```python
from arcengine import GameAction

GameAction.RESET      # reiniciar
GameAction.ACTION1    # arriba
GameAction.ACTION2    # abajo
GameAction.ACTION3    # izquierda
GameAction.ACTION4    # derecha
GameAction.ACTION5    # enter/espacio/delete
GameAction.ACTION6    # click (requiere x,y en data)
GameAction.ACTION7    # undo

# ACTION6 necesita coordenadas:
action = GameAction.ACTION6
env.step(action, data={"x": 32, "y": 32})

# Helpers:
action.is_simple()    # True para RESET, ACTION1-5, ACTION7
action.is_complex()   # True para ACTION6
```

### Observacion (FrameDataRaw)
```python
obs = env.step(GameAction.ACTION1)
obs.frame              # list[numpy.ndarray] — grillas 64x64, valores 0-15
obs.state              # GameState: NOT_PLAYED | IN_PROGRESS | WIN | GAME_OVER
obs.levels_completed   # int
obs.win_levels         # int — niveles necesarios para ganar
obs.available_actions  # list[int] — IDs de acciones disponibles
```

### Paleta de colores (16)
| Index | Color | Index | Color |
|-------|-------|-------|-------|
| 0 | White #FFFFFF | 8 | Red #F93C31 |
| 1 | Off-white #CCCCCC | 9 | Blue #1E93FF |
| 2 | Neutral light #999999 | 10 | Blue light #88D8F1 |
| 3 | Neutral #666666 | 11 | Yellow #FFDC00 |
| 4 | Off-black #333333 | 12 | Orange #FF851B |
| 5 | Black #000000 | 13 | Maroon #921231 |
| 6 | Magenta #E53AA3 | 14 | Green #4FCC30 |
| 7 | Magenta light #FF7BCC | 15 | Purple #A356D6 |

## Scoring (RHAE)
```
level_score = min(((baseline_actions / actions_taken) ** 2) * 100, 100.0)
game_score = weighted_avg(level_scores, weights=level_indices)  # nivel N pesa N
```

## Modos de operacion
| Modo | Descripcion |
|------|-------------|
| NORMAL | Local + API (default) |
| ONLINE | Solo API |
| OFFLINE | Solo local |
| COMPETITION | Solo API, scoring oficial, un solo make() por env |

## Ejemplo minimo
```python
import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")
print(env.action_space)
obs = env.step(GameAction.ACTION1)
print(arc.get_scorecard())
```

## Random agent completo
```python
import random
from arcengine import GameAction, GameState
import arc_agi

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")

for step in range(100):
    action = random.choice(env.action_space)
    data = {}
    if action.is_complex():
        data = {"x": random.randint(0, 63), "y": random.randint(0, 63)}
    obs = env.step(action, data=data)
    if obs and obs.state == GameState.WIN:
        print(f"Won at step {step}!")
        break
    elif obs and obs.state == GameState.GAME_OVER:
        env.reset()

scorecard = arc.get_scorecard()
print(f"Score: {scorecard.score}" if scorecard else "No scorecard")
```

## Render modes
```python
env = arc.make("ls20")                          # sin render (max FPS, 2000+)
env = arc.make("ls20", render_mode="terminal")  # texto, FPS limitado
env = arc.make("ls20", render_mode="human")     # matplotlib
```

---

## Repo de Agents (arcprize/ARC-AGI-3-Agents)

### Base class para agentes
```python
from abc import ABC, abstractmethod
from arcengine import FrameData, GameAction

class Agent(ABC):
    MAX_ACTIONS: int = 80

    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        raise NotImplementedError
```

### Agentes de ejemplo disponibles
| Agent | Approach | Descripcion |
|-------|----------|-------------|
| Random | Baseline | Acciones aleatorias |
| LLM | GPT function calling | Grilla como texto, loop de observacion |
| FastLLM | LLM sin observation step | Mas rapido, menos contexto |
| ReasoningLLM | o4-mini reasoning | Captura reasoning tokens |
| GuidedLLM | o3 con reglas explicitas | Prompts per game |
| MultiModalLLM | Vision model + PNG | Convierte grillas a imagenes |
| ReasoningAgent | o4-mini + hypothesis tracking | Image + grid, structured output |
| LangGraphFunc | LangGraph functional | Checkpointed conversation |
| SmolCodingAgent | HuggingFace smolagents | Code generation |

---

## Repo de Benchmarking (arcprize/arc-agi-3-benchmarking)

### Instalacion
```bash
git clone git@github.com:arcprize/arc-agi-3-benchmarking.git
cd arc-agi-3-benchmarking
uv venv && uv sync
```

### Base class MultimodalAgent
```python
from arcagi3.agent import MultimodalAgent
from arcagi3.schemas import GameStep
from arcagi3.utils.context import SessionContext

class MyAgent(MultimodalAgent):
    def step(self, context: SessionContext) -> GameStep:
        return GameStep(
            action={"action": "ACTION1"},
            reasoning={"agent": "my_agent"},
        )
```

### SessionContext (lo que recibe el agente)
- `context.game.game_id`, `context.game.current_score`, `context.game.current_state`
- `context.game.available_actions`, `context.game.action_counter`
- `context.frames.frame_grids` — grillas del frame actual
- `context.frames.previous_grids` — grillas del step anterior
- `context.frame_images` — PIL images
- `context.last_frame_grid`, `context.last_frame_image(resize=...)`
- `context.datastore` — key/value persistente (checkpointed)

### CLI
```bash
uv run python -m arcagi3.runner --check
uv run python -m arcagi3.runner --list-games
uv run python -m arcagi3.runner --game_id ls20 --config gpt-5-2-openrouter --max_actions 3
```

## Sources
- https://github.com/arcprize/ARC-AGI
- https://github.com/arcprize/ARC-AGI-3-Agents
- https://github.com/arcprize/arc-agi-3-benchmarking
- https://docs.arcprize.org/
