# ARC-AGI 3 — LLM Agent Research

LLM-based agent for [ARC Prize 2026](https://arcprize.org) interactive environments.
Uses Azure AI Foundry to call LLMs that observe a 64x64 grid and choose actions
to solve abstract reasoning puzzles.

## Setup

### 1. Clone and install

```bash
git clone https://github.com/lucaspecina/arcagi3.git
cd arcagi3
pip install -e .
```

### 2. Get API keys

- **Azure AI Foundry**: Get your API key from [Azure Portal](https://portal.azure.com)
- **ARC-AGI-3**: Register at [three.arcprize.org](https://three.arcprize.org) to get an `ARC_API_KEY`

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
AZURE_INFERENCE_CREDENTIAL=your-azure-api-key
AZURE_FOUNDRY_BASE_URL=https://your-resource.openai.azure.com/openai/v1
AZURE_MODEL=gpt-4o
ARC_API_KEY=your-arc-api-key
```

## Usage

### List available games

```bash
python -m arcagi3.run --list-games
```

### Run the agent on a game

```bash
# Default: game ls20, 80 actions, vision mode, terminal render
python -m arcagi3.run

# Specify game and max actions
python -m arcagi3.run --game ls20 --max-actions 100

# Use a different model
python -m arcagi3.run --model DeepSeek-V3.2

# Text-only mode (no vision)
python -m arcagi3.run --no-vision

# No rendering (fastest)
python -m arcagi3.run --render none

# Custom temperature
python -m arcagi3.run --temperature 0.5
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--game` | `ls20` | Game ID to play |
| `--model` | from `AZURE_MODEL` env | LLM model name |
| `--max-actions` | `80` | Maximum actions per game |
| `--no-vision` | `false` | Use text-only mode instead of vision |
| `--render` | `terminal` | Render mode: `terminal`, `human`, or `none` |
| `--temperature` | `0.7` | LLM sampling temperature |
| `--list-games` | - | List available games and exit |

## How it works

1. The agent connects to an ARC-AGI-3 environment via the official `arc-agi` toolkit
2. Each turn, it receives a 64x64 grid observation (16 colors)
3. The grid is converted to an image and sent to the LLM along with:
   - A diff highlighting what changed since last turn
   - Recent action history
   - Agent memory/scratchpad
4. The LLM responds with a JSON action (`UP`, `DOWN`, `LEFT`, `RIGHT`, `INTERACT`, `CLICK`, `UNDO`, `RESET`)
5. The action is executed and the loop continues until WIN, GAME_OVER, or max actions

## Project structure

```
src/arcagi3/
  agent.py         # LLM agent — observation loop, action parsing
  grid_utils.py    # Grid-to-image/text conversions, diff visualization
  run.py           # CLI entry point
research/          # Research notes and findings
pyproject.toml     # Dependencies
.env.example       # Environment variables template
```

## Available models (Azure Foundry)

Any model deployed on your Azure AI Foundry resource works. Pass it via
`AZURE_MODEL` env var or `--model` flag. Examples:

```bash
python -m arcagi3.run --model gpt-4o
python -m arcagi3.run --model DeepSeek-V3.2
python -m arcagi3.run --model gpt-5.4
```

## References

- [ARC Prize 2026](https://arcprize.org)
- [ARC-AGI-3 Documentation](https://docs.arcprize.org/)
- [ARC-AGI Toolkit](https://github.com/arcprize/ARC-AGI)
- [ARC-AGI-3 Agents (official examples)](https://github.com/arcprize/ARC-AGI-3-Agents)
