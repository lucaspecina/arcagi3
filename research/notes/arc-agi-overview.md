# ARC-AGI Overview — Research Notes

Fecha: 2026-03-27

## ARC-AGI-3 (nuevo formato interactivo)

### Que es
- Lanzado 2026-03-24 por ARC Prize Foundation
- Cambio fundamental: de puzzles estaticos a entornos interactivos turn-based
- Grilla 64x64 con 16 colores
- El agente debe explorar, modelar el mundo, inferir objetivos, y ejecutar acciones
- NO hay instrucciones, NO hay reglas escritas, NO hay goal explicito

### Evaluacion
- Metrica: RHAE (Relative Human Action Efficiency)
- Score = min(1.0, human_actions / AI_actions)^2
- Humanos: 100%, mejor IA: <1% (Gemini 3.1 Pro: 0.37%)

### Toolkit
- `pip install arc-agi` o `uv add arc-agi`
- API registration: https://three.arcprize.org
- Repos: github.com/arcprize/ARC-AGI, github.com/arcprize/ARC-AGI-3-Agents

### Approaches probados
- **StochasticGoose (1ro, 12.58%)**: CNN + RL, predice que acciones causan cambios de frame
- **Blind Squirrel (2do, 6.71%)**: State graph + ResNet18 value model
- **Fluxonian (8.04%)**: DSL + LLM combinacion
- **Play Zero Agent (4.37%)**: Random exploration + LLM video analysis

### Implicacion para nosotros
Los LLMs puros no funcionan (<1%). Se necesita RL/agentic. Esto es research line 2, no prioridad.

---

## ARC-AGI-2 / ARC-AGI-1 (formato estatico clasico)

### Formato
- JSON con `"train"` y `"test"` — pares input/output
- Grillas de listas de listas, integers 0-9
- Tamano 1x1 a 30x30
- Repos: github.com/fchollet/ARC-AGI (v1), github.com/arcprize/ARC-AGI-2 (v2)

### Performance de LLMs
- ARC-AGI-1: hasta ~54% (test-time training), ~71.6% (product of experts con augmentation)
- ARC-AGI-2: hasta ~24% (NVARC, 1er lugar ARC Prize 2025)
- Frontier AI scores significativos — hay traccion real

### Approaches LLM relevantes
1. **Transduccion directa**: LLM recibe ejemplos + test input, predice output grid
2. **Induccion (program synthesis)**: LLM genera programa Python que transforma input->output
3. **Test-time training (TTT)**: Fine-tuning en tiempo de inferencia sobre los ejemplos de cada tarea
4. **Product of Experts**: LLM como generador y scorer usando probabilidades de output
5. **SOAR**: Evolutionary program synthesis que fine-tunea un LLM sobre sus propios search traces

### Competicion Kaggle
- ARC Prize 2026 — ARC-AGI-2: kaggle.com/competitions/arc-prize-2026-arc-agi-2
- ARC Prize 2026 — ARC-AGI-3: kaggle.com/competitions/arc-prize-2026-arc-agi-3
- Timeline: Milestone 1 jun 30, Milestone 2 sep 30, cierre nov 2, resultados dic 4, 2026
- Premios: $850K (AGI-3), $425K (AGI-2)

### Decision inicial
Empezar por ARC-AGI-2 estatico con approach de transduccion directa via LLM (Azure Foundry).
Es el path mas rapido para tener un baseline funcional y medir.

## Sources
- arcprize.org/blog/arc-agi-3-launch
- arxiv.org/html/2603.24621 (ARC-AGI-3 Technical Report)
- docs.arcprize.org
- github.com/arcprize/ARC-AGI
- github.com/arcprize/ARC-AGI-3-Agents
- github.com/DriesSmit/ARC3-solution (StochasticGoose)
- arxiv.org/abs/2505.07859 (Product of Experts)
- arcprize.org/blog/arc-prize-2025-results-analysis
