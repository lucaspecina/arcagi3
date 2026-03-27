# ARC-AGI 3 — Research & Inference

## What is this?
Proyecto de investigacion para participar en ARC Prize 2026. Exploramos enfoques
de inferencia usando LLMs (via Azure AI Foundry) para resolver tareas ARC-AGI.
El foco inicial es ARC-AGI-2 (puzzles estaticos) donde los LLMs tienen traccion
demostrada, con una linea de investigacion paralela en ARC-AGI-3 (entornos interactivos).

## LA PREGUNTA
> Cual es el approach mas efectivo para resolver tareas ARC-AGI usando LLMs
> como motor de razonamiento, y como lo validamos rapido?
>
> Esta pregunta guia todas las decisiones: que probar primero, como evaluar,
> cuando pivotar de approach.

## Key concepts
- **ARC-AGI-2**: Puzzles estaticos (grillas JSON, input->output). LLMs llegan a ~54% en v1, ~24% en v2
- **ARC-AGI-3**: Entornos interactivos turn-based (64x64, 16 colores). LLMs <1%. Requiere RL/agentic
- **Induction vs Transduction**: Generar programa (induccion) vs predecir output directo (transduccion)
- **Test-time training (TTT)**: Fine-tuning en tiempo de inferencia sobre los ejemplos de cada tarea
- **Azure AI Foundry**: Plataforma para llamar LLMs (DeepSeek, GPT, Llama, etc.) via API

## Design principles
1. **Empezar simple, iterar rapido** — baseline LLM directo antes de approaches complejos
2. **Medir todo** — cada approach tiene que tener metricas claras contra el dataset
3. **Modular** — separar data loading, prompting, inference, evaluation
4. **Cost-conscious** — trackear costo por tarea, optimizar tokens

## Architecture overview (preliminary)
```
data/          <- datasets ARC-AGI (descargados)
src/
  loader.py    <- carga y parsea tareas ARC-AGI
  prompts.py   <- templates de prompts para LLMs
  inference.py <- llamadas a LLMs via Azure Foundry
  evaluate.py  <- evaluacion de respuestas vs ground truth
research/      <- investigacion y notas
```

## What success looks like
- Tener un pipeline funcional que toma tareas ARC-AGI, las pasa por un LLM, y evalua resultados
- Superar un baseline random significativamente
- Identificar que tipos de tareas los LLMs resuelven bien y cuales no
- Tener un framework para iterar rapido sobre prompts y approaches
