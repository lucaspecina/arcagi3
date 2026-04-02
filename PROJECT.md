# ARC-AGI 3 — Research & Inference

> Este documento es la estrella polar del proyecto.
> Toda decision debe alinearse con lo que dice aca.

## What is this?
Proyecto para **ganar ARC Prize 2026 en la categoria ARC-AGI-3**. Diseñamos un
harness/sistema que orquesta modelos frontier (GPT-5.4, otros, o combinaciones)
para resolver entornos interactivos ARC-AGI-3 de manera competitiva. El approach
es iterativo: research → implementacion → analisis → nuevo research.

## LA PREGUNTA
> Como diseñamos un sistema que orqueste LLMs frontier para resolver
> tareas ARC-AGI-3 de manera competitiva, y como iteramos rapido
> sobre approaches hasta encontrar uno ganador?
>
> Esta pregunta guia todas las decisiones: que probar, como evaluar,
> cuando pivotar de approach.

## The goal
**Ganar ARC-AGI-3.** No "explorar", no "investigar" — ganar. Todo lo que
hacemos tiene que acercarnos a ese objetivo. Si un approach no mueve la aguja
en las metricas, se descarta y se prueba otro.

## Key concepts
- **ARC-AGI-3**: Entornos interactivos turn-based (64x64, 16 colores). LLMs <1%. El desafio mas duro
- **Harness**: Sistema que orquesta uno o mas LLMs — prompting, memoria, planificacion, herramientas
- **Multi-model**: Usar varios modelos (GPT-5.4, DeepSeek, Claude, etc.) en conjunto, cada uno en su fortaleza
- **Induction vs Transduction**: Generar programa (induccion) vs predecir output directo (transduccion)
- **Test-time compute**: Dedicar mas compute en inferencia (search, verification, self-correction)
- **Azure AI Foundry**: Plataforma para llamar LLMs (catalogo completo) via API

## Design principles
1. **Ganar primero, elegancia despues** — lo que funcione en las metricas es lo que vale
2. **Iterar rapido** — ciclo research→impl→eval corto, no overengineer
3. **Medir todo** — cada approach tiene metricas claras, comparables, reproducibles
4. **Multi-model por diseño** — el harness puede usar cualquier modelo o combinacion
5. **Test-time compute** — usar todo el compute disponible en inferencia

## What success looks like
- **Score competitivo en ARC-AGI-3 leaderboard** — top positions
- Sistema que resuelve un % significativo de tareas que hoy ningun LLM resuelve (<1%)
- Framework para iterar rapido: nueva idea → implementar → medir → decidir en horas, no semanas
- Entender QUE tipos de razonamiento necesitan las tareas y COMO los LLMs pueden aportar
