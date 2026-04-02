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

## Core hypothesis
**Los LLMs frontier (GPT-5.4, etc.) YA tienen las capacidades cognitivas
necesarias para resolver ARC-AGI-3** — reconocer objetos, entender movimiento,
identificar patrones, inferir goals. El problema no es que no puedan, sino que
no logran usar esas capacidades de forma consistente y automatica.

**Nuestro trabajo es construir un harness que EXTRAIGA esas capacidades** de
forma inteligente: cuestionando, verificando, sintetizando, abstrayendo.
El harness convierte capacidades latentes en rendimiento real.

## EL OBJETIVO CENTRAL — NO PERDERLO NUNCA
**Construir un harness/scaffold de META-COGNICION que maximice las capacidades
de GPT-5.4 para ARC-AGI-3 de forma GENERALIZABLE a todos los juegos.**

Esto significa:
- **SIN informacion externa** — no leer source code, no hardcodear mecanicas,
  no inyectar conocimiento game-specific. El sistema tiene que aprender solo.
- **Meta-cognicion** — el harness FUERZA al modelo a extraer, mantener, combinar
  y mejorar sus propios razonamientos. El LLM es el CEREBRO, el harness lo
  potencia con estructura.
- **Generalizable** — cada pieza de codigo debe funcionar en TODOS los juegos.
  Si solo sirve para un juego, no sirve.

Las palancas del harness son:
- **Buenas preguntas** — prompts que fuerzan al LLM a razonar mejor
- **Memoria y abstracciones** — que el LLM construya conocimiento acumulativo
- **Multi-agente** — multiples runs en paralelo, sintesis de abstracciones,
  critic que cuestiona, debate entre perspectivas
- **Tools** — darle herramientas al LLM (diff, comparacion, verificacion)
- **Multi-run learning** — N intentos → sintetizar → abstraer → reintentar
  con mas conocimiento. Cada run mejora el siguiente.

Validacion: si le mostramos dos frames al LLM y le preguntamos especificamente
"para que lado se movio el objeto?", probablemente lo reconoce. El desafio es
que lo haga solo, sin que le preguntemos. El harness tiene que hacer que se
haga esas preguntas a si mismo.

## Key concepts
- **ARC-AGI-3**: Entornos interactivos turn-based (64x64, 16 colores). LLMs <1%. El desafio mas duro
- **Harness**: Sistema que orquesta uno o mas LLMs — prompting, memoria, planificacion, herramientas
- **Capacidades latentes**: Los LLMs tienen habilidades que no emergen con prompting naive.
  El harness las extrae via: preguntas dirigidas, verificacion, abstraccion, sintesis
- **Abstracciones**: No solo memoria cruda — el harness construye entendimiento de alto nivel
  ("esto es un juego de navegacion", "los rojos matan", "la barra es vida")
- **Multi-run learning**: Multiples intentos → sintetizar → abstraer → reintentar con mas conocimiento
- **Hipotesis, no hechos**: Toda afirmacion en memoria es una hipotesis que se cuestiona constantemente
- **Azure AI Foundry**: Plataforma para llamar LLMs (catalogo completo) via API

## Design principles
1. **Ganar primero, elegancia despues** — lo que funcione en las metricas es lo que vale
2. **Iterar rapido** — ciclo research→impl→eval corto, no overengineer
3. **Medir todo** — cada approach tiene metricas claras, comparables, reproducibles
4. **Extraer, no reemplazar** — el LLM puede, el harness extrae. No construir CNNs si el LLM ya ve
5. **Cuestionar todo** — ninguna abstraccion es definitiva, todo se re-verifica
6. **Multi-model por diseño** — el harness puede usar cualquier modelo o combinacion
7. **GENERALIZABLE siempre** — NUNCA codigo game-specific. Todo debe funcionar en todos los juegos
8. **El LLM es el cerebro** — no reemplazarlo con solvers programaticos. Potenciarlo con estructura

## What success looks like
- **Score competitivo en ARC-AGI-3 leaderboard** — top positions
- Sistema que resuelve un % significativo de tareas que hoy ningun LLM resuelve (<1%)
- Demostrar que el gap 12% vs <1% se cierra con mejor orquestacion, no con otro paradigma
- Framework para iterar rapido: nueva idea → implementar → medir → decidir en horas, no semanas
