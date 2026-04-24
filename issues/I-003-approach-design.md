---
id: 3
title: Diseño de approach para ARC-AGI-3
status: active
type: research
created: 2026-04-01
related: [I-001, I-002]
---

# I-003: Diseño de approach para ARC-AGI-3

## Status
- **Estado:** Implementacion de harness meta-cognitivo generalizable
- **Hipotesis actual:** Multi-run con sintesis + critic + abstracciones potencia al LLM
- **Ultimo resultado:** Autoresearch fallido (13 exp, todo 0.0) — se fue por game-specific. REVERTIDO.
- **Proximo paso:** Implementar multi-run + sintesis, critic agent, evaluar en multiples juegos

## OBJETIVO CENTRAL
Construir harness de META-COGNICION que extraiga capacidades latentes de GPT-5.4
de forma GENERALIZABLE. El LLM es el cerebro. El harness lo potencia con:
buenas preguntas, memoria/abstracciones, multi-agente, critic, tools, multi-run learning.
**PROHIBIDO: codigo game-specific, source code de juegos, solvers programaticos.**

## Pregunta
Cual es el approach mas efectivo para un harness que orqueste LLMs frontier
para resolver tareas ARC-AGI-3 de manera competitiva?

## Log

### 2026-04-01 · DISEÑO — Ideas iniciales de approach

#### Idea 1: Parallel runs + Memory synthesis + Voting
Correr N juegos en paralelo (ej: 10 runs del mismo juego). Cada run genera
sus propias memorias y abstracciones. Al terminar la tanda:
1. **Sintesis**: un LLM recibe TODAS las memorias y extrae abstracciones
   comunes — que controles hacen que, que reglas se repiten, que goals
   parecen correctos
2. **Votacion**: las abstracciones se rankean por frecuencia/consistencia.
   Las que aparecen en 8/10 runs son "alta confianza". Las que aparecen
   en 2/10 son "baja confianza" o descartadas.
3. **Siguiente ronda**: los 10 runs nuevos arrancan con las abstracciones
   votadas como contexto. Ciclo: run → sintetizar → votar → run.

**Abstracción como concepto clave**: no es solo "memoria" (datos crudos),
es construir abstracciones de nivel superior:
- "Este juego es de tipo navegación con obstáculos"
- "El avatar es el objeto que se mueve con ACTION1-4"
- "Los objetos rojos matan, los verdes son meta"

Las abstracciones son más transferibles entre runs que los datos crudos.

#### Idea 2: Critic model
Un segundo LLM (o el mismo con otro prompt) que actúa como **crítico**.

**Input del critic:**
- La grilla actual (imagen)
- Las últimas N acciones del agente
- El reasoning del agente en el último turno
- Las memorias/abstracciones actuales

**Rol del critic:**
- Señalar contradicciones ("dijiste que ACTION1 sube pero el objeto bajó")
- Cuestionar assumptions ("no tenés evidencia de que el goal sea X")
- Sugerir qué re-testear ("deberías verificar si ACTION3 realmente va a la izq")
- Detectar loops ("llevas 5 turnos haciendo lo mismo sin resultado")

**Cuándo actúa:** no cada turno (caro), sino:
- Cada N turnos (ej: cada 5)
- Después de un GAME_OVER
- Cuando el agente parece stuck (misma acción repetida)

#### Idea 3: OMPE (Observe-Model-Plan-Execute)
Separar las fases en vez de un loop monolítico:
1. **Observe**: exploración estructurada (~10 acciones), capturar diffs
2. **Model**: LLM recibe TODAS las observaciones de golpe, genera modelo del mundo
3. **Plan**: dado el modelo, planificar secuencia óptima (LLM o search)
4. **Execute**: correr el plan, si falla volver a fase 1

#### Idea 4: Multi-model pipeline
- **Observer** (vision model barato): describe qué ve en cada frame
- **Analyst** (reasoning model, GPT-5.4): razona sobre mecánicas
- **Coder** (code model): genera políticas como código Python
- **Critic**: cuestiona y verifica

#### Idea 5: Differential + Program Synthesis
Inspirado en StochasticGoose (1er lugar):
- Capturar transiciones (frame_before, action, frame_after)
- LLM sintetiza programa que predice frame_after dado frame_before + action
- Usar como world model para search

### Combinaciones prometedoras

**Combo A: Parallel OMPE + Critic + Synthesis**
- N runs paralelos usan OMPE
- Critic supervisa cada run
- Post-ronda: sintetizar abstracciones, votar, arrancar nueva ronda

**Combo B: Multi-model OMPE + Synthesis**
- Observer + Analyst + Coder como pipeline
- N pipelines paralelos
- Synthesis de abstracciones entre rondas

### 2026-04-01 · DISEÑO — Abstracciones cross-game

**Insight clave**: las abstracciones no solo se transfieren entre runs del MISMO
juego, sino entre JUEGOS DIFERENTES. Hay conocimiento que es universal a ARC-AGI-3:

**Abstracciones universales (cross-game):**
- El sistema de vidas (ej: 3 oportunidades antes de game over definitivo)
- Patrones de acciones (ACTION1-4 suelen ser movimiento direccional)
- Indicadores visuales (barras = vidas/progreso, colores = semántica)
- Estructura de niveles (nivel 1 = tutorial, niveles posteriores agregan mecánicas)
- Patrones de mecánicas (objetos que matan, objetos que son meta, obstáculos)

**Abstracciones game-specific:**
- Qué hace cada acción en ESTE juego
- Reglas específicas de este juego
- Goal de este juego

**Implicación para la arquitectura:**
Necesitamos DOS niveles de memoria/abstracción:
1. **Meta-knowledge** (cross-game): se acumula a lo largo de TODOS los juegos.
   Persiste entre juegos. Se enriquece con cada juego nuevo.
2. **Game-knowledge** (per-game): específico del juego actual. Se resetea al
   empezar un juego nuevo, pero arranca con el meta-knowledge como contexto.

El ciclo sería:
```
Juego 1 → game-knowledge → al terminar, extraer meta-abstracciones
Juego 2 → arranca con meta-knowledge + game-knowledge vacío → al terminar, enriquecer meta
Juego 3 → arranca con meta-knowledge mejorado → ...
```

Esto es similar a como un humano juega: ya sabe qué son vidas, qué es un
avatar, qué es un obstáculo. No redescubre esos conceptos en cada juego.

### Preguntas abiertas
- ¿Cuántos runs paralelos son suficientes para buena síntesis? (5? 10? 20?)
- ¿El critic debería ser el mismo modelo o uno diferente?
- ¿Costo por tarea? Budget de tokens por juego?
- ¿OMPE es viable con solo 80 acciones de budget?
- ¿Cómo estructurar el meta-knowledge? ¿Texto libre, JSON, embeddings?
- ¿En qué orden jugar los juegos para maximizar transfer de meta-knowledge?
- ¿Cuántas vidas da cada juego? ¿Es siempre 3 o varía?

### 2026-04-02 · DEBATE — Codex critique del approach

Codex revisó la arquitectura de 6 capas y el roadmap. Feedback duro:

**Veredicto:** "Su roadmap parece diseñado para un paper lindo, no para
subir leaderboard rápido."

**Criticas principales:**
1. **Sobre-arquitecturado.** 6 capas para un proyecto que no tiene ni eval harness.
   Cortar: critic, memory tiers, skill library, parallel synthesis. Son prematuros.
2. **Falta eval + instrumentacion.** Sin benchmark serio, cualquier mejora "parece"
   buena. Necesitamos per-game success, acciones desperdiciadas, novel states.
3. **No ser LLM-first.** Gap es 12% (non-LLM) vs <1% (LLM). Non-LLM infra primero,
   LLM como helper puntual.
4. **WorldCoder global: no.** Demasiada variedad. Si, para mecanicas locales/parciales.
5. **Subestimamos representacion del estado.** Frame hash + diff no alcanza. Necesitamos
   estado canonico orientado a objetos/eventos.
6. **Subestimamos ACTION6.** Espacio accion-coordenada es enorme. Necesita priors
   espaciales, saliency.
7. **Parallel runs ahora = 10x basura.** Primero un buen explorer.
8. **Cross-game priors pueden contaminar.** "ACTION1-4 = movimiento" puede ser falso.

**Re-priorizacion de Codex:**
- HACER: eval harness, state graph, canonical state/object extraction, spatial
  action proposal, non-LLM search, logs masivos
- NO HACER (aun): critic, reflection tiers, cross-game memory, parallel synthesis,
  WorldCoder completo, Voyager skill library

**Contra-argumentos (Claude):**
- GPT-5.4 es mucho mas capaz que lo que probaron otros (<1% fue con Gemini 3.1 Pro).
  Con buena infra el LLM podria aportar mas.
- Cross-game learning es lo que hacen los humanos, descartarlo del todo es riesgoso.

**Decision:** Repensar el approach. Evaluar pivot a non-LLM first con LLM como
helper. Definir MVP minimo y eval harness antes de features avanzados.

### 2026-04-06 noche · AUTORESEARCH session — prompt-tweak loop, sin breakthrough

**Setup:** Single-brain mode (Claude Code orquestando, benches en background
paralelo). ls20 únicamente, gpt-5.4-mini. Cap de 4-6 iteraciones.

**Baseline real medido:** mean=21.25 (n=4), std≈6. El 30/100 inicial era
suerte. La señal por debajo de ~10 puntos es ruido.

**5 iteraciones probadas:**

| iter | commit | qué cambió | resultado | veredicto |
|---|---|---|---|---|
| 1 | c6334bf | Reflector STEP 1: forzar full-grid scan | mean=20 (n=2) | noise, marginal |
| 2 | e134f73 | Analyzer: regla "INTERACT BEFORE NAVIGATE" | mean=25 (n=3) | weak keep, +3.75 (~0.6 std), judge highlight semantically alineado ("white + is interactive") |
| 3 | cd2a732 | Actor: priorizar tocar objetos distintivos no testeados | mean=12.5 (n=2) | DISCARD, -12 vs iter2. El actor abandonó exploración local útil |
| 4 | 73bee3b | Reflector STEP 1.5: SURPRISE DETECTION (cambios distantes como top causal) | mean=20 (n=2) | DISCARD, no mejora |
| 5 | 8abef4f | BarTracker: solo bars cerca de edges + dedupe filas adyacentes | mean=25 (n=2) | bug fix válido pero no movió la métrica |

**Lo único realmente útil que se descubrió:**
1. **Bug del shell timeout 1800.** Un bench de 3 chains tarda 38-42 min, no
   30. El timeout mataba python silenciosamente en chain 3 step 7-9. Costó
   ~1.5 hs de debugging. Fix en 6fd18fd: `timeout 3600 python -u`.
2. **Bug del BarTracker.** Reportaba 5 "RESOURCE BAR" falsas por step
   (paredes del laberinto interior). 7 warnings/step → 1. Fix en 8abef4f.
3. **Variance del baseline correctamente medida.** Antes solo había una
   muestra (30, lucky run); ahora sabemos que es 21.25 ± 6.

**Lo que NO se intentó (y debería haberse intentado):**
- Cambios estructurales al loop (multi-hipótesis, critic, debate, replay)
- Belief schema redesign (sigue siendo dict plano de strings)
- Tool calling
- Multi-modelo (¿el techo es el mini o el harness?)
- Self-consistency / muestreo del actor con vote
- Region-based observation (segmentar la grilla en zonas semánticas)

**Patrón del fracaso:** prompt-tweak loop. 4 de 5 iteraciones fueron
"agreguemos un párrafo a este system prompt". Esto se siente como
progreso porque hay commits y benches corriendo, pero el espacio de
búsqueda "adjetivo en un system prompt" es enorme, el ruido es alto y
los modelos mini suelen ignorar texto agregado. Termina siendo A/B
testing de inglés.

**Findings sobre dónde está el cuello de botella:**
- Cambios al ANALYZER produjeron la única mejora marginal (iter2)
- Cambios al ACTOR hicieron daño (iter3)
- Cambios al REFLECTOR fueron neutros (iter1, iter4)
- Hipótesis: el cuello de botella está en *qué percibe* el agente, no
  en *cómo decide*. El próximo ataque debería atacar la perception
  layer estructuralmente, no más prompts.

**Anti-patterns documentados** en `AUTORESEARCH.md` para que el próximo
autoresearch no caiga en lo mismo. Análisis profundo en
`research/notes/autoresearch-night-2026-04-06.md`.

**Estado del repo:** main = 8abef4f. iter2 (e134f73) y bug fixes
(6fd18fd, 8abef4f) en main. Iter 3 y 4 revertidos. AUTORESEARCH.md
status = OFF.

### 2026-04-24 · PLAN — Belief state estructurado (BLF-adapt)

**Source:** Paper de Kevin Murphy "Agentic Forecasting using Sequential
Bayesian Updating of Linguistic Beliefs" (arxiv 2604.18576, abril 2026).

**Ataque:** Reemplazar el dict plano de beliefs por un schema
estructurado con `goal_hypotheses` (evidence_for/against citando step,
status active/refuted/confirmed, next_test) y `belief_update_reasoning`
mandatory cada step. Plan completo en
`research/notes/belief-blf-plan-2026-04-24.md`.

**Why este ataque y no otro:**
- Estructural (cambia el dato que mantiene el agente), no prompt-tweak
- Mínimo (1 archivo, ~120 LOC) — cheap de validar
- BLF ablation: sin belief state estructurado, -3.0 BI (=quitar web search).
  Si transfiere, debería ser detectable con n=4 contra std≈6
- Si NO mueve el needle con n=4: evidencia útil de que el bottleneck
  no es el belief schema → pivotear a perception (region-based) o
  multi-modelo

**Próximo paso concreto:** implementar en `src/arcagi3/agent.py`,
branch `autoresearch/belief-blf-2026-04-24`, experimento según plan.

## Conclusion
<!-- Se llena al cerrar el issue -->
