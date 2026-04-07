# Autoresearch night 2026-04-06 — post-mortem

**TL;DR:** 5 iteraciones, ~8 hs de compute paralelo, 0 mejoras reales. La
sesión cayó en un *prompt-tweak loop* y nunca probó un cambio estructural.
Este documento existe para que el próximo autoresearch evite el mismo
patrón.

## Datos crudos

### Variance del baseline (5c0750c)
| sample | metric |
|---|---|
| 1 (single, optimista) | 30 |
| 2 (paralel x4) | 20 |
| 3 (paralel x4) | 15 |
| 4 (paralel x4) | 20 |

**mean = 21.25, std ≈ 6** sobre escala 0-100. Una sola muestra no dice
absolutamente nada.

### Iteraciones

| iter | commit | qué cambió | n | metric mean | Δ vs baseline | veredicto |
|---|---|---|---|---|---|---|
| 1 | c6334bf | REFLECTOR STEP 1: full-grid scan | 2 | 20.0 | -1.25 | NOISY |
| 2 | e134f73 | ANALYZER: INTERACT BEFORE NAVIGATE rule | 3 | 25.0 | +3.75 | weak KEEP |
| 3 | cd2a732 | ACTOR: priorizar tocar objetos distintivos | 2 | 12.5 | -8.75 | DISCARD |
| 4 | 73bee3b | REFLECTOR STEP 1.5: SURPRISE DETECTION | 2 | 20.0 | -1.25 | DISCARD |
| 5 | 8abef4f | BarTracker fix (edge bars + dedupe) | 2 | 25.0 | +3.75 | bug fix válido pero ~iter2 |

Iter 2 sigue en main porque mejoró marginalmente y semánticamente (el
judge específicamente mencionó "white plus as interactive object" en 2/3
runs, alineado con la regla agregada).

Iter 5 sigue en main porque es un bugfix de tracker (eliminó ruido del
contexto), aunque no movió la métrica del judge.

## Por qué falló

### 1. Prompt-tweak loop
4 de 5 iteraciones fueron "agregar un párrafo nuevo a un system prompt
de ANALYZER/REFLECTOR/ACTOR". Esto se siente como progreso porque cada
~40 min sale un commit y un resultado. Pero:

- El espacio de búsqueda "wording de inglés en un system prompt" es
  esencialmente infinito.
- gpt-5.4-mini suele ignorar instrucciones agregadas a prompts ya largos
  (los prompts ANALYZER/REFLECTOR ya tienen ~2000 tokens cada uno).
- El ruido del judge (~6 std) está en el mismo orden que cualquier
  mejora marginal posible vía prompt tweak.
- Termina siendo A/B testing semántico de inglés, no investigación de
  arquitectura.

### 2. Falta de muestras
Iter 1 sample 1 = 25, "parece mejora". Sample 2 = 15, "ah, fue suerte".
Iter 2 sample 1 = 30, "wow!", samples 2-3 = 20, 25 → mean=25, "ok marginal".

n=1 no permite ninguna conclusión. n=2 apenas separa diferencias de
~12 puntos. n=3 es el mínimo para decir algo. Muchas iteraciones se
hubieran ahorrado si hubiera empezado con n=2 desde el principio.

### 3. Tiempo perdido en infraestructura
- ~1.5 hs debugando el `timeout 1800` que mataba pythons silenciosamente
  en chain 3
- ~30 min revisando processes, pids, file mtimes, intentando entender por
  qué los logs se freezaban

Eso es ~25% del tiempo total de la sesión gastado en una herramienta rota.

### 4. No leí las trackers antes de iterar prompts
El BarTracker tenía un bug obvio (paredes interiores reportadas como
bars que decrecen, 5 warnings duplicados por step). Estuvo polluteando
el contexto del LLM en *todos* los runs anteriores y de tonight.

Si hubiera leído un log completo end-to-end antes de empezar, habría
visto el bug y arreglado eso primero. En cambio empecé tweakeando
prompts asumiendo que el harness le daba al LLM información limpia.

## Patrón emergente sobre dónde está el cuello de botella

Aunque los datos son ruidosos, hay un patrón cualitativo:

| módulo tocado | resultado | n |
|---|---|---|
| ANALYZER | única mejora marginal observada | 1 iter |
| REFLECTOR | neutro | 2 iters |
| ACTOR | hizo daño | 1 iter |
| Tracker (perception) | bugfix válido | 1 iter |

**Hipótesis:** el cuello de botella está en *qué percibe* el agente, no
en *cómo decide*. El judge consistentemente reporta que el agente
aprende avatar, movement, yellow bar — pero **nunca** descubre la
mecánica del "+" (Tier 2) ni el goal real (Tier 3). No es un problema
de razonamiento — es un problema de no notar las pistas.

Esto sugiere que los próximos ataques deberían enfocarse en la
**perception layer**, no en agregar más texto a los prompts.

## Lo que NO se intentó (lista de ataques estructurales)

Para que el próximo autoresearch tenga ideas concretas y no caiga en
prompt-tweak loop:

### Perception layer
- **Region-based observation**: segmentar la grilla en zonas semánticas
  (HUD top, HUD bottom, jugable, esquinas) y reportar cambios por zona
  con campos estructurados, no texto plano.
- **Frame diff con highlighting visual**: en lugar de texto "blue object
  moved from (36,33) to (36,28)", generar una imagen overlay con flechas
  rojas marcando los cambios y enviarla al analyzer.
- **Object tracker más rico**: además del avatar, trackear cada objeto
  isolado, su tamaño, color, posición, y emitir eventos
  "object_appeared", "object_disappeared", "object_changed_color".
- **HUD-aware diffing**: detectar zonas que cambian *consistentemente*
  pero NO son el avatar — esos son indicadores de estado oculto.

### Reasoning layer
- **Multi-hipótesis paralela**: en lugar de un belief state único,
  mantener N=3 líneas de juego con diferentes goals, ejecutar M steps,
  comparar evidencia, votar.
- **Critic / debate**: tercera llamada LLM entre reflector y actor cuyo
  único trabajo es atacar al reflector ("¿cuál es la evidencia más
  fuerte EN CONTRA de tu hipótesis top?").
- **Self-consistency en el actor**: muestrear 3 acciones con
  temperatura alta y elegir por mayoría.

### Memory layer
- **Belief graph en lugar de dict plano**: nodos = entidades, aristas =
  hipótesis causales, scores de confianza, punteros a evidencia.
- **Replay learning post-chain**: tras cada chain, una llamada LLM de
  "post-mortem" analiza el log completo + diffs y extrae lecciones
  estructuradas para el siguiente chain.
- **Cross-run abstraction memory**: lecciones que sobreviven entre
  intentos de un mismo juego.

### Stagnation handling
- **Modo "exploración loca"**: cuando `no_progress_count >= 3`, cambiar
  el actor a un modo radicalmente diferente — random walk, ACTION6 grid
  sweep, temperature=1.5 — hasta que algo cambie.
- **Reset estratégico**: cuando el chain está atascado, RESET y empezar
  con beliefs vacíos pero con un golden brief de "cosas que ya no
  funcionan".

### Tools
- **Tool calling**: dar al LLM tools reales: `compare_frames(a, b)`,
  `highlight_color(c)`, `list_isolated_objects(min_size, max_size)`,
  `move_to(x, y)`. Hoy el LLM tiene que derivar todo de texto crudo.

### Multi-modelo
- **Capacity test**: correr UNA iteración con gpt-5.4 (no mini) en TODO
  el loop, para saber si el techo es el modelo o el harness. Es caro
  pero responde una pregunta fundamental.
- **Mixed orchestration**: gpt-5.4 para el analyzer (perception), mini
  para reflector y actor (más volumen, menos calidad necesaria).

## Reglas para el próximo autoresearch

1. **Antes de empezar, leer un log completo de un run reciente
   end-to-end.** Buscar bugs en trackers, ruido en el contexto, info
   redundante. Arreglar eso PRIMERO.

2. **Cada iteración nueva tiene n>=2 desde el inicio.** Lanzar 2 benches
   en paralelo siempre. Si los dos no concuerdan, lanzar el tercero.

3. **Después de 3 iteraciones de prompt-tweak (mejoren o no), pivotear
   a un cambio estructural.** Tomar uno de la lista de "ataques
   estructurales" arriba.

4. **Cada iteración debe responder una pregunta de investigación, no
   "probemos esto a ver qué pasa".** Si no podés enunciar la pregunta,
   no es un experimento.

5. **No tocar ACTOR sin antes haber tocado ANALYZER y REFLECTOR.** El
   patrón emergente sugiere que cambios al actor son frágiles y
   tienden a hacer daño.

6. **Variance baseline siempre con n>=3.** Y revisar si tu "best so
   far" tiene n>=3 también, no solo n=1.

## Estado al cierre

- **main HEAD:** 8abef4f
- **Commits útiles esta noche:** 6fd18fd (timeout fix), 8abef4f
  (BarTracker fix), e134f73 (analyzer rule, weak keep)
- **Commits revertidos:** cd2a732 (iter3 actor), 73bee3b (iter4
  reflector surprise)
- **AUTORESEARCH.md:** Status OFF, anti-patterns documentados
- **results.tsv:** 5 iteraciones logueadas con sus muestras crudas
- **Próximo paso recomendado:** sesión de planning offline (sin
  benches) para diseñar UN ataque estructural concreto y bien
  motivado, antes de volver a turn ON el autoresearch.
