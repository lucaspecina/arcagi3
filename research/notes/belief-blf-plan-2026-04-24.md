# Plan: Belief state estructurado (BLF-adapt) — 2026-04-24

**TL;DR:** Adaptar el "linguistic belief state" de Murphy 2026 (BLF,
arxiv 2604.18576) al harness de ARC-AGI-3. Ataque estructural mínimo
— cambia el schema de beliefs y fuerza razonamiento explícito de update,
no agrega prompts. Medir si mueve el needle contra variance std≈6.

## Contexto

Post-mortem del 2026-04-06 (ver `autoresearch-night-2026-04-06.md`)
concluyó que 4 de 5 iteraciones fueron prompt-tweaks, ninguna estructural.
Lista de candidatos estructurales abierta. Este doc aterriza UNA:
rediseñar cómo el agente mantiene y actualiza beliefs.

## Paper-source

**Paper:** "Agentic Forecasting using Sequential Bayesian Updating of
Linguistic Beliefs" — Kevin Murphy, arxiv 2604.18576 (abril 2026).

**Idea central:** LLM forecasters que acumulan raw evidence en context
pierden atención y drift; en vez de eso, mantener un belief state
*semi-structured* donde el LLM destila evidencia cada step en campos
explícitos (prob, evidence_for/against con citas de fuente, open_questions,
update_reasoning). Ablación: sin belief state, -3.0 BI; mismo impacto
que quitar web search entero.

**Lo que transfiere:** estructura + update explícito. No: probabilidad
escalar única (forecasting binario ≠ decisiones secuenciales), platt
scaling (no emitimos probabilidades).

## Estado actual del harness

`src/arcagi3/agent.py` — loop Analyzer-Reflector-Actor.

Belief schema (REFLECTOR STEP 8, línea 305-314 aprox):
```json
"updated_beliefs": {
  "controls": {"ACTION1": "...", ...},
  "rules": [...],
  "causal_model": [...],
  "goal": "top hypothesis (confidence: N%)",
  "objects": [...],
  "dangers": [...],
  "unknowns": [...],
  "failed_approaches": [...]
}
```

Cada step, el reflector regenera el belief set completo desde cero
(STEP 3: review each belief KEEP/CHANGE/DROP, STEP 8: output complete
updated set). Problemas:
- Evidencias se pierden — no hay registro de *cuándo* se observó qué.
- Hipótesis activas no están separadas de hipótesis rechazadas.
- No hay push explícito hacia articular *por qué* el belief cambió.
- Drift: el modelo puede silenciosamente renombrar / reformular.

## Diff de schema propuesto

Reemplazar el campo `goal` (string) + `failed_approaches` por
`goal_hypotheses` estructurado. Agregar `belief_update_reasoning`
obligatorio. Resto igual.

```json
"updated_beliefs": {
  "controls": {...},          // igual
  "rules": [...],             // igual
  "causal_model": [...],      // igual
  "objects": [...],           // igual
  "dangers": [...],           // igual
  "unknowns": [...],          // igual
  "goal_hypotheses": [
    {
      "id": "h1",
      "text": "reach white object at (10,15)",
      "probability": 0.35,
      "evidence_for": [
        {"step": 3, "finding": "distinctive isolated color", "strength": "weak"},
        {"step": 8, "finding": "touching it decremented HUD bar by 1", "strength": "strong"}
      ],
      "evidence_against": [
        {"step": 12, "finding": "stepping on it did NOT end level", "strength": "strong"}
      ],
      "status": "active | refuted | confirmed",
      "next_test": "ACTION6 click @ (10,15)"
    }
  ],
  "belief_update_reasoning": "MANDATORY 1-3 sentences: qué cambió este step y por qué. Cita steps específicos."
}
```

### 3 reglas duras (parse-validadas)

1. **Cada evidence item cita `step` numérico** (int). Parse fail → warning.
2. **`evidence_for` y `evidence_against` acumulan, cap=6 por hypothesis.**
   Al exceder, reflector debe RESUMIR inline el más viejo y descartarlo.
   No es auto: el LLM lo hace como parte del update.
3. **`belief_update_reasoning` es required.** String no vacío. Parse fail
   → se loguea warning, memory NO se actualiza (se mantiene el step
   anterior).

### Cap de hipótesis activas

Max 3 con `status == "active"`. Refutadas pasan a `status: "refuted"`
pero se mantienen en la lista (evitar redescubrir lo mismo). `confirmed`
se promueven a goal principal.

## Código

Un solo archivo: `src/arcagi3/agent.py`.

- **REFLECTOR_PROMPT** (líneas 188-316): reescribir STEP 6 (goal
  hypotheses con evidence) y STEP 8 (schema nuevo + mandatory
  belief_update_reasoning). Agregar las 3 reglas duras en texto.
- **`_enumerate_beliefs()`** (líneas 631-681): cuando enumera
  hipótesis para el reflector, incluir evidencias citadas para que
  el LLM las vea y decida si añadir/resumir.
- **`format_memory()`** (líneas 883-910): imprimir evidencias en
  terminal para debugging.
- **Validación post-parse** en `run_reflector()` (línea 749-784):
  si `belief_update_reasoning` ausente o goal_hypotheses mal formado,
  log warning y mantener memory anterior. Si schema parse-ok pero
  evidence sin step#, log warning pero mantener.

LOC estimado: ~120 líneas. Riesgo bajo de romper pipeline (cambio
contenido en una función, parsing defensivo).

## Experimento

**Base:** `e3141ac` (HEAD actual), gpt-5.4-mini, ls20, 3-run chain,
30 max-actions.

**Pasos:**
1. Confirmar baseline variance: 4 samples más en `e3141ac`.
   (Ya tenemos n=4 original de 5c0750c; re-medir no duplica trabajo
   porque iter2 + bugfixes pueden haber desplazado el mean.)
2. Branch `autoresearch/belief-blf-2026-04-24` desde e3141ac,
   implementar cambio.
3. n=4 samples en el branch. Judge en modo normal.
4. Inspeccionar 2 reflexiones completas a mano:
   - ¿Las citas de `step` apuntan a steps reales?
   - ¿Al menos una hypothesis transita a `refuted` durante el run?
   - ¿El `belief_update_reasoning` es trivial ("no changes") o
     articulado?

**Umbral de decisión** (contra std≈6):

| Δmean | n=4 | acción |
|---|---|---|
| ≥ +9 | suficiente | KEEP, merge a main |
| +3 a +9 | ambiguo | n=3 adicionales en g50t antes de decidir |
| -3 a +3 | ruido | DISCARD |
| < -3 | daño | DISCARD rápido |

## Riesgos

- **Output token budget.** Reflector ya usa `max_completion_tokens=8000`.
  Evidencias acumuladas pueden empujar esto. Mitigación: cap 3 hyps
  activas, 6 evidencias/hyp. Si aun así trunca, subir a 10000 tokens.
- **Schema drift del modelo.** gpt-5.4-mini puede inventar campos o
  renombrar. Mitigación: validación defensiva post-parse, fallback al
  schema anterior si falla.
- **No es el ataque estructural más grande.** Sigue siendo cambio en
  cómo el modelo formatea output. Multi-hypothesis-paralelo, critic/debate,
  tool calling serían más radicales. Si n=4 no detecta nada, evidencia
  útil de que belief schema NO es el bottleneck → pivotear a
  perception (region-based observation) o multi-model.
- **Transferencia del resultado BLF.** BLF es forecasting con web search;
  nosotros somos decisión secuencial con visión. La ablación -3.0 BI
  puede no transferir. No hay garantía, solo indicio.

## Lo que NO hace este plan

- K trials paralelos con aggregation (otro ticket, más caro).
- Colapsar Reflector+Actor en 1 call (BLF lo hace; nosotros tenemos
  razón para separar — discusión futura si esto no anda).
- Tools, critic/debate, belief graph, replay learning.

## Decisión al cerrar

- Si **KEEP**: doc de findings en `research/synthesis/` + update TODO.
- Si **DISCARD**: update este doc con post-mortem + update TODO
  pivoteando al siguiente ataque (perception layer, candidato top).

## Referencias

- Paper: https://arxiv.org/abs/2604.18576
- Paper local (pdf): no commiteado, se re-fetchea con WebFetch si hace falta
- Post-mortem previo: `autoresearch-night-2026-04-06.md`
- Código: `src/arcagi3/agent.py:188-316` (REFLECTOR_PROMPT)
- Issue: I-003
