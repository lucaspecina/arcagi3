# Autoresearch

## Status: ON

## Session: 2026-04-02 — Belief Validation & Meta-Cognition Harness

### Objective
Diseñar e implementar un ciclo de validacion/actualizacion de creencias en cada
paso del agente. Forzar al LLM a revisar, cuestionar, y actualizar sus memorias
despues de cada accion. Meta-cognicion pura, 100% generalizable.

### Workflow
```
diseñar ciclo de validacion → implementar → testear en multiples juegos (N>1)
→ medir mejora vs baseline → iterar → repeat
```

### Rules
- **NEVER STOP** — seguir iterando
- **GENERALIZABLE** — todo debe funcionar en TODOS los juegos. ZERO game-specific.
- **EL LLM ES EL CEREBRO** — potenciarlo con estructura, no reemplazarlo
- **PIVOT FAST** — 2-3 tests sin mejora = cambiar approach
- **GASTAR BUDGET** — experimentar agresivamente. No ser tacaño.
- **BUILD, DON'T THEORIZE** — implementar y testear, no analizar infinitamente
- **N>1** — multiples runs por experimento, multiples juegos
- **COMMIT REGULARLY** — commit + push despues de cada milestone

### Budget
- Total: $70 USD shared resource
- Spent: ~$23 (32%)
- Available: ~$47
- Target: usar $15-20 en esta sesion
- Check: `bash scripts/check_budget.sh`

### What to build
Ciclo de validacion post-accion:
1. ¿Que cambio? (explicitar diffs)
2. ¿Que predije vs que paso? (contrastar)
3. Para cada creencia: ¿confirma, contradice, o irrelevante? (revisar)
4. Actualizar creencias (corregir, fortalecer, agregar)
5. Actuar con creencias validadas

### Stop conditions
- Budget agotado ($47 restante)
- Breakthrough (approach que resuelve >0 niveles en multiples juegos)
- Contexto agotado
