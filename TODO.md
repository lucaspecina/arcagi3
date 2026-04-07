# TODO

**RECORDAR: El objetivo es un harness de meta-cognicion GENERALIZABLE.
El LLM es el cerebro. El harness lo potencia con estructura.
NUNCA codigo game-specific.**

## NOW
- [ ] PIVOT: diseñar UN ataque estructural concreto (no más prompt tweaks) → I-003.
      Leer `research/notes/autoresearch-night-2026-04-06.md` y elegir uno de la
      lista "Ataques estructurales no intentados". Candidatos top:
      region-based observation, multi-hipótesis paralela, replay learning post-chain.
- [ ] Sesión de planning offline (sin benches) para motivar y planear el ataque
      antes de volver a Status: ON en autoresearch.

## NEXT
- [ ] Critic / debate module (3a llamada LLM entre reflector y actor) → I-003
- [ ] Belief schema redesign (graph en lugar de dict plano) → I-003
- [ ] Multi-run con sintesis de abstracciones (N runs → sintetizar → reintentar) → I-003
- [ ] Tools para el LLM (compare_frames, highlight_color, list_isolated_objects)
- [ ] Capacity test: correr 1 iter con gpt-5.4 (no mini) en todo el loop para saber
      si el techo es el modelo o el harness

## LATER
- [ ] Multi-model orchestration (combinar GPT-5.4 + otros)
- [ ] Cross-game meta-knowledge (abstracciones que transfieren entre juegos)
- [ ] Test-time compute optimization (search, verification, self-correction)
- [ ] Submission a ARC-AGI-3 leaderboard

## DONE (recent)
- [x] Autoresearch session 2026-04-06 noche: 5 iters, 0 mejoras reales,
      anti-patterns documentados → I-003 — 2026-04-07
- [x] Bug fix: shell timeout 1800 mataba chain 3 (now 3600) → I-003 — 2026-04-07
- [x] Bug fix: BarTracker false positives en paredes interiores → I-003 — 2026-04-07
- [x] Iter 2 keep weak: ANALYZER INTERACT BEFORE NAVIGATE rule (e134f73) → I-003 — 2026-04-07
- [x] Upgrade estructura a workflow estandar → I-002 — 2026-04-01
- [x] Implementar agente LLM basico para ARC-AGI-3 (vision + text) → I-001 — 2026-03-27
- [x] Research ARC-AGI-2 y ARC-AGI-3 → I-001 — 2026-03-27
- [x] Setup inicial del proyecto → I-001 — 2026-03-27
