# TODO

**RECORDAR: El objetivo es un harness de meta-cognicion GENERALIZABLE.
El LLM es el cerebro. El harness lo potencia con estructura.
NUNCA codigo game-specific.**

## NOW
- [ ] Iterar arquitectura generalizable del harness — testeando en ls20 (principal) y g50t (secundario) → I-003
- [ ] Implementar multi-run con sintesis de abstracciones (N runs → sintetizar → reintentar) → I-003
- [ ] Implementar critic agent (cuestiona razonamientos, detecta contradicciones) → I-003

## NEXT
- [ ] Memoria de abstracciones cross-run (que sobreviva entre intentos)
- [ ] Multi-agente: paralelo + debate + sintesis
- [ ] Tools para el LLM (comparacion de frames, verificacion de hipotesis)
- [ ] Mejorar prompts: preguntas que fuercen mejor razonamiento

## LATER
- [ ] Multi-model orchestration (combinar GPT-5.4 + otros)
- [ ] Cross-game meta-knowledge (abstracciones que transfieren entre juegos)
- [ ] Test-time compute optimization (search, verification, self-correction)
- [ ] Submission a ARC-AGI-3 leaderboard

## DONE (recent)
- [x] Upgrade estructura a workflow estandar → I-002 — 2026-04-01
- [x] Implementar agente LLM basico para ARC-AGI-3 (vision + text) → I-001 — 2026-03-27
- [x] Research ARC-AGI-2 y ARC-AGI-3 → I-001 — 2026-03-27
- [x] Setup inicial del proyecto → I-001 — 2026-03-27
