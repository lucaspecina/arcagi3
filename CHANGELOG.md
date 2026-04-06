# Changelog

## 2026-04-06
- Add LLM Judge Oracle for evaluating agent understanding vs golden thinking (I-003)
- Add bench runner with chained runs, belief transfer, and parallel games (I-003)
- Add golden thinking for ls20 (controls, mechanics, milestones, anti-patterns)
- Benchmark Azure Foundry models: gpt-5.4-mini selected for autoresearch (36s/5 actions)
- Fix Unicode encoding crash on Windows (cp1252 chars in agent.py)
- Add --judge and --judge-model flags to run.py CLI
- Conda env setup: arcagi3 with Python 3.12

## 2026-04-01
- Upgrade estructura a workflow estandar: issue tracking, experiments, autoresearch (I-002)
- Reescrito CLAUDE.md, TODO.md, CURRENT_STATE.md alineados con dev-workflow skills
- Creados issues/, experiments/, research/archive/, research/synthesis/
- Creado AUTORESEARCH.md (OFF)
- Mejorados skills /status y /test, creado /review

## 2026-03-27
- Setup inicial del proyecto con estructura de documentacion (I-001)
- Research completo: ARC-AGI-3 toolkit/agents/benchmarking + ARC-AGI-2 formatos/loaders (I-001)
- Implementar agente LLM para ARC-AGI-3: vision+text, Azure Foundry, message history (I-001)
- Utilidades de grilla: grid->image, grid->text hex, image diff (I-001)
