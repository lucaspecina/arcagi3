---
id: 2
title: Upgrade estructura a workflow estandar
status: closed
type: task
created: 2026-04-01
closed: 2026-04-01
related: [I-001]
---

# I-002: Upgrade estructura a workflow estandar

## Status
- **Estado:** Completo
- **Ultimo resultado:** Proyecto alineado con dev-workflow skills de usuario
- **Ultimo commit:** pendiente
- **Proximo paso:** Ver TODO.md

## Pregunta
Alinear la estructura del proyecto con las user-level skills (dev-workflow,
project-bootstrap, setup-claude-md) para habilitar issue tracking, experiments,
autoresearch, y Codex collaboration.

## Log

### 2026-04-01 · TASK — Upgrade completo de estructura

Comparacion de project skills vs user skills revelo gaps significativos:
- No habia issue tracking (issues/)
- No habia experiment framework (experiments/)
- No habia autoresearch config (AUTORESEARCH.md)
- CLAUDE.md faltaban secciones: issue tracking, autoresearch, cleanup rules, where to find what
- TODO.md usaba formato plano en vez de NOW/NEXT/BLOCKED/LATER/DONE
- CURRENT_STATE.md no era friendly enough
- No habia skill /review

Cambios realizados:
- Creados: issues/, experiments/, research/archive/, research/synthesis/
- Creado: AUTORESEARCH.md (OFF)
- Creados: I-001 (retroactivo), I-002 (este issue)
- Reescrito: CLAUDE.md (template setup-claude-md completo)
- Reescrito: TODO.md (formato board con I-NNN refs)
- Reescrito: CURRENT_STATE.md (formato friendly)
- Actualizado: CHANGELOG.md (I-NNN refs + entrada upgrade)
- Actualizado: PROJECT.md (alineacion menor)
- Actualizado: research/README.md (archive/ y synthesis/)
- Mejorados: /status y /test skills
- Creado: /review skill

**Decision:** No duplicar skills de usuario a nivel proyecto. Las skills de Azure,
dev-workflow, codex-collab, bootstrap quedan a nivel usuario (cross-project).
Solo se crean skills especificas del proyecto.

## Conclusion
Proyecto completamente alineado con el workflow estandar definido en las user skills.
Issue tracking, experiments, y autoresearch habilitados. Listo para trabajo autonomo.
