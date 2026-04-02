# Research — ARC-AGI

## Active research lines

### 1. LLM approaches para ARC-AGI-2 (estatico)
- **Pregunta**: Cual es el mejor approach LLM para puzzles ARC-AGI estaticos?
- **Leer**: `notes/arc-agi-2-loading.md` (como cargar), `notes/arc-agi-overview.md` (approaches)
- **Estado**: Research completo — listo para implementar
- **TODO relacionado**: Crear loader, disenar prompts, implementar pipeline

### 2. ARC-AGI-3 entornos interactivos
- **Pregunta**: Como interactuar con ARC-AGI-3 y pueden los LLMs ser utiles?
- **Leer**: `notes/arc-agi-3-toolkit.md` (API completa), `notes/arc-agi-overview.md` (approaches)
- **Estado**: Research completo — toolkit documentado, agente basico implementado
- **TODO relacionado**: Testear con API keys reales, analizar resultados

## Notes index
| File | Contenido |
|------|-----------|
| `notes/arc-agi-overview.md` | Overview general: ARC-AGI-3 vs 2, approaches, competencia |
| `notes/arc-agi-3-toolkit.md` | API completa del toolkit `arc-agi`, agents, benchmarking |
| `notes/arc-agi-2-loading.md` | Como cargar datasets ARC-AGI-1/2 en Python (arckit, DIY) |
| `notes/arc-agi-dataset-loading.md` | Detalles adicionales de carga de datasets |
| `notes/sota-abstractions-tools-agents.md` | SOTA: abstracciones, tool creation, world models, memory, competition approaches (I-003) |

## Synthesis index
| File | Contenido |
|------|-----------|
| *(vacio — se llena al cerrar issues con findings generales)* | |

## Archive
| File | Razon |
|------|-------|
| *(vacio — se mueven docs superados por synthesis o decisiones)* | |

## Rules
- Notes van en `notes/`, conclusiones consolidadas en `synthesis/`
- Docs superados se mueven a `archive/` (no se borran)
- Cuando un finding se vuelve decision de proyecto, se promueve a `PROJECT.md`
- Siempre actualizar este README al crear/mover docs
- Referenciar issues con `I-NNN` cuando el research nace de un issue
