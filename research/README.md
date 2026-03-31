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
- **Estado**: Research completo — toolkit documentado, agents de ejemplo identificados
- **TODO relacionado**: Instalar toolkit, probar con agent basico

## Notes index
| File | Contenido |
|------|-----------|
| `notes/arc-agi-overview.md` | Overview general: ARC-AGI-3 vs 2, approaches, competencia |
| `notes/arc-agi-3-toolkit.md` | API completa del toolkit `arc-agi`, agents, benchmarking |
| `notes/arc-agi-2-loading.md` | Como cargar datasets ARC-AGI-1/2 en Python (arckit, DIY) |
| `notes/arc-agi-dataset-loading.md` | Referencia completa: JSON format, arckit, arc-agi-core, DIY loaders |

## Rules
- Notes van en `notes/`, conclusiones consolidadas en `synthesis/`
- Cuando un finding se vuelve decision de proyecto, se promueve a `PROJECT.md`
- Siempre actualizar este README al crear/mover docs
