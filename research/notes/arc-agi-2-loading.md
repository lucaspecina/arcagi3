# ARC-AGI-2 / ARC-AGI-1 — Como cargar datasets

Fecha: 2026-03-27

## Formato JSON de tareas

Cada tarea es un `.json` (nombre: hex ID, ej `007bbfb7.json`):
```json
{
  "train": [
    {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
    {"input": [[2, 3], [3, 2]], "output": [[3, 2], [2, 3]]}
  ],
  "test": [
    {"input": [[4, 5], [5, 4]], "output": [[5, 4], [4, 5]]}
  ]
}
```

- Grillas: `list[list[int]]`, valores 0-9, tamano 1x1 a 30x30
- `train`: pares de ejemplo (2-10 tipicamente)
- `test`: pares para evaluar (1-3 tipicamente, output es ground truth)

## Estructura de repos

### ARC-AGI-1 (fchollet/ARC-AGI)
```
data/
  training/     # 400 tareas
  evaluation/   # 400 tareas
```

### ARC-AGI-2 (arcprize/ARC-AGI-2)
```
data/
  training/     # 1000 tareas (incluye las 400 de v1 + 600 nuevas)
  evaluation/   # 120 tareas
```

## Opciones para cargar en Python

### Opcion 1: arckit (RECOMENDADO)

```bash
pip install arckit
```

```python
import arckit

# Cargar ARC-AGI-1
train_set, eval_set = arckit.load_data()

# Cargar ARC-AGI-2
train_set, eval_set = arckit.load_data("arcagi2")

# Iterar tareas
for task in train_set:
    print(task.id)          # str: hex task ID
    for pair in task.train:
        print(pair[0])      # numpy.ndarray — input grid
        print(pair[1])      # numpy.ndarray — output grid
    for pair in task.test:
        print(pair[0])      # input
        print(pair[1])      # output (ground truth)

# Acceder tarea por ID
task = train_set["007bbfb7"]

# Visualizar
task.show()  # matplotlib
```

Ventajas: bundlea el JSON internamente (no necesita clonar repo), devuelve numpy arrays, tiene visualizacion.

### Opcion 2: arc-agi-core

```bash
pip install arc-agi-core
```

```python
import arc

# Descargar y cargar
dataset = arc.ARC2Training.download("./data")
for task in dataset:
    print(task.id)
    for pair in task.train:
        print(pair.input)   # Grid object
        print(pair.output)
```

### Opcion 3: DIY (cargar JSON directamente)

```python
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np

@dataclass
class Pair:
    input: np.ndarray
    output: np.ndarray

@dataclass
class Task:
    id: str
    train: list[Pair]
    test: list[Pair]

def load_task(path: Path) -> Task:
    with open(path) as f:
        data = json.load(f)
    return Task(
        id=path.stem,
        train=[Pair(np.array(p["input"]), np.array(p["output"])) for p in data["train"]],
        test=[Pair(np.array(p["input"]), np.array(p["output"])) for p in data["test"]],
    )

def load_dataset(directory: Path) -> list[Task]:
    return [load_task(p) for p in sorted(directory.glob("*.json"))]

# Uso:
tasks = load_dataset(Path("data/training"))
```

## Evaluacion

- **Match exacto**: todas las celdas deben coincidir, dimensiones correctas
- ARC-AGI-1: 3 intentos por test pair
- ARC-AGI-2: 2 intentos por test pair
- Tarea "resuelta" solo si TODOS los test pairs son correctos

```python
def evaluate(predicted: np.ndarray, expected: np.ndarray) -> bool:
    return predicted.shape == expected.shape and np.array_equal(predicted, expected)
```

## Sources
- https://github.com/fchollet/ARC-AGI
- https://github.com/arcprize/ARC-AGI-2
- https://pypi.org/project/arckit/
- https://pypi.org/project/arc-agi-core/
