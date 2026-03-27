# ARC-AGI Dataset Format & Python Loading — Research Notes

Fecha: 2026-03-27

---

## 1. Repository Structure

### ARC-AGI-1 (fchollet/ARC-AGI)

```
data/
├── training/       # 400 task JSON files
│   ├── 007bbfb7.json
│   ├── 00d62c1b.json
│   └── ...
└── evaluation/     # 400 task JSON files
    ├── 0a938d79.json
    └── ...
apps/
└── testing_interface.html
```

### ARC-AGI-2 (arcprize/ARC-AGI-2)

```
data/
├── training/       # 1,000 task JSON files (includes ARC-AGI-1 tasks + new ones)
│   ├── 00576224.json
│   └── ...
└── evaluation/     # 120 task JSON files
    ├── ...
    └── ...
```

Key differences:
- ARC-AGI-1: 400 training + 400 evaluation = 800 tasks total
- ARC-AGI-2: 1,000 training + 120 evaluation = 1,120 tasks total
- ARC-AGI-2 training set combines ARC-AGI-1 tasks + new tasks
- Evaluation set is smaller (120 vs 400) but rated at 66% average human performance

---

## 2. JSON Task Format

Each task is a single `.json` file named by hex ID (e.g., `007bbfb7.json`).

### Structure

```json
{
  "train": [
    {
      "input": [[0, 7, 7], [7, 7, 7], [0, 7, 7]],
      "output": [[0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], ...]
    },
    {
      "input": [[4, 0, 4], [0, 0, 0], [0, 4, 0]],
      "output": [[4, 0, 4, 0, 0, 0, 4, 0, 4], ...]
    }
    // ... typically 2-5 training examples
  ],
  "test": [
    {
      "input": [[7, 0, 7], [7, 0, 7], [7, 7, 0]],
      "output": [[7, 0, 7, 0, 0, 0, 7, 0, 7], ...]
    }
    // ... typically 1-2 test examples
  ]
}
```

### Grid Format
- **Type**: List of lists of integers (`list[list[int]]`)
- **Values**: Integers 0-9 (each mapped to a color for visualization)
- **Dimensions**: Rectangular, from 1x1 to 30x30
- **0 is typically "background"** (black in visualization)

### Real example (007bbfb7.json from ARC-AGI-1):

```json
{
  "train": [
    {
      "input": [[0, 7, 7], [7, 7, 7], [0, 7, 7]],
      "output": [
        [0, 0, 0, 0, 7, 7, 0, 7, 7],
        [0, 0, 0, 7, 7, 7, 7, 7, 7],
        [0, 0, 0, 0, 7, 7, 0, 7, 7],
        [0, 7, 7, 0, 7, 7, 0, 7, 7],
        [7, 7, 7, 7, 7, 7, 7, 7, 7],
        [0, 7, 7, 0, 7, 7, 0, 7, 7],
        [0, 0, 0, 0, 7, 7, 0, 7, 7],
        [0, 0, 0, 7, 7, 7, 7, 7, 7],
        [0, 0, 0, 0, 7, 7, 0, 7, 7]
      ]
    },
    // ... 4 more training examples
  ],
  "test": [
    {
      "input": [[7, 0, 7], [7, 0, 7], [7, 7, 0]],
      "output": [
        [7, 0, 7, 0, 0, 0, 7, 0, 7],
        [7, 0, 7, 0, 0, 0, 7, 0, 7],
        [7, 7, 0, 0, 0, 0, 7, 7, 0],
        [7, 0, 7, 0, 0, 0, 7, 0, 7],
        [7, 0, 7, 0, 0, 0, 7, 0, 7],
        [7, 7, 0, 0, 0, 0, 7, 7, 0],
        [7, 0, 7, 7, 0, 7, 0, 0, 0],
        [7, 0, 7, 7, 0, 7, 0, 0, 0],
        [7, 7, 0, 7, 7, 0, 0, 0, 0]
      ]
    }
  ]
}
```

### Real example (00576224.json from ARC-AGI-2):

```json
{
  "train": [
    {
      "input": [[7, 9], [4, 3]],
      "output": [
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3],
        [9, 7, 9, 7, 9, 7],
        [3, 4, 3, 4, 3, 4],
        [7, 9, 7, 9, 7, 9],
        [4, 3, 4, 3, 4, 3]
      ]
    }
  ],
  "test": [
    {
      "input": [[3, 2], [7, 8]],
      "output": [
        [3, 2, 3, 2, 3, 2],
        [7, 8, 7, 8, 7, 8],
        [2, 3, 2, 3, 2, 3],
        [8, 7, 8, 7, 8, 7],
        [3, 2, 3, 2, 3, 2],
        [7, 8, 7, 8, 7, 8]
      ]
    }
  ]
}
```

---

## 3. Evaluation Rules

- Must produce **exact** output grids — ALL cells must match
- ARC-AGI-1: **3 attempts** per test input
- ARC-AGI-2: **2 attempts** per test input
- Task is "solved" only if ALL test inputs in that task are correct
- Grid dimensions must also be correct
- Score = number of tasks fully solved / total tasks

---

## 4. Python Packages for Loading ARC-AGI

### 4a. `arckit` (recommended, most mature)

**Install**: `pip install -U arckit`
**GitHub**: https://github.com/mxbi/arckit
**License**: Apache 2.0

```python
import arckit

# Load datasets — returns (TaskSet, TaskSet) for train/eval
train_set, eval_set = arckit.load_data()          # ARC-AGI-2 (default, "latest")
train_set, eval_set = arckit.load_data("arcagi")   # ARC-AGI-1
train_set, eval_set = arckit.load_data("kaggle")   # Kaggle 2025 competition
train_set, eval_set = arckit.load_data("kaggle2024")  # Kaggle 2024

# Version aliases:
# "latest" / "arcagi2"    -> ARC-AGI-2
# "arcagi" / "arcagi1"    -> ARC-AGI-1
# "kaggle" / "kaggle2025" -> Kaggle 2025
# "kaggle2024"            -> Kaggle 2024
# "arc" / "kaggle2019"    -> Original ARC (2019)

# Access tasks
train_set                     # <TaskSet: 400 tasks>
task = train_set[0]           # Index by position
task = train_set['007bbfb7']  # Index by task ID
task = arckit.load_single('007bbfb7')  # Load one task

# Task properties
task.id        # '007bbfb7'
task.dataset   # 'train' or 'eval'
task.train     # List[Tuple[np.ndarray, np.ndarray]] — (input, output) pairs
task.test      # List[Tuple[np.ndarray, np.ndarray]]

# Access grids (NumPy arrays)
inp, out = task.train[0]   # First training example
inp.shape                  # e.g. (3, 3)
inp.dtype                  # int

# Convert back to JSON dict
task.to_dict()
# Returns: {"id": "007bbfb7", "train": [{"input": [...], "output": [...]}, ...], "test": [...]}

# Visualization
task.show()                      # Terminal display with ANSI colors
task.show(answer=True)           # Also show test outputs

import arckit.vis as vis
vis.draw_grid(inp, xmax=3, ymax=3)
vis.draw_task(task, width=10, height=6)
vis.output_drawing(drawing, "task.png")  # .svg, .pdf, .png
vis.print_grid(inp)              # Terminal output

# Score a Kaggle submission
score = eval_set.score_submission('submission.csv', topn=2)

# Check if prediction is correct
task.scoreA(predicted_output)  # Returns True/False for first test pair
```

#### Full arckit Task class source (key parts):

```python
class Task:
    def __init__(self, id, train, test, dataset=None, version=None):
        self.dataset = dataset
        self.version = version
        self.id = id
        self.train = [(np.array(example['input']), np.array(example['output'])) for example in train]
        self.test = [(np.array(example['input']), np.array(example['output'])) for example in test]

    @classmethod
    def from_json(cls, json_file):
        with open(json_file) as f:
            data = json.load(f)
            return cls(os.path.basename(json_file)[:-5], data['train'], data['test'])

    def to_dict(self):
        return {
            "id": self.id,
            "train": [{"input": input.tolist(), "output": output.tolist()} for input, output in self.train],
            "test": [{"input": input.tolist(), "output": output.tolist()} for input, output in self.test]
        }

    def scoreA(self, output):
        output = np.array(output).astype(int)
        if output.shape != self.test[0][1].shape:
            return False
        return (output == self.test[0][1]).all()

class TaskSet:
    def __init__(self, tasks):
        tasks = sorted(tasks)
        self.tasks = tasks
        self.task_dict = {task.id: task for task in tasks}

    def __getitem__(self, item):
        if isinstance(item, slice):
            return TaskSet(self.tasks[item])
        get = self.task_dict.get(item)
        if get is None:
            return self.tasks[item]
        return get

    def score_submission(self, fn, topn=2, return_correct=False):
        # Reads CSV with output_id column (taskid_testnum) and output column
        # output format: pipe-separated rows, space-separated predictions
        # e.g. "1234|5678 1234|5679"
        ...

def load_data(version='latest'):
    data = get_data_json(version)
    train_tasks = [Task(id, t['train'], t['test'], 'train', version=version)
                   for id, t in data['train'].items()]
    eval_tasks = [Task(id, t['train'], t['test'], 'eval', version=version)
                  for id, t in data['eval'].items()]
    return TaskSet(train_tasks), TaskSet(eval_tasks)
```

### 4b. `arc-agi-core` (newer, more structured)

**Install**: `pip install arc-agi-core`
**PyPI**: https://pypi.org/project/arc-agi-core/0.1.14/

```python
import arc_agi_core as arc

# Download and load ARC-AGI-2 training set
dataset = arc.ARC2Training.download("dataset/arc-agi-2/training")

# Sample a random task
task = dataset.sample()

# Access pairs
pair = task.train[0]
grid = pair.input    # Grid object
grid = pair.output   # Grid object

# Core classes:
# - Grid:    2D grid of symbols (0-9), with methods for JSON/NPY/SVG I/O
# - Pair:    input Grid + output Grid, supports censoring output for test pairs
# - Task:    collection of train Pairs + test Pairs
# - Dataset: collection of Tasks, supports lazy loading, subset, sampling, shuffling
```

Features:
- Flexible I/O: JSON, NumPy (.npy), SVG
- Visualization: terminal (ANSI), Jupyter (HTML), SVG
- Lazy loading from directories
- Dataset subclasses for downloading official ARC datasets
- Demo notebook: `examples/arc-agi-demo/demo.ipynb`

### 4c. `arc-agi` (ARC-AGI-3 toolkit, interactive format)

**Install**: `pip install arc-agi` or `uv add arc-agi`
**Note**: This is for ARC-AGI-3 (interactive, NOT static puzzles)

```python
import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")

for _ in range(10):
    env.step(GameAction.ACTION1)

print(arc.get_scorecard())
```

This is NOT relevant for ARC-AGI-1/2 static tasks.

---

## 5. DIY Loading (no dependencies)

### Minimal loader — plain Python + json

```python
import json
import os
from pathlib import Path

def load_tasks_from_directory(directory: str) -> dict:
    """Load all ARC tasks from a directory of JSON files."""
    tasks = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.json'):
            task_id = filename[:-5]  # Remove .json extension
            with open(os.path.join(directory, filename)) as f:
                tasks[task_id] = json.load(f)
    return tasks

# Usage:
training_tasks = load_tasks_from_directory("data/training")
evaluation_tasks = load_tasks_from_directory("data/evaluation")

# Access a specific task:
task = training_tasks["007bbfb7"]
train_examples = task["train"]   # list of {"input": [...], "output": [...]}
test_examples = task["test"]     # list of {"input": [...], "output": [...]}

# First training example:
inp = train_examples[0]["input"]   # list[list[int]]
out = train_examples[0]["output"]  # list[list[int]]

# Grid dimensions:
height = len(inp)
width = len(inp[0])
```

### With NumPy conversion

```python
import json
import numpy as np
from pathlib import Path

def load_arc_task(filepath):
    """Load a single ARC task and convert grids to numpy arrays."""
    with open(filepath) as f:
        data = json.load(f)

    result = {"train": [], "test": []}
    for split in ["train", "test"]:
        for pair in data[split]:
            result[split].append({
                "input": np.array(pair["input"], dtype=np.int8),
                "output": np.array(pair["output"], dtype=np.int8),
            })
    return result

def load_all_tasks(directory):
    """Load all tasks from a directory."""
    tasks = {}
    for path in sorted(Path(directory).glob("*.json")):
        task_id = path.stem
        tasks[task_id] = load_arc_task(path)
    return tasks
```

### With dataclasses (typed, clean)

```python
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

Grid = np.ndarray  # shape (H, W), dtype int, values 0-9

@dataclass
class Pair:
    input: Grid
    output: Grid

    @classmethod
    def from_dict(cls, d: dict) -> "Pair":
        return cls(
            input=np.array(d["input"], dtype=np.int8),
            output=np.array(d["output"], dtype=np.int8),
        )

    def to_dict(self) -> dict:
        return {
            "input": self.input.tolist(),
            "output": self.output.tolist(),
        }

@dataclass
class Task:
    task_id: str
    train: list[Pair]
    test: list[Pair]

    @classmethod
    def from_json(cls, filepath: str | Path) -> "Task":
        filepath = Path(filepath)
        with open(filepath) as f:
            data = json.load(f)
        return cls(
            task_id=filepath.stem,
            train=[Pair.from_dict(p) for p in data["train"]],
            test=[Pair.from_dict(p) for p in data["test"]],
        )

    @classmethod
    def from_dict(cls, task_id: str, data: dict) -> "Task":
        return cls(
            task_id=task_id,
            train=[Pair.from_dict(p) for p in data["train"]],
            test=[Pair.from_dict(p) for p in data["test"]],
        )

    def to_dict(self) -> dict:
        return {
            "train": [p.to_dict() for p in self.train],
            "test": [p.to_dict() for p in self.test],
        }

@dataclass
class Dataset:
    tasks: dict[str, Task] = field(default_factory=dict)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "Dataset":
        directory = Path(directory)
        tasks = {}
        for path in sorted(directory.glob("*.json")):
            task = Task.from_json(path)
            tasks[task.task_id] = task
        return cls(tasks=tasks)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, task_id: str) -> Task:
        return self.tasks[task_id]

    def __iter__(self):
        return iter(self.tasks.values())

# Usage:
training = Dataset.from_directory("data/training")
evaluation = Dataset.from_directory("data/evaluation")

task = training["007bbfb7"]
for pair in task.train:
    print(f"Input shape: {pair.input.shape}, Output shape: {pair.output.shape}")
```

---

## 6. Evaluation — Comparing Predicted Grid to Actual

### Basic comparison

```python
import numpy as np

def grids_match(predicted: list[list[int]], expected: list[list[int]]) -> bool:
    """Check if predicted grid exactly matches expected grid."""
    pred = np.array(predicted)
    exp = np.array(expected)
    if pred.shape != exp.shape:
        return False
    return np.array_equal(pred, exp)
```

### With multiple attempts (as in competition)

```python
def evaluate_task(task_data: dict, predictions: list[list[list[list[int]]]],
                  max_attempts: int = 2) -> bool:
    """
    Evaluate a task. Returns True if ALL test pairs are solved.

    predictions: for each test pair, a list of up to max_attempts predicted grids.
    Each grid is list[list[int]].
    """
    test_pairs = task_data["test"]

    for i, test_pair in enumerate(test_pairs):
        expected = np.array(test_pair["output"])
        solved = False
        for attempt in predictions[i][:max_attempts]:
            pred = np.array(attempt)
            if pred.shape == expected.shape and np.array_equal(pred, expected):
                solved = True
                break
        if not solved:
            return False
    return True

def score_dataset(tasks: dict, all_predictions: dict, max_attempts: int = 2) -> float:
    """Score predictions across all tasks. Returns fraction solved."""
    solved = 0
    for task_id, task_data in tasks.items():
        if task_id in all_predictions:
            if evaluate_task(task_data, all_predictions[task_id], max_attempts):
                solved += 1
    return solved / len(tasks)
```

### Kaggle submission format

The Kaggle competition uses a CSV with columns `output_id` and `output`:
- `output_id`: `{task_id}_{test_index}` (e.g., `007bbfb7_0`)
- `output`: Pipe-separated rows, space-separated predictions (e.g., `|12|34 |56|78`)

```python
# arckit's score_submission parses this format:
# Row format: "taskid_0", "|row1val1row1val2|row2val1row2val2 |alt_row1|alt_row2"
# Pipe separates rows within a grid, space separates alternative predictions
```

---

## 7. Color Mapping (for visualization)

Standard ARC color palette (index -> color):
```
0: Black (background)
1: Blue
2: Red
3: Green
4: Yellow
5: Gray/Grey
6: Magenta/Pink
7: Orange
8: Cyan/Azure
9: Maroon/Brown
```

---

## 8. Cloning the Data

```bash
# ARC-AGI-1 (original, 800 tasks)
git clone https://github.com/fchollet/ARC-AGI.git

# ARC-AGI-2 (expanded, 1120 tasks)
git clone https://github.com/arcprize/ARC-AGI-2.git
```

Or download directly from arckit (bundles the data):
```python
import arckit
train, eval = arckit.load_data("arcagi2")  # No clone needed
```

---

## Sources
- https://github.com/fchollet/ARC-AGI (ARC-AGI-1 repo)
- https://github.com/arcprize/ARC-AGI-2 (ARC-AGI-2 repo)
- https://github.com/mxbi/arckit (arckit library, full source)
- https://pypi.org/project/arc-agi-core/0.1.14/ (arc-agi-core package)
- https://pypi.org/project/arckit/ (arckit PyPI)
- https://docs.arcprize.org/ (ARC-AGI-3 docs)
- https://medium.com/bitgrit-data-science-publication/can-llms-reason-c58a45e17059
