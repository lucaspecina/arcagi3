# SOTA Research: Abstractions, Tool Creation, and Agent Architectures

Fecha: 2026-04-02
Issue: I-003

## Context
Survey de tecnicas SOTA para el diseño del harness ARC-AGI-3. Foco en
abstraccion, auto-creacion de tools, self-improvement, multi-agent,
world models, memory, y approaches competitivos.

---

## 1. Abstraction Learning in LLM Agents

### TheoryCoder-2 (arXiv:2602.00929, Jan 2026)
RL agent que usa LLM in-context learning para sintetizar abstracciones
reusables autonomamente, luego las integra en planificacion jerarquica.
Mucho mas sample-efficient que WorldCoder. Resuelve BabyAI, Minihack,
Sokoban. **El mas relevante para nuestro problema.**

### BARC Dataset (Ellis et al., 2024)
400K tareas ARC-like generadas via program synthesis. Base para entrenar
modelos especializados en ARC. Dos approaches: induccion (inferir programa)
y transduccion (predecir output directamente).

### Ryan Greenblatt (Redwood Research, 2024)
42-50% en ARC-AGI-1. GPT-4o genera miles de programas Python candidatos
(k=2048-8000), verifica contra demos. Accuracy escala log-linealmente con k.
Incluye revision step donde el LLM ve output real vs esperado.

### NVARC (NVIDIA, 1er lugar ARC Prize 2025, ~24% ARC-AGI-2)
Synthetic data a escala (103K puzzles validados) + test-time training.
Un modelo 4B supera modelos mucho mas grandes. Costo: ~20 cents/tarea.

### Product of Experts (arXiv:2505.07859, ICML 2025)
71.6% ARC-AGI-1. Un solo LLM fine-tuned en dual rol: generador (DFS para
soluciones diversas) y scorer (media geometrica de likelihoods sobre
augmentations). La augmentation-based scoring es la innovacion clave.

### SOAR (arXiv:2507.14172, ICML 2025)
Program synthesis evolutiva auto-mejorante. Alterna entre busqueda
evolutiva con LLM y hindsight learning (intentos fallidos se convierten
en training data relabeleando para tareas que accidentalmente resuelven).
52% ARC-AGI-1 sin DSLs humanos.

---

## 2. Auto Tool Creation / Tool Synthesis

### LATM (arXiv:2305.17126)
Framework fundacional. LLM fuerte crea tools Python reusables, LLM barato
los usa. Costo de creacion amortizado sobre muchos usos.

### Voyager (arXiv:2305.16291, NeurIPS 2023)
Agente Minecraft con 3 componentes:
1. Curriculum automatico (tareas progresivamente mas dificiles)
2. **Skill library** de programas ejecutables indexados por embedding
3. Prompting iterativo con feedback del ambiente
3.3x mas items, 15.3x mas rapido. Skills transfieren a mundos nuevos.
github.com/MineDojo/Voyager

### CREATOR (arXiv:2305.14318)
Separa creacion abstracta de tool de ejecucion concreta.

### Alita (arXiv:2505.20286)
75.15% en GAIA con tools minimos predefinidos. Genera MCPs dinamicamente.
"Minimal predefinition, maximal self-evolution."

### SMITH (arXiv:2512.11303, Dec 2025)
Unifica tool creation con memory cross-task (procedural + semantic +
episodic). Experience sharing via episodic memory retrieval.

---

## 3. Self-Evolving / Self-Improving Agents

### Reflexion (arXiv:2303.11366)
Verbal RL. Despues de fallo, genera reflexion en lenguaje natural,
la guarda en memoria episodica. En retry, la reflexion guia mejores
decisiones. +11% HumanEval, +22% ALFWorld. Limitacion: confirmation
bias por ser single-model.

### Multi-Agent Reflexion (arXiv:2512.20845, Dec 2025)
Resuelve limitacion de Reflexion separando roles: agentes distintos para
generacion, evaluacion y reflexion. Rompe el bias de single-model.

### DeepSeek-R1 (arXiv:2501.12948, Nature 2025)
Reasoning emerge de RL puro con rewards verificables (RLVR). Feedback
binario correcto/incorrecto es suficiente. Self-reflection, verificacion
y adaptacion de estrategia emergen como comportamientos durante training.

---

## 4. Multi-Agent / Ensemble

### Multi-Agent Debate (arXiv:2305.14325, ICML 2024)
Multiples LLMs proponen y critican mutuamente. Equipos heterogeneos
(modelos diferentes) dan 91% vs 82% en GSM-8K.

### Finding critico (ICLR 2025)
MAD actual no supera consistentemente votacion simple por mayoria.
Los rounds de debate solo aportan con heterogeneidad o reglas especificas.

### Best-of-N (BoN)
Generar N candidatos, scorer selecciona el mejor. Simple y efectivo.
Para ARC-AGI-3, el environment mismo es el verificador.

---

## 5. World Models + Planning con LLMs

### WorldCoder (arXiv:2402.12275, NeurIPS 2024)
**EL MAS RELEVANTE.** LLM agent que construye un programa Python como
world model a partir de interacciones. Sintetiza funciones de transicion
en codigo, explora con reward functions optimistas, planifica usando el
code world model. 10,000x mas rapido que deep RL. Code-based world models
son interpretables, editables, y transfieren entre tareas.

### GIF-MCTS (arXiv:2405.15383, NeurIPS 2024)
MCTS para sintetizar Code World Models. Una vez sintetizados, ejecutan
4-7 ordenes de magnitud mas rapido que LLM calls por paso de planning.

### Cost-Augmented MCTS (arXiv:2505.14656, 2025)
MCTS con costos explicitos y budget constraints. Relevante para RHAE
donde la eficiencia de acciones se penaliza cuadraticamente.

---

## 6. Memory Architectures

### Generative Agents (arXiv:2304.03442, UIST 2023)
Fundacional. Memory stream + reflection + planning. Retrieval combina
recency, relevance, importance.

### MemGPT (arXiv:2310.08560, 2023)
Dos tiers: main context (working memory en prompt) + archival storage
(retrieved on demand). Self-editing de que promover/demover.

### A-MEM (arXiv:2502.12110, NeurIPS 2025)
Memoria agentica tipo Zettelkasten. Notas estructuradas con links.
85-93% reduccion de tokens vs MemGPT.

### MACLA (arXiv:2512.18950, Dec 2025)
Memoria procedural jerarquica. Comprime 2,851 trayectorias en 187
procedimientos reusables.

---

## 7. ARC-AGI-3 Competition Approaches

### StochasticGoose (1ro, 12.58%)
CNN action prediction + sparse RL. Hierarchical sampling (action type
luego coordenada via convolucion). Experience buffer con deduplicacion
hash. Retraining iterativo entre niveles.
github.com/DriesSmit/ARC3-solution

### Blind Squirrel (2do, 6.71%)
State graph explicito. Poda acciones que crean loops o no cambian estado.
Back-labels con distancias cuando score mejora. ResNet18 value model.

### Graph-Based Exploration (3ro, arXiv:2512.24156)
Training-free, graph-based. Vision processing + systematic state-space
exploration. Segmenta frames, prioriza por saliencia visual.
github.com/dolphin-in-a-coma/arc-agi-3-just-explore

### Key learnings
1. Non-LLM approaches dominan (12% vs <1% LLM)
2. State deduplication es critica
3. Sparse RL funciona (level completion como reward)
4. 2D spatial bias importa (CNN > texto)
5. Exploracion sistematica > random

---

## 8. Proposed Architecture (synthesis)

### Layer 1: Perception (Non-LLM)
- CNN frame encoder, frame hashing, diff programatico, object segmentation

### Layer 2: State Management (Non-LLM)
- State graph dirigido, novelty detector, loop pruner

### Layer 3: World Model (LLM + Code)
- WorldCoder pattern: LLM escribe Python modelando transition function
- Se testea contra transiciones observadas, mismatch → revision

### Layer 4: Reasoning (LLM)
- Hypothesis generation, strategy selection, memory management, reflection

### Layer 5: Action Selection (Hybrid)
- System 1: graph search + novelty heuristic (exploracion)
- System 2: LLM planning (goal-directed)
- Critic: compara propuesta con prediccion del world model

### Layer 6: Learning (Cross-Level/Game)
- Skill library (Voyager), verified control mappings, procedural memories
- Hindsight relabeling de secuencias fallidas

---

## Priority Implementation Roadmap

1. **Execution-grounded perception** — computar diffs programaticamente,
   DECIRLE al LLM que cambio en vez de preguntarle. (Alto impacto, bajo esfuerzo)
2. **State graph + deduplication** — hash states, grafo dirigido, podar loops.
   (Alto impacto, esfuerzo medio. Probado por top 3.)
3. **WorldCoder code world model** — LLM escribe Python del transition function.
   (Muy alto impacto, alto esfuerzo. Mas novel.)
4. **Structured memory with reflection** — episodic/semantic/procedural tiers.
   (Medio impacto, esfuerzo medio.)
5. **Hybrid System 1/2** — non-LLM exploration + LLM reasoning.
   (Alto impacto, alto esfuerzo.)

---

## Key Repos

| Repo | Que es |
|---|---|
| github.com/DriesSmit/ARC3-solution | StochasticGoose (1ro ARC-AGI-3) |
| github.com/dolphin-in-a-coma/arc-agi-3-just-explore | Graph exploration (3ro) |
| github.com/MineDojo/Voyager | Skill library architecture |
| github.com/arcprize/ARC-AGI-3-Agents | Agentes oficiales de ejemplo |
| github.com/arcprize/arc-agi-3-benchmarking | Benchmarking oficial |

## Sources
- TheoryCoder-2: arXiv:2602.00929
- SOAR: arXiv:2507.14172
- Product of Experts: arXiv:2505.07859
- LATM: arXiv:2305.17126
- Voyager: arXiv:2305.16291
- SMITH: arXiv:2512.11303
- Reflexion: arXiv:2303.11366
- Multi-Agent Reflexion: arXiv:2512.20845
- DeepSeek-R1: arXiv:2501.12948
- Multi-Agent Debate: arXiv:2305.14325
- WorldCoder: arXiv:2402.12275
- GIF-MCTS: arXiv:2405.15383
- Generative Agents: arXiv:2304.03442
- MemGPT: arXiv:2310.08560
- A-MEM: arXiv:2502.12110
- MACLA: arXiv:2512.18950
- Graph Exploration ARC3: arXiv:2512.24156
- ARC-AGI-3 Technical Report: arXiv:2603.24621
- ARC-AGI-3 30-Day Learnings: arcprize.org/blog/arc-agi-3-preview-30-day-learnings
- StochasticGoose writeup: medium.com/@dries.epos
- Ryan Greenblatt: blog.redwoodresearch.org
- RLEF: arXiv:2410.02089
- Cost-Augmented MCTS: arXiv:2505.14656
