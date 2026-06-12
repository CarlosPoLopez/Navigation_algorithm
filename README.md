# Maze Solving with Reaction–Diffusion Systems (FitzHugh–Nagumo)

A pure-NumPy numerical simulation of the **FitzHugh–Nagumo (FHN)** reaction–diffusion
model used to solve mazes. A travelling wave (autowave) floods the maze from the
start, and a second retracting wave traces the shortest path back — a biologically
inspired alternative to classical graph search.

> Part of my Physics Bachelor's Thesis (TFG) at the University of Santiago de
> Compostela (2025–2026). A deep-learning extension (U-Net) that learns to reproduce
> these paths in milliseconds is developed in a separate repository.

## The model

The system integrates the coupled reaction–diffusion PDEs:

$$\frac{\partial u}{\partial t} = \epsilon (u - u^3 - v + F) + D_u \nabla^2 u$$

$$\frac{\partial v}{\partial t} = (u - \alpha v + \beta) + D_v \nabla^2 v$$

- **`u`** — activator (the propagating wavefront).
- **`v`** — inhibitor.
- **`F`** — forcing matrix encoding the maze geometry: it suppresses the activator on
  walls, so the wave can only travel along corridors.

## How it works

| Phase | Script | Numerical scheme | What it does |
|-------|--------|------------------|--------------|
| 1 — Expansion | `main_expansion.py` | Dufort–Frankel | Generates a solvable maze and propagates the autowave from start to exit. |
| 2 — Retraction | `main_retraction.py` | FTCS | Loads the phase-1 state and retracts the wave to reveal the optimal path. |

Core modules:

- **`solver.py`** — numerical engine (`dufort_frankel`, `FTCS`).
- **`maze.py`** — maze generators:
  - `generate_maze_multiroute` — builds a maze with several disjoint routes
    (a short one and longer distractors) between opposite corners. This is the
    generator used to build the thesis dataset.
  - `generate_maze_perfect` — classic perfect maze (single path between any
    two cells) via DFS backtracking.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Phase 1 — expansion. Creates frames_expansion/ and matrices_expansion/ (.npy state).
python main_expansion.py

# Phase 2 — retraction. Creates frames_retraction/ and matrices_retraction/ (.npy).
python main_retraction.py
```

You can also preview a maze on its own:

```bash
python maze.py
```

## Author

**Carlos Polo López** — BSc Physics, University of Santiago de Compostela.
Supervised by Alberto Pérez Muñuzuri and David García Selfa.
