# Force-Directed Graph Visualizer

A Python implementation of force-directed graph layout algorithms for computational geometry, specifically implementing the **Eades Spring Embedder** and **Fruchterman-Reingold** algorithms.

## Overview

This project implements force-directed graph drawing algorithms that treat nodes as electrically charged particles and edges as springs. The algorithms aim to minimize the energy in the system by balancing repulsive forces between all nodes and attractive forces between connected nodes.

### Algorithms Implemented

1. **Eades Spring Embedder**: Uses Coulomb-like repulsion and logarithmic spring attraction
2. **Fruchterman-Reingold**: Enhanced version with improved force calculations and cooling schedule

## Features

- **Interactive GUI**: Real-time visualization of graph layout evolution
- **Multiple Algorithms**: Compare different force-directed approaches
- **Customizable Parameters**: Adjust force constants, damping, and iteration counts
- **Sample Graphs**: Included example graphs for testing
- **Performance Metrics**: Track convergence and layout quality

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd force_directed_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python main.py
```

### Programmatic Usage

```python
from main import ForceDirectedLayout
from common.graph_utils import create_sample_graph

# Create a sample graph
graph = create_sample_graph()

# Initialize the layout engine
layout = ForceDirectedLayout(graph)

# Run Eades algorithm
positions = layout.eades_spring_embedder(max_iterations=1000)

# Run Fruchterman-Reingold algorithm
positions = layout.fruchterman_reingold(max_iterations=1000)
```

## Algorithm Details

### Eades Spring Embedder

- **Repulsive Force**: F_rep = k_rep / d² (Coulomb-like)
- **Attractive Force**: F_att = k_att * log(d / l₀) (spring-like)
- **Goal**: Minimize total system energy

### Fruchterman-Reingold

- **Repulsive Force**: F_rep = k² / d (where k is optimal distance)
- **Attractive Force**: F_att = d² / k (for connected vertices)
- **Cooling Schedule**: Gradually reduces movement to find stable layout

## Project Structure

```
force_directed_project/
├── main.py              # Main algorithm implementation
├── common/              # Shared utilities
│   ├── graph_utils.py   # Graph data structures
│   └── graph_gui.py     # GUI components
├── examples/            # Sample graph files
├── docs/               # Documentation
├── course_content/     # Lecture materials
└── requirements.txt    # Python dependencies
```

## Dependencies

- matplotlib (visualization)
- numpy (numerical computations)
- tkinter (GUI framework)

## Academic Context

This project is based on computational geometry lecture materials covering:
- **Lecture 9**: Spring Embedder by Eades (slides 5-1 to 5-12)
- **Lecture 10**: Fruchterman & Reingold Variant (slides 2-1 to 3-4)

The algorithms address the general layout problem by optimizing for:
- Adjacent vertices close together
- Non-adjacent vertices far apart
- Short and straight edges
- Even vertex distribution
- Minimal edge crossings

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational purposes as part of a computational geometry course.
