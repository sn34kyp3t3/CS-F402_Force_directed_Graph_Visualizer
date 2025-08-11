#!/usr/bin/env python3
"""
Script to load and run algorithms on a saved graph from JSON file.
Usage: python run_saved_graph.py <path_to_graph.json>
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))

from graph_utils import parse_graph_from_file, visualize_graph
from main import ForceDirectedLayout

def run_saved_graph(json_file_path):
    """Load a saved graph from JSON and run force-directed algorithms on it."""
    
    print(f"Loading graph from: {json_file_path}")
    
    # Load the graph from JSON file
    try:
        graph = parse_graph_from_file(json_file_path)
        print(f"Successfully loaded graph with {len(graph.vertices)} vertices and {len(graph.edges)} edges")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return
    
    # Show initial layout
    print("\nInitial graph layout:")
    visualize_graph(graph, "Initial Graph Layout")
    
    # Randomize positions for better visualization
    print("\nRandomizing initial positions...")
    graph.randomize_positions(width=10, height=10)
    visualize_graph(graph, "Randomized Initial Layout")
    
    # Run Eades Spring Embedder
    print("\nRunning Eades Spring Embedder...")
    layout = ForceDirectedLayout(graph)
    positions = layout.eades_spring_embedder(max_iterations=100)
    
    # Show final layout
    print("Final layout (Eades Spring Embedder):")
    visualize_graph(graph, "Graph Layout - Eades Spring Embedder")
    
    # Reset and run Fruchterman-Reingold
    print("\nRunning Fruchterman-Reingold...")
    graph.randomize_positions(width=10, height=10)
    positions = layout.fruchterman_reingold(max_iterations=100)
    
    # Show final layout
    print("Final layout (Fruchterman-Reingold):")
    visualize_graph(graph, "Graph Layout - Fruchterman-Reingold")

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_saved_graph.py <path_to_graph.json>")
        print("Example: python run_saved_graph.py examples/sample_graph.json")
        return
    
    json_file_path = sys.argv[1]
    
    if not os.path.exists(json_file_path):
        print(f"Error: File '{json_file_path}' not found")
        return
    
    if not json_file_path.endswith('.json'):
        print("Warning: File doesn't have .json extension")
    
    run_saved_graph(json_file_path)

if __name__ == "__main__":
    main()
