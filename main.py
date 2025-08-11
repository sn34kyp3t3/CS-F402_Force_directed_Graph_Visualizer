import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from graph_utils import Graph, Vertex, visualize_graph, create_sample_graph

class ForceDirectedLayout:
    """Implements force-directed graph layout algorithms."""
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.ideal_edge_length = 1.0
        self.repulsion_constant = 1.0
        self.attraction_constant = 1.0
        self.damping = 0.95
        self.max_iterations = 1000
        self.tolerance = 1e-6
    
    def eades_spring_embedder(self, max_iterations: int = None) -> List[Tuple[float, float]]:
        """
        Implements Eades spring embedder algorithm.
        
        Force calculations:
        - Repulsive force: F_rep = k_rep / d^2 (Coulomb-like)
        - Attractive force: F_att = k_att * log(d / l_0) (spring-like)
        """
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        vertices = self.graph.get_all_vertices()
        n = len(vertices)
        
        # Initialize forces
        forces = {v.id: [0.0, 0.0] for v in vertices}
        
        for iteration in range(max_iterations):
            # Reset forces
            for v_id in forces:
                forces[v_id] = [0.0, 0.0]
            
            # Calculate repulsive forces between all pairs
            for i, v1 in enumerate(vertices):
                for j, v2 in enumerate(vertices):
                    if i >= j:
                        continue
                    
                    dx = v2.x - v1.x
                    dy = v2.y - v1.y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance < 1e-10:  # Avoid division by zero
                        distance = 1e-10
                    
                    # Repulsive force (Coulomb-like)
                    force_magnitude = self.repulsion_constant / (distance * distance)
                    
                    # Apply repulsive force
                    fx = force_magnitude * dx / distance
                    fy = force_magnitude * dy / distance
                    
                    forces[v1.id][0] -= fx
                    forces[v1.id][1] -= fy
                    forces[v2.id][0] += fx
                    forces[v2.id][1] += fy
            
            # Calculate attractive forces for connected vertices
            for v1_id, v2_id in self.graph.get_all_edges():
                v1 = self.graph.get_vertex(v1_id)
                v2 = self.graph.get_vertex(v2_id)
                
                dx = v2.x - v1.x
                dy = v2.y - v1.y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 1e-10:
                    distance = 1e-10
                
                # Attractive force (spring-like)
                force_magnitude = self.attraction_constant * math.log(distance / self.ideal_edge_length)
                
                # Apply attractive force
                fx = force_magnitude * dx / distance
                fy = force_magnitude * dy / distance
                
                forces[v1_id][0] += fx
                forces[v1_id][1] += fy
                forces[v2_id][0] -= fx
                forces[v2_id][1] -= fy
            
            # Update positions
            max_displacement = 0.0
            for vertex in vertices:
                fx, fy = forces[vertex.id]
                
                # Apply damping
                fx *= self.damping
                fy *= self.damping
                
                # Update position
                vertex.x += fx
                vertex.y += fy
                
                # Track maximum displacement
                displacement = math.sqrt(fx*fx + fy*fy)
                max_displacement = max(max_displacement, displacement)
            
            # Check convergence
            if max_displacement < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return [(v.x, v.y) for v in vertices]
    
    def fruchterman_reingold(self, max_iterations: int = None) -> List[Tuple[float, float]]:
        """
        Implements Fruchterman-Reingold algorithm.
        
        Force calculations:
        - Repulsive force: F_rep = k^2 / d (simplified)
        - Attractive force: F_att = d^2 / k (spring-like)
        """
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        vertices = self.graph.get_all_vertices()
        n = len(vertices)
        
        # Calculate optimal distance
        area = 1000.0  # Area of the drawing
        k = math.sqrt(area / n)  # Optimal distance between vertices
        
        # Initialize forces
        forces = {v.id: [0.0, 0.0] for v in vertices}
        
        for iteration in range(max_iterations):
            # Calculate temperature (cooling schedule)
            temperature = max(0.1, 1.0 - iteration / max_iterations)
            
            # Reset forces
            for v_id in forces:
                forces[v_id] = [0.0, 0.0]
            
            # Calculate repulsive forces between all pairs
            for i, v1 in enumerate(vertices):
                for j, v2 in enumerate(vertices):
                    if i >= j:
                        continue
                    
                    dx = v2.x - v1.x
                    dy = v2.y - v1.y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance < 1e-10:
                        distance = 1e-10
                    
                    # Repulsive force
                    force_magnitude = k * k / distance
                    
                    # Apply repulsive force
                    fx = force_magnitude * dx / distance
                    fy = force_magnitude * dy / distance
                    
                    forces[v1.id][0] -= fx
                    forces[v1.id][1] -= fy
                    forces[v2.id][0] += fx
                    forces[v2.id][1] += fy
            
            # Calculate attractive forces for connected vertices
            for v1_id, v2_id in self.graph.get_all_edges():
                v1 = self.graph.get_vertex(v1_id)
                v2 = self.graph.get_vertex(v2_id)
                
                dx = v2.x - v1.x
                dy = v2.y - v1.y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 1e-10:
                    distance = 1e-10
                
                # Attractive force
                force_magnitude = distance * distance / k
                
                # Apply attractive force
                fx = force_magnitude * dx / distance
                fy = force_magnitude * dy / distance
                
                forces[v1_id][0] += fx
                forces[v1_id][1] += fy
                forces[v2_id][0] -= fx
                forces[v2_id][1] -= fy
            
            # Update positions with temperature-based movement
            max_displacement = 0.0
            for vertex in vertices:
                fx, fy = forces[vertex.id]
                
                # Limit movement by temperature
                displacement = math.sqrt(fx*fx + fy*fy)
                if displacement > temperature:
                    fx = fx * temperature / displacement
                    fy = fy * temperature / displacement
                
                # Update position
                vertex.x += fx
                vertex.y += fy
                
                # Keep vertices within bounds
                vertex.x = max(-50, min(50, vertex.x))
                vertex.y = max(-50, min(50, vertex.y))
                
                max_displacement = max(max_displacement, displacement)
            
            # Check convergence
            if max_displacement < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return [(v.x, v.y) for v in vertices]

def main():
    import sys
    import os
    
    print("Force-Directed Visualiser (Eades & Fruchterman-Reingold)")
    
    # Check if a JSON file was provided as argument
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
        
        if not os.path.exists(json_file_path):
            print(f"Error: File '{json_file_path}' not found")
            print("Usage: python main.py [path_to_graph.json]")
            print("If no file is provided, a sample graph will be used.")
            return
        
        if not json_file_path.endswith('.json'):
            print("Warning: File doesn't have .json extension")
        
        # Load graph from JSON file
        print(f"Loading graph from: {json_file_path}")
        try:
            from graph_utils import parse_graph_from_file
            graph = parse_graph_from_file(json_file_path)
            print(f"Successfully loaded graph with {len(graph.vertices)} vertices and {len(graph.edges)} edges")
        except Exception as e:
            print(f"Error loading graph: {e}")
            return
    else:
        # Create a sample graph
        graph = create_sample_graph()
        print(f"Created sample graph with {len(graph.vertices)} vertices and {len(graph.edges)} edges")
    
    # Randomize initial positions
    graph.randomize_positions(width=10, height=10)
    
    # Show initial layout
    print("Initial layout:")
    visualize_graph(graph, "Initial Graph Layout")
    
    # Apply Eades spring embedder
    print("\nApplying Eades spring embedder...")
    layout = ForceDirectedLayout(graph)
    positions = layout.eades_spring_embedder(max_iterations=100)
    
    # Show final layout
    print("Final layout (Eades):")
    visualize_graph(graph, "Graph Layout - Eades Spring Embedder")
    
    # Reset and apply Fruchterman-Reingold
    print("\nApplying Fruchterman-Reingold...")
    graph.randomize_positions(width=10, height=10)
    positions = layout.fruchterman_reingold(max_iterations=100)
    
    # Show final layout
    print("Final layout (Fruchterman-Reingold):")
    visualize_graph(graph, "Graph Layout - Fruchterman-Reingold")

if __name__ == "__main__":
    main()
