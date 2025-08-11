import math
import random
import json
import re
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Vertex:
    """Represents a vertex in the graph with position and metadata."""
    id: int
    x: float = 0.0
    y: float = 0.0
    
    def distance_to(self, other: 'Vertex') -> float:
        """Calculate Euclidean distance to another vertex."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __hash__(self):
        return hash(self.id)

class Graph:
    """Represents a graph with vertices and edges."""
    
    def __init__(self):
        self.vertices: Dict[int, Vertex] = {}
        self.edges: List[Tuple[int, int]] = []
        self.adjacency_list: Dict[int, Set[int]] = {}
    
    def add_vertex(self, vertex_id: int, x: float = None, y: float = None) -> Vertex:
        """Add a vertex to the graph."""
        if x is None:
            x = random.uniform(-1, 1)
        if y is None:
            y = random.uniform(-1, 1)
        
        vertex = Vertex(vertex_id, x, y)
        self.vertices[vertex_id] = vertex
        self.adjacency_list[vertex_id] = set()
        return vertex
    
    def add_edge(self, vertex1_id: int, vertex2_id: int):
        """Add an edge between two vertices."""
        if vertex1_id not in self.vertices or vertex2_id not in self.vertices:
            raise ValueError(f"Vertices {vertex1_id} and {vertex2_id} must exist")
        
        self.edges.append((vertex1_id, vertex2_id))
        self.adjacency_list[vertex1_id].add(vertex2_id)
        self.adjacency_list[vertex2_id].add(vertex1_id)
    
    def get_vertex(self, vertex_id: int) -> Vertex:
        """Get a vertex by ID."""
        return self.vertices[vertex_id]
    
    def get_neighbors(self, vertex_id: int) -> Set[int]:
        """Get neighbors of a vertex."""
        return self.adjacency_list.get(vertex_id, set())
    
    def get_all_vertices(self) -> List[Vertex]:
        """Get all vertices in the graph."""
        return list(self.vertices.values())
    
    def get_all_edges(self) -> List[Tuple[int, int]]:
        """Get all edges in the graph."""
        return self.edges.copy()
    
    def randomize_positions(self, width: float = 10.0, height: float = 10.0):
        """Randomize vertex positions within given bounds."""
        for vertex in self.vertices.values():
            vertex.x = random.uniform(-width/2, width/2)
            vertex.y = random.uniform(-height/2, height/2)

def create_sample_graph() -> Graph:
    """Create a sample graph for testing."""
    graph = Graph()
    
    # Add vertices
    for i in range(6):
        graph.add_vertex(i)
    
    # Add edges to create a simple connected graph
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4), (2, 5)]
    for v1, v2 in edges:
        graph.add_edge(v1, v2)
    
    return graph

# Graph parsing functions for formal input
def parse_adjacency_matrix(matrix_str: str) -> Graph:
    """Parse graph from adjacency matrix string."""
    lines = matrix_str.strip().split('\n')
    n = len(lines)
    
    graph = Graph()
    
    # Add vertices
    for i in range(n):
        graph.add_vertex(i)
    
    # Parse adjacency matrix
    for i, line in enumerate(lines):
        values = line.strip().split()
        if len(values) != n:
            raise ValueError(f"Invalid adjacency matrix: row {i} has {len(values)} values, expected {n}")
        
        for j, value in enumerate(values):
            if value == '1':
                if i != j:  # Avoid self-loops
                    graph.add_edge(i, j)
    
    return graph

def parse_edge_list(edge_list_str: str) -> Graph:
    """Parse graph from edge list string."""
    lines = edge_list_str.strip().split('\n')
    
    graph = Graph()
    vertices = set()
    
    # First pass: collect all vertices
    for line in lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 2:
                v1, v2 = int(parts[0]), int(parts[1])
                vertices.add(v1)
                vertices.add(v2)
    
    # Add vertices
    for vertex_id in sorted(vertices):
        graph.add_vertex(vertex_id)
    
    # Second pass: add edges
    for line in lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 2:
                v1, v2 = int(parts[0]), int(parts[1])
                graph.add_edge(v1, v2)
    
    return graph

def parse_adjacency_list(adj_list_str: str) -> Graph:
    """Parse graph from adjacency list string."""
    lines = adj_list_str.strip().split('\n')
    
    graph = Graph()
    vertices = set()
    
    # First pass: collect all vertices
    for line in lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 1:
                vertex_id = int(parts[0])
                vertices.add(vertex_id)
                # Add neighbors to vertices set
                for neighbor in parts[1:]:
                    vertices.add(int(neighbor))
    
    # Add vertices
    for vertex_id in sorted(vertices):
        graph.add_vertex(vertex_id)
    
    # Second pass: add edges
    for line in lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 2:
                vertex_id = int(parts[0])
                for neighbor in parts[1:]:
                    neighbor_id = int(neighbor)
                    graph.add_edge(vertex_id, neighbor_id)
    
    return graph

def parse_json_graph(json_str: str) -> Graph:
    """Parse graph from JSON format."""
    data = json.loads(json_str)
    
    graph = Graph()
    
    # Add vertices
    if 'vertices' in data:
        for vertex_data in data['vertices']:
            vertex_id = vertex_data.get('id', vertex_data.get('vertex', 0))
            x = vertex_data.get('x', None)
            y = vertex_data.get('y', None)
            graph.add_vertex(vertex_id, x, y)
    elif 'nodes' in data:
        for node_data in data['nodes']:
            node_id = node_data.get('id', node_data.get('node', 0))
            x = node_data.get('x', None)
            y = node_data.get('y', None)
            graph.add_vertex(node_id, x, y)
    
    # Add edges
    if 'edges' in data:
        for edge_data in data['edges']:
            if isinstance(edge_data, list) and len(edge_data) >= 2:
                v1, v2 = edge_data[0], edge_data[1]
            elif isinstance(edge_data, dict):
                v1 = edge_data.get('from', edge_data.get('source', 0))
                v2 = edge_data.get('to', edge_data.get('target', 0))
            else:
                continue
            graph.add_edge(v1, v2)
    
    return graph

def parse_graph_from_file(filename: str) -> Graph:
    """Parse graph from file based on file extension and content."""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Try to detect format based on file extension and content
    if filename.endswith('.json'):
        return parse_json_graph(content)
    elif filename.endswith('.txt') or filename.endswith('.edges'):
        # Try edge list format first
        try:
            return parse_edge_list(content)
        except:
            # Try adjacency list format
            return parse_adjacency_list(content)
    elif filename.endswith('.matrix') or filename.endswith('.adj'):
        # For .adj files, try adjacency list first, then matrix
        if filename.endswith('.adj'):
            try:
                return parse_adjacency_list(content)
            except:
                return parse_adjacency_matrix(content)
        else:
            return parse_adjacency_matrix(content)
    else:
        # Try to auto-detect format
        lines = content.strip().split('\n')
        if not lines:
            raise ValueError("Empty file")
        
        # Check if it looks like JSON
        if content.strip().startswith('{'):
            return parse_json_graph(content)
        
        # Check if it looks like adjacency matrix (square)
        first_line = lines[0].strip().split()
        if len(lines) == len(first_line):
            # Additional check: if all lines have same number of values as number of lines
            try:
                for line in lines:
                    if len(line.strip().split()) != len(lines):
                        raise ValueError("Not a square matrix")
                return parse_adjacency_matrix(content)
            except:
                # If not a valid square matrix, try adjacency list
                return parse_adjacency_list(content)
        
        # Check if it looks like edge list (pairs of numbers)
        try:
            for line in lines[:3]:  # Check first few lines
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 2:
                        int(parts[0]), int(parts[1])
            return parse_edge_list(content)
        except:
            # Default to adjacency list
            return parse_adjacency_list(content)

def save_graph_to_file(graph: Graph, filename: str, format: str = 'auto'):
    """Save graph to file in specified format."""
    if format == 'auto':
        if filename.endswith('.json'):
            format = 'json'
        elif filename.endswith('.txt') or filename.endswith('.edges'):
            format = 'edge_list'
        elif filename.endswith('.matrix') or filename.endswith('.adj'):
            format = 'adjacency_matrix'
        else:
            format = 'edge_list'  # Default
    
    with open(filename, 'w') as f:
        if format == 'json':
            data = {
                'vertices': [{'id': v.id, 'x': v.x, 'y': v.y} for v in graph.get_all_vertices()],
                'edges': list(graph.get_all_edges())
            }
            json.dump(data, f, indent=2)
        
        elif format == 'edge_list':
            for v1, v2 in graph.get_all_edges():
                f.write(f"{v1} {v2}\n")
        
        elif format == 'adjacency_list':
            for vertex in graph.get_all_vertices():
                neighbors = list(graph.get_neighbors(vertex.id))
                f.write(f"{vertex.id} {' '.join(map(str, neighbors))}\n")
        
        elif format == 'adjacency_matrix':
            vertices = graph.get_all_vertices()
            n = len(vertices)
            matrix = [[0] * n for _ in range(n)]
            
            for v1, v2 in graph.get_all_edges():
                matrix[v1][v2] = 1
                matrix[v2][v1] = 1  # Undirected graph
            
            for row in matrix:
                f.write(' '.join(map(str, row)) + '\n')

def visualize_graph(graph: Graph, title: str = "Graph Visualization", 
                   vertex_size: int = 300, edge_width: float = 1.0):
    """Visualize the graph using matplotlib."""
    plt.figure(figsize=(10, 8))
    
    # Draw edges
    for v1_id, v2_id in graph.edges:
        v1 = graph.get_vertex(v1_id)
        v2 = graph.get_vertex(v2_id)
        plt.plot([v1.x, v2.x], [v1.y, v2.y], 'b-', linewidth=edge_width, alpha=0.6)
    
    # Draw vertices
    x_coords = [v.x for v in graph.get_all_vertices()]
    y_coords = [v.y for v in graph.get_all_vertices()]
    plt.scatter(x_coords, y_coords, c='red', s=vertex_size, zorder=5)
    
    # Add vertex labels
    for vertex in graph.get_all_vertices():
        plt.annotate(str(vertex.id), (vertex.x, vertex.y), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

def save_graph_positions(graph: Graph, filename: str):
    """Save graph positions to a file."""
    with open(filename, 'w') as f:
        f.write("# Graph positions\n")
        f.write("# vertex_id x y\n")
        for vertex in graph.get_all_vertices():
            f.write(f"{vertex.id} {vertex.x:.6f} {vertex.y:.6f}\n")

def load_graph_positions(graph: Graph, filename: str):
    """Load graph positions from a file."""
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) == 3:
                vertex_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                if vertex_id in graph.vertices:
                    graph.vertices[vertex_id].x = x
                    graph.vertices[vertex_id].y = y
