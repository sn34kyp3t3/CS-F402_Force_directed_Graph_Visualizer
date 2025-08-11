import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from typing import Optional, Tuple, List
from graph_utils import Graph, Vertex, save_graph_to_file

class GraphEditorGUI:
    """Interactive GUI for creating and editing graphs."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Graph Editor - Computational Geometry")
        self.root.geometry("800x600")
        
        # Graph data
        self.graph = Graph()
        self.next_vertex_id = 0
        self.selected_vertex = None
        self.drawing_edge = False
        self.edge_start = None
        
        # Grid mode settings
        self.grid_mode = tk.BooleanVar(value=True)  # True = grid only, False = continuous
        self.grid_size = 20  # Grid cell size
        
        # Canvas for graph display
        self.canvas = tk.Canvas(root, bg='white', width=600, height=500)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Control panel
        self.create_control_panel()
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Draw grid
        self.draw_grid()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Click to add vertices, drag to create edges")
        status_bar = tk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_control_panel(self):
        """Create the control panel with buttons and options."""
        control_frame = tk.Frame(self.root, width=200)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Title
        title_label = tk.Label(control_frame, text="Graph Editor", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Graph info
        info_frame = tk.LabelFrame(control_frame, text="Graph Info", padx=5, pady=5)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.vertex_count_var = tk.StringVar(value="Vertices: 0")
        self.edge_count_var = tk.StringVar(value="Edges: 0")
        
        tk.Label(info_frame, textvariable=self.vertex_count_var).pack()
        tk.Label(info_frame, textvariable=self.edge_count_var).pack()
        
        # Actions
        actions_frame = tk.LabelFrame(control_frame, text="Actions", padx=5, pady=5)
        actions_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(actions_frame, text="Clear Graph", command=self.clear_graph).pack(fill=tk.X, pady=2)
        tk.Button(actions_frame, text="Randomize Positions", command=self.randomize_positions).pack(fill=tk.X, pady=2)
        tk.Button(actions_frame, text="Delete Selected", command=self.delete_selected).pack(fill=tk.X, pady=2)
        
        # File operations
        file_frame = tk.LabelFrame(control_frame, text="File Operations", padx=5, pady=5)
        file_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(file_frame, text="Load Graph", command=self.load_graph).pack(fill=tk.X, pady=2)
        tk.Button(file_frame, text="Save Graph", command=self.save_graph).pack(fill=tk.X, pady=2)
        tk.Button(file_frame, text="Export for Algorithms", command=self.export_graph).pack(fill=tk.X, pady=2)
        
        # Grid mode toggle
        grid_frame = tk.LabelFrame(control_frame, text="Placement Mode", padx=5, pady=5)
        grid_frame.pack(fill=tk.X, pady=5)
        
        self.grid_toggle = tk.Checkbutton(
            grid_frame, 
            text="Grid Only", 
            variable=self.grid_mode,
            command=self.update_grid_mode
        )
        self.grid_toggle.pack(fill=tk.X, pady=2)
        
        self.grid_label = tk.Label(grid_frame, text="Mode: Grid Only (discrete)")
        self.grid_label.pack(pady=2)
        
        # Algorithm buttons
        algo_frame = tk.LabelFrame(control_frame, text="Run Algorithms", padx=5, pady=5)
        algo_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(algo_frame, text="Force-Directed", command=self.run_force_directed).pack(fill=tk.X, pady=2)
        tk.Button(algo_frame, text="Orthogonal Layout", command=self.run_orthogonal).pack(fill=tk.X, pady=2)
        
        # Instructions
        instructions_frame = tk.LabelFrame(control_frame, text="Instructions", padx=5, pady=5)
        instructions_frame.pack(fill=tk.X, pady=5)
        
        instructions = """
• Click empty space to add vertices
• Click and drag from vertex to create edges
• Click vertex to select it
• Use buttons to perform operations
        """
        tk.Label(instructions_frame, text=instructions, justify=tk.LEFT).pack()
    
    def draw_grid(self):
        """Draw a grid on the canvas."""
        for i in range(0, 600, 20):
            self.canvas.create_line(i, 0, i, 500, fill='lightgray', width=1)
        for i in range(0, 500, 20):
            self.canvas.create_line(0, i, 600, i, fill='lightgray', width=1)
    
    def update_info(self):
        """Update the graph information display."""
        self.vertex_count_var.set(f"Vertices: {len(self.graph.vertices)}")
        self.edge_count_var.set(f"Edges: {len(self.graph.edges)}")
    
    def clear_canvas(self):
        """Clear the canvas (except grid)."""
        self.canvas.delete("vertex", "edge", "label")
    
    def redraw_graph(self):
        """Redraw the entire graph on the canvas."""
        self.clear_canvas()
        
        # Draw edges
        for v1_id, v2_id in self.graph.edges:
            v1 = self.graph.get_vertex(v1_id)
            v2 = self.graph.get_vertex(v2_id)
            x1, y1 = self.world_to_canvas(v1.x, v1.y)
            x2, y2 = self.world_to_canvas(v2.x, v2.y)
            
            # Highlight edge if connected to selected vertex
            color = 'blue'
            width = 2
            if self.selected_vertex and (v1_id == self.selected_vertex or v2_id == self.selected_vertex):
                color = 'red'
                width = 3
            
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, tags="edge")
        
        # Draw vertices
        for vertex in self.graph.get_all_vertices():
            x, y = self.world_to_canvas(vertex.x, vertex.y)
            
            # Highlight selected vertex
            color = 'red' if vertex.id == self.selected_vertex else 'green'
            size = 8 if vertex.id == self.selected_vertex else 6
            
            self.canvas.create_oval(x-size, y-size, x+size, y+size, 
                                  fill=color, outline='black', width=2, tags="vertex")
            self.canvas.create_text(x, y-15, text=str(vertex.id), 
                                  font=("Arial", 10, "bold"), tags="label")
    
    def world_to_canvas(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to canvas coordinates."""
        # Scale and center the graph
        scale = 50  # pixels per unit
        cx, cy = 300, 250  # canvas center
        
        canvas_x = int(cx + x * scale)
        canvas_y = int(cy - y * scale)  # Flip Y axis
        
        return canvas_x, canvas_y
    
    def canvas_to_world(self, canvas_x: int, canvas_y: int) -> Tuple[float, float]:
        """Convert canvas coordinates to world coordinates."""
        scale = 50
        cx, cy = 300, 250
        
        x = (canvas_x - cx) / scale
        y = -(canvas_y - cy) / scale  # Flip Y axis
        
        return x, y
    
    def find_vertex_at(self, canvas_x: int, canvas_y: int) -> Optional[int]:
        """Find vertex at given canvas coordinates."""
        for vertex in self.graph.get_all_vertices():
            vx, vy = self.world_to_canvas(vertex.x, vertex.y)
            distance = ((canvas_x - vx) ** 2 + (canvas_y - vy) ** 2) ** 0.5
            if distance <= 10:  # Click radius
                return vertex.id
        return None
    
    def update_grid_mode(self):
        """Update the grid mode label and behavior."""
        if self.grid_mode.get():
            self.grid_label.config(text="Mode: Grid Only (discrete)")
            self.status_var.set("Grid mode: Only grid intersections allowed for vertices.")
        else:
            self.grid_label.config(text="Mode: Continuous (any point)")
            self.status_var.set("Continuous mode: Vertices can be placed anywhere.")

    def snap_to_grid(self, x, y):
        """Snap coordinates to the nearest grid intersection."""
        gx = round(x / self.grid_size) * self.grid_size
        gy = round(y / self.grid_size) * self.grid_size
        return gx, gy

    def on_canvas_click(self, event):
        """Handle canvas click for adding/selecting vertices or completing edges."""
        x, y = event.x, event.y
        if self.grid_mode.get():
            x, y = self.snap_to_grid(x, y)
        world_x, world_y = self.canvas_to_world(x, y)
        v_id = self.find_vertex_at(x, y)
        if self.drawing_edge:
            # Complete edge
            if v_id is not None and v_id != self.edge_start:
                self.graph.add_edge(self.edge_start, v_id)
            self.drawing_edge = False
            self.edge_start = None
            self.status_var.set("Edge created.")
        elif v_id is not None:
            # Select vertex
            self.selected_vertex = v_id
            self.status_var.set(f"Selected vertex {v_id}")
        else:
            # Add vertex
            self.graph.add_vertex(self.next_vertex_id, world_x, world_y)
            self.selected_vertex = self.next_vertex_id
            self.next_vertex_id += 1
            self.status_var.set(f"Added vertex {self.selected_vertex}")
        self.redraw_graph()
        self.update_info()
    
    def on_canvas_motion(self, event):
        """Handle canvas motion events."""
        if self.drawing_edge and self.edge_start is not None:
            # Draw temporary edge line
            self.canvas.delete("temp_edge")
            start_vertex = self.graph.get_vertex(self.edge_start)
            start_x, start_y = self.world_to_canvas(start_vertex.x, start_vertex.y)
            self.canvas.create_line(start_x, start_y, event.x, event.y, 
                                  fill='orange', width=2, dash=(5, 5), tags="temp_edge")
    
    def on_canvas_release(self, event):
        """Handle canvas release events."""
        if self.drawing_edge:
            # Start edge drawing
            vertex_id = self.find_vertex_at(event.x, event.y)
            if vertex_id is not None:
                self.edge_start = vertex_id
                self.drawing_edge = True
                self.status_var.set(f"Drag from vertex {vertex_id} to create edge")
    
    def clear_graph(self):
        """Clear the entire graph."""
        if messagebox.askyesno("Clear Graph", "Are you sure you want to clear the graph?"):
            self.graph = Graph()
            self.next_vertex_id = 0
            self.selected_vertex = None
            self.drawing_edge = False
            self.edge_start = None
            self.redraw_graph()
            self.update_info()
            self.status_var.set("Graph cleared")
    
    def randomize_positions(self):
        """Randomize vertex positions."""
        self.graph.randomize_positions(width=8, height=6)
        self.redraw_graph()
        self.status_var.set("Positions randomized")
    
    def delete_selected(self):
        """Delete the selected vertex."""
        if self.selected_vertex is not None:
            # Remove edges connected to this vertex
            edges_to_remove = []
            for v1, v2 in self.graph.edges:
                if v1 == self.selected_vertex or v2 == self.selected_vertex:
                    edges_to_remove.append((v1, v2))
            
            for edge in edges_to_remove:
                self.graph.edges.remove(edge)
            
            # Remove vertex
            del self.graph.vertices[self.selected_vertex]
            del self.graph.adjacency_list[self.selected_vertex]
            
            self.selected_vertex = None
            self.redraw_graph()
            self.update_info()
            self.status_var.set("Vertex deleted")
    
    def load_graph(self):
        """Load graph from file."""
        filename = filedialog.askopenfilename(
            title="Load Graph",
            filetypes=[
                ("JSON files", "*.json"),
                ("Text files", "*.txt"),
                ("Edge files", "*.edges"),
                ("Matrix files", "*.matrix"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                from graph_utils import parse_graph_from_file
                self.graph = parse_graph_from_file(filename)
                self.next_vertex_id = max(self.graph.vertices.keys()) + 1 if self.graph.vertices else 0
                self.redraw_graph()
                self.update_info()
                self.status_var.set(f"Loaded graph from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load graph: {str(e)}")
    
    def save_graph(self):
        """Save graph to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Graph",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("Text files", "*.txt"),
                ("Edge files", "*.edges"),
                ("Matrix files", "*.matrix")
            ]
        )
        
        if filename:
            try:
                save_graph_to_file(self.graph, filename)
                self.status_var.set(f"Saved graph to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save graph: {str(e)}")
    
    def export_graph(self):
        """Export graph for algorithm processing."""
        if not self.graph.vertices:
            messagebox.showwarning("Warning", "No graph to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Graph for Algorithms",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if filename:
            try:
                save_graph_to_file(self.graph, filename, 'json')
                self.status_var.set(f"Exported graph to {filename}")
                messagebox.showinfo("Success", f"Graph exported to {filename}\nYou can now use this file with the algorithms.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export graph: {str(e)}")
    
    def run_force_directed(self):
        """Run force-directed algorithm on the current graph."""
        if not self.graph.vertices:
            messagebox.showwarning("Warning", "No graph to process!")
            return
        
        try:
            # Import and run force-directed algorithm
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'force_directed'))
            from force_directed.main import ForceDirectedLayout
            
            # Create a copy of the graph
            graph_copy = Graph()
            for vertex in self.graph.get_all_vertices():
                graph_copy.add_vertex(vertex.id, vertex.x, vertex.y)
            for v1, v2 in self.graph.get_all_edges():
                graph_copy.add_edge(v1, v2)
            
            # Run algorithm
            layout = ForceDirectedLayout(graph_copy)
            layout.eades_spring_embedder(max_iterations=50)
            
            # Update our graph with the results
            for vertex in graph_copy.get_all_vertices():
                self.graph.get_vertex(vertex.id).x = vertex.x
                self.graph.get_vertex(vertex.id).y = vertex.y
            
            self.redraw_graph()
            self.status_var.set("Force-directed layout applied")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run force-directed algorithm: {str(e)}")
    
    def run_orthogonal(self):
        """Run orthogonal layout algorithm on the current graph."""
        if not self.graph.vertices:
            messagebox.showwarning("Warning", "No graph to process!")
            return
        
        try:
            # Import and run orthogonal algorithm
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'orthogonal_bend_min'))
            from orthogonal_bend_min.main import OrthogonalLayout
            
            # Create a copy of the graph
            graph_copy = Graph()
            for vertex in self.graph.get_all_vertices():
                graph_copy.add_vertex(vertex.id, vertex.x, vertex.y)
            for v1, v2 in self.graph.get_all_edges():
                graph_copy.add_edge(v1, v2)
            
            # Run algorithm
            layout = OrthogonalLayout(graph_copy)
            coordinates = layout.optimize_layout("greedy")
            
            # Update our graph with the results
            for vertex_id, (x, y) in coordinates.items():
                if vertex_id in self.graph.vertices:
                    self.graph.get_vertex(vertex_id).x = x
                    self.graph.get_vertex(vertex_id).y = y
            
            self.redraw_graph()
            self.status_var.set("Orthogonal layout applied")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run orthogonal algorithm: {str(e)}")

def main():
    """Launch the graph editor GUI."""
    root = tk.Tk()
    app = GraphEditorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 