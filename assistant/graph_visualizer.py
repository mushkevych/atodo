import base64
import networkx as nx
import panel as pn
from pyvis.network import Network
from langgraph.graph import StateGraph

from assistant.inf_graph_todo import RouteListener

# Define colors
DEFAULT_COLOR = 'lightblue'
SELECTED_COLOR = 'red'

class GraphVisualizer(pn.pane.HTML):
    def __init__(self, graph: StateGraph, **params):
        """Initialize with a LangGraph instance and set up the PyVis network."""
        super().__init__('', sizing_mode='stretch_both', **params)  # Start with empty HTML

        self.graph = graph
        self.G = nx.DiGraph()
        self.nt = Network(height='800px', width='100%', directed=True)
        self.node_colors = {}
        self.layout_positions = None  # Store node positions

        # Build the graph from LangGraph edges
        self._build_graph()
        self._update_html()

    def _build_graph(self) -> None:
        """Constructs the initial NetworkX graph from LangGraph."""
        edges = self.graph.get_graph(xray=1).edges
        for edge in edges:
            self.G.add_edge(edge[0], edge[1])

        # Initialize all nodes with the default color
        for node in self.G.nodes:
            self.node_colors[node] = DEFAULT_COLOR

        # Compute positions manually to avoid KeyError
        pos = nx.spring_layout(self.G)  # Assign initial positions

        self.nt.from_nx(self.G)

        # Store node positions manually
        if self.layout_positions is None:
            self.layout_positions = {
                node: {
                    'x': pos[node][0] * 1000,
                    'y': pos[node][1] * 1000
                }
                for node in self.G.nodes
            }

        # Assign stored positions to PyVis
        for node, position in self.layout_positions.items():
            self.nt.get_node(node)['x'] = position['x']
            self.nt.get_node(node)['y'] = position['y']

        # Disable zoom but keep drag enabled
        self.nt.options.physics.enabled = True
        self.nt.options.interaction.zoomView = False
        self.nt.options.interaction.dragNodes = False

        self._apply_colors()

    def _apply_colors(self):
        """Apply colors to nodes in PyVis network."""
        for node in self.G.nodes:
            self.nt.get_node(node)['color'] = self.node_colors[node]

    def _update_html(self):
        """Generate and update the HTML content in the Panel app."""
        net_html = self.nt.generate_html()
        net_html_b64 = base64.b64encode(net_html.encode()).decode()

        # Assign a new iframe, ensuring it updates correctly
        self.object = f"""
        <iframe 
            src="data:text/html;base64,{net_html_b64}"
            style="border:none; height: 100vh; width: 100%; display: block;">
        </iframe>
        """

    def clear_graph(self):
        """Removes the iframe to prevent layout issues."""
        self.object = None  # Clears the iframe to free up the UI

    def restore_graph(self):
        """Restores the iframe by regenerating the HTML."""
        self._update_html()  # Calls the update function to regenerate the iframe

    def update_node_color(self, selected_node: str) -> None:
        """Change the color of a node dynamically without reloading layout."""
        # Reset all nodes to default
        for node in self.G.nodes:
            self.node_colors[node] = DEFAULT_COLOR

        # Highlight the selected node
        if selected_node in self.node_colors:
            self.node_colors[selected_node] = SELECTED_COLOR

        self._apply_colors()
        self._update_html()


class NodeColorizer(RouteListener):
    def __init__(self, graph_visualizer: GraphVisualizer, **params):
        self.graph_visualizer = graph_visualizer

    def update(self, next_node: str) -> None:
        self.graph_visualizer.update_node_color(next_node)
