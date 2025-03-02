import base64

import networkx as nx
import panel as pn
from langgraph.graph import StateGraph
from pyvis.network import Network

from assistant.inf_graph_todo import RouteListener

# Define colors
DEFAULT_NODE_COLOR = 'lightblue'
TARGET_NODE_COLOR = 'red'
SOURCE_NODE_COLOR = 'green'


class GraphVisualizer(pn.pane.HTML):
    def __init__(self, graph: StateGraph, **params):
        """Initialize with a LangGraph instance and set up the PyVis network."""
        super().__init__('', sizing_mode='stretch_both', **params)  # Start with empty HTML

        self.graph = graph
        self.nx_G = nx.DiGraph()
        self.pyvis_network = Network(height='800px', width='100%', directed=True)
        self.node_colors: dict[str, str] = dict()
        self.layout_positions: dict[str, dict[str, float]] = dict()  # Store node positions

        # Build the graph from LangGraph edges
        self._build_graph()
        self._update_html()

    def _build_graph(self) -> None:
        """Constructs the initial NetworkX graph from LangGraph."""
        edges = self.graph.get_graph(xray=1).edges
        for edge in edges:
            self.nx_G.add_edge(edge[0], edge[1])

        # Initialize all nodes with the default color
        for node in self.nx_G.nodes:
            self.node_colors[node] = DEFAULT_NODE_COLOR

        # Assign node positions to `nx_G`
        pos = nx.spring_layout(self.nx_G)
        self.pyvis_network.from_nx(self.nx_G)

        # Store node positions manually
        if self.layout_positions is None:
            self.layout_positions = {
                node: {
                    'x': pos[node][0] * 1000,
                    'y': pos[node][1] * 1000
                }
                for node in self.nx_G.nodes
            }

        # Assign stored positions to PyVis
        for node, position in self.layout_positions.items():
            self.pyvis_network.get_node(node)['x'] = position['x']
            self.pyvis_network.get_node(node)['y'] = position['y']

        # Disable zoom but keep drag enabled
        self.pyvis_network.options.physics.enabled = True
        self.pyvis_network.options.interaction.zoomView = False
        self.pyvis_network.options.interaction.dragNodes = False

        self._apply_colors()

    def _apply_colors(self):
        """Apply colors to nodes in PyVis network."""
        for node in self.nx_G.nodes:
            self.pyvis_network.get_node(node)['color'] = self.node_colors[node]

    def _update_html(self):
        """Generate and update the HTML content in the Panel app."""
        net_html = self.pyvis_network.generate_html()
        net_html_b64 = base64.b64encode(net_html.encode()).decode()

        # Assign a new iframe, ensuring it updates correctly
        self.object = f"""
        <iframe 
            src="data:text/html;base64,{net_html_b64}"
            style="border:none; height: 100vh; width: 100%; display: block;">
        </iframe>
        """

    def update_node_color(self, source_node: str, target_node: str) -> None:
        """Change the color of source_node, target_node, and the specific edge between them dynamically."""

        # Update node colors
        for node in self.nx_G.nodes:
            if node == source_node:
                node_color = SOURCE_NODE_COLOR
            elif node == target_node:
                node_color = TARGET_NODE_COLOR
            else:
                node_color = DEFAULT_NODE_COLOR
            self.node_colors[node] = node_color
        self._apply_colors()

        # Update edge colors
        for edge in self.pyvis_network.get_edges():
            if edge['from'] == source_node and edge['to'] == target_node:
                edge['color'] = SOURCE_NODE_COLOR
            else:
                edge['color'] = DEFAULT_NODE_COLOR

        # Update the visualization
        self._update_html()


class NodeColorizer(RouteListener):
    def __init__(self, graph_visualizer: GraphVisualizer, **params):
        self.graph_visualizer = graph_visualizer

    def update(self, current_node: str = None, next_node: str = None) -> None:
        self.graph_visualizer.update_node_color(source_node=current_node, target_node=next_node)
