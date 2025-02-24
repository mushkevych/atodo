import base64
from typing import Any

import networkx as nx
import panel as pn
from langchain_core.messages import HumanMessage, convert_to_messages
from langchain_openai import OpenAI
from langgraph.graph import StateGraph
from pyvis.network import Network

from assistant.inf_graph_todo import graph as graph_todo
from assistant.models import UserProfile, ToDo, UpdateMemory
from utils.fs_utils import load_api_key

PAGE_NAME_CHAT = 'Chat'
PAGE_NAME_DETAILS = 'Details'

SAMPLE_JSON = {
    'dict'  : {'key': 'value'},
    'float' : 3.14,
    'int'   : 1,
    'list'  : [1, 2, 3],
    'string': 'A very-very-very long-long-long string',
}


def generate_graph_html(graph: StateGraph) -> pn.pane.HTML:
    """Generates a PyVis graph from the LangGraph instance, embedded via a Base64 data URI."""
    # Build a DiGraph from the LangGraph edges
    G: nx.DiGraph = nx.DiGraph()
    edges = graph.get_graph(xray=1).edges
    for edge in edges:
        G.add_edge(edge[0], edge[1])

    # Create a PyVis network
    nt: Network = Network(height='500px', width='500px', directed=True)
    nt.from_nx(G)

    # Get the raw HTML for the PyVis network
    net_html: str = nt.generate_html()

    # Convert the HTML to a Base64 data URI and embed it in an iframe
    net_html_b64: str = base64.b64encode(net_html.encode()).decode()
    iframe_html: str = f"""
        <iframe 
            src="data:text/html;base64,{net_html_b64}"
            height="500px" 
            width="500px" 
            style="border:none;"
        ></iframe>
        """
    return pn.pane.HTML(iframe_html, sizing_mode='stretch_both')


class AssistantApp:
    def __init__(self) -> None:
        self.llm = OpenAI(api_key=load_api_key('openai.api_key'))
        self.conversation_thread = {'configurable': {'thread_id': '1'}}

        # -----------------------------
        # Construct main page
        # -----------------------------
        self.chat_feed = pn.chat.ChatFeed()
        self.chat_input = pn.chat.ChatAreaInput()
        self.chat_input.param.watch(self.submit_message_action, 'value')

        self.chat_interface = pn.Column(
            self.chat_feed,
            self.chat_input,
            sizing_mode='stretch_both'
        )

        self.panel_main = pn.Column(
            self.chat_interface,
            sizing_mode='stretch_both',
            margin=10
        )

        # -----------------------------
        # Construct "details" page
        # -----------------------------
        self.je_user_profile = pn.widgets.JSONEditor(value=SAMPLE_JSON, mode='view', sizing_mode='stretch_both')

        self.je_todos = pn.widgets.JSONEditor(value=SAMPLE_JSON, mode='view', sizing_mode='stretch_both')

        self.je_instructions = pn.widgets.JSONEditor(value=SAMPLE_JSON, mode='view', sizing_mode='stretch_both')

        html_graph_todo: pn.pane.HTML = generate_graph_html(graph_todo)
        self.panel_details = pn.Column(
            pn.Tabs(
                ('graph', html_graph_todo),
                ('mem: User Profile', self.je_user_profile),
                ('mem: ToDos', self.je_todos),
                ('mem: Instructions', self.je_instructions),
                dynamic=True
            ),
            sizing_mode='stretch_both',
            margin=10
        )

        # -----------------------------
        # the dashboard
        # -----------------------------
        self.navigation_bar = pn.widgets.ToggleGroup(
            name='NavigationBar',
            options=[PAGE_NAME_CHAT, PAGE_NAME_DETAILS],
            behavior='radio',
            button_style='outline',
            button_type='primary'
        )
        self.navigation_bar.param.watch(self.on_navigation_change, 'value')

        self.dashboard = pn.Column(
            self.navigation_bar,
            self.panel_main
        )

    def on_navigation_change(self, event):
        """Handle toggle switch event and update panel display."""
        if event.new == PAGE_NAME_CHAT:
            self.dashboard[:] = [self.navigation_bar, self.panel_main]  # Show Chat panel
        else:
            self.dashboard[:] = [self.navigation_bar, self.panel_details]  # Show Details panel

    def submit_message_action(self, event: Any = None) -> None:
        """Handles message submission and updates the chat feed."""
        user_message = event.new
        if user_message:
            msg_settings = dict(
                show_avatar=True, show_user=False, show_timestamp=True, show_copy_icon=False, show_edit_icon=False, reaction_icons={}
            )

            self.chat_feed.append(pn.chat.ChatMessage(user_message, avatar=chr(0xC6C3), user='user', **msg_settings))
            response = self.get_llm_response(user_message)
            self.chat_feed.append(pn.chat.ChatMessage(response, avatar=chr(0x2728), user='ai', **msg_settings))

    def get_llm_response(self, message: str) -> str:
        """Invokes the inference graph; collects the response."""
        response: str = ''
        for event in graph_todo.stream(
            input={'messages': [HumanMessage(content=message)]},
            config=self.conversation_thread,
            stream_mode='values'
        ):
            if event.get('event') == 'values':
                state = event.data
                response += convert_to_messages(state['messages'])[-1].pretty_repr()

        return response

    def get_dashboard(self) -> pn.Column:
        """Returns the Panel dashboard."""
        return self.dashboard
