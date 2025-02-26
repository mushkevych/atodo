from typing import Any

import panel as pn
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAI

from assistant.graph_visualizer import GraphVisualizer, NodeColorizer
from assistant.inf_graph_todo import graph as graph_todo, route_listeners
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
            sizing_mode='stretch_both',
            styles={'border': '1px solid black', 'padding': '10px', 'border-radius': '5px'},
        )

        self.graph_visualizer = GraphVisualizer(graph_todo)
        route_listeners.add(NodeColorizer(self.graph_visualizer))

        self.panel_main = pn.Row(
            self.chat_interface,
            self.graph_visualizer,
            sizing_mode='stretch_both',
            margin=10
        )

        # -----------------------------
        # Construct "details" page
        # -----------------------------
        self.je_user_profile = pn.widgets.JSONEditor(value=SAMPLE_JSON, mode='view', sizing_mode='stretch_both')

        self.je_todos = pn.widgets.JSONEditor(value=SAMPLE_JSON, mode='view', sizing_mode='stretch_both')

        self.je_instructions = pn.widgets.JSONEditor(value=SAMPLE_JSON, mode='view', sizing_mode='stretch_both')

        self.panel_details = pn.Column(
            pn.Tabs(
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

        self._init_profile()

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
            if (comm := event['messages'][-1]) and comm.type == 'ai':
                response += comm.content

        return response

    def _init_profile(self):
        # self.chat_input.value = 'What tasks can I get done today?'
        human_messages = [
            "I am Dan. I live in Beaverton, Oregon, and like to ride my bicycle.",
            "Create or update few ToDos: 1) Buy rye bread from nearby Whole Foods store by 2025-06-10. 2) Upload AToDo agentic app to the Github by March of 2025."
        ]
        for message in human_messages:
            self.get_llm_response(message)

    def get_dashboard(self) -> pn.Column:
        """Returns the Panel dashboard."""
        return self.dashboard
