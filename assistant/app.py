from collections import namedtuple
from datetime import datetime, timedelta

import panel as pn
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAI
from param.parameterized import Event

from assistant.graph_visualizer import GraphVisualizer, NodeColorizer
from assistant.inf_graph_todo import graph as graph_todo, route_listeners, across_thread_memory
from assistant.models import Configuration, MemoryType
from utils.fs_utils import load_api_key

PAGE_NAME_CHAT = 'Chat'
PAGE_NAME_DETAILS = 'Details'

TAB_USER_PROFILE = 'mem: User Profile'
TAB_TODO = 'mem: ToDos'
TAB_INSTRUCTIONS = 'mem: Instructions'

EMPTY_JSON = {}
MockEvent = namedtuple(typename='MockEvent', field_names=['name', 'old', 'new'])

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
        self.btn_simulate_conv = pn.widgets.Button(
            name='Simulate conversation', button_type='default', button_style='outline'
        )
        self.btn_simulate_conv.on_click(self.simulate_conversation)

        self.chat_interface = pn.Column(
            self.chat_feed,
            pn.Row(self.chat_input, self.btn_simulate_conv),
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
        self.je_user_profile = pn.widgets.JSONEditor(value=EMPTY_JSON, mode='view', sizing_mode='stretch_both')
        self.je_todos = pn.widgets.JSONEditor(value=EMPTY_JSON, mode='view', sizing_mode='stretch_both')
        self.je_instructions = pn.widgets.JSONEditor(value=EMPTY_JSON, mode='view', sizing_mode='stretch_both')

        self.tabs_details = pn.Tabs(
            (TAB_USER_PROFILE, self.je_user_profile),
            (TAB_TODO, self.je_todos),
            (TAB_INSTRUCTIONS, self.je_instructions),
            dynamic=True
        )
        self.tabs_details.param.watch(self.on_details_change, 'active')

        self.panel_details = pn.Column(
            self.tabs_details,
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

        self.panel_main.visible = True
        self.panel_details.visible = False
        self.dashboard = pn.Column(
            self.navigation_bar,
            self.panel_main,
            self.panel_details
        )

    def on_navigation_change(self, event: Event):
        if event.new == PAGE_NAME_CHAT:
            self.panel_main.visible = True
            self.panel_details.visible = False
        else:
            self.panel_main.visible = False
            self.panel_details.visible = True
            # self.on_details_change(MockEvent(name='active', old=None, new=0))

    def on_details_change(self, event: Event | MockEvent):
        tab_mapping = {
            0: TAB_USER_PROFILE,
            1: TAB_TODO,
            2: TAB_INSTRUCTIONS
        }

        selected_tab = tab_mapping.get(event.new, None)
        if selected_tab == TAB_USER_PROFILE:
            namespace = (MemoryType.USER_PROFILE.value, Configuration.assistant_type, Configuration.user_id)
            component = self.je_user_profile
        elif selected_tab == TAB_TODO:
            namespace = (MemoryType.TODO.value, Configuration.assistant_type, Configuration.user_id)
            component = self.je_todos
        elif selected_tab == TAB_INSTRUCTIONS:
            namespace = (MemoryType.INSTRUCTIONS.value, Configuration.assistant_type, Configuration.user_id)
            component = self.je_instructions
        else:
            raise ValueError(f'Unknown event {event.new}')

        existing_memory = across_thread_memory.search(namespace)
        component.value = [entry.value for entry in existing_memory]

    def submit_message_action(self, event: Event | MockEvent) -> None:
        """Handles message submission and updates the chat feed."""
        self.btn_simulate_conv.disabled = True  # any interaction with the Input Field disables the Simulation Button


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

    def simulate_conversation(self, event: Event | MockEvent) -> None:
        human_messages = [
            'I am Dan. I live in Beaverton, Oregon, and like to ride my bicycle.',
            """
Consider following instructions:
- When providing a 'todo summary':
  1. List all current tasks grouped by deadline (overdue, today, this week, future)
  2. Highlight any tasks missing deadlines and gently encourage adding them
  3. Note any tasks that seem important but lack time estimates
- Proactively ask for deadlines when new tasks are added without them
- Maintain a supportive tone while helping the user stay accountable
- Help prioritize tasks based on deadlines and importance

Your communication style should be encouraging and helpful, never judgmental. 

When tasks are missing deadlines, respond with something like "I notice [task] doesn't have a deadline yet. Would you like to add one to help us track it better?
            """,
            f"""
Current time is: {datetime.now().isoformat(timespec='minutes')}.
Create or update few ToDos: 
1) Buy rye bread from the nearby Whole Foods store by {(datetime.now() + timedelta(hours=3)).isoformat(timespec='minutes')}. 
2) Upload AToDo agentic app to the Github by {(datetime.now() + timedelta(days=10)).isoformat(timespec='minutes')}.
3) Register for Friends Of Trees event in my neighbourhood.
            """,
            f"""by {(datetime.now() + timedelta(days=30)).isoformat(timespec='minutes')}""",  # provide missing deadline
            # 'Please show me my current tasks',
        ]
        for message in human_messages:
            self.submit_message_action(MockEvent(name='value', old=None, new=message))

    def get_dashboard(self) -> pn.Column:
        """Returns the Panel dashboard."""
        return self.dashboard
