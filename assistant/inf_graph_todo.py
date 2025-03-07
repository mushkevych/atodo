import uuid
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, merge_message_runs, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from trustcall import create_extractor

import assistant.models
from assistant.models import UserProfile, ToDo, UpdateMemory, MemoryType
from assistant.services import llm_4o, llm_llama3_1_8b
from assistant.inspector import ToolInvocationInspector, extract_tool_info

# Chatbot instruction for choosing:
# - what to update: user_profile, list of todos or instructions
# - which tool to call: "user_profile", "todo" or "instructions"
INSTRUCTION_MEMORY_TOOL_AND_RESPONSE = """{assistant_role} 

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them) 
2. The user's ToDo list
3. General instructions for updating the ToDo list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user_profile`
- If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`
- If the user has specified preferences for how to update the ToDo list, update the instructions by calling UpdateMemory tool with type `instructions`

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user when you updated the ToDo list
- Do not tell the user that you have updated instructions

4. Err on the side of updating the todo list. No need to ask for explicit permission.

5. After a tool call has been made to save memories, or if no tool call was needed, respond naturally to the user."""

# Parallel trustcall: how to update memories about the user
INSTRUCTION_USER_MEMORY_UPDATE = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

# Instructions for updating the ToDo list
INSTRUCTION_INSTRUCTIONS_MEMORY_UPDATE = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""

the_model = llm_llama3_1_8b

## Create the Trustcall extractors for updating the user profile and ToDo list
profile_extractor = create_extractor(
    the_model,
    tools=[UserProfile],
    tool_choice=UserProfile.__name__,
)


## Node definitions
def task_controller(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memories from the Memory Store and use them to personalize the chatbot's response."""

    # Get the user ID from the config
    configurable = assistant.models.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    assistant_type = configurable.assistant_type
    assistant_role = configurable.assistant_role

    # Retrieve profile memory from the store
    namespace = (MemoryType.USER_PROFILE.value, assistant_type, user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # Retrieve ToDos memories from the store
    namespace = (MemoryType.TODO.value, assistant_type, user_id)
    memories = store.search(namespace)
    todo = '\n'.join(f'{mem.value}' for mem in memories)

    # Retrieve custom instructions
    namespace = (MemoryType.INSTRUCTIONS.value, assistant_type, user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ''

    system_msg = INSTRUCTION_MEMORY_TOOL_AND_RESPONSE.format(
        assistant_role=assistant_role,
        user_profile=user_profile,
        todo=todo,
        instructions=instructions
    )

    # Respond using memory as well as the chat history
    response = the_model.bind_tools(
        tools=[UpdateMemory]  # , parallel_tool_calls=False
    ).invoke(
        [SystemMessage(content=system_msg)] + state['messages']
    )

    return {'messages': [response]}


def tool_update_user_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update User Profile memory collection."""

    # Get the user ID from the config
    configurable = assistant.models.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    assistant_type = configurable.assistant_type

    # Define the namespace for the memories
    namespace = (MemoryType.USER_PROFILE.value, assistant_type, user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = UserProfile.__name__
    existing_memories = (
        [(existing_item.key, tool_name, existing_item.value) for existing_item in existing_items]
        if existing_items else None
    )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = INSTRUCTION_USER_MEMORY_UPDATE.format(time=datetime.now().isoformat())
    updated_messages: list[BaseMessage] = merge_message_runs(
        messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state['messages'][:-1]
    )

    # Invoke the extractor
    result = profile_extractor.invoke(input={
        'messages': updated_messages,
        'existing': existing_memories
    })

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result['responses'], result['response_metadata']):
        store.put(
            namespace=namespace,
            key=rmeta.get('json_doc_id', str(uuid.uuid4())),
            value=r.model_dump(mode='json'),
        )

    tool_calls = state['messages'][-1].tool_calls

    # Return tool message with update verification
    return {
        'messages': [
            {
                'role': 'tool',
                'content': 'updated profile',
                'tool_call_id': tool_calls[0]['id']
            }
        ]
    }


def tool_update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""

    # Get the user ID from the config
    configurable = assistant.models.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    assistant_type = configurable.assistant_type

    # Define the namespace for the memories
    namespace = (MemoryType.TODO.value, assistant_type, user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = ToDo.__name__
    existing_memories = (
        [(existing_item.key, tool_name, existing_item.value) for existing_item in existing_items]
        if existing_items else None
    )

    # Merge the chat history and the instruction
    INSTRUCTIONS_USER_MEMORY_UPDATE_FMT = INSTRUCTION_USER_MEMORY_UPDATE.format(time=datetime.now().isoformat())

    updated_messages = merge_message_runs(
        messages=[SystemMessage(content=INSTRUCTIONS_USER_MEMORY_UPDATE_FMT)] + state['messages'][:-1]
    )

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = ToolInvocationInspector()

    # Create the Trustcall extractor for updating the ToDo list
    todo_extractor = create_extractor(
        the_model,
        tools=[ToDo],
        tool_choice=tool_name,
        enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = todo_extractor.invoke({
        'messages': updated_messages,
        'existing': existing_memories
    })

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result['responses'], result['response_metadata']):
        store.put(
            namespace,
            rmeta.get('json_doc_id', str(uuid.uuid4())),
            r.model_dump(mode='json'),
        )

    # Respond to the tool call made in task_controller, confirming the update
    tool_calls = state['messages'][-1].tool_calls

    # Extract the changes made by Trustcall and add the ToolMessage returned to task_controller
    todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {
        'messages': [
            {
                'role': 'tool',
                'content': todo_update_msg,
                'tool_call_id': tool_calls[0]['id']
            }
        ]
    }


def tool_update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""
    key = 'user_instructions'

    # Get the user ID from the config
    configurable = assistant.models.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    assistant_type = configurable.assistant_type

    namespace = (MemoryType.INSTRUCTIONS.value, assistant_type, user_id)
    existing_memory = store.get(namespace=namespace, key=key)

    # Format the memory in the system prompt
    system_msg = INSTRUCTION_INSTRUCTIONS_MEMORY_UPDATE.format(
        current_instructions=existing_memory.value if existing_memory else None
    )
    new_memory = the_model.invoke(
        [SystemMessage(content=system_msg)]
        + state['messages'][:-1]
        + [HumanMessage(content='Please update the instructions based on the conversation')]
    )

    # Overwrite the existing memory in the store
    store.put(
        namespace=namespace, key=key, value={'memory': new_memory.content}
    )
    tool_calls = state['messages'][-1].tool_calls
    # Return tool message with update verification
    return {
        'messages': [
            {
                'role': 'tool',
                'content': 'updated instructions',
                'tool_call_id': tool_calls[0]['id']
            }
        ]
    }


class RouteListener:
    def update(self, current_node: str = None, next_node: str = None) -> None:
        raise NotImplementedError()


route_listeners: set[RouteListener] = set()


# Conditional edge
def route_message(
    state: MessagesState, config: RunnableConfig, store: BaseStore
) -> Literal[END, tool_update_todos.__name__, tool_update_instructions.__name__, tool_update_user_profile.__name__]:
    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]
    if len(message.tool_calls) == 0:
        selected_node = END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == MemoryType.USER_PROFILE.value:
            selected_node = tool_update_user_profile.__name__
        elif tool_call['args']['update_type'] == MemoryType.TODO.value:
            selected_node = tool_update_todos.__name__
        elif tool_call['args']['update_type'] == MemoryType.INSTRUCTIONS.value:
            selected_node = tool_update_instructions.__name__
        else:
            raise ValueError(f'Unknown update_type: {tool_call["args"]["update_type"]}')

    global route_listeners
    for route_listener in route_listeners:
        route_listener.update(current_node=config['metadata']['langgraph_node'], next_node=selected_node)
    return selected_node


def build_graph() -> StateGraph:
    # Create the graph + all nodes
    builder = StateGraph(MessagesState, config_schema=assistant.models.Configuration)

    # Define the flow of the memory extraction process
    builder.add_node(task_controller)
    builder.add_node(tool_update_todos)
    builder.add_node(tool_update_user_profile)
    builder.add_node(tool_update_instructions)

    # Define the flow
    builder.add_edge(START, task_controller.__name__)
    builder.add_conditional_edges(task_controller.__name__, route_message)
    builder.add_edge(tool_update_todos.__name__, task_controller.__name__)
    builder.add_edge(tool_update_user_profile.__name__, task_controller.__name__)
    builder.add_edge(tool_update_instructions.__name__, task_controller.__name__)
    return builder


across_thread_memory = InMemoryStore()  # Store for long-term (across-thread) memory
within_thread_memory = MemorySaver()  # Checkpointer for short-term (within-thread) memory
graph = build_graph().compile(checkpointer=within_thread_memory, store=across_thread_memory)
