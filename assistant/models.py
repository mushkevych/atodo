import os
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from typing import Optional, Literal, Self, Any

from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description='Personal connection of the user, such as family members, friends, or coworkers',
        default_factory=list
    )
    interests: list[str] = Field(
        description='Interests that the user has',
        default_factory=list
    )


class ToDo(BaseModel):
    task: str = Field(description='The task to be completed.')
    time_to_complete: Optional[int] = Field(description='Estimated time to complete the task (minutes).')
    deadline: Optional[datetime] = Field(
        description='When the task needs to be completed by (if applicable)',
        default=None
    )
    solutions: list[str] = Field(
        description='List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)',
        min_items=1,
        default_factory=list
    )
    status: Literal['not started', 'in progress', 'done', 'archived'] = Field(
        description='Current status of the task',
        default='not started'
    )


class MemoryType(Enum):
    USER_PROFILE = 'user_profile'
    TODO = 'todo'
    INSTRUCTIONS = 'instructions'


class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal[MemoryType.USER_PROFILE.value, MemoryType.TODO.value, MemoryType.INSTRUCTIONS.value]


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    user_id: str = 'default-user'
    assistant_type: str = 'general'
    atodo_assistant_role: str = "You are a helpful task management assistant. You help to create, organize, and track the user's ToDo list."

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> Self:
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config['configurable'] if config and 'configurable' in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls) if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
