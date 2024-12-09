from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from typing import List, Callable, Union, Optional
from enum import Enum

# Third-party imports
from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"


AgentFunction = Callable[[], Union[str, "Agent", dict]]


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4"
    provider: ModelProvider = ModelProvider.OPENAI
    deployment_name: Optional[str] = None  # For Azure deployments
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: Optional[str] = None
    parallel_tool_calls: bool = True

    @property
    def model_identifier(self) -> str:
        """Returns the appropriate model identifier based on the provider."""
        if self.provider == ModelProvider.AZURE:
            # Use deployment name if provided, otherwise convert standard name to Azure format
            if self.deployment_name:
                return self.deployment_name
            # Convert standard OpenAI model names to Azure format
            return self.model.replace("gpt-3.5-turbo", "gpt-35-turbo")
        return self.model


class Response(BaseModel):
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}
