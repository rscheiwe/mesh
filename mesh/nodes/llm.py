"""LLM node implementation for AI provider endpoints."""

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, Field

from mesh.core.node import NodeConfig
from mesh.nodes.base import BaseNode
from mesh.state.state import GraphState


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CUSTOM = "custom"


class Message(BaseModel):
    """Represents a message in a conversation."""

    role: str  # system, user, assistant
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class LLMConfig(NodeConfig):
    """Configuration for LLM nodes."""

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    api_key: Optional[str] = None  # API key for the provider
    api_base: Optional[str] = None  # Custom API endpoint
    organization: Optional[str] = None  # For providers that support org IDs
    stream: bool = False  # Whether to stream responses
    use_async: bool = True  # Whether to use async or sync methods
    provider_options: Dict[str, Any] = field(
        default_factory=dict
    )  # Additional provider-specific options
    can_be_terminal: bool = True  # LLM nodes can be terminal nodes


class LLMNode(BaseNode):
    """Node for interacting with LLM providers."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.llm_config: LLMConfig = config
        self._client = None

    async def _execute_impl(
        self, input_data: Dict[str, Any], state: Optional[GraphState] = None
    ) -> Dict[str, Any]:
        """Execute LLM call.

        Args:
            input_data: Should contain 'messages' or 'prompt'
            state: Optional shared state

        Returns:
            Dict containing 'response' and other metadata
        """
        # Extract messages from input
        messages = self._prepare_messages(input_data, state)

        # Make LLM call
        response = await self._call_llm(messages)

        # If streaming is enabled and we got an iterator, store it
        if self.llm_config.stream and hasattr(response, "__aiter__"):
            # Store the stream for the executor to handle
            self._stream_iterator = response
            return {
                "response": None,  # Will be populated by streaming
                "messages": messages,
                "model": self.llm_config.model,
                "provider": self.llm_config.provider.value,
                "streaming": True,
            }

        return {
            "response": response,
            "messages": messages,
            "model": self.llm_config.model,
            "provider": self.llm_config.provider.value,
        }

    def _prepare_messages(
        self, input_data: Dict[str, Any], state: Optional[GraphState] = None
    ) -> List[Message]:
        """Prepare messages for LLM call.

        Args:
            input_data: Input data containing messages or prompt
            state: Optional shared state

        Returns:
            List of messages
        """
        messages = []

        # Add system prompt if configured
        if self.llm_config.system_prompt:
            messages.append(
                Message(role="system", content=self.llm_config.system_prompt)
            )

        # Handle different input formats
        if "messages" in input_data:
            # Already formatted as messages
            for msg in input_data["messages"]:
                if isinstance(msg, dict):
                    messages.append(Message(**msg))
                elif isinstance(msg, Message):
                    messages.append(msg)

        elif "prompt" in input_data:
            # Single prompt string
            messages.append(Message(role="user", content=input_data["prompt"]))

        elif "user_message" in input_data:
            # User message
            messages.append(Message(role="user", content=input_data["user_message"]))

        # Add conversation history from state if available
        if state and "conversation_history" in state.data:
            history = state.data["conversation_history"]
            if isinstance(history, list):
                for msg in history:
                    if isinstance(msg, dict):
                        messages.append(Message(**msg))

        return messages

    async def _call_llm(
        self, messages: List[Message]
    ) -> Union[str, AsyncIterator[str]]:
        """Make the actual LLM API call.

        Args:
            messages: List of messages to send

        Returns:
            Response string or async iterator of chunks if streaming
        """
        # Get or create provider
        provider = await self._get_provider()

        if self.llm_config.use_async:
            # Async call
            response = await provider.chat_completion(
                messages=messages,
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
                stream=self.llm_config.stream,
            )

            if self.llm_config.stream:
                # Return async iterator for streaming
                return response  # Return the stream directly
            else:
                return response.content
        else:
            # Sync call
            response = provider.chat_completion_sync(
                messages=messages,
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
                stream=self.llm_config.stream,
            )

            if self.llm_config.stream:
                # Collect sync stream into string
                return self._collect_stream_sync(response)
            else:
                return response.content

    async def _collect_stream_async(self, stream: AsyncIterator) -> str:
        """Collect async stream chunks into a single response."""
        chunks = []
        async for chunk in stream:
            chunks.append(chunk.content)
        return "".join(chunks)

    def _collect_stream_sync(self, stream: Iterator) -> str:
        """Collect sync stream chunks into a single response."""
        chunks = []
        for chunk in stream:
            chunks.append(chunk.content)
        return "".join(chunks)

    async def _get_provider(self):
        """Get the appropriate provider instance."""
        if self._client is None:
            # Import providers
            from mesh.providers.openai_provider import OpenAIConfig, OpenAIProvider

            # Create provider based on config
            if self.llm_config.provider == LLMProvider.OPENAI:
                config = OpenAIConfig(
                    api_key=self.llm_config.api_key,
                    base_url=self.llm_config.api_base,
                    organization=self.llm_config.organization,
                    **self.llm_config.provider_options,
                )
                self._client = OpenAIProvider(config)
                await self._client.initialize()
            elif self.llm_config.provider == LLMProvider.ANTHROPIC:
                # Would import and use AnthropicProvider
                raise NotImplementedError("Anthropic provider not yet implemented")
            elif self.llm_config.provider == LLMProvider.GOOGLE:
                # Would import and use GoogleProvider
                raise NotImplementedError("Google provider not yet implemented")
            else:
                raise ValueError(f"Unknown provider: {self.llm_config.provider}")

        return self._client
