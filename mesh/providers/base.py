"""Base provider interface for AI services."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel

from mesh.nodes.llm import Message


@dataclass
class ProviderConfig:
    """Base configuration for providers."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    extra_headers: Dict[str, str] = None


class ProviderResponse(BaseModel):
    """Standard response from providers."""

    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = {}


class StreamChunk(BaseModel):
    """A chunk of streaming response."""

    content: str
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = {}


class BaseProvider(ABC):
    """Abstract base class for AI providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client = None

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider client."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ProviderResponse, AsyncIterator[StreamChunk]]:
        """Make a chat completion request.

        Args:
            messages: List of messages in the conversation
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            ProviderResponse if not streaming, AsyncIterator[StreamChunk] if streaming
        """
        pass

    def chat_completion_sync(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ProviderResponse, Iterator[StreamChunk]]:
        """Synchronous version of chat_completion.

        Args:
            messages: List of messages in the conversation
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            ProviderResponse if not streaming, Iterator[StreamChunk] if streaming
        """
        raise NotImplementedError("Synchronous chat completion not implemented")

    @abstractmethod
    async def chat_completion_with_tools(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[str] = "auto",
        **kwargs,
    ) -> ProviderResponse:
        """Make a chat completion request with tool support.

        Args:
            messages: List of messages in the conversation
            tools: List of tool definitions
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tool_choice: Tool selection strategy
            **kwargs: Additional provider-specific parameters

        Returns:
            ProviderResponse with completion result and tool calls
        """
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Get list of supported models.

        Returns:
            List of model identifiers
        """
        pass

    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            if hasattr(self._client, "close"):
                await self._client.close()
