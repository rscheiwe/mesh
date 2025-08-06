"""OpenAI provider implementation."""

import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from mesh.nodes.llm import Message
from mesh.providers.base import (
    BaseProvider,
    ProviderConfig,
    ProviderResponse,
    StreamChunk,
)


@dataclass
class OpenAIConfig(ProviderConfig):
    """Configuration specific to OpenAI."""

    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    organization: Optional[str] = None
    default_model: str = "gpt-3.5-turbo"

    def __post_init__(self):
        # Try to get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""

    def __init__(self, config: Optional[OpenAIConfig] = None):
        if config is None:
            config = OpenAIConfig()
        super().__init__(config)
        self.config: OpenAIConfig = config
        self._client = None
        self._sync_client = None
        self._models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install mesh[openai]"
            )

        if not self.config.api_key:
            raise ValueError("OpenAI API key not provided")

        # Initialize async client
        self._client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            organization=self.config.organization,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

        # Initialize sync client
        self._sync_client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            organization=self.config.organization,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

    async def chat_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        top_p: Optional[float] = None,
        n: int = 1,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Union[ProviderResponse, AsyncIterator[StreamChunk]]:
        """Make a chat completion request to OpenAI.

        Args:
            messages: List of messages in the conversation
            model: Model to use (defaults to config.default_model)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            stop: Stop sequences
            presence_penalty: Penalize new topics (-2.0 to 2.0)
            frequency_penalty: Penalize repeated tokens (-2.0 to 2.0)
            seed: Random seed for reproducibility
            response_format: Response format (e.g., {"type": "json_object"})
            **kwargs: Additional parameters

        Returns:
            ProviderResponse with completion result
        """
        if not self._client:
            await self.initialize()

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            msg_dict = {"role": msg.role, "content": msg.content}
            # Add name for function messages
            if msg.role == "function" and msg.metadata and "name" in msg.metadata:
                msg_dict["name"] = msg.metadata["name"]
            openai_messages.append(msg_dict)

        # Build request parameters
        params = {
            "model": model or self.config.default_model,
            "messages": openai_messages,
            "temperature": temperature,
            "n": n,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }

        # Add optional parameters
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if stop is not None:
            params["stop"] = stop
        if seed is not None:
            params["seed"] = seed
        if response_format is not None:
            params["response_format"] = response_format

        # Add any additional kwargs
        params.update(kwargs)

        # Make the API call
        if stream:
            # Return async generator for streaming
            return self._stream_chat_completion(params)
        else:
            # Non-streaming response
            response = await self._client.chat.completions.create(**params)

            # Extract the first choice
            choice = response.choices[0]

            return ProviderResponse(
                content=choice.message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                finish_reason=choice.finish_reason,
                metadata={
                    "id": response.id,
                    "created": response.created,
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                },
            )

    async def _stream_chat_completion(
        self, params: Dict[str, Any]
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion response."""
        params["stream"] = True
        stream = await self._client.chat.completions.create(**params)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(
                    content=chunk.choices[0].delta.content,
                    model=chunk.model,
                    finish_reason=chunk.choices[0].finish_reason,
                    metadata={
                        "id": chunk.id,
                        "created": chunk.created,
                    },
                )

    async def chat_completion_with_tools(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[str] = "auto",
        **kwargs,
    ) -> ProviderResponse:
        """Make a chat completion request with tools.

        Args:
            messages: List of messages in the conversation
            tools: List of tool definitions
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tool_choice: Tool selection strategy ("auto", "none", or specific tool)
            **kwargs: Additional parameters

        Returns:
            ProviderResponse with completion result and tool calls
        """
        if not self._client:
            await self.initialize()

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            msg_dict = {"role": msg.role, "content": msg.content}
            # Add name for function messages
            if msg.role == "function" and msg.metadata and "name" in msg.metadata:
                msg_dict["name"] = msg.metadata["name"]
            openai_messages.append(msg_dict)

        # Convert tools to OpenAI format
        openai_tools = []
        for tool in tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    },
                }
            )

        # Build request parameters
        params = {
            "model": model or self.config.default_model,
            "messages": openai_messages,
            "tools": openai_tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Add any additional kwargs
        params.update(kwargs)

        # Make the API call
        response = await self._client.chat.completions.create(**params)

        # Extract the first choice
        choice = response.choices[0]

        # Extract tool calls if present
        tool_calls = []
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        return ProviderResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=choice.finish_reason,
            metadata={
                "id": response.id,
                "created": response.created,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "tool_calls": tool_calls,
            },
        )

    def get_supported_models(self) -> List[str]:
        """Get list of supported OpenAI models.

        Returns:
            List of model identifiers
        """
        return self._models.copy()

    def chat_completion_sync(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ProviderResponse, Iterator[StreamChunk]]:
        """Synchronous chat completion.

        Args:
            messages: List of messages in the conversation
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            ProviderResponse if not streaming, Iterator[StreamChunk] if streaming
        """
        if not self._sync_client:
            # Initialize sync client if needed
            import openai

            self._sync_client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                organization=self.config.organization,
            )

        # Convert messages
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # Build parameters
        params = {
            "model": model or self.config.default_model,
            "messages": openai_messages,
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        params.update(kwargs)

        if stream:
            # Return streaming iterator
            return self._stream_chat_completion_sync(params)
        else:
            # Non-streaming response
            response = self._sync_client.chat.completions.create(**params)
            choice = response.choices[0]

            return ProviderResponse(
                content=choice.message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                finish_reason=choice.finish_reason,
                metadata={
                    "id": response.id,
                    "created": response.created,
                },
            )

    def _stream_chat_completion_sync(
        self, params: Dict[str, Any]
    ) -> Iterator[StreamChunk]:
        """Synchronous streaming."""
        stream = self._sync_client.chat.completions.create(**params)

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(
                    content=chunk.choices[0].delta.content,
                    model=chunk.model,
                    finish_reason=chunk.choices[0].finish_reason,
                    metadata={"id": chunk.id},
                )
