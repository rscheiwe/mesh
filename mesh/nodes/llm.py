"""LLM node for direct model calls with streaming.

This node makes direct calls to LLM providers (OpenAI, Anthropic, etc.)
without using an agent framework. It supports streaming token-by-token.
"""

from typing import Any, Dict, Optional, List
import os

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext
from mesh.core.events import ExecutionEvent, EventType, OpenAIEventTranslator
from mesh.utils.variables import VariableResolver


class LLMNode(BaseNode):
    """Direct LLM call node with streaming support.

    This node makes API calls to LLM providers and streams the response
    token-by-token. It supports:
    - Variable resolution in prompts
    - System prompts with templates
    - Chat history integration
    - Multiple providers (currently OpenAI)

    Example:
        >>> llm = LLMNode(
        ...     id="summarizer",
        ...     model="gpt-4",
        ...     system_prompt="Summarize: {{agent_node.content}}",
        ... )
    """

    def __init__(
        self,
        id: str,
        model: str = "gpt-4",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        provider: str = "openai",
        config: Dict[str, Any] = None,
    ):
        """Initialize LLM node.

        Args:
            id: Node identifier
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            system_prompt: System prompt template (can contain {{variables}})
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            provider: LLM provider ("openai", "anthropic", etc.)
            config: Additional configuration
        """
        super().__init__(id, config or {})
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider

        # Initialize provider client
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the provider client."""
        if self.provider == "openai":
            try:
                from openai import AsyncOpenAI

                api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
                self._client = AsyncOpenAI(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute LLM call with streaming.

        Args:
            input: Input data (string or dict with 'content')
            context: Execution context

        Returns:
            NodeResult with LLM response
        """
        # Resolve variables in system prompt
        resolver = VariableResolver(context)
        resolved_prompt = ""
        if self.system_prompt:
            resolved_prompt = await resolver.resolve(self.system_prompt)

        # Build messages
        messages = []
        if resolved_prompt:
            messages.append({"role": "system", "content": resolved_prompt})

        # Extract user message from input
        user_message = self._extract_message(input)
        messages.append({"role": "user", "content": user_message})

        # Stream response
        if self.provider == "openai":
            return await self._execute_openai(messages, context)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _execute_openai(
        self,
        messages: List[Dict[str, str]],
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute OpenAI completion with streaming.

        Args:
            messages: List of chat messages
            context: Execution context

        Returns:
            NodeResult with response
        """
        full_response = ""
        translator = OpenAIEventTranslator()

        # Prepare kwargs
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        # Stream response
        stream = await self._client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token

                # Emit token event
                event = translator.from_openai_chunk(chunk)
                event.node_id = self.id
                await context.emit_event(event)

        # Build chat history entry
        chat_history = [
            {"role": "user", "content": messages[-1]["content"]},
            {"role": "assistant", "content": full_response},
        ]

        return NodeResult(
            output={"content": full_response},
            chat_history=chat_history,
            metadata={
                "model": self.model,
                "provider": self.provider,
                "tokens": len(full_response.split()),  # Rough estimate
            },
        )

    def _extract_message(self, input: Any) -> str:
        """Extract message content from various input formats.

        Args:
            input: Input data

        Returns:
            Message string
        """
        if isinstance(input, str):
            return input
        elif isinstance(input, dict):
            # Try common keys
            for key in ["content", "message", "text", "input", "question"]:
                if key in input:
                    return str(input[key])
            # Fallback to str representation
            return str(input)
        else:
            return str(input)
