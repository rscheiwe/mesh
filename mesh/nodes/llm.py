"""LLM node for direct model calls with streaming.

This node makes direct calls to LLM providers (OpenAI, Anthropic, etc.)
without using an agent framework. It supports streaming token-by-token.
"""

from typing import Any, Dict, Optional, List
import os

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext
from mesh.core.events import ExecutionEvent, EventType, transform_event_for_transient_mode
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
        event_mode: str = "full",
        auto_inject_context: bool = False,
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
            event_mode: Event emission mode (default: "full")
                - "full": All events - streams to chat
                - "status_only": Only progress indicators (mesh-node-start/complete)
                - "transient_events": All events prefixed with data-llm-node-*
                - "silent": No events
            auto_inject_context: If True, automatically inject upstream DataHandler,
                Tool, and RAG node outputs into the system prompt as structured context.
                The user does not need to manually reference {{variables}} for these nodes.
            config: Additional configuration
        """
        super().__init__(id, config or {})
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self.event_mode = event_mode
        self.auto_inject_context = auto_inject_context

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
        elif self.provider == "anthropic":
            # Anthropic uses direct HTTP with httpx (no official async SDK)
            try:
                import httpx
                self._client = None  # Will use httpx directly
                self._api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
                if not self._api_key:
                    raise ValueError("Anthropic API key not provided")
            except ImportError:
                raise ImportError(
                    "httpx package not installed. Install with: pip install httpx"
                )
        elif self.provider == "gemini":
            # Google Gemini uses google-generativeai SDK
            try:
                import google.generativeai as genai
                self._api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY")
                if not self._api_key:
                    raise ValueError("Google API key not provided")
                genai.configure(api_key=self._api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Install with: pip install google-generativeai"
                )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _emit_event_if_enabled(self, context: ExecutionContext, event: "ExecutionEvent") -> None:
        """Emit event based on event_mode.

        Args:
            context: Execution context
            event: Event to emit
        """
        # Silent mode - no events
        if self.event_mode == "silent":
            return

        # Status only - skip regular events, only custom events emitted separately
        if self.event_mode == "status_only":
            return

        # Transient events - transform all events with data-llm-node-* prefix
        if self.event_mode == "transient_events":
            transformed_event = transform_event_for_transient_mode(event, "llm")
            await context.emit_event(transformed_event)
        else:
            # Full mode - emit events normally
            await context.emit_event(event)

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

        # Auto-inject upstream context if enabled (appended after user's prompt)
        if self.auto_inject_context:
            from mesh.utils.context_injection import UpstreamContextBuilder
            upstream_context = UpstreamContextBuilder.build(context, self.system_prompt or "")
            if upstream_context:
                resolved_prompt = resolved_prompt + "\n\n" + upstream_context

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
        elif self.provider == "anthropic":
            return await self._execute_anthropic(messages, context)
        elif self.provider == "gemini":
            return await self._execute_gemini(messages, context)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _execute_openai(
        self,
        messages: List[Dict[str, str]],
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute OpenAI completion with streaming using Vel translator.

        Args:
            messages: List of chat messages
            context: Execution context

        Returns:
            NodeResult with response
        """
        try:
            from vel import get_openai_api_translator
            from mesh.utils.translator_orchestrator import SimpleTranslatorOrchestrator
        except ImportError:
            raise ImportError(
                "Vel SDK required for LLMNode event translation. "
                "Install with: pip install vel @ git+https://github.com/rscheiwe/vel.git"
            )

        full_response = ""
        usage_data = None

        # Initialize translator and orchestrator
        translator = get_openai_api_translator()
        orchestrator = SimpleTranslatorOrchestrator(translator)

        # Prepare kwargs
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},  # Include usage in final chunk
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        # Create async generator from OpenAI chunks
        async def chunk_stream():
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                # Convert chunk to dict for translator
                yield chunk.model_dump()

        # Stream through orchestrator (adds start/finish events)
        async for event_dict in orchestrator.stream(chunk_stream()):
            event_type = event_dict.get("type")

            # Handle orchestration events
            if event_type == "start":
                # Generation started - emit as NODE_START with raw_event
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.NODE_START,
                        node_id=self.id,
                        metadata={
                            "node_type": "llm",
                            "provider": "openai",
                            "model": self.model,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "start-step":
                # Step started (single-step for LLM, AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.START_STEP,
                        node_id=self.id,
                        metadata={
                            "step_index": 0,
                            "node_type": "llm",
                            "model": self.model,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "finish-step":
                # Step finished with usage (AI SDK format)
                usage_data = event_dict.get("usage")
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.FINISH_STEP,
                        node_id=self.id,
                        metadata={
                            "step_index": 0,
                            "finish_reason": event_dict.get("finishReason"),
                            "usage": usage_data,
                            "response": event_dict.get("response"),
                            "node_type": "llm",
                            "model": self.model,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "finish":
                # Overall generation complete (AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.FINISH,
                        node_id=self.id,
                        metadata={
                            "finish_reason": event_dict.get("finishReason"),
                            "total_usage": event_dict.get("totalUsage"),
                            "node_type": "llm",
                            "model": self.model,
                        },
                        raw_event=event_dict,
                    )
                )

            # Handle content events
            elif event_type == "text-start":
                # Text block started (AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TEXT_START,
                        node_id=self.id,
                        metadata={
                            "block_id": event_dict.get("blockId"),
                            "node_type": "llm",
                            "model": self.model,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "text-delta":
                # Token received (AI SDK format)
                delta = event_dict.get("delta", "")
                full_response += delta
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TEXT_DELTA,
                        node_id=self.id,
                        delta=delta,  # AI SDK field
                        metadata={
                            "node_type": "llm",
                            "model": self.model,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "text-end":
                # Text block ended (AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TEXT_END,
                        node_id=self.id,
                        metadata={
                            "block_id": event_dict.get("blockId"),
                            "node_type": "llm",
                            "model": self.model,
                        },
                        raw_event=event_dict,
                    )
                )

            # Handle reasoning events (o1/o3 models, AI SDK format)
            elif event_type == "reasoning-start":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.REASONING_START,
                        node_id=self.id,
                        metadata={
                            "block_id": event_dict.get("blockId"),
                            "node_type": "llm",
                            "model": self.model,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "reasoning-delta":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.REASONING_DELTA,
                        node_id=self.id,
                        delta=event_dict.get("delta", ""),  # AI SDK field
                        metadata={
                            "block_id": event_dict.get("blockId"),
                            "node_type": "llm",
                            "model": self.model,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "reasoning-end":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.REASONING_END,
                        node_id=self.id,
                        metadata={
                            "block_id": event_dict.get("blockId"),
                            "node_type": "llm",
                            "model": self.model,
                        },
                        raw_event=event_dict,
                    )
                )

        # Build chat history entry
        chat_history = [
            {"role": "user", "content": messages[-1]["content"]},
            {"role": "assistant", "content": full_response},
        ]

        # Build metadata with actual usage data
        metadata = {
            "model": self.model,
            "provider": self.provider,
        }
        if usage_data:
            metadata["usage"] = usage_data

        return NodeResult(
            output={"content": full_response},
            chat_history=chat_history,
            metadata=metadata,
        )

    async def _execute_anthropic(
        self,
        messages: List[Dict[str, str]],
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute Anthropic completion with streaming using Vel translator.

        Args:
            messages: List of chat messages
            context: Execution context

        Returns:
            NodeResult with response
        """
        try:
            import httpx
            import json
            from vel import get_anthropic_translator
            from mesh.utils.translator_orchestrator import SimpleTranslatorOrchestrator
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}. "
                "Install with: pip install httpx vel"
            )

        full_response = ""
        usage_data = None

        # Initialize translator and orchestrator
        translator = get_anthropic_translator()
        orchestrator = SimpleTranslatorOrchestrator(translator)

        # Prepare Anthropic-specific format
        system_message = ""
        anthropic_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                system_message = content
            else:
                anthropic_messages.append({"role": role, "content": content})

        # Prepare payload
        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens or 4096,
            "temperature": self.temperature,
            "stream": True,
        }
        if system_message:
            payload["system"] = system_message

        # Create async generator from Anthropic SSE events
        async def event_stream():
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self._api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            try:
                                data = json.loads(data_str)
                                yield data
                            except json.JSONDecodeError:
                                continue

        # Stream through orchestrator (adds start/finish events)
        async for event_dict in orchestrator.stream(event_stream()):
            event_type = event_dict.get("type")

            # Handle orchestration events
            if event_type == "start":
                # Generation started - emit as NODE_START with raw_event
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.NODE_START,
                        node_id=self.id,
                        metadata={
                            "node_type": "llm",
                            "provider": "anthropic",
                            "model": self.model,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "start-step":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.START_STEP,
                        node_id=self.id,
                        metadata={
                            "step_index": 0,
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "anthropic",
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "finish-step":
                usage_data = event_dict.get("usage")
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.FINISH_STEP,
                        node_id=self.id,
                        metadata={
                            "step_index": 0,
                            "finish_reason": event_dict.get("finishReason"),
                            "usage": usage_data,
                            "response": event_dict.get("response"),
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "anthropic",
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "finish":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.FINISH,
                        node_id=self.id,
                        metadata={
                            "finish_reason": event_dict.get("finishReason"),
                            "total_usage": event_dict.get("totalUsage"),
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "anthropic",
                        },
                        raw_event=event_dict,
                    )
                )

            # Handle content events
            elif event_type == "text-start":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TEXT_START,
                        node_id=self.id,
                        metadata={
                            "block_id": event_dict.get("blockId"),
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "anthropic",
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "text-delta":
                delta = event_dict.get("delta", "")
                full_response += delta
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TEXT_DELTA,
                        node_id=self.id,
                        delta=delta,  # AI SDK field
                        metadata={
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "anthropic",
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "text-end":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TEXT_END,
                        node_id=self.id,
                        metadata={
                            "block_id": event_dict.get("blockId"),
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "anthropic",
                        },
                        raw_event=event_dict,
                    )
                )

            # Handle reasoning events (Extended Thinking, AI SDK format)
            elif event_type == "reasoning-start":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.REASONING_START,
                        node_id=self.id,
                        metadata={
                            "block_id": event_dict.get("blockId"),
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "anthropic",
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "reasoning-delta":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.REASONING_DELTA,
                        node_id=self.id,
                        delta=event_dict.get("delta", ""),  # AI SDK field
                        metadata={
                            "block_id": event_dict.get("blockId"),
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "anthropic",
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "reasoning-end":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.REASONING_END,
                        node_id=self.id,
                        metadata={
                            "block_id": event_dict.get("blockId"),
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "anthropic",
                        },
                        raw_event=event_dict,
                    )
                )

        # Build chat history entry
        chat_history = [
            {"role": "user", "content": anthropic_messages[-1]["content"]},
            {"role": "assistant", "content": full_response},
        ]

        # Build metadata with actual usage data
        metadata = {
            "model": self.model,
            "provider": self.provider,
        }
        if usage_data:
            metadata["usage"] = usage_data

        return NodeResult(
            output={"content": full_response},
            chat_history=chat_history,
            metadata=metadata,
        )

    async def _execute_gemini(
        self,
        messages: List[Dict[str, str]],
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute Gemini completion with streaming using Vel translator.

        Args:
            messages: List of chat messages
            context: Execution context

        Returns:
            NodeResult with response
        """
        try:
            from vel import get_gemini_translator
            from mesh.utils.translator_orchestrator import SimpleTranslatorOrchestrator
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}. "
                "Install with: pip install vel google-generativeai"
            )

        full_response = ""
        usage_data = None

        # Initialize translator and orchestrator
        translator = get_gemini_translator()
        orchestrator = SimpleTranslatorOrchestrator(translator)

        # Convert messages to Gemini chat history format
        # Gemini only supports 'user' and 'model' roles
        chat_history = []
        for msg in messages[:-1]:  # All except last message
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
                continue
            gemini_role = "model" if role == "assistant" else "user"
            chat_history.append({"role": gemini_role, "parts": [content]})

        # Last message is the prompt
        user_message = messages[-1]["content"]

        # Start chat with history
        chat = self._client.start_chat(history=chat_history)

        # Create async generator from Gemini chunks
        async def chunk_stream():
            response = await chat.send_message_async(
                user_message,
                stream=True,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
                if self.max_tokens
                else {"temperature": self.temperature},
            )
            async for chunk in response:
                yield chunk

        # Stream through orchestrator (adds start/finish events)
        async for event_dict in orchestrator.stream(chunk_stream()):
            event_type = event_dict.get("type")

            # Handle orchestration events
            if event_type == "start":
                # Generation started - emit as NODE_START with raw_event
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.NODE_START,
                        node_id=self.id,
                        metadata={
                            "node_type": "llm",
                            "provider": "gemini",
                            "model": self.model,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "start-step":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.START_STEP,
                        node_id=self.id,
                        metadata={
                            "step_index": 0,
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "gemini",
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "finish-step":
                usage_data = event_dict.get("usage")
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.FINISH_STEP,
                        node_id=self.id,
                        metadata={
                            "step_index": 0,
                            "finish_reason": event_dict.get("finishReason"),
                            "usage": usage_data,
                            "response": event_dict.get("response"),
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "gemini",
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "finish":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.FINISH,
                        node_id=self.id,
                        metadata={
                            "finish_reason": event_dict.get("finishReason"),
                            "total_usage": event_dict.get("totalUsage"),
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "gemini",
                        },
                        raw_event=event_dict,
                    )
                )

            # Handle content events
            elif event_type == "text-start":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TEXT_START,
                        node_id=self.id,
                        metadata={
                            "block_id": event_dict.get("blockId"),
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "gemini",
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "text-delta":
                delta = event_dict.get("delta", "")
                full_response += delta
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TEXT_DELTA,
                        node_id=self.id,
                        delta=delta,
                        metadata={
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "gemini",
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "text-end":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TEXT_END,
                        node_id=self.id,
                        metadata={
                            "block_id": event_dict.get("blockId"),
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "gemini",
                        },
                        raw_event=event_dict,
                    )
                )

            # Handle grounding sources (Gemini-specific)
            elif event_type == "source":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.SOURCE,
                        node_id=self.id,
                        metadata={
                            "url": event_dict.get("url"),
                            "title": event_dict.get("title"),
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "gemini",
                        },
                        raw_event=event_dict,
                    )
                )

            # Handle file attachments (multi-modal)
            elif event_type == "file":
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.FILE,
                        node_id=self.id,
                        metadata={
                            "name": event_dict.get("name"),
                            "mime_type": event_dict.get("mimeType"),
                            "data": event_dict.get("data"),
                            "node_type": "llm",
                            "model": self.model,
                            "provider": "gemini",
                        },
                        raw_event=event_dict,
                    )
                )

        # Build chat history entry
        chat_history_entry = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": full_response},
        ]

        # Build metadata with actual usage data
        metadata = {
            "model": self.model,
            "provider": self.provider,
        }
        if usage_data:
            metadata["usage"] = usage_data

        return NodeResult(
            output={"content": full_response},
            chat_history=chat_history_entry,
            metadata=metadata,
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
