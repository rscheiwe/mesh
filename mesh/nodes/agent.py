"""Agent node for wrapping Vel and OpenAI Agents SDK.

This node provides a unified interface for executing agents from different
frameworks. It auto-detects whether an agent is from Vel SDK or OpenAI
Agents SDK and handles streaming appropriately.
"""

from typing import Any, Dict, Optional, Union
import os

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext
from mesh.core.events import ExecutionEvent, EventType, VelEventTranslator, OpenAIEventTranslator
from mesh.utils.variables import VariableResolver
from mesh.utils.translator_orchestrator import TranslatorOrchestrator


class AgentNode(BaseNode):
    """Wrap Vel or OpenAI Agents SDK agents for execution.

    This node automatically detects the agent type (Vel vs OpenAI) and
    handles streaming token-by-token with provider-agnostic events.

    Responsibilities:
    - Detect agent type (Vel vs OpenAI)
    - Stream token-by-token
    - Translate events to Mesh format
    - Inject context into agent runtime

    Example:
        >>> from vel import Agent as VelAgent
        >>> vel_agent = VelAgent(name="assistant", model="gpt-4")
        >>>
        >>> agent_node = AgentNode(
        ...     id="my_agent",
        ...     agent=vel_agent,
        ... )
    """

    def __init__(
        self,
        id: str,
        agent: Any,
        system_prompt: Optional[str] = None,
        use_native_events: bool = False,
        config: Dict[str, Any] = None,
    ):
        """Initialize agent node.

        Args:
            id: Node identifier
            agent: Agent instance (Vel or OpenAI Agents SDK)
            system_prompt: Optional system prompt override
            use_native_events: If True, use provider's native events. If False (default),
                             use Vel's translated events for consistent event handling
            config: Additional configuration
        """
        super().__init__(id, config or {})
        self.agent = agent
        self.system_prompt = system_prompt
        self.use_native_events = use_native_events
        self.agent_type = self._detect_agent_type(agent)

        # Initialize Vel SDK translator if available and not using native events
        self.vel_sdk_translator = None
        if not use_native_events and self.agent_type != "vel":
            # Try to use Vel's SDK event translator for non-Vel agents
            try:
                # Map agent type to SDK translator
                if self.agent_type == "openai":
                    from vel import get_openai_agents_translator
                    self.vel_sdk_translator = get_openai_agents_translator()
                # Add other SDK translators as they become available
            except ImportError:
                # Vel not available, use native events
                self.use_native_events = True

        # Initialize legacy event translators (for backwards compatibility)
        if self.agent_type == "vel":
            self.event_translator = VelEventTranslator()
        else:
            self.event_translator = OpenAIEventTranslator()

    def _detect_agent_type(self, agent: Any) -> str:
        """Determine if agent is Vel or OpenAI Agents SDK.

        Args:
            agent: Agent instance

        Returns:
            "vel" or "openai"

        Raises:
            ValueError: If agent type cannot be determined
        """
        agent_class_name = agent.__class__.__name__
        agent_module = agent.__class__.__module__

        # Check for Vel agent
        if "vel" in agent_module.lower():
            return "vel"

        # Check for OpenAI Agents SDK (from 'agents' package)
        if "agents" in agent_module.lower() and hasattr(agent, "name") and hasattr(agent, "instructions"):
            return "openai"

        # Check for method signatures (Vel pattern)
        if hasattr(agent, "stream") or hasattr(agent, "run_stream"):
            return "vel"

        # Fallback: raise error
        raise ValueError(
            f"Unsupported agent type: {agent_class_name} from {agent_module}. "
            "Supported types: Vel Agent (from 'vel' module), OpenAI Agents SDK (from 'agents' module)"
        )

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute agent with streaming.

        Args:
            input: Input message
            context: Execution context

        Returns:
            NodeResult with agent response
        """
        # Extract message from input
        message = self._extract_message(input)

        # Resolve system prompt if provided
        if self.system_prompt:
            resolver = VariableResolver(context)
            resolved_prompt = await resolver.resolve(self.system_prompt)
            # Update agent system prompt if possible
            self._update_system_prompt(resolved_prompt)

        # Execute based on agent type
        if self.agent_type == "vel":
            return await self._execute_vel_agent(message, context)
        elif self.agent_type == "openai":
            return await self._execute_openai_agent(message, context)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

    async def _execute_vel_agent(
        self,
        message: str,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute Vel agent with native event streaming.

        Vel's API: run_stream(input: Dict, session_id: str) -> AsyncGenerator[Dict]

        Args:
            message: Input message
            context: Execution context

        Returns:
            NodeResult with agent response
        """
        full_response = ""
        chat_history = []

        try:
            # Vel expects input as a Dict
            input_data = {"message": message}

            # Call run_stream with session_id
            event_stream = self.agent.run_stream(
                input=input_data,
                session_id=context.session_id,
            )

            async for event in event_stream:
                # Emit token events
                from mesh.core.events import ExecutionEvent, EventType

                # Handle Vel stream protocol events
                if isinstance(event, dict):
                    event_type = event.get("type", "")

                    if event_type == "start":
                        # Message generation begins
                        await context.emit_event(
                            ExecutionEvent(
                                type=EventType.MESSAGE_START,
                                node_id=self.id,
                                metadata={"message_id": event.get("messageId")},
                            )
                        )

                    elif event_type == "text-start":
                        # Text block starts
                        await context.emit_event(
                            ExecutionEvent(
                                type=EventType.MESSAGE_START,
                                node_id=self.id,
                                metadata={"text_block_id": event.get("id")},
                            )
                        )

                    elif event_type == "text-delta":
                        # Token streaming
                        delta = event.get("delta", "")
                        if delta:
                            full_response += delta
                            await context.emit_event(
                                ExecutionEvent(
                                    type=EventType.TOKEN,
                                    node_id=self.id,
                                    content=delta,
                                    metadata={"text_block_id": event.get("id")},
                                )
                            )

                    elif event_type == "text-end":
                        # Text block completes
                        await context.emit_event(
                            ExecutionEvent(
                                type=EventType.MESSAGE_COMPLETE,
                                node_id=self.id,
                                metadata={"text_block_id": event.get("id")},
                            )
                        )

                    elif event_type == "tool-input-start":
                        # Tool call begins
                        await context.emit_event(
                            ExecutionEvent(
                                type=EventType.TOOL_CALL_START,
                                node_id=self.id,
                                metadata={
                                    "tool_call_id": event.get("toolCallId"),
                                    "tool_name": event.get("toolName"),
                                },
                            )
                        )

                    elif event_type == "tool-input-delta":
                        # Tool argument chunk streaming
                        delta = event.get("inputTextDelta", "")
                        if delta:
                            await context.emit_event(
                                ExecutionEvent(
                                    type=EventType.TOKEN,
                                    node_id=self.id,
                                    content=delta,
                                    metadata={
                                        "tool_call_id": event.get("toolCallId"),
                                        "event_subtype": "tool_input",
                                    },
                                )
                            )

                    elif event_type == "tool-input-available":
                        # Tool arguments complete and ready
                        await context.emit_event(
                            ExecutionEvent(
                                type=EventType.TOOL_CALL_START,
                                node_id=self.id,
                                metadata={
                                    "tool_call_id": event.get("toolCallId"),
                                    "tool_name": event.get("toolName"),
                                    "input": event.get("input"),
                                },
                            )
                        )

                    elif event_type == "tool-output-available":
                        # Tool execution result
                        await context.emit_event(
                            ExecutionEvent(
                                type=EventType.TOOL_CALL_COMPLETE,
                                node_id=self.id,
                                output=event.get("output"),
                                metadata={
                                    "tool_call_id": event.get("toolCallId"),
                                },
                            )
                        )

                    elif event_type == "finish-message":
                        # Message generation complete
                        finish_reason = event.get("finishReason", "unknown")
                        await context.emit_event(
                            ExecutionEvent(
                                type=EventType.MESSAGE_COMPLETE,
                                node_id=self.id,
                                metadata={"finish_reason": finish_reason},
                            )
                        )

                    elif event_type == "error":
                        # Error occurred
                        error_msg = event.get("error", "Unknown error")
                        await context.emit_event(
                            ExecutionEvent(
                                type=EventType.NODE_ERROR,
                                node_id=self.id,
                                error=error_msg,
                            )
                        )
                        raise RuntimeError(f"Vel agent error: {error_msg}")

                    else:
                        # Unknown event type - log for debugging
                        pass

                elif hasattr(event, "content"):
                    # Legacy fallback for non-dict events
                    content = event.content
                    if content:
                        full_response += content
                        await context.emit_event(
                            ExecutionEvent(
                                type=EventType.TOKEN,
                                node_id=self.id,
                                content=content,
                            )
                        )

                elif hasattr(event, "delta"):
                    content = event.delta
                    if content:
                        full_response += content
                        await context.emit_event(
                            ExecutionEvent(
                                type=EventType.TOKEN,
                                node_id=self.id,
                                content=content,
                            )
                        )

                elif isinstance(event, str):
                    full_response += event
                    await context.emit_event(
                        ExecutionEvent(
                            type=EventType.TOKEN,
                            node_id=self.id,
                            content=event,
                        )
                    )

        except Exception as e:
            raise RuntimeError(f"Vel agent execution failed: {str(e)}") from e

        # Build chat history
        chat_history = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": full_response},
        ]

        return NodeResult(
            output={"content": full_response},
            chat_history=chat_history,
            metadata={
                "agent_type": "vel",
                "agent_id": getattr(self.agent, "id", "unknown"),
            },
        )

    async def _execute_openai_agent(
        self,
        message: str,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute OpenAI Agents SDK agent with streaming.

        If vel_sdk_translator is available, uses Vel's SDK event translator to
        convert native OpenAI Agents SDK events to Vel's standardized format.
        Otherwise, uses native OpenAI Agents SDK events directly.

        Args:
            message: Input message
            context: Execution context

        Returns:
            NodeResult with agent response
        """
        # If Vel SDK translator is available, use it for standardized events
        if self.vel_sdk_translator:
            return await self._execute_openai_with_vel_translation(message, context)

        # Otherwise use native OpenAI Agents SDK events
        try:
            from agents import Runner
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: pip install openai-agents"
            )

        full_response = ""

        # Execute agent with streaming
        result = Runner.run_streamed(
            self.agent,
            message,
        )

        # Stream native events
        async for event in result.stream_events():
            from mesh.core.events import ExecutionEvent, EventType

            # Handle raw response events (token-by-token)
            if event.type == "raw_response_event":
                # Token streaming
                delta = getattr(event.data, "delta", "")
                if delta:
                    full_response += delta
                    await context.emit_event(
                        ExecutionEvent(
                            type=EventType.TOKEN,
                            node_id=self.id,
                            content=delta,
                        )
                    )

            # Handle run item stream events (higher-level updates)
            elif event.type == "run_item_stream_event":
                item = event.item

                # Message output items
                if hasattr(item, "type") and item.type == "message_output_item":
                    # Message generation started/completed
                    if hasattr(item, "status"):
                        if item.status == "completed":
                            await context.emit_event(
                                ExecutionEvent(
                                    type=EventType.MESSAGE_COMPLETE,
                                    node_id=self.id,
                                    metadata={"item_id": getattr(item, "id", None)},
                                )
                            )

                # Tool call items
                elif hasattr(item, "type") and "tool" in item.type.lower():
                    tool_name = getattr(item, "name", "unknown")

                    if hasattr(item, "status"):
                        if item.status == "in_progress":
                            await context.emit_event(
                                ExecutionEvent(
                                    type=EventType.TOOL_CALL_START,
                                    node_id=self.id,
                                    metadata={
                                        "tool_name": tool_name,
                                        "item_id": getattr(item, "id", None),
                                    },
                                )
                            )
                        elif item.status == "completed":
                            await context.emit_event(
                                ExecutionEvent(
                                    type=EventType.TOOL_CALL_COMPLETE,
                                    node_id=self.id,
                                    output=getattr(item, "output", None),
                                    metadata={
                                        "tool_name": tool_name,
                                        "item_id": getattr(item, "id", None),
                                    },
                                )
                            )

            # Handle agent updated events
            elif event.type == "agent_updated_stream_event":
                # Agent state changed
                pass

        # If no streaming content was captured, try to get final output
        if not full_response and hasattr(result, "final_output"):
            full_response = str(result.final_output)

        return NodeResult(
            output={"content": full_response},
            chat_history=[
                {"role": "user", "content": message},
                {"role": "assistant", "content": full_response},
            ],
            metadata={
                "agent_type": "openai",
                "agent_name": self.agent.name,
                "event_translation": "native",
            },
        )

    async def _execute_openai_with_vel_translation(
        self,
        message: str,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute OpenAI Agents SDK agent with Vel event translation and orchestration.

        This uses the actual OpenAI Agents SDK (Runner.run_streamed) and translates
        its native events to Vel's standardized stream protocol format using Vel's
        SDK event translator. The TranslatorOrchestrator fills event gaps (start-step,
        finish-step) to properly track multi-step execution.

        Args:
            message: Input message
            context: Execution context

        Returns:
            NodeResult with agent response
        """
        from mesh.core.events import ExecutionEvent, EventType

        try:
            from agents import Runner
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: pip install openai-agents"
            )

        full_response = ""
        total_usage = {}
        steps_completed = 0

        # Execute agent with native SDK
        result = Runner.run_streamed(
            self.agent,
            message,
        )

        # Wrap with orchestrator to fill event gaps
        orchestrator = TranslatorOrchestrator(self.vel_sdk_translator, max_steps=10)

        # Stream orchestrated events (includes start-step, finish-step, etc.)
        async for event_dict in orchestrator.stream(
            result.stream_events(),
            emit_start=False,  # Mesh executor already emits NODE_START
        ):
            event_type = event_dict.get("type", "")

            # Handle orchestration events (step boundaries)
            if event_type == "start":
                # Execution start (orchestrator emits this)
                # We don't need to emit anything - executor handles this
                pass

            elif event_type == "start-step":
                # Step begins
                step_index = event_dict.get("stepIndex", 0)
                await context.emit_event(
                    ExecutionEvent(
                        type=EventType.STEP_START,
                        node_id=self.id,
                        metadata={"step_index": step_index},
                    )
                )

            elif event_type == "finish-step":
                # Step completes with metadata
                step_index = event_dict.get("stepIndex", 0)
                finish_reason = event_dict.get("finishReason", "stop")
                usage = event_dict.get("usage")
                response_meta = event_dict.get("response")

                await context.emit_event(
                    ExecutionEvent(
                        type=EventType.STEP_COMPLETE,
                        node_id=self.id,
                        metadata={
                            "step_index": step_index,
                            "finish_reason": finish_reason,
                            "usage": usage,
                            "response": response_meta,
                            "had_tool_calls": event_dict.get("hadToolCalls", False),
                        },
                    )
                )
                steps_completed = step_index + 1

            elif event_type == "finish":
                # Overall execution complete
                total_usage = event_dict.get("totalUsage", {})
                # We don't emit anything here - executor will emit NODE_COMPLETE

            # Handle content events
            elif event_type == "text-start":
                # Text block starts
                await context.emit_event(
                    ExecutionEvent(
                        type=EventType.MESSAGE_START,
                        node_id=self.id,
                        metadata={"text_block_id": event_dict.get("id")},
                    )
                )

            elif event_type == "text-delta":
                # Token streaming
                delta = event_dict.get("delta", "")
                if delta:
                    full_response += delta
                    await context.emit_event(
                        ExecutionEvent(
                            type=EventType.TOKEN,
                            node_id=self.id,
                            content=delta,
                            metadata={"text_block_id": event_dict.get("id")},
                        )
                    )

            elif event_type == "text-end":
                # Text block completes
                await context.emit_event(
                    ExecutionEvent(
                        type=EventType.MESSAGE_COMPLETE,
                        node_id=self.id,
                        metadata={"text_block_id": event_dict.get("id")},
                    )
                )

            elif event_type == "tool-input-start":
                # Tool call begins
                await context.emit_event(
                    ExecutionEvent(
                        type=EventType.TOOL_CALL_START,
                        node_id=self.id,
                        metadata={
                            "tool_call_id": event_dict.get("toolCallId"),
                            "tool_name": event_dict.get("toolName"),
                        },
                    )
                )

            elif event_type == "tool-input-delta":
                # Tool argument chunk streaming
                delta = event_dict.get("inputTextDelta", "")
                if delta:
                    await context.emit_event(
                        ExecutionEvent(
                            type=EventType.TOKEN,
                            node_id=self.id,
                            content=delta,
                            metadata={
                                "tool_call_id": event_dict.get("toolCallId"),
                                "event_subtype": "tool_input",
                            },
                        )
                    )

            elif event_type == "tool-input-available":
                # Tool arguments complete and ready
                await context.emit_event(
                    ExecutionEvent(
                        type=EventType.TOOL_CALL_START,
                        node_id=self.id,
                        metadata={
                            "tool_call_id": event_dict.get("toolCallId"),
                            "tool_name": event_dict.get("toolName"),
                            "input": event_dict.get("input"),
                        },
                    )
                )

            elif event_type == "tool-output-available":
                # Tool execution result
                await context.emit_event(
                    ExecutionEvent(
                        type=EventType.TOOL_CALL_COMPLETE,
                        node_id=self.id,
                        output=event_dict.get("output"),
                        metadata={
                            "tool_call_id": event_dict.get("toolCallId"),
                        },
                    )
                )

            elif event_type == "error":
                # Error occurred
                error_msg = event_dict.get("error", "Unknown error")
                await context.emit_event(
                    ExecutionEvent(
                        type=EventType.NODE_ERROR,
                        node_id=self.id,
                        error=error_msg,
                    )
                )
                raise RuntimeError(f"OpenAI agent error: {error_msg}")

        # If no streaming content was captured, try to get final output
        if not full_response and hasattr(result, "final_output"):
            full_response = str(result.final_output)

        return NodeResult(
            output={"content": full_response},
            chat_history=[
                {"role": "user", "content": message},
                {"role": "assistant", "content": full_response},
            ],
            metadata={
                "agent_type": "openai",
                "agent_name": self.agent.name,
                "event_translation": "vel_orchestrated",
                "steps_completed": steps_completed,
                "total_usage": total_usage,
            },
        )

    def _extract_message(self, input: Any) -> str:
        """Extract message from input.

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
            return str(input)
        else:
            return str(input)

    def _update_system_prompt(self, prompt: str) -> None:
        """Update agent's system prompt if possible.

        Args:
            prompt: New system prompt
        """
        if self.agent_type == "vel":
            if hasattr(self.agent, "system_prompt"):
                self.agent.system_prompt = prompt
        elif self.agent_type == "openai":
            if hasattr(self.agent, "instructions"):
                self.agent.instructions = prompt

    def __repr__(self) -> str:
        return f"AgentNode(id='{self.id}', type='{self.agent_type}')"
