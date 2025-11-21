"""Agent node for wrapping Vel and OpenAI Agents SDK.

This node provides a unified interface for executing agents from different
frameworks. It auto-detects whether an agent is from Vel SDK or OpenAI
Agents SDK and handles streaming appropriately.
"""

from typing import Any, Dict, Optional, Union
import os

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext
from mesh.core.events import ExecutionEvent, EventType
from mesh.utils.variables import VariableResolver
from mesh.utils.translator_orchestrator import TranslatorOrchestrator
from mesh.utils.input_parser import (
    should_parse_input,
    detect_input_variables,
    parse_natural_language_input
)
from mesh.core.events import (
    create_mesh_node_start_event,
    create_mesh_node_complete_event,
    transform_event_for_transient_mode
)


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
        event_mode: str = "full",
        config: Dict[str, Any] = None,
    ):
        """Initialize agent node.

        Args:
            id: Node identifier
            agent: Agent instance (Vel or OpenAI Agents SDK)
            system_prompt: Optional system prompt override
            use_native_events: If True, use provider's native events. If False (default),
                             use Vel's translated events for consistent event handling
            event_mode: Event emission mode:
                - "full": All events (text-delta, tool-*, etc.) - streams to chat
                - "status_only": Only data-mesh-node-start/complete - progress indicators only
                - "transient_events": All events but prefixed with data-agent-node-* - render differently
                - "silent": No events to FE - invisible execution
            config: Additional configuration
        """
        super().__init__(id, config or {})
        self.agent = agent
        self.system_prompt = system_prompt
        self.use_native_events = use_native_events
        self.event_mode = event_mode
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

        # Note: Vel agents don't need translators - they already emit Vel protocol.
        # Only non-Vel agents (OpenAI Agents SDK, etc.) need translation.

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
        # Emit custom node start event if graph metadata available
        # Always emit for status_only and transient_events, not for silent
        if context.graph_metadata and self.event_mode != "silent":
            is_final = self.id in context.graph_metadata.final_nodes
            is_intermediate = self.id in context.graph_metadata.intermediate_nodes

            start_event = create_mesh_node_start_event(
                node_id=self.id,
                node_type="agent",
                is_final=is_final,
                is_intermediate=is_intermediate
            )
            await context.emit_event(start_event)

        try:
            # Extract message from input
            message = self._extract_message(input)

            # Auto-parse natural language input if system_prompt has multiple {{$input.X}} variables
            if self.system_prompt and should_parse_input(self.system_prompt, input):
                field_names = detect_input_variables(self.system_prompt)

                # Get model config for parser (use fast model for efficiency)
                parser_model_config = {
                    "provider": "openai",
                    "model": "gpt-4o-mini"  # Fast, cheap model for parsing
                }

                # Parse natural language into structured data
                parsed_data = await parse_natural_language_input(
                    message,
                    field_names,
                    parser_model_config
                )

                # Update the START node output in executed_data so VariableResolver can access it
                if context.executed_data and len(context.executed_data) > 0:
                    # Replace the START node output with parsed structured data
                    context.executed_data[0]["output"] = parsed_data

                # Also update input for downstream processing
                input = parsed_data

            # Resolve system prompt if provided
            if self.system_prompt:
                resolver = VariableResolver(context)
                resolved_prompt = await resolver.resolve(self.system_prompt)
                # Update agent system prompt if possible
                self._update_system_prompt(resolved_prompt)

            # Execute based on agent type
            if self.agent_type == "vel":
                result = await self._execute_vel_agent(message, context)
            elif self.agent_type == "openai":
                result = await self._execute_openai_agent(message, context)
            else:
                raise ValueError(f"Unknown agent type: {self.agent_type}")

            # Emit custom node complete event if graph metadata available
            # Always emit for status_only and transient_events, not for silent
            if context.graph_metadata and self.event_mode != "silent":
                is_final = self.id in context.graph_metadata.final_nodes
                is_intermediate = self.id in context.graph_metadata.intermediate_nodes

                # Get output preview (first 100 chars)
                output_preview = None
                if result.output:
                    output_str = str(result.output)
                    output_preview = output_str[:100] + "..." if len(output_str) > 100 else output_str

                complete_event = create_mesh_node_complete_event(
                    node_id=self.id,
                    node_type="agent",
                    is_final=is_final,
                    is_intermediate=is_intermediate,
                    output_preview=output_preview
                )
                await context.emit_event(complete_event)

            return result

        except Exception as e:
            # Emit completion event even on error
            # Always emit for status_only and transient_events, not for silent
            if context.graph_metadata and self.event_mode != "silent":
                is_final = self.id in context.graph_metadata.final_nodes
                is_intermediate = self.id in context.graph_metadata.intermediate_nodes

                complete_event = create_mesh_node_complete_event(
                    node_id=self.id,
                    node_type="agent",
                    is_final=is_final,
                    is_intermediate=is_intermediate,
                    output_preview="ERROR"
                )
                await context.emit_event(complete_event)

            raise

    async def _emit_event_if_enabled(self, context: ExecutionContext, event: "ExecutionEvent") -> None:
        """Emit event based on event_mode configuration.

        Args:
            context: Execution context
            event: Event to emit
        """
        if self.event_mode == "silent":
            # No events emitted
            return

        if self.event_mode == "status_only":
            # Only emit custom data-mesh-node-* events (handled separately)
            # Don't emit regular streaming events
            return

        if self.event_mode == "transient_events":
            # Transform ALL events to data-agent-node-* format
            transformed_event = transform_event_for_transient_mode(event, "agent")
            await context.emit_event(transformed_event)
        else:
            # event_mode == "full" - emit normal events
            await context.emit_event(event)

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
                        # Generation started - emit as NODE_START with metadata
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.NODE_START,
                                node_id=self.id,
                                metadata={
                                    "message_id": event.get("messageId"),
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "text-start":
                        # Text block starts (AI SDK format)
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.TEXT_START,
                                node_id=self.id,
                                metadata={
                                    "text_block_id": event.get("id"),
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "text-delta":
                        # Token streaming (AI SDK format)
                        delta = event.get("delta", "")
                        if delta:
                            full_response += delta
                            await self._emit_event_if_enabled(context,
                                ExecutionEvent(
                                    type=EventType.TEXT_DELTA,
                                    node_id=self.id,
                                    delta=delta,  # AI SDK field
                                    metadata={
                                        "text_block_id": event.get("id"),
                                        "node_type": "agent",
                                        "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                    },
                                    raw_event=event,
                                )
                            )

                    elif event_type == "text-end":
                        # Text block completes (AI SDK format)
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.TEXT_END,
                                node_id=self.id,
                                metadata={
                                    "text_block_id": event.get("id"),
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "tool-input-start":
                        # Tool call begins (AI SDK format)
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.TOOL_INPUT_START,
                                node_id=self.id,
                                metadata={
                                    "tool_call_id": event.get("toolCallId"),
                                    "tool_name": event.get("toolName"),
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "tool-input-delta":
                        # Tool argument chunk streaming (AI SDK format)
                        delta = event.get("inputTextDelta", "")
                        if delta:
                            await self._emit_event_if_enabled(context,
                                ExecutionEvent(
                                    type=EventType.TOOL_INPUT_DELTA,
                                    node_id=self.id,
                                    delta=delta,  # AI SDK field
                                    metadata={
                                        "tool_call_id": event.get("toolCallId"),
                                        "node_type": "agent",
                                        "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                    },
                                    raw_event=event,
                                )
                            )

                    elif event_type == "tool-input-available":
                        # Tool arguments complete and ready (AI SDK format)
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.TOOL_INPUT_AVAILABLE,
                                node_id=self.id,
                                metadata={
                                    "tool_call_id": event.get("toolCallId"),
                                    "tool_name": event.get("toolName"),
                                    "input": event.get("input"),
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "tool-output-available":
                        # Tool execution result (AI SDK format)
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.TOOL_OUTPUT_AVAILABLE,
                                node_id=self.id,
                                output=event.get("output"),
                                metadata={
                                    "tool_call_id": event.get("toolCallId"),
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "finish-message":
                        # Message generation complete (AI SDK format)
                        finish_reason = event.get("finishReason", "unknown")
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.FINISH_MESSAGE,
                                node_id=self.id,
                                metadata={
                                    "finish_reason": finish_reason,
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "error":
                        # Error occurred (AI SDK format)
                        error_msg = event.get("error", "Unknown error")
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.ERROR,
                                node_id=self.id,
                                error=error_msg,
                                metadata={
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )
                        raise RuntimeError(f"Vel agent error: {error_msg}")

                    elif event_type == "start-step":
                        # Step begins (multi-step execution, AI SDK format)
                        step_index = event.get("stepIndex", 0)
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.START_STEP,
                                node_id=self.id,
                                metadata={
                                    "step_index": step_index,
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "finish-step":
                        # Step completes with usage and metadata (AI SDK format)
                        step_index = event.get("stepIndex", 0)
                        finish_reason = event.get("finishReason", "stop")
                        usage = event.get("usage")
                        response_meta = event.get("response")

                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.FINISH_STEP,
                                node_id=self.id,
                                metadata={
                                    "step_index": step_index,
                                    "finish_reason": finish_reason,
                                    "usage": usage,
                                    "response": response_meta,
                                    "had_tool_calls": event.get("hadToolCalls", False),
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "finish":
                        # Overall generation complete (AI SDK format)
                        finish_reason = event.get("finishReason", "stop")
                        total_usage = event.get("totalUsage")

                        # Emit as finish event (AI SDK)
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.FINISH,
                                node_id=self.id,
                                metadata={
                                    "finish_reason": finish_reason,
                                    "total_usage": total_usage,
                                    "steps_completed": event.get("stepsCompleted"),
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "reasoning-start":
                        # Reasoning block starts (o1/o3/Claude Extended Thinking, AI SDK format)
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.REASONING_START,
                                node_id=self.id,
                                metadata={
                                    "block_id": event.get("id"),
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "reasoning-delta":
                        # Reasoning token streaming (AI SDK format)
                        delta = event.get("delta", "")
                        if delta:
                            await self._emit_event_if_enabled(context,
                                ExecutionEvent(
                                    type=EventType.REASONING_DELTA,
                                    node_id=self.id,
                                    delta=delta,  # AI SDK field
                                    metadata={
                                        "block_id": event.get("id"),
                                        "node_type": "agent",
                                        "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                    },
                                    raw_event=event,
                                )
                            )

                    elif event_type == "reasoning-end":
                        # Reasoning block completes (AI SDK format)
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.REASONING_END,
                                node_id=self.id,
                                metadata={
                                    "block_id": event.get("id"),
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "response-metadata":
                        # Usage statistics, model info, timing (AI SDK format)
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.RESPONSE_METADATA,
                                node_id=self.id,
                                metadata={
                                    "id": event.get("id"),
                                    "model_id": event.get("modelId"),
                                    "usage": event.get("usage"),
                                    "timestamp": event.get("timestamp"),
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "source":
                        # Citations and grounding (Gemini, AI SDK format)
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.SOURCE,
                                node_id=self.id,
                                metadata={
                                    "sources": event.get("sources", []),
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type == "file":
                        # File attachment (multi-modal, AI SDK format)
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.FILE,
                                node_id=self.id,
                                metadata={
                                    "name": event.get("name"),
                                    "mime_type": event.get("mimeType"),
                                    "content": event.get("content"),
                                    "node_type": "agent",
                                    "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                                },
                                raw_event=event,
                            )
                        )

                    elif event_type.startswith("data-"):
                        # Custom data events (passthrough)
                        # Includes: data-*, data-rlm-*, etc.
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.CUSTOM_DATA,
                                node_id=self.id,
                                content=event.get("data"),
                                metadata={
                                    "data_type": event_type,
                                    "transient": event.get("transient", False),
                                },
                            raw_event=event,
                            )
                        )

                    else:
                        # Unknown event type - log for debugging
                        pass

                elif hasattr(event, "content"):
                    # Legacy fallback for non-dict events
                    content = event.content
                    if content:
                        full_response += content
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.TOKEN,
                                node_id=self.id,
                                content=content,
                            raw_event=event,
                            )
                        )

                elif hasattr(event, "delta"):
                    content = event.delta
                    if content:
                        full_response += content
                        await self._emit_event_if_enabled(context,
                            ExecutionEvent(
                                type=EventType.TOKEN,
                                node_id=self.id,
                                content=content,
                            raw_event=event,
                            )
                        )

                elif isinstance(event, str):
                    full_response += event
                    await self._emit_event_if_enabled(context,
                        ExecutionEvent(
                            type=EventType.TOKEN,
                            node_id=self.id,
                            content=event,
                        raw_event=event,
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
                    await self._emit_event_if_enabled(context,
                        ExecutionEvent(
                            type=EventType.TOKEN,
                            node_id=self.id,
                            content=delta,
                        raw_event=event,
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
                            await self._emit_event_if_enabled(context,
                                ExecutionEvent(
                                    type=EventType.MESSAGE_COMPLETE,
                                    node_id=self.id,
                                    metadata={"item_id": getattr(item, "id", None)},
                                raw_event=event,
                                )
                            )

                # Tool call items
                elif hasattr(item, "type") and "tool" in item.type.lower():
                    tool_name = getattr(item, "name", "unknown")

                    if hasattr(item, "status"):
                        if item.status == "in_progress":
                            await self._emit_event_if_enabled(context,
                                ExecutionEvent(
                                    type=EventType.TOOL_CALL_START,
                                    node_id=self.id,
                                    metadata={
                                        "tool_name": tool_name,
                                        "item_id": getattr(item, "id", None),
                                    },
                                raw_event=event,
                                )
                            )
                        elif item.status == "completed":
                            await self._emit_event_if_enabled(context,
                                ExecutionEvent(
                                    type=EventType.TOOL_CALL_COMPLETE,
                                    node_id=self.id,
                                    output=getattr(item, "output", None),
                                    metadata={
                                        "tool_name": tool_name,
                                        "item_id": getattr(item, "id", None),
                                    },
                                raw_event=event,
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
                # Generation started - emit as NODE_START with metadata
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.NODE_START,
                        node_id=self.id,
                        metadata={
                            "message_id": event_dict.get("messageId"),
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "start-step":
                # Step begins (AI SDK format)
                step_index = event_dict.get("stepIndex", 0)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.START_STEP,
                        node_id=self.id,
                        metadata={
                            "step_index": step_index,
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "finish-step":
                # Step completes with metadata (AI SDK format)
                step_index = event_dict.get("stepIndex", 0)
                finish_reason = event_dict.get("finishReason", "stop")
                usage = event_dict.get("usage")
                response_meta = event_dict.get("response")

                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.FINISH_STEP,
                        node_id=self.id,
                        metadata={
                            "step_index": step_index,
                            "finish_reason": finish_reason,
                            "usage": usage,
                            "response": response_meta,
                            "had_tool_calls": event_dict.get("hadToolCalls", False),
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )
                steps_completed = step_index + 1

            elif event_type == "finish":
                # Overall execution complete (AI SDK format)
                total_usage = event_dict.get("totalUsage", {})
                finish_reason = event_dict.get("finishReason", "stop")

                # Emit finish event (AI SDK)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.FINISH,
                        node_id=self.id,
                        metadata={
                            "finish_reason": finish_reason,
                            "total_usage": total_usage,
                            "steps_completed": event_dict.get("stepsCompleted"),
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )

            # Handle content events
            elif event_type == "text-start":
                # Text block starts (AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TEXT_START,
                        node_id=self.id,
                        metadata={
                            "text_block_id": event_dict.get("id"),
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "text-delta":
                # Token streaming (AI SDK format)
                delta = event_dict.get("delta", "")
                if delta:
                    full_response += delta
                    await self._emit_event_if_enabled(context,
                        ExecutionEvent(
                            type=EventType.TEXT_DELTA,
                            node_id=self.id,
                            delta=delta,  # AI SDK field
                            metadata={
                                "text_block_id": event_dict.get("id"),
                                "node_type": "agent",
                                "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                            },
                            raw_event=event_dict,
                        )
                    )

            elif event_type == "text-end":
                # Text block completes (AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TEXT_END,
                        node_id=self.id,
                        metadata={
                            "text_block_id": event_dict.get("id"),
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "tool-input-start":
                # Tool call begins (AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TOOL_INPUT_START,
                        node_id=self.id,
                        metadata={
                            "tool_call_id": event_dict.get("toolCallId"),
                            "tool_name": event_dict.get("toolName"),
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "tool-input-delta":
                # Tool argument chunk streaming (AI SDK format)
                delta = event_dict.get("inputTextDelta", "")
                if delta:
                    await self._emit_event_if_enabled(context,
                        ExecutionEvent(
                            type=EventType.TOOL_INPUT_DELTA,
                            node_id=self.id,
                            delta=delta,  # AI SDK field
                            metadata={
                                "tool_call_id": event_dict.get("toolCallId"),
                                "node_type": "agent",
                                "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                            },
                            raw_event=event_dict,
                        )
                    )

            elif event_type == "tool-input-available":
                # Tool arguments complete and ready (AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TOOL_INPUT_AVAILABLE,
                        node_id=self.id,
                        metadata={
                            "tool_call_id": event_dict.get("toolCallId"),
                            "tool_name": event_dict.get("toolName"),
                            "input": event_dict.get("input"),
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "tool-output-available":
                # Tool execution result (AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.TOOL_OUTPUT_AVAILABLE,
                        node_id=self.id,
                        output=event_dict.get("output"),
                        metadata={
                            "tool_call_id": event_dict.get("toolCallId"),
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "error":
                # Error occurred (AI SDK format)
                error_msg = event_dict.get("error", "Unknown error")
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.ERROR,
                        node_id=self.id,
                        error=error_msg,
                        metadata={
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )
                raise RuntimeError(f"OpenAI agent error: {error_msg}")

            elif event_type == "reasoning-start":
                # Reasoning block starts (o1/o3/Claude Extended Thinking, AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.REASONING_START,
                        node_id=self.id,
                        metadata={
                            "block_id": event_dict.get("id"),
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "reasoning-delta":
                # Reasoning token streaming (AI SDK format)
                delta = event_dict.get("delta", "")
                if delta:
                    await self._emit_event_if_enabled(context,
                        ExecutionEvent(
                            type=EventType.REASONING_DELTA,
                            node_id=self.id,
                            delta=delta,  # AI SDK field
                            metadata={
                                "block_id": event_dict.get("id"),
                                "node_type": "agent",
                                "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                            },
                            raw_event=event_dict,
                        )
                    )

            elif event_type == "reasoning-end":
                # Reasoning block completes (AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.REASONING_END,
                        node_id=self.id,
                        metadata={
                            "block_id": event_dict.get("id"),
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "response-metadata":
                # Usage statistics, model info, timing (AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.RESPONSE_METADATA,
                        node_id=self.id,
                        metadata={
                            "id": event_dict.get("id"),
                            "model_id": event_dict.get("modelId"),
                            "usage": event_dict.get("usage"),
                            "timestamp": event_dict.get("timestamp"),
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "source":
                # Citations and grounding (Gemini, AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.SOURCE,
                        node_id=self.id,
                        metadata={
                            "sources": event_dict.get("sources", []),
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type == "file":
                # File attachment (multi-modal, AI SDK format)
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.FILE,
                        node_id=self.id,
                        metadata={
                            "name": event_dict.get("name"),
                            "mime_type": event_dict.get("mimeType"),
                            "content": event_dict.get("content"),
                            "node_type": "agent",
                            "agent_id": self.agent.id if hasattr(self.agent, 'id') else None,
                        },
                        raw_event=event_dict,
                    )
                )

            elif event_type.startswith("data-"):
                # Custom data events (passthrough)
                # Includes: data-*, data-rlm-*, etc.
                await self._emit_event_if_enabled(context,
                    ExecutionEvent(
                        type=EventType.CUSTOM_DATA,
                        node_id=self.id,
                        content=event_dict.get("data"),
                        metadata={
                            "data_type": event_type,
                            "transient": event_dict.get("transient", False),
                        },
                    raw_event=event_dict,
                    )
                )

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
            # Vel agents use 'instruction' parameter for system prompts
            self.agent.instruction = prompt
        elif self.agent_type == "openai":
            if hasattr(self.agent, "instructions"):
                self.agent.instructions = prompt

    def __repr__(self) -> str:
        return f"AgentNode(id='{self.id}', type='{self.agent_type}')"
