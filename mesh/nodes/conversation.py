"""Conversation node for multi-turn parameter extraction.

This node extends AgentNode to support multi-turn conversations that pause
execution until extraction conditions are met (e.g., all required parameters
have been collected from the user).

Similar to ApprovalNode but instead of a single approve/reject decision,
it loops with the user until structured output is complete.
"""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel

from mesh.nodes.agent import AgentNode
from mesh.nodes.base import NodeResult
from mesh.core.state import ExecutionContext
from mesh.core.events import ExecutionEvent, EventType


class ConversationNode(AgentNode):
    """Multi-turn conversational agent that pauses until extraction is complete.

    This node wraps a Vel agent and adds conversation loop logic:
    1. Agent responds to user input (may ask clarifying questions)
    2. If extraction is not complete, returns conversation_pending=True
    3. Executor pauses and waits for user's next message
    4. Resume with user's reply, agent continues conversation
    5. Repeat until agent returns valid structured output
    6. When complete, returns extracted parameters as output

    The extraction is considered complete when:
    - Agent calls the special `submit_extraction` tool with valid data, OR
    - Agent's response contains valid JSON matching the output_schema

    Config:
        agent: Vel agent instance (with optional tools like lookup_account)
        output_schema: Pydantic model defining required extraction fields
        extraction_prompt: System prompt with extraction instructions
        max_turns: Maximum conversation turns (default: 10)
        conversation_id: Unique identifier for this conversation point

    Example:
        >>> from pydantic import BaseModel
        >>> from vel import Agent as VelAgent
        >>>
        >>> class RevenueParams(BaseModel):
        ...     account_id: str
        ...     period: str  # Q1, Q2, Q3, Q4
        ...     year: int
        >>>
        >>> agent = VelAgent(
        ...     model={"provider": "openai", "model": "gpt-4o"},
        ...     instruction="Extract account and time period for revenue investigation."
        ... )
        >>>
        >>> conversation_node = ConversationNode(
        ...     id="param_extractor",
        ...     agent=agent,
        ...     output_schema=RevenueParams,
        ...     max_turns=10,
        ... )

    Graph Usage:
        START → ConversationNode → DataHandler1 → DataHandler2 → AnalysisAgent → END

        The ConversationNode will pause/resume until params are extracted,
        then flow continues with extracted values available as:
        {{param_extractor.account_id}}, {{param_extractor.period}}, etc.
    """

    def __init__(
        self,
        id: str,
        agent: Any,
        output_schema: Optional[Type[BaseModel]] = None,
        extraction_fields: Optional[List[str]] = None,
        max_turns: int = 10,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        event_mode: str = "full",
        config: Dict[str, Any] = None,
    ):
        """Initialize conversation node.

        Args:
            id: Node identifier
            agent: Vel agent instance (can have tools attached)
            output_schema: Pydantic model for structured output validation.
                          When agent outputs valid JSON matching this schema,
                          extraction is considered complete.
            extraction_fields: Alternative to output_schema - list of field names
                              that must all be present in agent's JSON output.
                              Use this for simple cases without Pydantic.
            max_turns: Maximum conversation turns before forced completion
            conversation_id: Unique ID for this conversation (defaults to node id)
            system_prompt: Override system prompt (usually set on agent)
            event_mode: Event emission mode
            config: Additional configuration
        """
        # Initialize parent AgentNode
        super().__init__(
            id=id,
            agent=agent,
            system_prompt=system_prompt,
            event_mode=event_mode,
            streaming=True,  # Always stream for conversations
            config=config or {},
        )

        self.output_schema = output_schema
        self.extraction_fields = extraction_fields or []
        self.max_turns = max_turns
        self.conversation_id = conversation_id or f"conversation_{id}"

        # If output_schema provided, set it on the agent for structured output
        if output_schema and hasattr(agent, 'output_type'):
            # Note: We don't set output_type on agent here because we want
            # the agent to be able to respond conversationally first.
            # We'll check for structured output in the response manually.
            pass

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute conversation turn and check for extraction completion.

        Uses stateless messages array pattern - we manage history ourselves
        and pass the full conversation to Vel each time.

        Args:
            input: User message or initial input
            context: Execution context (contains conversation state)

        Returns:
            NodeResult with either:
            - conversation_pending=True (via approval_pending) if more turns needed
            - Extracted parameters as output if complete
        """
        # Get conversation history - prefer messages from client (stateless pattern)
        # The client manages history and passes the full array with each request
        history_key = f"_conversation_history_{self.id}"

        # Check if input contains messages array from client
        if isinstance(input, dict) and "messages" in input and input["messages"]:
            # Client passed full history - use it directly
            conversation_history = list(input["messages"])  # Make a copy
            # Get user message (last user message or from explicit field)
            user_message = input.get("message") or input.get("user_message") or ""
            if not user_message:
                # Extract from last message if it's a user message
                for msg in reversed(conversation_history):
                    if msg.get("role") == "user":
                        user_message = msg.get("content", "")
                        break
        else:
            # Fallback to context.state for backwards compatibility
            conversation_history = context.state.get(history_key, [])

            # Extract the user message from input
            if isinstance(input, str):
                user_message = input
            elif isinstance(input, dict):
                user_message = input.get("message") or input.get("user_message") or input.get("content") or str(input)
            else:
                user_message = str(input)

            # Add user message to history (only if not already from client)
            if user_message:
                conversation_history.append({"role": "user", "content": user_message})

        # Get current turn count
        turn_count = len([m for m in conversation_history if m.get("role") == "user"])

        # Check max turns
        if turn_count > self.max_turns:
            return NodeResult(
                output={
                    "error": "max_turns_exceeded",
                    "message": f"Conversation exceeded {self.max_turns} turns",
                    "partial_data": context.state.get(f"_conversation_data_{self.id}", {}),
                },
                state={
                    history_key: conversation_history,
                    f"_conversation_complete_{self.id}": True,
                },
                metadata={
                    "conversation_id": self.conversation_id,
                    "turn_count": turn_count,
                    "status": "max_turns_exceeded",
                },
            )

        # Execute agent with full message history (stateless pattern)
        agent_result = await self._execute_with_history(conversation_history, context)

        # Add assistant response to history
        assistant_response = agent_result.output
        if isinstance(assistant_response, str):
            conversation_history.append({"role": "assistant", "content": assistant_response})
        elif isinstance(assistant_response, dict) and "content" in assistant_response:
            conversation_history.append({"role": "assistant", "content": assistant_response["content"]})
        else:
            conversation_history.append({"role": "assistant", "content": str(assistant_response)})

        # Check if extraction is complete
        extracted_data = self._check_extraction_complete(agent_result.output, context)

        if extracted_data is not None:
            # Extraction complete! Return the structured data
            # Emit completion event
            if self.event_mode != "silent":
                await context.emit_event(
                    ExecutionEvent(
                        type=EventType.CUSTOM_DATA,
                        node_id=self.id,
                        content={
                            "conversation_complete": True,
                            "extracted_data": extracted_data,
                            "turn_count": turn_count,
                        },
                        metadata={
                            "data_type": "data-conversation-complete",
                            "conversation_id": self.conversation_id,
                        },
                    )
                )

            return NodeResult(
                output=extracted_data,
                state={
                    history_key: conversation_history,
                    f"_conversation_complete_{self.id}": True,
                    f"_conversation_data_{self.id}": extracted_data,
                },
                chat_history=conversation_history,
                metadata={
                    "conversation_id": self.conversation_id,
                    "turn_count": turn_count,
                    "status": "complete",
                    "agent_type": agent_result.metadata.get("agent_type") if agent_result.metadata else None,
                },
            )

        # Extraction not complete - pause for next user message
        # Emit pending event
        if self.event_mode != "silent":
            await context.emit_event(
                ExecutionEvent(
                    type=EventType.APPROVAL_PENDING,  # Reuse approval event type
                    node_id=self.id,
                    output=agent_result.output,
                    metadata={
                        "conversation_id": self.conversation_id,
                        "conversation_pending": True,
                        "turn_count": turn_count,
                        "max_turns": self.max_turns,
                        "node_type": "conversation",
                        "awaiting_user_input": True,
                    },
                )
            )

        # Return with approval_pending=True to pause execution
        # The executor will pause and wait for resume() call
        return NodeResult(
            output=agent_result.output,
            approval_pending=True,  # Reuse approval mechanism for pause
            approval_id=self.conversation_id,
            approval_data={
                "conversation_id": self.conversation_id,
                "conversation_pending": True,
                "turn_count": turn_count,
                "max_turns": self.max_turns,
                "agent_response": agent_result.output,
                "awaiting_user_input": True,
            },
            state={
                history_key: conversation_history,  # Persist conversation history
                f"_conversation_data_{self.id}": context.state.get(
                    f"_conversation_data_{self.id}", {}
                ),
            },
            chat_history=conversation_history,
            metadata={
                "conversation_id": self.conversation_id,
                "turn_count": turn_count,
                "status": "awaiting_input",
                "node_type": "conversation",
            },
        )

    async def _execute_with_history(
        self,
        messages: List[Dict[str, Any]],
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute agent with full message history (stateless pattern).

        Forwards ALL Vel events to properly support AI SDK useChat hook.
        This includes start-step, finish-step, text-start, text-end, etc.

        Args:
            messages: Full conversation history as messages array
            context: Execution context

        Returns:
            NodeResult with agent response
        """
        from mesh.core.events import ExecutionEvent, EventType

        full_response = ""

        # Emit node start event (data-node-start) - required for FE to track current streaming node
        if self.event_mode != "silent":
            await context.emit_event(
                ExecutionEvent(
                    type=EventType.NODE_START,
                    node_id=self.id,
                    metadata={
                        "node_type": "conversation",
                        "agent_id": f"{self.id}_agent",
                    },
                )
            )

        try:
            # Add system prompt to messages if not already present
            messages_with_system = messages.copy()
            has_system = any(m.get("role") == "system" for m in messages_with_system)
            if not has_system and self.system_prompt:
                messages_with_system.insert(0, {"role": "system", "content": self.system_prompt})

            # Call Vel agent with messages array (stateless pattern)
            event_stream = self.agent.run_stream({"messages": messages_with_system})

            async for event in event_stream:
                if isinstance(event, dict):
                    event_type = event.get("type", "")

                    # Forward ALL AI SDK events for proper useChat hook support
                    if event_type == "start-step":
                        # Step begins (AI SDK format) - REQUIRED for useChat
                        step_index = event.get("stepIndex", 0)
                        if self.event_mode == "full":
                            await context.emit_event(
                                ExecutionEvent(
                                    type=EventType.START_STEP,
                                    node_id=self.id,
                                    metadata={
                                        "step_index": step_index,
                                        "node_type": "conversation",
                                        "agent_id": f"{self.id}_agent",
                                    },
                                    raw_event=event,
                                )
                            )

                    elif event_type == "text-start":
                        # Text block starts (AI SDK format) - REQUIRED for useChat
                        if self.event_mode == "full":
                            await context.emit_event(
                                ExecutionEvent(
                                    type=EventType.TEXT_START,
                                    node_id=self.id,
                                    metadata={
                                        "text_block_id": event.get("id"),
                                        "node_type": "conversation",
                                        "agent_id": f"{self.id}_agent",
                                    },
                                    raw_event=event,
                                )
                            )

                    elif event_type == "text-delta":
                        delta = event.get("delta", "")
                        if delta:
                            full_response += delta
                            # Emit streaming event
                            if self.event_mode == "full":
                                await context.emit_event(
                                    ExecutionEvent(
                                        type=EventType.TEXT_DELTA,
                                        node_id=self.id,
                                        delta=delta,
                                        metadata={
                                            "text_block_id": event.get("id"),
                                            "node_type": "conversation",
                                            "agent_id": f"{self.id}_agent",
                                        },
                                        raw_event=event,
                                    )
                                )

                    elif event_type == "text-end":
                        # Text block completes (AI SDK format) - REQUIRED for useChat
                        if self.event_mode == "full":
                            await context.emit_event(
                                ExecutionEvent(
                                    type=EventType.TEXT_END,
                                    node_id=self.id,
                                    metadata={
                                        "text_block_id": event.get("id"),
                                        "node_type": "conversation",
                                        "agent_id": f"{self.id}_agent",
                                    },
                                    raw_event=event,
                                )
                            )

                    elif event_type == "finish-step":
                        # Step completes (AI SDK format) - REQUIRED for useChat
                        step_index = event.get("stepIndex", 0)
                        finish_reason = event.get("finishReason", "stop")
                        usage = event.get("usage")
                        if self.event_mode == "full":
                            await context.emit_event(
                                ExecutionEvent(
                                    type=EventType.FINISH_STEP,
                                    node_id=self.id,
                                    metadata={
                                        "step_index": step_index,
                                        "finish_reason": finish_reason,
                                        "usage": usage,
                                        "had_tool_calls": event.get("hadToolCalls", False),
                                        "node_type": "conversation",
                                        "agent_id": f"{self.id}_agent",
                                    },
                                    raw_event=event,
                                )
                            )

                    elif event_type == "finish":
                        # Overall generation complete (AI SDK format)
                        finish_reason = event.get("finishReason", "stop")
                        total_usage = event.get("totalUsage")
                        if self.event_mode == "full":
                            await context.emit_event(
                                ExecutionEvent(
                                    type=EventType.FINISH,
                                    node_id=self.id,
                                    metadata={
                                        "finish_reason": finish_reason,
                                        "total_usage": total_usage,
                                        "steps_completed": event.get("stepsCompleted"),
                                        "node_type": "conversation",
                                        "agent_id": f"{self.id}_agent",
                                    },
                                    raw_event=event,
                                )
                            )

                    elif event_type == "finish-message":
                        # Message generation complete (AI SDK format)
                        finish_reason = event.get("finishReason", "unknown")
                        if self.event_mode == "full":
                            await context.emit_event(
                                ExecutionEvent(
                                    type=EventType.FINISH_MESSAGE,
                                    node_id=self.id,
                                    metadata={
                                        "finish_reason": finish_reason,
                                        "node_type": "conversation",
                                        "agent_id": f"{self.id}_agent",
                                    },
                                    raw_event=event,
                                )
                            )

                    elif event_type == "error":
                        # Error occurred (AI SDK format)
                        error_msg = event.get("errorText", "Unknown error")
                        if self.event_mode == "full":
                            await context.emit_event(
                                ExecutionEvent(
                                    type=EventType.ERROR,
                                    node_id=self.id,
                                    error=error_msg,
                                    metadata={
                                        "node_type": "conversation",
                                        "agent_id": f"{self.id}_agent",
                                    },
                                    raw_event=event,
                                )
                            )
                        raise RuntimeError(f"Conversation agent error: {error_msg}")

                    elif event_type.startswith("data-"):
                        # Custom data events (passthrough)
                        if self.event_mode == "full":
                            await context.emit_event(
                                ExecutionEvent(
                                    type=EventType.CUSTOM_DATA,
                                    node_id=self.id,
                                    content=event.get("data"),
                                    metadata={
                                        "data_type": event_type,
                                        "transient": event.get("transient", False),
                                        "node_type": "conversation",
                                    },
                                    raw_event=event,
                                )
                            )

            # Emit node complete event
            if context.graph_metadata and self.event_mode != "silent":
                from mesh.core.events import create_mesh_node_complete_event
                is_final = self.id in context.graph_metadata.final_nodes
                is_intermediate = self.id in context.graph_metadata.intermediate_nodes

                output_preview = full_response[:100] + "..." if len(full_response) > 100 else full_response
                complete_event = create_mesh_node_complete_event(
                    node_id=self.id,
                    node_type="conversation",
                    is_final=is_final,
                    is_intermediate=is_intermediate,
                    output_preview=output_preview
                )
                await context.emit_event(complete_event)

            return NodeResult(
                output=full_response,
                metadata={"agent_type": "vel"},
            )

        except Exception as e:
            # Emit error event
            if context.graph_metadata and self.event_mode != "silent":
                from mesh.core.events import create_mesh_node_complete_event
                complete_event = create_mesh_node_complete_event(
                    node_id=self.id,
                    node_type="conversation",
                    is_final=False,
                    is_intermediate=False,
                    output_preview="ERROR"
                )
                await context.emit_event(complete_event)
            raise

    def _check_extraction_complete(
        self,
        agent_output: Any,
        context: ExecutionContext,
    ) -> Optional[Dict[str, Any]]:
        """Check if the agent's output contains complete extraction.

        Tries multiple strategies:
        1. Check if agent output is already a dict with all required fields
        2. Try to parse JSON from agent's text response
        3. Validate against output_schema if provided

        Args:
            agent_output: Output from agent execution
            context: Execution context

        Returns:
            Extracted data dict if complete, None if more conversation needed
        """
        import json

        # Extract the actual content - handle various output formats
        content = ""
        if isinstance(agent_output, dict):
            # Check if the dict itself has the required fields (structured output)
            if self._has_required_fields(agent_output):
                return self._validate_and_extract(agent_output)

            # Try to get content from various keys
            content = (
                agent_output.get("content") or
                agent_output.get("text") or
                agent_output.get("message") or
                agent_output.get("output") or
                ""
            )

            # If content is still empty, try stringifying the whole dict
            if not content and agent_output:
                content = json.dumps(agent_output)
        elif isinstance(agent_output, str):
            content = agent_output
        else:
            content = str(agent_output) if agent_output else ""

        # Try to find JSON in the response content
        if content:
            json_data = self._extract_json_from_text(content)
            if json_data and self._has_required_fields(json_data):
                return self._validate_and_extract(json_data)

        return None

    def _has_required_fields(self, data: Dict[str, Any]) -> bool:
        """Check if data has all required fields.

        Args:
            data: Data dictionary to check

        Returns:
            True if all required fields present and non-empty
        """
        if self.output_schema:
            # Get required fields from Pydantic schema
            try:
                if hasattr(self.output_schema, 'model_fields'):
                    # Pydantic v2
                    required_fields = [
                        name for name, field in self.output_schema.model_fields.items()
                        if field.is_required()
                    ]
                elif hasattr(self.output_schema, '__fields__'):
                    # Pydantic v1
                    required_fields = [
                        name for name, field in self.output_schema.__fields__.items()
                        if field.required
                    ]
                else:
                    required_fields = self.extraction_fields
            except Exception:
                required_fields = self.extraction_fields
        else:
            required_fields = self.extraction_fields

        # Check all required fields are present and non-empty
        for field in required_fields:
            if field not in data:
                return False
            value = data[field]
            if value is None or value == "" or value == []:
                return False

        return len(required_fields) > 0  # Must have at least one field

    def _validate_and_extract(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate data against schema and return extracted fields.

        Args:
            data: Data to validate

        Returns:
            Validated data dict or None if validation fails
        """
        if self.output_schema:
            try:
                # Validate with Pydantic
                if hasattr(self.output_schema, 'model_validate'):
                    # Pydantic v2
                    validated = self.output_schema.model_validate(data)
                    return validated.model_dump()
                elif hasattr(self.output_schema, 'parse_obj'):
                    # Pydantic v1
                    validated = self.output_schema.parse_obj(data)
                    return validated.dict()
            except Exception:
                # Validation failed
                return None

        # No schema - just return the data with required fields
        if self.extraction_fields:
            return {k: data[k] for k in self.extraction_fields if k in data}

        return data

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to extract JSON object from text.

        Handles:
        - Pure JSON
        - JSON in markdown code blocks
        - JSON embedded in text (including multi-line)

        Args:
            text: Text that may contain JSON

        Returns:
            Parsed JSON dict or None
        """
        import json
        import re

        if not text:
            return None

        text = text.strip()

        # Try direct parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code blocks (```json ... ``` or ``` ... ```)
        code_block_pattern = r'```(?:json)?\s*\n([\s\S]*?)\n\s*```'
        matches = re.findall(code_block_pattern, text)
        for match in matches:
            try:
                data = json.loads(match.strip())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue

        # Try to find JSON object in text using brace matching
        # Find the first { and try to find the matching }
        start_idx = text.find('{')
        while start_idx != -1:
            # Try to find matching closing brace
            brace_count = 0
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found matching brace
                        potential_json = text[start_idx:i+1]
                        try:
                            data = json.loads(potential_json)
                            if isinstance(data, dict):
                                return data
                        except json.JSONDecodeError:
                            pass
                        break
            # Try next opening brace
            start_idx = text.find('{', start_idx + 1)

        return None

    def __repr__(self) -> str:
        schema_name = self.output_schema.__name__ if self.output_schema else "None"
        return f"ConversationNode(id='{self.id}', schema={schema_name})"


# Convenience function for creating conversation result to continue
def continue_conversation(
    user_message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a result to continue a conversation.

    This is passed to executor.resume() to continue a conversation
    after the ConversationNode has paused.

    Args:
        user_message: The user's next message
        metadata: Additional metadata

    Returns:
        Dict suitable for resume() call

    Example:
        >>> result = continue_conversation("account 123 for Q3")
        >>> async for event in executor.resume(context, result):
        ...     print(event)
    """
    return {
        "approved": True,  # Always "approve" to continue
        "modified_data": {
            "user_message": user_message,
            "message": user_message,  # Alias
        },
        "metadata": metadata or {},
    }
