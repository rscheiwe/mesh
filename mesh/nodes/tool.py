"""Tool node for executing Python functions.

This node wraps arbitrary Python functions (sync or async) and executes them
as part of the graph workflow. It supports automatic parameter injection from
context and input data.

Output Behavior:
- If tool returns a dict: stored directly, access via {{node_id.field}}
- If tool returns a primitive: wrapped as {"output": value}, access via {{node_id.output}}

Examples:
    # Tool returns dict {"result": 3, "status": "ok"}
    # Access: {{my_tool.result}}, {{my_tool.status}}

    # Tool returns primitive "hello"
    # Access: {{my_tool.output}}
"""

from typing import Callable, Any, Dict, Optional
import inspect
import asyncio

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext
from mesh.core.events import ExecutionEvent, EventType, transform_event_for_transient_mode


class ToolNode(BaseNode):
    """Execute arbitrary Python functions as tools.

    This node can wrap any Python function (sync or async) and execute it
    as part of the graph. It automatically inspects the function signature
    and injects parameters from input data and context.

    Supported parameter names (auto-injected):
    - input: The input data
    - context: The ExecutionContext
    - state: The state dictionary
    - variables: The variables dictionary
    - Custom parameters from config["bindings"]

    Example:
        >>> def my_tool(input: str, multiplier: int = 2) -> str:
        ...     return input * multiplier
        >>>
        >>> tool = ToolNode(
        ...     id="repeat_tool",
        ...     tool_fn=my_tool,
        ...     config={"bindings": {"multiplier": 3}}
        ... )
    """

    def __init__(
        self,
        id: str,
        tool_fn: Callable,
        event_mode: str = "full",
        config: Dict[str, Any] = None,
    ):
        """Initialize tool node.

        Args:
            id: Node identifier
            tool_fn: Function to execute (sync or async)
            event_mode: Event emission mode (default: "full")
                - "full": All events - streams to chat
                - "status_only": Only progress indicators (tool-start/complete)
                - "transient_events": All events prefixed with data-tool-node-*
                - "silent": No events
            config: Configuration including parameter bindings
        """
        super().__init__(id, config or {})
        self.tool_fn = tool_fn
        self.event_mode = event_mode
        self.is_async = inspect.iscoroutinefunction(tool_fn)
        self.signature = inspect.signature(tool_fn)

        # Store function metadata
        self.function_name = tool_fn.__name__
        self.function_doc = inspect.getdoc(tool_fn) or ""

    def set_tool_function(self, tool_fn: Callable) -> None:
        """Inject tool function after node creation.

        This allows backends to load tools from databases and inject them
        after graph parsing but before execution.

        Args:
            tool_fn: Tool function to execute
        """
        self.tool_fn = tool_fn
        self.is_async = inspect.iscoroutinefunction(tool_fn)
        self.signature = inspect.signature(tool_fn)
        self.function_name = tool_fn.__name__
        self.function_doc = inspect.getdoc(tool_fn) or ""

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
            # Only emit if it's a custom data event (tool-start/complete)
            if event.type == EventType.CUSTOM_DATA:
                await context.emit_event(event)
            return

        # Transient events - transform all events with data-tool-node-* prefix
        if self.event_mode == "transient_events":
            transformed_event = transform_event_for_transient_mode(event, "tool")
            await context.emit_event(transformed_event)
        else:
            # Full mode - emit events normally
            await context.emit_event(event)

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute tool function.

        Args:
            input: Input data
            context: Execution context

        Returns:
            NodeResult with function output
        """
        # Emit start event
        await self._emit_event_if_enabled(
            context,
            ExecutionEvent(
                type=EventType.NODE_START,
                node_id=self.id,
                metadata={
                    "tool_name": self.function_name,
                    "node_type": "tool",
                },
            )
        )

        # Build kwargs from function signature
        kwargs = self._build_kwargs(input, context)

        # Execute function
        try:
            if self.is_async:
                result = await self.tool_fn(**kwargs)
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: self.tool_fn(**kwargs))
        except Exception as e:
            # Emit error event
            await self._emit_event_if_enabled(
                context,
                ExecutionEvent(
                    type=EventType.NODE_ERROR,
                    node_id=self.id,
                    error=str(e),
                    metadata={
                        "tool_name": self.function_name,
                        "node_type": "tool",
                    },
                )
            )
            raise RuntimeError(
                f"Tool function '{self.function_name}' failed: {str(e)}"
            ) from e

        # Return result directly for intuitive variable access
        # Users can access via {{tool_id.field}} (e.g., {{tool_id.result}})
        # If result is a dict, its keys are directly accessible
        # If result is a primitive, wrap it for consistent access as {{tool_id.output}}
        if isinstance(result, dict):
            output = result
        else:
            output = {"output": result}

        return NodeResult(
            output=output,
            metadata={
                "tool_name": self.function_name,
                "tool_doc": self.function_doc,
                "node_type": "tool",
            },
        )

    def _build_kwargs(self, input: Any, context: ExecutionContext) -> Dict[str, Any]:
        """Build function kwargs from input and context.

        Args:
            input: Input data
            context: Execution context

        Returns:
            Dictionary of kwargs for function call
        """
        kwargs = {}
        bindings = self.config.get("bindings", {})

        for param_name, param in self.signature.parameters.items():
            # Special parameter names
            if param_name == "input":
                kwargs["input"] = input
            elif param_name == "context":
                kwargs["context"] = context
            elif param_name == "state":
                kwargs["state"] = context.state
            elif param_name == "variables":
                kwargs["variables"] = context.variables
            elif param_name == "chat_history":
                kwargs["chat_history"] = context.chat_history
            # Custom bindings
            elif param_name in bindings:
                kwargs[param_name] = bindings[param_name]
            # Try to extract from input if it's a dict
            elif isinstance(input, dict) and param_name in input:
                kwargs[param_name] = input[param_name]
            # Use default if available
            elif param.default != inspect.Parameter.empty:
                kwargs[param_name] = param.default

        return kwargs

    def __repr__(self) -> str:
        return f"ToolNode(id='{self.id}', tool='{self.function_name}')"
