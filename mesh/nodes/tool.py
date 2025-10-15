"""Tool node for executing Python functions.

This node wraps arbitrary Python functions (sync or async) and executes them
as part of the graph workflow. It supports automatic parameter injection from
context and input data.
"""

from typing import Callable, Any, Dict, Optional
import inspect
import asyncio

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext


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
        config: Dict[str, Any] = None,
    ):
        """Initialize tool node.

        Args:
            id: Node identifier
            tool_fn: Function to execute (sync or async)
            config: Configuration including parameter bindings
        """
        super().__init__(id, config or {})
        self.tool_fn = tool_fn
        self.is_async = inspect.iscoroutinefunction(tool_fn)
        self.signature = inspect.signature(tool_fn)

        # Store function metadata
        self.function_name = tool_fn.__name__
        self.function_doc = inspect.getdoc(tool_fn) or ""

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
            raise RuntimeError(
                f"Tool function '{self.function_name}' failed: {str(e)}"
            ) from e

        return NodeResult(
            output=result,
            metadata={
                "tool_name": self.function_name,
                "tool_doc": self.function_doc,
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
