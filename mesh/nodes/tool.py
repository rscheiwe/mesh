"""ToolNode implementation for executing functions and providing context."""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from mesh.core.node import NodeConfig
from mesh.nodes.base import BaseNode
from mesh.state.state import GraphState


@dataclass
class ToolNodeConfig(NodeConfig):
    """Configuration for ToolNode."""

    tool_name: str = "tool"  # Default tool name
    tool_description: Optional[str] = None
    output_key: Optional[str] = None  # Key to store result in output
    store_in_state: bool = True  # Whether to store result in graph state
    state_key: Optional[str] = None  # Custom key for state storage
    can_be_terminal: bool = True  # Tool nodes can be terminal nodes


class ToolNode(BaseNode):
    """Node that executes a tool/function and provides results as context.

    Unlike AgentNode which decides when to use tools, ToolNode always
    executes its function and passes the result forward in the graph.
    This is useful for:
    - Data retrieval before AI processing
    - Preprocessing steps
    - API calls to gather context
    - Database queries
    - File operations
    """

    def __init__(
        self,
        tool_func: Callable,
        config: Optional[ToolNodeConfig] = None,
        extract_args: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """Initialize ToolNode.

        Args:
            tool_func: The function to execute
            config: Node configuration
            extract_args: Optional function to extract arguments from input data
        """
        if config is None:
            config = ToolNodeConfig(tool_name="ToolNode")
        super().__init__(config)

        self.tool_func = tool_func
        self.tool_config: ToolNodeConfig = config
        self.extract_args = extract_args or self._default_extract_args

    def _default_extract_args(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Default argument extraction - passes through tool_args if present."""
        return input_data.get("tool_args", {})

    async def _execute_impl(
        self, input_data: Dict[str, Any], state: Optional[GraphState] = None
    ) -> Dict[str, Any]:
        """Execute the tool function.

        Args:
            input_data: Input data containing arguments for the tool
            state: Optional graph state

        Returns:
            Dict with tool execution results
        """
        # Extract arguments for the tool
        tool_args = self.extract_args(input_data)

        # Execute the tool
        try:
            if asyncio.iscoroutinefunction(self.tool_func):
                result = await self.tool_func(**tool_args)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: self.tool_func(**tool_args)
                )

            # Store in state if configured
            if state and self.tool_config.store_in_state:
                state_key = (
                    self.tool_config.state_key or f"{self.tool_config.tool_name}_result"
                )
                await state.set(state_key, result)

            # Prepare output
            output_key = self.tool_config.output_key or "tool_result"

            # Start with input data as base
            output = dict(input_data)

            # Then add/override with tool results
            output.update(
                {
                    output_key: result,
                    "tool_name": self.tool_config.tool_name,
                    "success": True,
                    "input_data": input_data,  # Preserve original input separately
                }
            )

            return output

        except Exception as e:
            # Start with input data as base
            error_output = dict(input_data)

            # Add error information
            output_key = self.tool_config.output_key or "tool_result"
            error_output.update(
                {
                    "error": str(e),
                    "tool_name": self.tool_config.tool_name,
                    "success": False,
                    "input_data": input_data,
                    output_key: None,
                }
            )

            return error_output


class MultiToolNode(BaseNode):
    """Node that executes multiple tools in parallel or sequence."""

    def __init__(
        self,
        tools: List[Union[Callable, tuple[str, Callable]]],
        parallel: bool = True,
        config: Optional[NodeConfig] = None,
    ):
        """Initialize MultiToolNode.

        Args:
            tools: List of functions or (name, function) tuples
            parallel: Whether to execute tools in parallel
            config: Node configuration
        """
        # Set can_be_terminal=True for MultiToolNode
        if config is None:
            config = NodeConfig(can_be_terminal=True)
        elif not hasattr(config, 'can_be_terminal'):
            config.can_be_terminal = True
        super().__init__(config)

        self.tools = []
        for tool in tools:
            if isinstance(tool, tuple):
                name, func = tool
                self.tools.append((name, func))
            else:
                # Use function name as tool name
                name = getattr(tool, "__name__", f"tool_{len(self.tools)}")
                self.tools.append((name, tool))

        self.parallel = parallel

    async def _execute_impl(
        self, input_data: Dict[str, Any], state: Optional[GraphState] = None
    ) -> Dict[str, Any]:
        """Execute multiple tools.

        Args:
            input_data: Input data for tools
            state: Optional graph state

        Returns:
            Dict with all tool results
        """
        results = {}

        if self.parallel:
            # Execute tools in parallel
            tasks = []
            for name, func in self.tools:
                if asyncio.iscoroutinefunction(func):
                    task = func(**input_data.get("tool_args", {}))
                else:
                    loop = asyncio.get_event_loop()
                    task = loop.run_in_executor(
                        None, lambda f=func: f(**input_data.get("tool_args", {}))
                    )
                tasks.append((name, task))

            # Gather results
            for name, task in tasks:
                try:
                    results[name] = await task
                except Exception as e:
                    results[name] = {"error": str(e)}
        else:
            # Execute tools sequentially
            for name, func in self.tools:
                try:
                    if asyncio.iscoroutinefunction(func):
                        results[name] = await func(**input_data.get("tool_args", {}))
                    else:
                        loop = asyncio.get_event_loop()
                        results[name] = await loop.run_in_executor(
                            None, lambda: func(**input_data.get("tool_args", {}))
                        )
                except Exception as e:
                    results[name] = {"error": str(e)}

        # Store results in state
        if state:
            for name, result in results.items():
                await state.set(f"tool_{name}_result", result)

        # Start with input data as base
        output = dict(input_data)

        # Add tool results
        output.update(
            {
                "tool_results": results,
                "tools_executed": list(results.keys()),
                "input_data": input_data,
            }
        )

        return output
