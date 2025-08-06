"""Agent node implementation with tool support."""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from mesh.nodes.llm import LLMConfig, LLMNode, Message
from mesh.state.state import GraphState


class ToolResult(BaseModel):
    """Result from a tool execution."""

    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    """Abstract base class for tools that agents can use."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, *args, **kwargs) -> ToolResult:
        """Execute the tool.

        Returns:
            ToolResult with the execution outcome
        """
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for the LLM.

        Returns:
            Schema describing the tool's parameters
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema(),
        }

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema.

        Subclasses can override to provide detailed schemas.

        Returns:
            Parameters schema
        """
        return {"type": "object", "properties": {}}


class FunctionTool(Tool):
    """Tool that wraps a Python function."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters_schema: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, description)
        self.func = func
        self.parameters_schema = parameters_schema or {}

    async def execute(self, *args, **kwargs) -> ToolResult:
        """Execute the wrapped function."""
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(*args, **kwargs)
            else:
                result = self.func(*args, **kwargs)

            return ToolResult(success=True, output=result)

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema."""
        return self.parameters_schema


@dataclass
class AgentConfig(LLMConfig):
    """Configuration for agent nodes."""

    tools: List[Tool] = field(default_factory=list)
    max_iterations: int = 10
    allow_parallel_tool_calls: bool = True
    tool_choice: Optional[str] = None  # auto, none, or specific tool name


class AgentNode(LLMNode):
    """Node that represents an LLM with tool capabilities."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.agent_config: AgentConfig = config
        self.tools_map = {tool.name: tool for tool in config.tools}

    async def _execute_impl(
        self, input_data: Dict[str, Any], state: Optional[GraphState] = None
    ) -> Dict[str, Any]:
        """Execute agent with tool support.

        Args:
            input_data: Input data for the agent
            state: Optional shared state

        Returns:
            Dict containing agent response and tool execution results
        """
        messages = self._prepare_messages(input_data, state)
        tool_results = []
        iterations = 0

        while iterations < self.agent_config.max_iterations:
            iterations += 1

            # Get LLM response with tool schemas
            response = await self._call_llm_with_tools(messages)

            # Check if LLM wants to use tools
            tool_calls = self._extract_tool_calls(response)

            if not tool_calls:
                # No tools requested, return final response
                return {
                    "response": response,
                    "messages": messages,
                    "tool_results": tool_results,
                    "iterations": iterations,
                }

            # Add the assistant's message that requested tools
            messages.append(
                Message(role="assistant", content=response.get("content", ""))
            )

            # Execute requested tools
            batch_results = await self._execute_tools(tool_calls)
            tool_results.extend(batch_results)

            # Add tool results to conversation
            for result in batch_results:
                messages.append(
                    Message(
                        role="function",
                        content=f"{result['result'].output}",
                        metadata={"name": result["tool_name"]},
                    )
                )

        # Max iterations reached
        return {
            "response": "Max iterations reached",
            "messages": messages,
            "tool_results": tool_results,
            "iterations": iterations,
            "max_iterations_reached": True,
        }

    async def _call_llm_with_tools(self, messages: List[Message]) -> Dict[str, Any]:
        """Call LLM with tool schemas included.

        Args:
            messages: Conversation messages

        Returns:
            Dict with LLM response and tool calls
        """
        # Get provider
        provider = await self._get_provider()

        # Convert tools to provider format
        tool_schemas = [tool.get_schema() for tool in self.agent_config.tools]

        # Make the API call with tools
        response = await provider.chat_completion_with_tools(
            messages=messages,
            tools=tool_schemas,
            model=self.agent_config.model,
            temperature=self.agent_config.temperature,
            max_tokens=self.agent_config.max_tokens,
            tool_choice=self.agent_config.tool_choice or "auto",
        )

        return {
            "content": response.content,
            "tool_calls": response.metadata.get("tool_calls", []),
        }

    def _extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response.

        Args:
            response: LLM response dict

        Returns:
            List of tool calls to execute
        """
        tool_calls = []

        # Extract tool calls from response
        for tc in response.get("tool_calls", []):
            if tc.get("type") == "function":
                func = tc.get("function", {})
                tool_calls.append(
                    {
                        "tool_name": func.get("name"),
                        "arguments": json.loads(func.get("arguments", "{}")),
                    }
                )

        return tool_calls

    async def _execute_tools(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute a batch of tool calls.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool execution results
        """
        results = []

        if self.agent_config.allow_parallel_tool_calls:
            # Execute tools in parallel
            import asyncio

            tasks = []

            for call in tool_calls:
                tool = self.tools_map.get(call["tool_name"])
                if tool:
                    task = self._execute_single_tool(tool, call["arguments"])
                    tasks.append(task)

            if tasks:
                tool_results = await asyncio.gather(*tasks)
                for i, result in enumerate(tool_results):
                    results.append(
                        {"tool_name": tool_calls[i]["tool_name"], "result": result}
                    )
        else:
            # Execute tools sequentially
            for call in tool_calls:
                tool = self.tools_map.get(call["tool_name"])
                if tool:
                    result = await self._execute_single_tool(tool, call["arguments"])
                    results.append({"tool_name": call["tool_name"], "result": result})

        return results

    async def _execute_single_tool(
        self, tool: Tool, arguments: Dict[str, Any]
    ) -> ToolResult:
        """Execute a single tool.

        Args:
            tool: Tool to execute
            arguments: Arguments for the tool

        Returns:
            ToolResult
        """
        try:
            return await tool.execute(**arguments)
        except Exception as e:
            return ToolResult(
                success=False, output=None, error=f"Tool execution error: {str(e)}"
            )
