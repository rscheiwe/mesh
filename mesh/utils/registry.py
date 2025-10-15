"""Node and agent registry for managing graph components.

This module provides a registry system for storing and retrieving
nodes, agents, and tools by name or ID.
"""

from typing import Dict, Any, Optional, Callable, Type
from mesh.nodes.base import Node
from mesh.utils.errors import InvalidNodeTypeError


class NodeRegistry:
    """Registry for managing nodes, agents, and tools.

    The registry allows you to:
    - Register node classes by type name
    - Register agent instances by name
    - Register tool functions by name
    - Retrieve registered components for graph building

    This is especially useful when parsing React Flow JSON, where
    node configurations reference agents and tools by string identifiers.

    Example:
        >>> registry = NodeRegistry()
        >>> registry.register_agent("research_agent", my_vel_agent)
        >>> registry.register_tool("summarize", summarize_function)
        >>>
        >>> # Later, during graph construction:
        >>> agent = registry.get_agent("research_agent")
        >>> tool = registry.get_tool("summarize")
    """

    def __init__(self):
        """Initialize empty registry."""
        self._node_types: Dict[str, Type[Node]] = {}
        self._agents: Dict[str, Any] = {}
        self._tools: Dict[str, Callable] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}

        # Register built-in node types
        self._register_builtin_nodes()

    def _register_builtin_nodes(self):
        """Register built-in node types."""
        from mesh.nodes.start import StartNode
        from mesh.nodes.end import EndNode
        from mesh.nodes.agent import AgentNode
        from mesh.nodes.llm import LLMNode
        from mesh.nodes.tool import ToolNode
        from mesh.nodes.condition import ConditionNode
        from mesh.nodes.loop import LoopNode

        self._node_types["start"] = StartNode
        self._node_types["end"] = EndNode
        self._node_types["agent"] = AgentNode
        self._node_types["llm"] = LLMNode
        self._node_types["tool"] = ToolNode
        self._node_types["condition"] = ConditionNode
        self._node_types["loop"] = LoopNode

    def register_node_type(self, type_name: str, node_class: Type[Node]) -> None:
        """Register a custom node type.

        Args:
            type_name: Type identifier (e.g., "custom_processor")
            node_class: Node class to register
        """
        self._node_types[type_name] = node_class

    def register_agent(self, name: str, agent: Any) -> None:
        """Register an agent instance.

        Args:
            name: Agent identifier
            agent: Agent instance (Vel, OpenAI, etc.)
        """
        self._agents[name] = agent

    def register_tool(self, name: str, tool_fn: Callable) -> None:
        """Register a tool function.

        Args:
            name: Tool identifier
            tool_fn: Callable function
        """
        self._tools[name] = tool_fn

    def register_config(self, name: str, config: Dict[str, Any]) -> None:
        """Register a configuration preset.

        Args:
            name: Config identifier
            config: Configuration dictionary
        """
        self._configs[name] = config

    def get_node_type(self, type_name: str) -> Type[Node]:
        """Get a node class by type name.

        Args:
            type_name: Node type identifier

        Returns:
            Node class

        Raises:
            InvalidNodeTypeError: If type not found
        """
        if type_name not in self._node_types:
            raise InvalidNodeTypeError(
                f"Node type '{type_name}' not registered. "
                f"Available types: {', '.join(self._node_types.keys())}"
            )
        return self._node_types[type_name]

    def get_agent(self, name: str) -> Any:
        """Get a registered agent.

        Args:
            name: Agent identifier

        Returns:
            Agent instance

        Raises:
            KeyError: If agent not found
        """
        if name not in self._agents:
            raise KeyError(
                f"Agent '{name}' not registered. "
                f"Available agents: {', '.join(self._agents.keys())}"
            )
        return self._agents[name]

    def get_tool(self, name: str) -> Callable:
        """Get a registered tool function.

        Args:
            name: Tool identifier

        Returns:
            Tool function

        Raises:
            KeyError: If tool not found
        """
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' not registered. "
                f"Available tools: {', '.join(self._tools.keys())}"
            )
        return self._tools[name]

    def get_config(self, name: str) -> Dict[str, Any]:
        """Get a registered configuration.

        Args:
            name: Config identifier

        Returns:
            Configuration dictionary

        Raises:
            KeyError: If config not found
        """
        if name not in self._configs:
            raise KeyError(
                f"Config '{name}' not registered. "
                f"Available configs: {', '.join(self._configs.keys())}"
            )
        return self._configs[name]

    def has_agent(self, name: str) -> bool:
        """Check if agent is registered."""
        return name in self._agents

    def has_tool(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def has_node_type(self, type_name: str) -> bool:
        """Check if node type is registered."""
        return type_name in self._node_types

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_node_types(self) -> list[str]:
        """List all registered node types."""
        return list(self._node_types.keys())

    def clear(self) -> None:
        """Clear all registrations (except built-in nodes)."""
        self._agents.clear()
        self._tools.clear()
        self._configs.clear()

    def __repr__(self) -> str:
        return (
            f"NodeRegistry("
            f"agents={len(self._agents)}, "
            f"tools={len(self._tools)}, "
            f"node_types={len(self._node_types)})"
        )
