"""LangGraph-style declarative graph builder.

This module provides a fluent API for building graphs programmatically,
inspired by LangGraph's StateGraph pattern.
"""

from typing import Callable, Optional, List, Dict, Any, Union
from datetime import datetime
from pathlib import Path

from mesh.core.graph import ExecutionGraph, Edge
from mesh.nodes.base import Node
from mesh.nodes.start import StartNode
from mesh.nodes.end import EndNode
from mesh.nodes.agent import AgentNode
from mesh.nodes.llm import LLMNode
from mesh.nodes.tool import ToolNode
from mesh.nodes.condition import ConditionNode, Condition
from mesh.nodes.loop import LoopNode, ForEachNode
from mesh.utils.errors import GraphValidationError
from mesh.utils.mermaid import generate_mermaid_code, save_mermaid_image, get_default_visualization_dir


class StateGraph:
    """LangGraph-style declarative graph builder.

    This provides a fluent API for building graphs programmatically
    with method chaining and clear semantics.

    Example:
        >>> graph = StateGraph()
        >>> graph.add_node("agent", agent_instance, node_type="agent")
        >>> graph.add_node("tool", tool_function, node_type="tool")
        >>> graph.add_edge("START", "agent")
        >>> graph.add_edge("agent", "tool")
        >>> graph.add_edge("tool", "END")
        >>> graph.set_entry_point("agent")
        >>> compiled = graph.compile()
    """

    def __init__(self):
        """Initialize empty graph builder."""
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._entry_point: Optional[str] = None

    def add_node(
        self,
        node_id: str,
        node_or_config: Union[Node, Any],
        node_type: Optional[str] = None,
        **kwargs,
    ) -> "StateGraph":
        """Add a node to the graph.

        This method is flexible and can handle:
        1. Pre-constructed Node instances
        2. Agent/tool instances with node_type specified
        3. Configuration for creating nodes

        Args:
            node_id: Unique identifier for the node
            node_or_config: Either a Node instance, or an object to wrap
            node_type: Type of node ('agent', 'tool', 'llm', etc.)
            **kwargs: Additional configuration for node creation

        Returns:
            Self for method chaining

        Example:
            >>> # Add pre-constructed node
            >>> graph.add_node("start", StartNode("start"))
            >>>
            >>> # Add agent (auto-wrapped)
            >>> graph.add_node("agent", my_agent, node_type="agent")
            >>>
            >>> # Add tool
            >>> graph.add_node("tool", my_function, node_type="tool")
        """
        # If already a Node, add directly
        if isinstance(node_or_config, Node):
            self._nodes[node_id] = node_or_config
            return self

        # Otherwise, create node based on type
        if not node_type:
            raise GraphValidationError(
                f"node_type must be specified when adding non-Node objects. "
                f"Got: {type(node_or_config)}"
            )

        # Create node based on type
        if node_type == "agent":
            node = AgentNode(
                id=node_id,
                agent=node_or_config,
                system_prompt=kwargs.get("system_prompt"),
                use_native_events=kwargs.get("use_native_events", False),
                event_mode=kwargs.get("event_mode", "full"),
                config=kwargs.get("config", {}),
            )
        elif node_type == "tool":
            node = ToolNode(
                id=node_id,
                tool_fn=node_or_config,
                event_mode=kwargs.get("event_mode", "full"),
                config=kwargs.get("config", {}),
            )
        elif node_type == "llm":
            node = LLMNode(
                id=node_id,
                model=kwargs.get("model", "gpt-4"),
                system_prompt=kwargs.get("system_prompt"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens"),
                provider=kwargs.get("provider", "openai"),
                event_mode=kwargs.get("event_mode", "full"),
                config=kwargs.get("config", {}),
            )
        elif node_type == "condition":
            # Unified condition node - supports deterministic or AI routing
            condition_routing = kwargs.get("condition_routing", "deterministic")

            if condition_routing == "deterministic":
                # Deterministic mode: requires conditions
                node = ConditionNode(
                    id=node_id,
                    condition_routing="deterministic",
                    conditions=node_or_config,
                    default_target=kwargs.get("default_target"),
                    event_mode=kwargs.get("event_mode", "full"),
                    config=kwargs.get("config", {}),
                )
            elif condition_routing == "ai":
                # AI mode: requires model, instructions, scenarios
                node = ConditionNode(
                    id=node_id,
                    condition_routing="ai",
                    model=kwargs.get("model"),
                    instructions=kwargs.get("instructions"),
                    scenarios=kwargs.get("scenarios"),
                    default_target=kwargs.get("default_target"),
                    event_mode=kwargs.get("event_mode", "full"),
                    config=kwargs.get("config", {}),
                )
            else:
                raise GraphValidationError(
                    f"condition_routing must be 'deterministic' or 'ai', got: {condition_routing}"
                )
        elif node_type == "loop":
            # Flowise-style loop: backward jump to a previously executed node
            if "loop_back_to" not in kwargs:
                raise GraphValidationError(
                    f"Loop node '{node_id}' requires 'loop_back_to' parameter"
                )
            node = LoopNode(
                id=node_id,
                loop_back_to=kwargs["loop_back_to"],
                max_loop_count=kwargs.get("max_loop_count", 5),
                event_mode=kwargs.get("event_mode", "full"),
                config=kwargs.get("config", {}),
            )
        elif node_type == "foreach":
            # Array iteration node (previously called "loop")
            node = ForEachNode(
                id=node_id,
                array_path=kwargs.get("array_path", "$.items"),
                max_iterations=kwargs.get("max_iterations", 100),
                event_mode=kwargs.get("event_mode", "full"),
                config=kwargs.get("config", {}),
            )
        elif node_type == "start":
            node = StartNode(
                id=node_id,
                event_mode=kwargs.get("event_mode", "full"),
                config=kwargs.get("config", {})
            )
        elif node_type == "end":
            node = EndNode(id=node_id, config=kwargs.get("config", {}))
        else:
            raise GraphValidationError(f"Unknown node type: {node_type}")

        self._nodes[node_id] = node
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        is_loop_edge: bool = False,
        loop_condition: Optional[Any] = None,
        max_iterations: Optional[int] = None,
    ) -> "StateGraph":
        """Add a direct edge between two nodes.

        Args:
            source: Source node ID
            target: Target node ID
            is_loop_edge: Whether this edge is part of a cycle (allows controlled loops)
            loop_condition: Callable (state, output) -> bool to continue loop
            max_iterations: Maximum iterations for this loop edge

        Returns:
            Self for method chaining

        Example:
            >>> # Regular edge
            >>> graph.add_edge("START", "agent")

            >>> # Loop edge with condition
            >>> graph.add_edge("check", "process",
            ...     is_loop_edge=True,
            ...     loop_condition=lambda state, output: state.get("count", 0) < 10)

            >>> # Loop edge with max iterations
            >>> graph.add_edge("check", "process",
            ...     is_loop_edge=True,
            ...     max_iterations=100)
        """
        self._edges.append(
            Edge(
                source=source,
                target=target,
                is_loop_edge=is_loop_edge,
                loop_condition=loop_condition,
                max_iterations=max_iterations,
            )
        )
        return self

    def add_conditional_edges(
        self,
        source: str,
        condition_fn: Callable[[Any], str],
        mapping: Dict[str, str],
        default: Optional[str] = None,
    ) -> "StateGraph":
        """Add conditional branching from a node.

        Args:
            source: Source node ID
            condition_fn: Function that returns a key from mapping
            mapping: Dict mapping condition results to target nodes
            default: Default target if condition doesn't match any key

        Returns:
            Self for method chaining

        Example:
            >>> graph.add_conditional_edges(
            ...     "agent",
            ...     lambda x: "success" if "ok" in str(x) else "failure",
            ...     {"success": "next_node", "failure": "error_handler"}
            ... )
        """
        # Create conditions from mapping
        conditions = []
        for key, target in mapping.items():
            predicate = lambda x, k=key: condition_fn(x) == k
            conditions.append(
                Condition(
                    name=key,
                    predicate=predicate,
                    target_node=target,
                )
            )

        # Create synthetic condition node (always deterministic mode)
        condition_node_id = f"{source}_condition"
        condition_node = ConditionNode(
            id=condition_node_id,
            condition_routing="deterministic",
            conditions=conditions,
            default_target=default,
        )

        # Add condition node
        self._nodes[condition_node_id] = condition_node

        # Add edges
        self._edges.append(Edge(source=source, target=condition_node_id))

        # Add edges from condition node to targets
        for target in mapping.values():
            self._edges.append(Edge(source=condition_node_id, target=target))

        if default:
            self._edges.append(Edge(source=condition_node_id, target=default))

        return self

    def set_entry_point(self, node_id: str) -> "StateGraph":
        """Set the starting node of the graph.

        Args:
            node_id: ID of the entry node

        Returns:
            Self for method chaining
        """
        self._entry_point = node_id
        return self

    def compile(self) -> ExecutionGraph:
        """Compile the graph into an executable ExecutionGraph.

        Returns:
            ExecutionGraph ready for execution

        Raises:
            GraphValidationError: If graph is invalid
        """
        if not self._entry_point:
            raise GraphValidationError(
                "Entry point not set. Call set_entry_point() before compiling."
            )

        # Add synthetic START node if not present
        if "START" not in self._nodes:
            self._nodes["START"] = StartNode(id="START")
            self._edges.insert(0, Edge(source="START", target=self._entry_point))

        # Note: END nodes are optional - the executor will identify
        # nodes with no children as ending points automatically

        # Build execution graph
        graph = ExecutionGraph.from_nodes_and_edges(self._nodes, self._edges)

        # Validate
        graph.validate()

        return graph

    def add_sequence(self, node_ids: List[str]) -> "StateGraph":
        """Add a sequence of nodes connected linearly.

        Args:
            node_ids: List of node IDs to connect in sequence

        Returns:
            Self for method chaining

        Example:
            >>> graph.add_sequence(["node1", "node2", "node3"])
            >>> # Creates: node1 -> node2 -> node3
        """
        for i in range(len(node_ids) - 1):
            self.add_edge(node_ids[i], node_ids[i + 1])
        return self

    def mermaid_code(
        self,
        title: Optional[str] = None,
        direction: str = "TD",
    ) -> str:
        """Generate Mermaid flowchart code for this graph.

        This compiles the graph and generates a Mermaid diagram representation.
        Different node types are styled with different colors for clarity.

        Args:
            title: Optional title to display above the diagram
            direction: Flowchart direction - "TD" (top-down) or "LR" (left-right)

        Returns:
            String containing Mermaid flowchart code

        Example:
            >>> graph = StateGraph()
            >>> graph.add_node("agent", my_agent, node_type="agent")
            >>> graph.add_edge("START", "agent")
            >>> graph.set_entry_point("agent")
            >>> code = graph.mermaid_code(title="My Agent Graph")
            >>> print(code)
        """
        # Compile graph to get ExecutionGraph
        compiled = self.compile()

        # Generate Mermaid code
        return generate_mermaid_code(compiled, title=title, direction=direction)

    def save_visualization(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        image_format: str = "png",
        direction: str = "TD",
    ) -> str:
        """Generate and save a Mermaid diagram visualization of this graph.

        This compiles the graph, generates Mermaid code, and saves it as an image
        using the mermaid.ink API.

        Args:
            output_path: Path where image should be saved. If None, saves to
                mesh/visualizations/ with auto-generated filename
            title: Optional title to display above the diagram
            image_format: Output format - "png", "svg", or "pdf" (default: "png")
            direction: Flowchart direction - "TD" (top-down) or "LR" (left-right)

        Returns:
            Path to saved image file

        Raises:
            httpx.HTTPError: If image generation fails
            GraphValidationError: If graph is invalid

        Example:
            >>> graph = StateGraph()
            >>> graph.add_node("check", check_fn, node_type="tool")
            >>> graph.add_node("increment", increment_fn, node_type="tool")
            >>> graph.add_edge("START", "check")
            >>> graph.add_edge("check", "increment")
            >>> graph.add_edge("increment", "check", is_loop_edge=True, max_iterations=10)
            >>> graph.set_entry_point("check")
            >>> path = graph.save_visualization(title="Divisible By 5 Graph")
            >>> print(f"Saved to: {path}")
        """
        # Generate Mermaid code
        mermaid_code = self.mermaid_code(title=title, direction=direction)

        # Determine output path
        if output_path is None:
            # Auto-generate filename with timestamp
            vis_dir = get_default_visualization_dir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            graph_name = title.replace(" ", "_") if title else "graph"
            filename = f"{graph_name}_{timestamp}.{image_format}"
            output_path = str(vis_dir / filename)

        # Save image
        return save_mermaid_image(
            mermaid_code,
            output_path,
            image_format=image_format,
        )

    def __repr__(self) -> str:
        return (
            f"StateGraph(nodes={len(self._nodes)}, edges={len(self._edges)}, "
            f"entry='{self._entry_point}')"
        )
