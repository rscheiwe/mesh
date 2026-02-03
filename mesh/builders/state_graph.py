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
        # Interrupt configurations by node_id
        self._interrupt_before: Dict[str, Dict[str, Any]] = {}
        self._interrupt_after: Dict[str, Dict[str, Any]] = {}
        # Parallel execution configurations
        self._parallel_branches: List[Dict[str, Any]] = []  # ParallelBranch configs
        self._fan_in_nodes: Dict[str, List[str]] = {}  # target -> sources it waits for
        self._fan_in_aggregators: Dict[str, Callable] = {}  # target -> aggregator fn

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
        elif node_type == "subgraph":
            from mesh.nodes.subgraph import SubgraphNode
            from mesh.subgraph import Subgraph
            # node_or_config should be a Subgraph instance
            if not isinstance(node_or_config, Subgraph):
                raise GraphValidationError(
                    f"subgraph node type requires a Subgraph instance, got: {type(node_or_config)}"
                )
            node = SubgraphNode(
                id=node_id,
                subgraph=node_or_config,
                config=kwargs.get("config", {}),
            )
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
        interrupt_before: bool = False,
        interrupt_after: bool = False,
        interrupt_condition: Optional[Callable[[Any, Any], bool]] = None,
    ) -> "StateGraph":
        """Add a direct edge between two nodes.

        Args:
            source: Source node ID
            target: Target node ID
            is_loop_edge: Whether this edge is part of a cycle (allows controlled loops)
            loop_condition: Callable (state, output) -> bool to continue loop
            max_iterations: Maximum iterations for this loop edge
            interrupt_before: Pause before executing target node for human review
            interrupt_after: Pause after executing target node for human review
            interrupt_condition: Callable (state, input/output) -> bool to trigger interrupt

        Returns:
            Self for method chaining

        Example:
            >>> # Regular edge
            >>> graph.add_edge("START", "agent")

            >>> # Loop edge with condition
            >>> graph.add_edge("check", "process",
            ...     is_loop_edge=True,
            ...     loop_condition=lambda state, output: state.get("count", 0) < 10)

            >>> # Interrupt before a critical node
            >>> graph.add_edge("researcher", "reviewer", interrupt_before=True)

            >>> # Conditional interrupt
            >>> graph.add_edge("agent", "tool", interrupt_before=True,
            ...     interrupt_condition=lambda state, input: state.get("requires_approval"))
        """
        self._edges.append(
            Edge(
                source=source,
                target=target,
                is_loop_edge=is_loop_edge,
                loop_condition=loop_condition,
                max_iterations=max_iterations,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                interrupt_condition=interrupt_condition,
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

        # Build execution graph with all configurations
        graph = ExecutionGraph.from_nodes_and_edges(
            self._nodes,
            self._edges,
            interrupt_before=self._interrupt_before,
            interrupt_after=self._interrupt_after,
            parallel_branches=self._parallel_branches,
            fan_in_nodes=self._fan_in_nodes,
            fan_in_aggregators=self._fan_in_aggregators,
        )

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

    def set_interrupt_before(
        self,
        node_id: str,
        condition: Optional[Callable[[Any, Any], bool]] = None,
        metadata_extractor: Optional[Callable[[Any, Any], Dict[str, Any]]] = None,
        timeout: Optional[float] = None,
    ) -> "StateGraph":
        """Configure an interrupt point before a node executes.

        This enables human-in-the-loop review before the node runs.
        The executor will pause and emit an INTERRUPT event, allowing
        the caller to review/modify state before resuming.

        Args:
            node_id: ID of node to interrupt before
            condition: Optional callable (state, input) -> bool to trigger conditionally
            metadata_extractor: Optional callable to extract review data
            timeout: Optional timeout in seconds (None = wait forever)

        Returns:
            Self for method chaining

        Example:
            >>> # Always interrupt before critical_node
            >>> graph.set_interrupt_before("critical_node")

            >>> # Conditional interrupt
            >>> graph.set_interrupt_before(
            ...     "reviewer",
            ...     condition=lambda state, inp: state.get("requires_approval", False)
            ... )
        """
        self._interrupt_before[node_id] = {
            "condition": condition,
            "metadata_extractor": metadata_extractor,
            "timeout": timeout,
        }
        return self

    def set_interrupt_after(
        self,
        node_id: str,
        condition: Optional[Callable[[Any, Any], bool]] = None,
        metadata_extractor: Optional[Callable[[Any, Any], Dict[str, Any]]] = None,
        timeout: Optional[float] = None,
    ) -> "StateGraph":
        """Configure an interrupt point after a node executes.

        This enables human-in-the-loop review after the node runs.
        The executor will pause and emit an INTERRUPT event, allowing
        the caller to review/modify state before continuing.

        Args:
            node_id: ID of node to interrupt after
            condition: Optional callable (state, output) -> bool to trigger conditionally
            metadata_extractor: Optional callable to extract review data
            timeout: Optional timeout in seconds (None = wait forever)

        Returns:
            Self for method chaining

        Example:
            >>> # Always interrupt after agent to review output
            >>> graph.set_interrupt_after("agent")

            >>> # Conditional interrupt on high-risk actions
            >>> graph.set_interrupt_after(
            ...     "action_node",
            ...     condition=lambda state, out: out.get("risk_level") == "high"
            ... )
        """
        self._interrupt_after[node_id] = {
            "condition": condition,
            "metadata_extractor": metadata_extractor,
            "timeout": timeout,
        }
        return self

    def add_parallel_edges(
        self,
        source: str,
        targets: List[str],
    ) -> "StateGraph":
        """Add fan-out edges from source to multiple targets (executed in parallel).

        This creates parallel branches where all targets execute concurrently
        after the source completes. Use with add_fan_in_edge to collect results.

        Args:
            source: Source node ID
            targets: List of target node IDs to execute concurrently

        Returns:
            Self for method chaining

        Example:
            >>> graph.add_parallel_edges("START", ["worker_1", "worker_2", "worker_3"])
            >>> # Creates: START -> (worker_1 | worker_2 | worker_3) in parallel
        """
        if len(targets) < 2:
            raise GraphValidationError("add_parallel_edges requires at least 2 targets")

        # Record parallel branch configuration
        self._parallel_branches.append({
            "source": source,
            "targets": list(targets),
            "is_dynamic": False,
        })

        # Also add regular edges for graph structure/validation
        for target in targets:
            self.add_edge(source, target)

        return self

    def add_fan_in_edge(
        self,
        sources: List[str],
        target: str,
        aggregator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> "StateGraph":
        """Add fan-in edge where target waits for all sources to complete.

        This creates a synchronization point where the target node only
        executes after all source nodes have completed. Results are aggregated
        and passed to the target.

        Args:
            sources: List of source node IDs that must complete
            target: Target node ID that receives aggregated results
            aggregator: Optional function to aggregate results (default: merge dicts)

        Returns:
            Self for method chaining

        Example:
            >>> graph.add_fan_in_edge(
            ...     ["worker_1", "worker_2", "worker_3"],
            ...     "consolidator",
            ...     aggregator=lambda results: {"all_findings": results}
            ... )
        """
        if len(sources) < 2:
            raise GraphValidationError("add_fan_in_edge requires at least 2 sources")

        # Record fan-in configuration
        self._fan_in_nodes[target] = list(sources)

        if aggregator:
            self._fan_in_aggregators[target] = aggregator

        # Add edges for graph structure
        for source in sources:
            self.add_edge(source, target)

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
