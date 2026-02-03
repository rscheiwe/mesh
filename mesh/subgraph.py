"""Subgraph composition for Mesh.

This module provides the ability to embed graphs within graphs for:
- Reusable workflow components
- Subagent isolation
- Modular graph design
- Dynamic graph composition

Example:
    >>> from mesh import StateGraph, Subgraph, SubgraphConfig
    >>>
    >>> # Define reusable subgraph
    >>> research_graph = StateGraph()
    >>> research_graph.add_node("search", search_tool, node_type="tool")
    >>> research_graph.add_node("analyze", analyzer_agent, node_type="agent")
    >>> research_graph.add_edge("START", "search")
    >>> research_graph.add_edge("search", "analyze")
    >>> research_graph.set_entry_point("search")
    >>> research_compiled = research_graph.compile()
    >>>
    >>> # Use in parent graph
    >>> main_graph = StateGraph()
    >>> main_graph.add_node("research", Subgraph(research_compiled), node_type="subgraph")
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mesh.core.graph import ExecutionGraph
    from mesh.core.state import ExecutionContext
    from mesh.core.events import ExecutionEvent


@dataclass
class SubgraphConfig:
    """Configuration for subgraph execution.

    Attributes:
        input_mapping: Map parent state keys to subgraph input keys
        output_mapping: Map subgraph output keys to parent state keys
        isolated: If True, subgraph gets clean state (default: True)
        inherit_keys: Keys to copy from parent state if isolated
        checkpoint_on_entry: Create checkpoint before entering subgraph
        checkpoint_on_exit: Create checkpoint after exiting subgraph
        prefix_events: Prefix subgraph event node_ids with subgraph name
    """

    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    isolated: bool = True
    inherit_keys: List[str] = field(default_factory=list)
    checkpoint_on_entry: bool = False
    checkpoint_on_exit: bool = False
    prefix_events: bool = True


class Subgraph:
    """Wrapper for embedding a compiled graph as a node in another graph.

    A Subgraph allows you to treat an entire ExecutionGraph as a single node
    within a parent graph. This enables modular, reusable workflow components.

    Attributes:
        graph: The compiled ExecutionGraph to embed
        config: Configuration for input/output mapping and isolation
        name: Name for this subgraph (used in event prefixing)

    Example:
        >>> # Create and compile a research subgraph
        >>> research = StateGraph()
        >>> research.add_node("search", search_fn, node_type="tool")
        >>> research.set_entry_point("search")
        >>> research_compiled = research.compile()
        >>>
        >>> # Embed in parent graph
        >>> parent = StateGraph()
        >>> parent.add_node("researcher", Subgraph(research_compiled), node_type="subgraph")
    """

    def __init__(
        self,
        graph: "ExecutionGraph",
        config: Optional[SubgraphConfig] = None,
        name: Optional[str] = None,
    ):
        """Initialize subgraph.

        Args:
            graph: Compiled ExecutionGraph to embed
            config: Optional configuration for mapping and isolation
            name: Optional name (defaults to auto-generated ID)
        """
        self.graph = graph
        self.config = config or SubgraphConfig()
        self.name = name or f"subgraph_{id(graph)}"

    def _map_input(self, parent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Map parent state to subgraph input.

        Args:
            parent_state: State from parent context

        Returns:
            Input dict for subgraph
        """
        if not self.config.input_mapping:
            # Default: pass entire parent state
            return dict(parent_state)

        subgraph_input = {}
        for parent_key, subgraph_key in self.config.input_mapping.items():
            if parent_key in parent_state:
                subgraph_input[subgraph_key] = parent_state[parent_key]

        return subgraph_input

    def _map_output(self, subgraph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Map subgraph output to parent state updates.

        Args:
            subgraph_state: Final state from subgraph execution

        Returns:
            State updates for parent context
        """
        if not self.config.output_mapping:
            # Default: return final subgraph state
            return subgraph_state

        parent_updates = {}
        for subgraph_key, parent_key in self.config.output_mapping.items():
            if subgraph_key in subgraph_state:
                parent_updates[parent_key] = subgraph_state[subgraph_key]

        return parent_updates

    def _create_subgraph_context(
        self,
        input_data: Dict[str, Any],
        parent_context: "ExecutionContext",
    ) -> "ExecutionContext":
        """Create isolated context for subgraph execution.

        Args:
            input_data: Input to the subgraph
            parent_context: Parent execution context

        Returns:
            New ExecutionContext for subgraph
        """
        from mesh.core.state import ExecutionContext

        # Create isolated or inherited state
        if self.config.isolated:
            subgraph_state = {}
            for key in self.config.inherit_keys:
                if key in parent_context.state:
                    subgraph_state[key] = parent_context.state[key]
        else:
            subgraph_state = dict(parent_context.state)

        # Map input
        mapped_input = self._map_input(input_data if isinstance(input_data, dict) else {"input": input_data})
        subgraph_state.update(mapped_input)

        # Create subgraph context with hierarchical IDs
        return ExecutionContext(
            graph_id=f"{parent_context.graph_id}.{self.name}",
            session_id=f"{parent_context.session_id}.{self.name}",
            chat_history=[],  # Isolated chat history
            variables=dict(parent_context.variables),
            state=subgraph_state,
        )

    async def execute(
        self,
        input_data: Any,
        parent_context: "ExecutionContext",
    ) -> AsyncIterator["ExecutionEvent"]:
        """Execute subgraph with proper context isolation.

        Yields events from subgraph execution, optionally prefixed with
        subgraph name for traceability.

        Args:
            input_data: Input to the subgraph
            parent_context: Parent execution context

        Yields:
            ExecutionEvent: Events from subgraph execution
        """
        from mesh.core.executor import Executor
        from mesh.core.events import ExecutionEvent, EventType
        from mesh.backends.memory import MemoryBackend
        from datetime import datetime

        # Create subgraph context
        subgraph_context = self._create_subgraph_context(input_data, parent_context)

        # Create executor for subgraph
        # Note: Using MemoryBackend for isolation; parent backend handles persistence
        executor = Executor(self.graph, MemoryBackend())

        # Map input
        mapped_input = self._map_input(
            input_data if isinstance(input_data, dict) else {"input": input_data}
        )

        # Emit subgraph start event
        yield ExecutionEvent(
            type=EventType.SUBGRAPH_START,
            node_id=self.name,
            timestamp=datetime.now(),
            metadata={"input": mapped_input, "isolated": self.config.isolated},
        )

        # Execute subgraph and yield events
        final_output = None
        async for event in executor.execute(mapped_input, subgraph_context):
            # Prefix event node_id with subgraph name if configured
            if self.config.prefix_events and event.node_id:
                event.node_id = f"{self.name}.{event.node_id}"

            # Capture final output
            if event.type == EventType.EXECUTION_COMPLETE:
                final_output = event.output
                # Don't yield the inner EXECUTION_COMPLETE, we'll emit SUBGRAPH_COMPLETE instead
                continue

            yield event

        # Map output back to parent
        output = self._map_output(subgraph_context.state)
        if final_output:
            output["_subgraph_output"] = final_output

        # Emit subgraph complete event
        yield ExecutionEvent(
            type=EventType.SUBGRAPH_COMPLETE,
            node_id=self.name,
            output=output,
            timestamp=datetime.now(),
            metadata={"final_state": subgraph_context.state},
        )

    def __repr__(self) -> str:
        return f"Subgraph(name='{self.name}', nodes={len(self.graph.nodes)})"


class SubgraphBuilder:
    """Utility for building subgraphs with fluent API.

    Example:
        >>> subgraph = (
        ...     SubgraphBuilder("research")
        ...     .add_node("search", search_fn, node_type="tool")
        ...     .add_node("analyze", analyze_fn, node_type="tool")
        ...     .add_edge("search", "analyze")
        ...     .set_entry_point("search")
        ...     .with_input_mapping({"query": "search_query"})
        ...     .with_output_mapping({"findings": "research_results"})
        ...     .build()
        ... )
    """

    def __init__(self, name: str):
        """Initialize builder.

        Args:
            name: Name for the subgraph
        """
        from mesh.builders.state_graph import StateGraph

        self.name = name
        self._graph = StateGraph()
        self._config = SubgraphConfig()

    def add_node(self, node_id: str, node_or_config: Any, node_type: Optional[str] = None, **kwargs) -> "SubgraphBuilder":
        """Add a node to the subgraph."""
        self._graph.add_node(node_id, node_or_config, node_type, **kwargs)
        return self

    def add_edge(self, source: str, target: str, **kwargs) -> "SubgraphBuilder":
        """Add an edge to the subgraph."""
        self._graph.add_edge(source, target, **kwargs)
        return self

    def set_entry_point(self, node_id: str) -> "SubgraphBuilder":
        """Set the entry point for the subgraph."""
        self._graph.set_entry_point(node_id)
        return self

    def with_input_mapping(self, mapping: Dict[str, str]) -> "SubgraphBuilder":
        """Configure input mapping."""
        self._config.input_mapping = mapping
        return self

    def with_output_mapping(self, mapping: Dict[str, str]) -> "SubgraphBuilder":
        """Configure output mapping."""
        self._config.output_mapping = mapping
        return self

    def with_isolation(self, isolated: bool = True, inherit_keys: Optional[List[str]] = None) -> "SubgraphBuilder":
        """Configure state isolation."""
        self._config.isolated = isolated
        if inherit_keys:
            self._config.inherit_keys = inherit_keys
        return self

    def with_checkpoints(self, on_entry: bool = False, on_exit: bool = False) -> "SubgraphBuilder":
        """Configure checkpoint behavior."""
        self._config.checkpoint_on_entry = on_entry
        self._config.checkpoint_on_exit = on_exit
        return self

    def build(self) -> Subgraph:
        """Build and return the Subgraph."""
        compiled = self._graph.compile()
        return Subgraph(compiled, self._config, self.name)
