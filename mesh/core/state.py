"""State management for graph execution.

This module provides state tracking and persistence coordination during execution.
"""

from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field
import uuid

if TYPE_CHECKING:
    from mesh.core.graph import GraphMetadata


@dataclass
class ExecutionContext:
    """Runtime context passed to nodes during execution.

    This contains all the information needed by nodes to execute properly,
    including graph state, variables, chat history, and iteration context.

    Attributes:
        graph_id: Identifier for the graph being executed
        session_id: Session identifier for state persistence
        chat_history: List of chat messages
        variables: Global variables accessible via {{$vars.*}}
        state: Mutable state dictionary
        iteration_context: Context for loop iterations
        trace_id: Unique identifier for this execution trace
        executed_data: List of node execution results
        loop_iterations: Tracking for cyclic edges (edge_key -> iteration_count)
        graph_metadata: Metadata about graph structure (for FE rendering hints)
    """

    graph_id: str
    session_id: str
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    iteration_context: Optional[Dict[str, Any]] = None
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    executed_data: List[Dict[str, Any]] = field(default_factory=list)
    loop_iterations: Dict[str, int] = field(default_factory=dict)  # Track loop edge iterations
    graph_metadata: Optional["GraphMetadata"] = None  # Graph structure metadata

    # Event emitter reference (set by executor)
    _event_emitter: Optional[Any] = field(default=None, repr=False)

    async def emit_event(self, event: Any) -> None:
        """Emit an event through the execution context.

        Args:
            event: ExecutionEvent to emit
        """
        if self._event_emitter:
            await self._event_emitter.emit(event)

    def set_iteration_context(
        self, index: int, value: Any, total: int
    ) -> None:
        """Set iteration context for loop execution.

        Args:
            index: Current iteration index (0-based)
            value: Current iteration value
            total: Total number of iterations
        """
        self.iteration_context = {
            "index": index,
            "value": value,
            "is_first": index == 0,
            "is_last": index == total - 1,
            "total": total,
        }

    def clear_iteration_context(self) -> None:
        """Clear iteration context after loop completion."""
        self.iteration_context = None

    def get_loop_iteration(self, edge_key: str) -> int:
        """Get current iteration count for a loop edge.

        Args:
            edge_key: Unique key for the edge (e.g., "source->target")

        Returns:
            Current iteration count (0 if never executed)
        """
        return self.loop_iterations.get(edge_key, 0)

    def increment_loop_iteration(self, edge_key: str) -> int:
        """Increment and return iteration count for a loop edge.

        Args:
            edge_key: Unique key for the edge (e.g., "source->target")

        Returns:
            New iteration count after incrementing
        """
        current = self.loop_iterations.get(edge_key, 0)
        self.loop_iterations[edge_key] = current + 1
        return current + 1

    def add_executed_node(self, node_id: str, output: Any, status: str = "FINISHED") -> None:
        """Record a node execution result.

        Args:
            node_id: ID of the executed node
            output: Output from the node
            status: Execution status
        """
        self.executed_data.append(
            {
                "node_id": node_id,
                "output": output,
                "status": status,
                "timestamp": None,  # Could add timestamp if needed
            }
        )

    def get_node_output(self, node_id: str) -> Optional[Any]:
        """Get the output of a previously executed node.

        Args:
            node_id: ID of the node

        Returns:
            Node output or None if not found
        """
        # Search in reverse to get most recent execution (important for loops)
        for exec_data in reversed(self.executed_data):
            if exec_data["node_id"] == node_id:
                return exec_data.get("output")
        return None

    def append_to_state(self, key: str, value: Any) -> List[Any]:
        """Append a value to a list in state, creating the list if needed.

        This is useful for accumulating observations, messages, or other
        items across multiple node executions without overwriting.

        Args:
            key: State key to append to
            value: Value to append

        Returns:
            The updated list

        Example:
            >>> context.append_to_state("observations", "Found relevant document")
            >>> context.append_to_state("observations", "Verified with source")
            >>> context.state["observations"]
            ['Found relevant document', 'Verified with source']
        """
        if key not in self.state:
            self.state[key] = []
        elif not isinstance(self.state[key], list):
            # Convert existing value to list
            self.state[key] = [self.state[key]]

        self.state[key].append(value)
        return self.state[key]

    def get_from_state(self, key: str, default: Any = None) -> Any:
        """Get a value from state with optional default.

        Args:
            key: State key to retrieve
            default: Default value if key not found

        Returns:
            Value at key or default
        """
        return self.state.get(key, default)

    def set_in_state(self, key: str, value: Any) -> None:
        """Set a value in state.

        Args:
            key: State key to set
            value: Value to set
        """
        self.state[key] = value


class StateManager:
    """Manager for execution state and persistence.

    This class coordinates state management during execution, including
    loading initial state, tracking updates, and persisting final state.
    """

    def __init__(self, backend: Optional[Any] = None):
        """Initialize state manager.

        Args:
            backend: StateBackend instance for persistence
        """
        self.backend = backend

    async def load_state(self, session_id: str) -> Dict[str, Any]:
        """Load state for a session.

        Args:
            session_id: Session identifier

        Returns:
            State dictionary
        """
        if self.backend:
            state = await self.backend.load(session_id)
            return state if state is not None else {}
        return {}

    async def save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        """Save state for a session.

        Args:
            session_id: Session identifier
            state: State dictionary to save
        """
        if self.backend:
            await self.backend.save(session_id, state)

    async def update_state(
        self, session_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update state with new values.

        Args:
            session_id: Session identifier
            updates: Dictionary of updates to apply

        Returns:
            Updated state dictionary
        """
        state = await self.load_state(session_id)
        state.update(updates)
        await self.save_state(session_id, state)
        return state

    async def delete_state(self, session_id: str) -> None:
        """Delete state for a session.

        Args:
            session_id: Session identifier
        """
        if self.backend:
            await self.backend.delete(session_id)
