"""Checkpointing system for execution state snapshots.

This module provides full state checkpointing with restore, branch, and replay
capabilities for graph execution - achieving LangGraph parity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import uuid


@dataclass
class Checkpoint:
    """Full snapshot of execution state at a point in time.

    Captures everything needed to restore or branch execution:
    - Execution context (state, chat_history, variables)
    - Execution position (current node, queue, waiting nodes)
    - Loop iteration tracking
    - Branching metadata

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint
        session_id: Session this checkpoint belongs to
        graph_id: Graph being executed
        created_at: Timestamp of checkpoint creation

        state: Mutable state dictionary at checkpoint time
        chat_history: Chat messages at checkpoint time
        variables: Global variables at checkpoint time
        executed_data: List of node execution results
        loop_iterations: Loop edge iteration counts

        current_node_id: Node being executed when checkpoint created
        pending_queue: Nodes waiting to execute [(node_id, inputs), ...]
        waiting_nodes: Multi-parent nodes waiting for inputs

        parent_checkpoint_id: Parent checkpoint if branched
        state_hash: SHA-256 hash for integrity verification
        tags: Optional tags for filtering/searching
        metadata: Additional metadata (e.g., label, description)
    """

    checkpoint_id: str
    session_id: str
    graph_id: str
    created_at: datetime

    # Execution context state
    state: Dict[str, Any]
    chat_history: List[Dict[str, str]]
    variables: Dict[str, Any]
    executed_data: List[Dict[str, Any]]
    loop_iterations: Dict[str, int]

    # Execution position
    current_node_id: Optional[str] = None
    pending_queue: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    waiting_nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Branching and integrity
    parent_checkpoint_id: Optional[str] = None
    state_hash: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute state hash if not provided."""
        if not self.state_hash:
            self.state_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of state for integrity checking.

        Returns:
            Hex string of first 16 chars of SHA-256 hash
        """
        content = json.dumps(
            {
                "state": self.state,
                "chat_history": self.chat_history,
                "variables": self.variables,
                "executed_data": self.executed_data,
                "loop_iterations": self.loop_iterations,
            },
            sort_keys=True,
            default=str,  # Handle datetime and other non-serializable types
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify checkpoint integrity by recomputing hash.

        Returns:
            True if current state matches stored hash
        """
        return self._compute_hash() == self.state_hash

    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint to dictionary for storage.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "graph_id": self.graph_id,
            "created_at": self.created_at.isoformat(),
            "state": self.state,
            "chat_history": self.chat_history,
            "variables": self.variables,
            "executed_data": self.executed_data,
            "loop_iterations": self.loop_iterations,
            "current_node_id": self.current_node_id,
            "pending_queue": self.pending_queue,
            "waiting_nodes": self.waiting_nodes,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "state_hash": self.state_hash,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Deserialize checkpoint from dictionary.

        Args:
            data: Dictionary representation from to_dict()

        Returns:
            Checkpoint instance
        """
        # Handle datetime parsing
        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            checkpoint_id=data["checkpoint_id"],
            session_id=data["session_id"],
            graph_id=data["graph_id"],
            created_at=created_at,
            state=data["state"],
            chat_history=data["chat_history"],
            variables=data["variables"],
            executed_data=data["executed_data"],
            loop_iterations=data["loop_iterations"],
            current_node_id=data.get("current_node_id"),
            pending_queue=data.get("pending_queue", []),
            waiting_nodes=data.get("waiting_nodes", {}),
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            state_hash=data.get("state_hash", ""),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def create(
        cls,
        session_id: str,
        graph_id: str,
        state: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        variables: Dict[str, Any],
        executed_data: List[Dict[str, Any]],
        loop_iterations: Dict[str, int],
        current_node_id: Optional[str] = None,
        pending_queue: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
        waiting_nodes: Optional[Dict[str, Dict[str, Any]]] = None,
        parent_checkpoint_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Checkpoint":
        """Create a new checkpoint with auto-generated ID and timestamp.

        Args:
            session_id: Session identifier
            graph_id: Graph identifier
            state: Current state dictionary
            chat_history: Current chat history
            variables: Current variables
            executed_data: Executed node data
            loop_iterations: Loop iteration counts
            current_node_id: Currently executing node
            pending_queue: Queue of pending nodes
            waiting_nodes: Nodes waiting for multi-parent inputs
            parent_checkpoint_id: Parent if branching
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            New Checkpoint instance
        """
        return cls(
            checkpoint_id=str(uuid.uuid4()),
            session_id=session_id,
            graph_id=graph_id,
            created_at=datetime.now(),
            state=dict(state),  # Copy to avoid mutations
            chat_history=list(chat_history),
            variables=dict(variables),
            executed_data=list(executed_data),
            loop_iterations=dict(loop_iterations),
            current_node_id=current_node_id,
            pending_queue=list(pending_queue) if pending_queue else [],
            waiting_nodes=dict(waiting_nodes) if waiting_nodes else {},
            parent_checkpoint_id=parent_checkpoint_id,
            tags=list(tags) if tags else [],
            metadata=dict(metadata) if metadata else {},
        )


@dataclass
class CheckpointConfig:
    """Configuration for automatic checkpointing behavior.

    Attributes:
        auto_checkpoint: Whether to create checkpoints automatically
        checkpoint_interval: Create checkpoint every N nodes (0 = disabled)
        checkpoint_nodes: Specific nodes to checkpoint after
        max_checkpoints: Maximum checkpoints to retain per session
        retention_days: Delete checkpoints older than this (0 = keep forever)
    """

    auto_checkpoint: bool = False
    checkpoint_interval: int = 0  # 0 = disabled, N = every N nodes
    checkpoint_nodes: List[str] = field(default_factory=list)  # Checkpoint after these
    max_checkpoints: int = 100  # Max per session
    retention_days: int = 30  # 0 = keep forever


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""

    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when checkpoint is not found."""

    def __init__(self, checkpoint_id: str):
        self.checkpoint_id = checkpoint_id
        super().__init__(f"Checkpoint not found: {checkpoint_id}")


class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint fails integrity verification."""

    def __init__(self, checkpoint_id: str, expected_hash: str, actual_hash: str):
        self.checkpoint_id = checkpoint_id
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            f"Checkpoint integrity check failed: {checkpoint_id}. "
            f"Expected hash {expected_hash}, got {actual_hash}"
        )
