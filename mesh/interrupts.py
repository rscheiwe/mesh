"""Interrupt system for human-in-the-loop workflows.

This module provides interrupt_before and interrupt_after functionality
for pausing execution at strategic points and allowing human review/modification.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import uuid


@dataclass
class InterruptConfig:
    """Configuration for an interrupt point.

    Attributes:
        node_id: Node to interrupt before/after
        position: "before" or "after" node execution
        condition: Optional callable (state, input/output) -> bool
        metadata_extractor: Optional callable to extract data for review
        timeout: Optional timeout in seconds (None = wait forever)
    """

    node_id: str
    position: str  # "before" or "after"
    condition: Optional[Callable[[Dict[str, Any], Any], bool]] = None
    metadata_extractor: Optional[Callable[[Dict[str, Any], Any], Dict[str, Any]]] = None
    timeout: Optional[float] = None


@dataclass
class InterruptState:
    """State captured when an interrupt is triggered.

    Contains all information needed to resume execution after human review.

    Attributes:
        interrupt_id: Unique identifier for this interrupt
        node_id: Node where interrupt occurred
        position: "before" or "after"
        state: Current execution state at interrupt
        input_data: Input to the node (always present)
        output_data: Output from node (only for "after" interrupts)
        metadata: Extracted metadata for review
        pending_queue: Nodes waiting to execute
        waiting_nodes: Multi-parent nodes waiting for inputs
        loop_counts: Loop iteration counters
        created_at: Timestamp of interrupt
    """

    interrupt_id: str
    node_id: str
    position: str  # "before" or "after"
    state: Dict[str, Any]
    input_data: Any
    output_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    pending_queue: List[tuple] = field(default_factory=list)
    waiting_nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    loop_counts: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(
        cls,
        node_id: str,
        position: str,
        state: Dict[str, Any],
        input_data: Any,
        output_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        pending_queue: Optional[List[tuple]] = None,
        waiting_nodes: Optional[Dict[str, Dict[str, Any]]] = None,
        loop_counts: Optional[Dict[str, int]] = None,
    ) -> "InterruptState":
        """Create a new interrupt state with auto-generated ID."""
        return cls(
            interrupt_id=str(uuid.uuid4()),
            node_id=node_id,
            position=position,
            state=dict(state),
            input_data=input_data,
            output_data=output_data,
            metadata=dict(metadata) if metadata else {},
            pending_queue=list(pending_queue) if pending_queue else [],
            waiting_nodes=dict(waiting_nodes) if waiting_nodes else {},
            loop_counts=dict(loop_counts) if loop_counts else {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "interrupt_id": self.interrupt_id,
            "node_id": self.node_id,
            "position": self.position,
            "state": self.state,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metadata": self.metadata,
            "pending_queue": self.pending_queue,
            "waiting_nodes": self.waiting_nodes,
            "loop_counts": self.loop_counts,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterruptState":
        """Deserialize from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            interrupt_id=data["interrupt_id"],
            node_id=data["node_id"],
            position=data["position"],
            state=data["state"],
            input_data=data["input_data"],
            output_data=data.get("output_data"),
            metadata=data.get("metadata", {}),
            pending_queue=data.get("pending_queue", []),
            waiting_nodes=data.get("waiting_nodes", {}),
            loop_counts=data.get("loop_counts", {}),
            created_at=created_at or datetime.now(),
        )


@dataclass
class InterruptResume:
    """Instructions for resuming from an interrupt.

    Attributes:
        modified_state: State modifications to apply before resuming
        modified_input: Modified input for the node (before interrupts only)
        skip_node: If True, skip the interrupted node entirely
        metadata: Additional metadata about the resume action
    """

    modified_state: Optional[Dict[str, Any]] = None
    modified_input: Optional[Any] = None
    skip_node: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterruptReject:
    """Instructions for rejecting/aborting at an interrupt.

    Attributes:
        reason: Human-readable reason for rejection
        metadata: Additional metadata about the rejection
    """

    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class InterruptError(Exception):
    """Base exception for interrupt operations."""

    pass


class InterruptNotFoundError(InterruptError):
    """Raised when interrupt state is not found."""

    def __init__(self, interrupt_id: str):
        self.interrupt_id = interrupt_id
        super().__init__(f"Interrupt not found: {interrupt_id}")


class InterruptTimeoutError(InterruptError):
    """Raised when interrupt times out waiting for response."""

    def __init__(self, node_id: str, timeout: float):
        self.node_id = node_id
        self.timeout = timeout
        super().__init__(f"Interrupt at node '{node_id}' timed out after {timeout}s")
