"""State management for graph execution."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

T = TypeVar("T", bound="GraphState")


@dataclass
class StateSnapshot:
    """Snapshot of the graph state at a point in time."""

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "node_outputs": self.node_outputs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateSnapshot":
        """Create snapshot from dictionary."""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
            node_outputs=data["node_outputs"],
            metadata=data.get("metadata", {}),
        )


class GraphState(BaseModel):
    """Mutable state object passed through graph execution."""

    data: Dict[str, Any] = Field(default_factory=dict)
    node_outputs: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._history: List[StateSnapshot] = []
        self._lock: asyncio.Lock = asyncio.Lock()

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from state data.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Value from state or default
        """
        async with self._lock:
            return self.data.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        """Set a value in state data.

        Args:
            key: Key to set
            value: Value to set
        """
        async with self._lock:
            self.data[key] = value

    async def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple values in state data.

        Args:
            updates: Dictionary of updates to apply
        """
        async with self._lock:
            self.data.update(updates)

    async def add_node_output(self, node_id: str, output: Any) -> None:
        """Add output from a node execution.

        Args:
            node_id: ID of the node
            output: Output from the node
        """
        async with self._lock:
            self.node_outputs[node_id] = output

    async def get_node_output(self, node_id: str) -> Optional[Any]:
        """Get output from a specific node.

        Args:
            node_id: ID of the node

        Returns:
            Node output if available
        """
        async with self._lock:
            return self.node_outputs.get(node_id)

    async def snapshot(self) -> StateSnapshot:
        """Create a snapshot of the current state.

        Returns:
            StateSnapshot of current state
        """
        async with self._lock:
            return StateSnapshot(
                data=self.data.copy(),
                node_outputs=self.node_outputs.copy(),
                metadata=self.metadata.copy(),
            )

    async def restore(self, snapshot: StateSnapshot) -> None:
        """Restore state from a snapshot.

        Args:
            snapshot: Snapshot to restore from
        """
        async with self._lock:
            self.data = snapshot.data.copy()
            self.node_outputs = snapshot.node_outputs.copy()
            self.metadata = snapshot.metadata.copy()

    async def add_to_history(self, snapshot: Optional[StateSnapshot] = None) -> None:
        """Add current state or snapshot to history.

        Args:
            snapshot: Optional snapshot to add, creates new if None
        """
        if snapshot is None:
            snapshot = await self.snapshot()
        self._history.append(snapshot)

    def get_history(self) -> List[StateSnapshot]:
        """Get state history.

        Returns:
            List of state snapshots
        """
        return self._history.copy()

    async def save_to_file(self, filepath: Path) -> None:
        """Save state to file.

        Args:
            filepath: Path to save state to
        """
        snapshot = await self.snapshot()
        filepath.write_text(json.dumps(snapshot.to_dict(), indent=2))

    @classmethod
    async def load_from_file(cls: Type[T], filepath: Path) -> T:
        """Load state from file.

        Args:
            filepath: Path to load state from

        Returns:
            GraphState instance
        """
        data = json.loads(filepath.read_text())
        snapshot = StateSnapshot.from_dict(data)

        state = cls()
        await state.restore(snapshot)
        return state

    def clear(self) -> None:
        """Clear all state data."""
        self.data.clear()
        self.node_outputs.clear()
        self.metadata.clear()
        self._history.clear()
