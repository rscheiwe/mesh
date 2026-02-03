"""In-memory state backend for testing and development.

This backend stores state in memory and is useful for testing,
development, and scenarios where persistence is not required.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING

from mesh.backends.base import StateBackend

if TYPE_CHECKING:
    from mesh.checkpoints import Checkpoint


class MemoryBackend:
    """In-memory state storage backend.

    This backend stores all state in a dictionary in memory.
    State is lost when the process terminates.

    Useful for:
    - Testing
    - Development
    - Ephemeral workflows
    - Single-request executions
    """

    def __init__(self):
        """Initialize memory backend with empty storage."""
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._checkpoints: Dict[str, "Checkpoint"] = {}  # checkpoint_id -> Checkpoint

    async def save(self, session_id: str, state: Dict[str, Any]) -> None:
        """Save state to memory.

        Args:
            session_id: Session identifier
            state: State dictionary to store
        """
        # Make a copy to avoid external mutations
        self._storage[session_id] = dict(state)

    async def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load state from memory.

        Args:
            session_id: Session identifier

        Returns:
            State dictionary or None if not found
        """
        if session_id in self._storage:
            # Return a copy to avoid external mutations
            return dict(self._storage[session_id])
        return None

    async def delete(self, session_id: str) -> None:
        """Delete state from memory.

        Args:
            session_id: Session identifier
        """
        if session_id in self._storage:
            del self._storage[session_id]

    async def exists(self, session_id: str) -> bool:
        """Check if state exists in memory.

        Args:
            session_id: Session identifier

        Returns:
            True if state exists
        """
        return session_id in self._storage

    async def list_sessions(self) -> list[str]:
        """List all session IDs in memory.

        Returns:
            List of session IDs
        """
        return list(self._storage.keys())

    def clear_all(self) -> None:
        """Clear all stored state and checkpoints.

        Useful for testing and cleanup.
        """
        self._storage.clear()
        self._checkpoints.clear()

    # Checkpoint methods

    async def save_checkpoint(self, checkpoint: "Checkpoint") -> None:
        """Save a checkpoint to memory.

        Args:
            checkpoint: Checkpoint object to store
        """
        from mesh.checkpoints import Checkpoint as CheckpointClass

        # Store a copy to avoid mutations
        self._checkpoints[checkpoint.checkpoint_id] = CheckpointClass.from_dict(
            checkpoint.to_dict()
        )

    async def load_checkpoint(self, checkpoint_id: str) -> Optional["Checkpoint"]:
        """Load a checkpoint from memory.

        Args:
            checkpoint_id: Unique checkpoint identifier

        Returns:
            Checkpoint object or None if not found
        """
        from mesh.checkpoints import Checkpoint as CheckpointClass

        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint:
            # Return a copy to avoid mutations
            return CheckpointClass.from_dict(checkpoint.to_dict())
        return None

    async def list_checkpoints(
        self, session_id: str, limit: int = 100
    ) -> List["Checkpoint"]:
        """List checkpoints for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number to return

        Returns:
            List of Checkpoints ordered by created_at descending
        """
        from mesh.checkpoints import Checkpoint as CheckpointClass

        session_checkpoints = [
            cp for cp in self._checkpoints.values() if cp.session_id == session_id
        ]
        # Sort by created_at descending (newest first)
        session_checkpoints.sort(key=lambda cp: cp.created_at, reverse=True)

        # Return copies
        return [
            CheckpointClass.from_dict(cp.to_dict())
            for cp in session_checkpoints[:limit]
        ]

    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a checkpoint from memory.

        Args:
            checkpoint_id: Unique checkpoint identifier
        """
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]

    async def get_latest_checkpoint(self, session_id: str) -> Optional["Checkpoint"]:
        """Get the most recent checkpoint for a session.

        Args:
            session_id: Session identifier

        Returns:
            Latest Checkpoint or None
        """
        checkpoints = await self.list_checkpoints(session_id, limit=1)
        return checkpoints[0] if checkpoints else None

    def __repr__(self) -> str:
        return f"MemoryBackend(sessions={len(self._storage)}, checkpoints={len(self._checkpoints)})"
