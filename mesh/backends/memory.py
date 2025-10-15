"""In-memory state backend for testing and development.

This backend stores state in memory and is useful for testing,
development, and scenarios where persistence is not required.
"""

from typing import Dict, Any, Optional

from mesh.backends.base import StateBackend


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
        """Clear all stored state.

        Useful for testing and cleanup.
        """
        self._storage.clear()

    def __repr__(self) -> str:
        return f"MemoryBackend(sessions={len(self._storage)})"
