"""Base protocol for state persistence backends.

This module defines the StateBackend protocol that all storage backends
must implement. It provides a consistent interface for state persistence.
"""

from typing import Protocol, Any, Dict, List, Optional, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from mesh.checkpoints import Checkpoint


@runtime_checkable
class StateBackend(Protocol):
    """Protocol for state persistence backends.

    All state backends must implement these methods to provide
    session state persistence.
    """

    async def save(self, session_id: str, state: Dict[str, Any]) -> None:
        """Save state for a session.

        Args:
            session_id: Session identifier
            state: State dictionary to persist

        Raises:
            Exception: If save operation fails
        """
        ...

    async def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load state for a session.

        Args:
            session_id: Session identifier

        Returns:
            State dictionary or None if not found

        Raises:
            Exception: If load operation fails
        """
        ...

    async def delete(self, session_id: str) -> None:
        """Delete state for a session.

        Args:
            session_id: Session identifier

        Raises:
            Exception: If delete operation fails
        """
        ...

    async def exists(self, session_id: str) -> bool:
        """Check if state exists for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if state exists, False otherwise
        """
        ...

    async def list_sessions(self) -> list[str]:
        """List all session IDs with stored state.

        Returns:
            List of session IDs

        Raises:
            Exception: If list operation fails
        """
        ...

    # Checkpoint methods (optional - for backends that support checkpointing)

    async def save_checkpoint(self, checkpoint: "Checkpoint") -> None:
        """Save a checkpoint.

        Args:
            checkpoint: Checkpoint object to persist

        Raises:
            Exception: If save operation fails
        """
        ...

    async def load_checkpoint(self, checkpoint_id: str) -> Optional["Checkpoint"]:
        """Load a checkpoint by ID.

        Args:
            checkpoint_id: Unique checkpoint identifier

        Returns:
            Checkpoint object or None if not found

        Raises:
            Exception: If load operation fails
        """
        ...

    async def list_checkpoints(
        self, session_id: str, limit: int = 100
    ) -> List["Checkpoint"]:
        """List checkpoints for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of checkpoints to return

        Returns:
            List of Checkpoint objects, ordered by created_at descending

        Raises:
            Exception: If list operation fails
        """
        ...

    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a checkpoint.

        Args:
            checkpoint_id: Unique checkpoint identifier

        Raises:
            Exception: If delete operation fails
        """
        ...

    async def get_latest_checkpoint(self, session_id: str) -> Optional["Checkpoint"]:
        """Get the most recent checkpoint for a session.

        Args:
            session_id: Session identifier

        Returns:
            Latest Checkpoint or None if no checkpoints exist

        Raises:
            Exception: If operation fails
        """
        ...
