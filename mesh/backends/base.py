"""Base protocol for state persistence backends.

This module defines the StateBackend protocol that all storage backends
must implement. It provides a consistent interface for state persistence.
"""

from typing import Protocol, Any, Dict, Optional, runtime_checkable


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
