"""SQLite state backend for persistent storage.

This backend uses SQLite for async state persistence, providing
durability across process restarts.
"""

import aiosqlite
import json
from typing import Dict, Any, Optional
from pathlib import Path

from mesh.backends.base import StateBackend


class SQLiteBackend:
    """SQLite-based state persistence backend.

    This backend stores state in a SQLite database with async support.
    It provides durable storage that survives process restarts.

    The database schema:
    - session_id: TEXT PRIMARY KEY
    - state: TEXT (JSON-encoded state)
    - created_at: TIMESTAMP
    - updated_at: TIMESTAMP
    """

    def __init__(self, db_path: str = "mesh_state.db"):
        """Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure database and table exist."""
        if self._initialized:
            return

        # Create directory if needed
        db_dir = Path(self.db_path).parent
        if db_dir and not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS mesh_state (
                    session_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Create index on updated_at for efficient queries
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_updated_at
                ON mesh_state(updated_at)
                """
            )
            await db.commit()

        self._initialized = True

    async def save(self, session_id: str, state: Dict[str, Any]) -> None:
        """Save state to SQLite database.

        Args:
            session_id: Session identifier
            state: State dictionary to persist
        """
        await self._ensure_initialized()

        state_json = json.dumps(state)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO mesh_state (session_id, state, created_at, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id)
                DO UPDATE SET
                    state = ?,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, state_json, state_json),
            )
            await db.commit()

    async def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load state from SQLite database.

        Args:
            session_id: Session identifier

        Returns:
            State dictionary or None if not found
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT state FROM mesh_state WHERE session_id = ?",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None

    async def delete(self, session_id: str) -> None:
        """Delete state from SQLite database.

        Args:
            session_id: Session identifier
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM mesh_state WHERE session_id = ?",
                (session_id,),
            )
            await db.commit()

    async def exists(self, session_id: str) -> bool:
        """Check if state exists in database.

        Args:
            session_id: Session identifier

        Returns:
            True if state exists
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT 1 FROM mesh_state WHERE session_id = ? LIMIT 1",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
                return row is not None

    async def list_sessions(self) -> list[str]:
        """List all session IDs in database.

        Returns:
            List of session IDs ordered by updated_at (newest first)
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT session_id FROM mesh_state ORDER BY updated_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]

    async def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """Delete sessions older than specified age.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of sessions deleted
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                DELETE FROM mesh_state
                WHERE updated_at < datetime('now', '-' || ? || ' days')
                """,
                (max_age_days,),
            )
            deleted = cursor.rowcount
            await db.commit()
            return deleted

    def __repr__(self) -> str:
        return f"SQLiteBackend(db_path='{self.db_path}')"
