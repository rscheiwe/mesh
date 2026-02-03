"""Tests for checkpointing system."""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

from mesh.checkpoints import (
    Checkpoint,
    CheckpointConfig,
    CheckpointNotFoundError,
    CheckpointIntegrityError,
)
from mesh.backends.memory import MemoryBackend
from mesh.backends.sqlite import SQLiteBackend
from mesh.core.state import ExecutionContext
from mesh.core.executor import Executor
from mesh.core.events import EventType
from mesh.builders.state_graph import StateGraph
from mesh.nodes.base import BaseNode, NodeResult


# Test fixtures

@pytest.fixture
def sample_checkpoint_data():
    """Sample data for creating checkpoints."""
    return {
        "session_id": "test-session-123",
        "graph_id": "test-graph",
        "state": {"counter": 5, "results": ["a", "b"]},
        "chat_history": [{"role": "user", "content": "Hello"}],
        "variables": {"api_key": "test"},
        "executed_data": [
            {"node_id": "node1", "output": "result1", "status": "FINISHED"}
        ],
        "loop_iterations": {"node1->node2": 3},
    }


@pytest.fixture
def memory_backend():
    """Create fresh memory backend."""
    return MemoryBackend()


@pytest.fixture
async def sqlite_backend(tmp_path):
    """Create temporary SQLite backend."""
    db_path = str(tmp_path / "test_checkpoints.db")
    backend = SQLiteBackend(db_path)
    return backend


# Checkpoint class tests

class TestCheckpointClass:
    """Tests for Checkpoint dataclass."""

    def test_checkpoint_create(self, sample_checkpoint_data):
        """Test creating a checkpoint with auto-generated fields."""
        checkpoint = Checkpoint.create(**sample_checkpoint_data)

        assert checkpoint.checkpoint_id is not None
        assert len(checkpoint.checkpoint_id) == 36  # UUID format
        assert checkpoint.session_id == "test-session-123"
        assert checkpoint.graph_id == "test-graph"
        assert checkpoint.state == {"counter": 5, "results": ["a", "b"]}
        assert isinstance(checkpoint.created_at, datetime)
        assert checkpoint.state_hash != ""

    def test_checkpoint_hash_computation(self, sample_checkpoint_data):
        """Test state hash is computed correctly."""
        cp1 = Checkpoint.create(**sample_checkpoint_data)
        cp2 = Checkpoint.create(**sample_checkpoint_data)

        # Same state should produce same hash
        assert cp1.state_hash == cp2.state_hash

        # Different state should produce different hash
        sample_checkpoint_data["state"]["counter"] = 10
        cp3 = Checkpoint.create(**sample_checkpoint_data)
        assert cp3.state_hash != cp1.state_hash

    def test_checkpoint_verify_integrity(self, sample_checkpoint_data):
        """Test integrity verification."""
        checkpoint = Checkpoint.create(**sample_checkpoint_data)

        # Original should pass
        assert checkpoint.verify_integrity() is True

        # Modify state after creation
        checkpoint.state["tampered"] = True

        # Should fail integrity check
        assert checkpoint.verify_integrity() is False

    def test_checkpoint_serialization(self, sample_checkpoint_data):
        """Test to_dict and from_dict round-trip."""
        original = Checkpoint.create(**sample_checkpoint_data)
        data = original.to_dict()

        # Check serialization
        assert data["session_id"] == "test-session-123"
        assert "checkpoint_id" in data
        assert "created_at" in data

        # Deserialize
        restored = Checkpoint.from_dict(data)

        assert restored.checkpoint_id == original.checkpoint_id
        assert restored.session_id == original.session_id
        assert restored.state == original.state
        assert restored.state_hash == original.state_hash

    def test_checkpoint_with_branching(self, sample_checkpoint_data):
        """Test checkpoint with parent (branching)."""
        parent = Checkpoint.create(**sample_checkpoint_data)

        child = Checkpoint.create(
            **sample_checkpoint_data,
            parent_checkpoint_id=parent.checkpoint_id,
            tags=["branch", "experiment"],
            metadata={"branch_name": "exploration"},
        )

        assert child.parent_checkpoint_id == parent.checkpoint_id
        assert "branch" in child.tags
        assert child.metadata["branch_name"] == "exploration"


class TestCheckpointConfig:
    """Tests for CheckpointConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CheckpointConfig()

        assert config.auto_checkpoint is False
        assert config.checkpoint_interval == 0
        assert config.checkpoint_nodes == []
        assert config.max_checkpoints == 100
        assert config.retention_days == 30

    def test_custom_config(self):
        """Test custom configuration."""
        config = CheckpointConfig(
            auto_checkpoint=True,
            checkpoint_interval=5,
            checkpoint_nodes=["reviewer", "approver"],
            max_checkpoints=50,
            retention_days=7,
        )

        assert config.auto_checkpoint is True
        assert config.checkpoint_interval == 5
        assert config.checkpoint_nodes == ["reviewer", "approver"]


# Backend tests

class TestMemoryBackendCheckpoints:
    """Tests for MemoryBackend checkpoint methods."""

    @pytest.mark.asyncio
    async def test_save_and_load_checkpoint(self, memory_backend, sample_checkpoint_data):
        """Test saving and loading a checkpoint."""
        checkpoint = Checkpoint.create(**sample_checkpoint_data)

        await memory_backend.save_checkpoint(checkpoint)
        loaded = await memory_backend.load_checkpoint(checkpoint.checkpoint_id)

        assert loaded is not None
        assert loaded.checkpoint_id == checkpoint.checkpoint_id
        assert loaded.state == checkpoint.state

    @pytest.mark.asyncio
    async def test_load_nonexistent_checkpoint(self, memory_backend):
        """Test loading checkpoint that doesn't exist."""
        loaded = await memory_backend.load_checkpoint("nonexistent-id")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, memory_backend, sample_checkpoint_data):
        """Test listing checkpoints for a session."""
        # Create multiple checkpoints
        for i in range(5):
            data = dict(sample_checkpoint_data)
            data["state"] = {"counter": i}
            checkpoint = Checkpoint.create(**data)
            await memory_backend.save_checkpoint(checkpoint)
            await asyncio.sleep(0.01)  # Small delay for ordering

        checkpoints = await memory_backend.list_checkpoints("test-session-123")

        assert len(checkpoints) == 5
        # Should be ordered by created_at descending
        for i in range(len(checkpoints) - 1):
            assert checkpoints[i].created_at >= checkpoints[i + 1].created_at

    @pytest.mark.asyncio
    async def test_list_checkpoints_with_limit(self, memory_backend, sample_checkpoint_data):
        """Test listing checkpoints with limit."""
        for i in range(10):
            data = dict(sample_checkpoint_data)
            data["state"] = {"counter": i}
            checkpoint = Checkpoint.create(**data)
            await memory_backend.save_checkpoint(checkpoint)

        checkpoints = await memory_backend.list_checkpoints("test-session-123", limit=3)
        assert len(checkpoints) == 3

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, memory_backend, sample_checkpoint_data):
        """Test deleting a checkpoint."""
        checkpoint = Checkpoint.create(**sample_checkpoint_data)
        await memory_backend.save_checkpoint(checkpoint)

        # Verify it exists
        loaded = await memory_backend.load_checkpoint(checkpoint.checkpoint_id)
        assert loaded is not None

        # Delete
        await memory_backend.delete_checkpoint(checkpoint.checkpoint_id)

        # Verify it's gone
        loaded = await memory_backend.load_checkpoint(checkpoint.checkpoint_id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint(self, memory_backend, sample_checkpoint_data):
        """Test getting the latest checkpoint."""
        # Create checkpoints with different timestamps
        for i in range(3):
            data = dict(sample_checkpoint_data)
            data["state"] = {"counter": i}
            checkpoint = Checkpoint.create(**data)
            await memory_backend.save_checkpoint(checkpoint)
            await asyncio.sleep(0.01)  # Ensure different timestamps
            last_checkpoint = checkpoint

        latest = await memory_backend.get_latest_checkpoint("test-session-123")

        assert latest is not None
        assert latest.checkpoint_id == last_checkpoint.checkpoint_id

    @pytest.mark.asyncio
    async def test_clear_all_includes_checkpoints(self, memory_backend, sample_checkpoint_data):
        """Test that clear_all clears checkpoints too."""
        checkpoint = Checkpoint.create(**sample_checkpoint_data)
        await memory_backend.save_checkpoint(checkpoint)
        await memory_backend.save("session", {"key": "value"})

        memory_backend.clear_all()

        assert await memory_backend.load_checkpoint(checkpoint.checkpoint_id) is None
        assert await memory_backend.load("session") is None


class TestSQLiteBackendCheckpoints:
    """Tests for SQLiteBackend checkpoint methods."""

    @pytest.mark.asyncio
    async def test_save_and_load_checkpoint(self, sqlite_backend, sample_checkpoint_data):
        """Test saving and loading a checkpoint."""
        checkpoint = Checkpoint.create(**sample_checkpoint_data)

        await sqlite_backend.save_checkpoint(checkpoint)
        loaded = await sqlite_backend.load_checkpoint(checkpoint.checkpoint_id)

        assert loaded is not None
        assert loaded.checkpoint_id == checkpoint.checkpoint_id
        assert loaded.state == checkpoint.state
        assert loaded.state_hash == checkpoint.state_hash

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, sqlite_backend, sample_checkpoint_data):
        """Test listing checkpoints."""
        for i in range(5):
            data = dict(sample_checkpoint_data)
            data["state"] = {"counter": i}
            checkpoint = Checkpoint.create(**data)
            await sqlite_backend.save_checkpoint(checkpoint)

        checkpoints = await sqlite_backend.list_checkpoints("test-session-123")
        assert len(checkpoints) == 5

    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints(self, sqlite_backend, sample_checkpoint_data):
        """Test cleaning up old checkpoints."""
        # Create 10 checkpoints
        for i in range(10):
            data = dict(sample_checkpoint_data)
            data["state"] = {"counter": i}
            checkpoint = Checkpoint.create(**data)
            await sqlite_backend.save_checkpoint(checkpoint)

        # Keep only 3
        deleted = await sqlite_backend.cleanup_old_checkpoints(
            "test-session-123", keep_count=3
        )

        assert deleted == 7
        remaining = await sqlite_backend.list_checkpoints("test-session-123")
        assert len(remaining) == 3


# Executor checkpoint tests

class SimpleToolNode(BaseNode):
    """Simple node for testing."""

    def __init__(self, id: str, output_value: Any = "result"):
        super().__init__(id, {"type": "tool"})
        self.output_value = output_value

    async def _execute_impl(self, input: Any, context: ExecutionContext) -> NodeResult:
        # Update state
        counter = context.state.get("counter", 0) + 1
        return NodeResult(
            output={"value": self.output_value, "counter": counter},
            state={"counter": counter},
        )


class TestExecutorCheckpoints:
    """Tests for Executor checkpoint methods."""

    @pytest.fixture
    def simple_graph_with_tool(self):
        """Create a simple graph with tool nodes."""
        graph = StateGraph()
        graph.add_node("tool1", SimpleToolNode("tool1", "result1"), node_type="tool")
        graph.add_node("tool2", SimpleToolNode("tool2", "result2"), node_type="tool")
        graph.add_edge("START", "tool1")
        graph.add_edge("tool1", "tool2")
        graph.set_entry_point("tool1")
        return graph.compile()

    @pytest.mark.asyncio
    async def test_checkpoint_creation(self, simple_graph_with_tool, memory_backend):
        """Test creating a checkpoint during execution."""
        executor = Executor(simple_graph_with_tool, memory_backend)
        context = ExecutionContext(
            graph_id="test-graph",
            session_id="test-session",
            chat_history=[],
            variables={},
            state={},
        )

        # Execute first node
        events = []
        async for event in executor.execute("test input", context):
            events.append(event)

        # Create checkpoint
        checkpoint_id = await executor.checkpoint(context)

        assert checkpoint_id is not None
        assert len(checkpoint_id) == 36  # UUID

        # Verify checkpoint was saved
        checkpoint = await memory_backend.load_checkpoint(checkpoint_id)
        assert checkpoint is not None
        assert checkpoint.session_id == "test-session"

    @pytest.mark.asyncio
    async def test_checkpoint_restore(self, simple_graph_with_tool, memory_backend):
        """Test restoring from a checkpoint."""
        executor = Executor(simple_graph_with_tool, memory_backend)
        context = ExecutionContext(
            graph_id="test-graph",
            session_id="test-session",
            chat_history=[],
            variables={"key": "value"},
            state={"initial": True},
        )

        # Execute and checkpoint
        async for _ in executor.execute("test", context):
            pass

        checkpoint_id = await executor.checkpoint(context)

        # Restore
        restored_context = await executor.restore(checkpoint_id)

        assert restored_context.session_id == "test-session"
        assert restored_context.graph_id == "test-graph"
        assert "counter" in restored_context.state
        assert restored_context.variables["key"] == "value"

    @pytest.mark.asyncio
    async def test_checkpoint_restore_not_found(self, simple_graph_with_tool, memory_backend):
        """Test restore raises error for missing checkpoint."""
        executor = Executor(simple_graph_with_tool, memory_backend)

        with pytest.raises(CheckpointNotFoundError) as exc_info:
            await executor.restore("nonexistent-id")

        assert "nonexistent-id" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_checkpoint_branch(self, simple_graph_with_tool, memory_backend):
        """Test branching from a checkpoint."""
        executor = Executor(simple_graph_with_tool, memory_backend)
        context = ExecutionContext(
            graph_id="test-graph",
            session_id="main-session",
            chat_history=[],
            variables={},
            state={"branch_point": True},
        )

        # Execute and checkpoint
        async for _ in executor.execute("test", context):
            pass

        checkpoint_id = await executor.checkpoint(context)

        # Create branch
        branch_context = await executor.branch(
            checkpoint_id, "branch-session", branch_name="experiment-1"
        )

        assert branch_context.session_id == "branch-session"
        assert branch_context.state["branch_point"] is True
        assert "counter" in branch_context.state

        # Verify branch checkpoint was created
        branch_checkpoints = await memory_backend.list_checkpoints("branch-session")
        assert len(branch_checkpoints) == 1
        assert branch_checkpoints[0].parent_checkpoint_id == checkpoint_id

    @pytest.mark.asyncio
    async def test_checkpoint_without_backend(self, simple_graph_with_tool):
        """Test checkpoint raises error without backend."""
        executor = Executor(simple_graph_with_tool, state_backend=None)
        context = ExecutionContext(
            graph_id="test",
            session_id="test",
            chat_history=[],
            variables={},
            state={},
        )

        with pytest.raises(ValueError, match="no state_backend"):
            await executor.checkpoint(context)

    @pytest.mark.asyncio
    async def test_checkpoint_with_tags(self, simple_graph_with_tool, memory_backend):
        """Test creating checkpoint with tags and metadata."""
        executor = Executor(simple_graph_with_tool, memory_backend)
        context = ExecutionContext(
            graph_id="test-graph",
            session_id="test-session",
            chat_history=[],
            variables={},
            state={},
        )

        async for _ in executor.execute("test", context):
            pass

        checkpoint_id = await executor.checkpoint(
            context,
            tags=["review", "important"],
            metadata={"label": "Before review step"},
        )

        checkpoint = await memory_backend.load_checkpoint(checkpoint_id)
        assert "review" in checkpoint.tags
        assert checkpoint.metadata["label"] == "Before review step"


class TestCheckpointReplay:
    """Tests for checkpoint replay functionality."""

    @pytest.fixture
    def multi_node_graph(self):
        """Create graph with multiple nodes for replay testing."""
        graph = StateGraph()
        graph.add_node("node1", SimpleToolNode("node1", "result1"), node_type="tool")
        graph.add_node("node2", SimpleToolNode("node2", "result2"), node_type="tool")
        graph.add_node("node3", SimpleToolNode("node3", "result3"), node_type="tool")
        graph.add_edge("START", "node1")
        graph.add_edge("node1", "node2")
        graph.add_edge("node2", "node3")
        graph.set_entry_point("node1")
        return graph.compile()

    @pytest.mark.asyncio
    async def test_replay_from_checkpoint(self, multi_node_graph, memory_backend):
        """Test replaying execution from a checkpoint."""
        executor = Executor(multi_node_graph, memory_backend)
        context = ExecutionContext(
            graph_id="test",
            session_id="test",
            chat_history=[],
            variables={},
            state={},
        )

        # Execute full graph
        async for _ in executor.execute("test", context):
            pass

        # Create checkpoint with pending queue (simulating mid-execution)
        from mesh.core.executor import NodeQueueItem
        queue = [NodeQueueItem(node_id="node2", inputs={"value": "test"})]

        checkpoint_id = await executor.checkpoint(context, queue=queue)

        # Replay
        events = []
        async for event in executor.replay(checkpoint_id):
            events.append(event)

        # Should have replay events
        event_types = [e.type for e in events]
        assert EventType.REPLAY_START in event_types
        assert EventType.REPLAY_COMPLETE in event_types


class TestCheckpointIntegrity:
    """Tests for checkpoint integrity features."""

    @pytest.mark.asyncio
    async def test_integrity_error_on_tampered_checkpoint(
        self, memory_backend, sample_checkpoint_data
    ):
        """Test that tampered checkpoints are detected."""
        checkpoint = Checkpoint.create(**sample_checkpoint_data)
        await memory_backend.save_checkpoint(checkpoint)

        # Manually tamper with stored checkpoint
        stored = memory_backend._checkpoints[checkpoint.checkpoint_id]
        stored.state["tampered"] = True

        # Create executor and try to restore
        from mesh.builders.state_graph import StateGraph

        graph = StateGraph()
        graph.add_node("tool", SimpleToolNode("tool"), node_type="tool")
        graph.add_edge("START", "tool")
        graph.set_entry_point("tool")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        with pytest.raises(CheckpointIntegrityError):
            await executor.restore(checkpoint.checkpoint_id)


class TestCheckpointEdgeCases:
    """Tests for edge cases in checkpointing."""

    @pytest.mark.asyncio
    async def test_checkpoint_with_loop_iterations(self, memory_backend):
        """Test checkpoint preserves loop iteration state."""
        data = {
            "session_id": "test",
            "graph_id": "test",
            "state": {},
            "chat_history": [],
            "variables": {},
            "executed_data": [],
            "loop_iterations": {"nodeA->nodeB": 5, "nodeB->nodeC": 3},
        }

        checkpoint = Checkpoint.create(**data)
        await memory_backend.save_checkpoint(checkpoint)

        loaded = await memory_backend.load_checkpoint(checkpoint.checkpoint_id)
        assert loaded.loop_iterations["nodeA->nodeB"] == 5
        assert loaded.loop_iterations["nodeB->nodeC"] == 3

    @pytest.mark.asyncio
    async def test_checkpoint_with_nested_state(self, memory_backend):
        """Test checkpoint with deeply nested state."""
        data = {
            "session_id": "test",
            "graph_id": "test",
            "state": {
                "level1": {
                    "level2": {
                        "level3": {"value": [1, 2, 3], "nested": {"deep": True}}
                    }
                }
            },
            "chat_history": [],
            "variables": {},
            "executed_data": [],
            "loop_iterations": {},
        }

        checkpoint = Checkpoint.create(**data)
        await memory_backend.save_checkpoint(checkpoint)

        loaded = await memory_backend.load_checkpoint(checkpoint.checkpoint_id)
        assert loaded.state["level1"]["level2"]["level3"]["nested"]["deep"] is True

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolation(self, memory_backend, sample_checkpoint_data):
        """Test checkpoints are isolated by session."""
        # Session 1
        data1 = dict(sample_checkpoint_data)
        data1["session_id"] = "session-1"
        cp1 = Checkpoint.create(**data1)
        await memory_backend.save_checkpoint(cp1)

        # Session 2
        data2 = dict(sample_checkpoint_data)
        data2["session_id"] = "session-2"
        cp2 = Checkpoint.create(**data2)
        await memory_backend.save_checkpoint(cp2)

        # List should be isolated
        session1_checkpoints = await memory_backend.list_checkpoints("session-1")
        session2_checkpoints = await memory_backend.list_checkpoints("session-2")

        assert len(session1_checkpoints) == 1
        assert len(session2_checkpoints) == 1
        assert session1_checkpoints[0].checkpoint_id != session2_checkpoints[0].checkpoint_id
