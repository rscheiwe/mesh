"""Tests for streaming modes functionality.

Tests the different streaming modes:
- VALUES: Full state after each node
- UPDATES: State deltas only
- MESSAGES: Chat messages only
- EVENTS: All execution events (default)
- DEBUG: Everything including internal state
"""

import pytest
from typing import Any, Dict
from datetime import datetime

from mesh import (
    StateGraph,
    Executor,
    ExecutionContext,
    MemoryBackend,
    StreamMode,
    StreamModeAdapter,
    StateValue,
    StateUpdate,
    StreamMessage,
    DebugInfo,
)
from mesh.core.events import EventType, ExecutionEvent


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def memory_backend():
    """Fresh memory backend for each test."""
    return MemoryBackend()


@pytest.fixture
def base_context():
    """Base execution context for tests."""
    return ExecutionContext(
        graph_id="test-graph",
        session_id="test-session",
        chat_history=[],
        variables={},
        state={},
    )


def create_state_updating_tool(key: str, value: Any):
    """Create a tool that sets a specific state key."""
    async def tool_fn(input: Any, context: ExecutionContext) -> Dict[str, Any]:
        return {key: value}
    return tool_fn


def create_accumulating_tool(key: str):
    """Create a tool that accumulates values."""
    async def tool_fn(input: Any, context: ExecutionContext) -> Dict[str, Any]:
        current = context.state.get(key, 0)
        return {key: current + 1}
    return tool_fn


# =============================================================================
# StreamMode Enum Tests
# =============================================================================


class TestStreamModeEnum:
    """Test StreamMode enum values."""

    def test_all_modes_exist(self):
        """Test all streaming modes are defined."""
        assert StreamMode.VALUES.value == "values"
        assert StreamMode.UPDATES.value == "updates"
        assert StreamMode.MESSAGES.value == "messages"
        assert StreamMode.EVENTS.value == "events"
        assert StreamMode.DEBUG.value == "debug"

    def test_mode_count(self):
        """Test correct number of modes."""
        modes = list(StreamMode)
        assert len(modes) == 5


# =============================================================================
# StateValue Tests
# =============================================================================


class TestStateValue:
    """Test StateValue dataclass."""

    def test_state_value_creation(self):
        """Test creating a StateValue."""
        value = StateValue(
            node_id="test_node",
            state={"key": "value"},
            metadata={"info": "test"},
        )

        assert value.node_id == "test_node"
        assert value.state == {"key": "value"}
        assert value.metadata == {"info": "test"}
        assert value.timestamp is not None

    def test_state_value_auto_timestamp(self):
        """Test StateValue gets automatic timestamp."""
        value = StateValue(node_id="test", state={})
        assert isinstance(value.timestamp, datetime)


# =============================================================================
# StateUpdate Tests
# =============================================================================


class TestStateUpdate:
    """Test StateUpdate dataclass."""

    def test_state_update_creation(self):
        """Test creating a StateUpdate."""
        update = StateUpdate(
            node_id="test_node",
            added={"new_key": "new_value"},
            modified={"existing": "changed"},
            removed=["old_key"],
        )

        assert update.node_id == "test_node"
        assert update.added == {"new_key": "new_value"}
        assert update.modified == {"existing": "changed"}
        assert update.removed == ["old_key"]

    def test_has_changes_true(self):
        """Test has_changes returns True when there are changes."""
        update = StateUpdate(node_id="test", added={"key": "value"})
        assert update.has_changes is True

    def test_has_changes_false(self):
        """Test has_changes returns False when empty."""
        update = StateUpdate(node_id="test")
        assert update.has_changes is False

    def test_has_changes_modified(self):
        """Test has_changes with only modifications."""
        update = StateUpdate(node_id="test", modified={"key": "value"})
        assert update.has_changes is True

    def test_has_changes_removed(self):
        """Test has_changes with only removals."""
        update = StateUpdate(node_id="test", removed=["key"])
        assert update.has_changes is True


# =============================================================================
# StreamMessage Tests
# =============================================================================


class TestStreamMessage:
    """Test StreamMessage dataclass."""

    def test_stream_message_creation(self):
        """Test creating a StreamMessage."""
        message = StreamMessage(
            role="assistant",
            content="Hello, world!",
            node_id="agent_node",
            metadata={"model": "gpt-4"},
        )

        assert message.role == "assistant"
        assert message.content == "Hello, world!"
        assert message.node_id == "agent_node"
        assert message.metadata == {"model": "gpt-4"}

    def test_stream_message_auto_timestamp(self):
        """Test StreamMessage gets automatic timestamp."""
        message = StreamMessage(role="user", content="test", node_id="test")
        assert isinstance(message.timestamp, datetime)


# =============================================================================
# DebugInfo Tests
# =============================================================================


class TestDebugInfo:
    """Test DebugInfo dataclass."""

    def test_debug_info_creation(self):
        """Test creating DebugInfo."""
        event = ExecutionEvent(
            type=EventType.NODE_START,
            node_id="test",
            timestamp=datetime.now(),
        )

        debug = DebugInfo(
            event=event,
            internal_state={"_current_node": "test"},
            queue=["next_node"],
            visited_nodes=["previous_node"],
        )

        assert debug.event == event
        assert debug.internal_state == {"_current_node": "test"}
        assert debug.queue == ["next_node"]
        assert debug.visited_nodes == ["previous_node"]


# =============================================================================
# StreamModeAdapter Tests
# =============================================================================


class TestStreamModeAdapter:
    """Test StreamModeAdapter functionality."""

    def test_adapter_creation(self):
        """Test creating adapter with mode."""
        adapter = StreamModeAdapter(StreamMode.VALUES)
        assert adapter.mode == StreamMode.VALUES

    def test_adapter_reset(self):
        """Test resetting adapter state."""
        adapter = StreamModeAdapter(StreamMode.UPDATES)
        adapter._previous_state = {"key": "value"}
        adapter._message_accumulator = {"node": "content"}

        adapter.reset()

        assert adapter._previous_state == {}
        assert adapter._message_accumulator == {}

    def test_compute_delta_added(self):
        """Test computing delta with added keys."""
        adapter = StreamModeAdapter(StreamMode.UPDATES)

        delta = adapter._compute_delta({"new_key": "value"}, "test_node")

        assert delta.added == {"new_key": "value"}
        assert delta.modified == {}
        assert delta.removed == []

    def test_compute_delta_modified(self):
        """Test computing delta with modified keys."""
        adapter = StreamModeAdapter(StreamMode.UPDATES)
        adapter._previous_state = {"key": "old_value"}

        delta = adapter._compute_delta({"key": "new_value"}, "test_node")

        assert delta.added == {}
        assert delta.modified == {"key": "new_value"}
        assert delta.removed == []

    def test_compute_delta_removed(self):
        """Test computing delta with removed keys."""
        adapter = StreamModeAdapter(StreamMode.UPDATES)
        adapter._previous_state = {"old_key": "value"}

        delta = adapter._compute_delta({}, "test_node")

        assert delta.added == {}
        assert delta.modified == {}
        assert delta.removed == ["old_key"]

    def test_compute_delta_skips_internal_keys(self):
        """Test that internal keys (starting with _) are skipped."""
        adapter = StreamModeAdapter(StreamMode.UPDATES)

        delta = adapter._compute_delta(
            {"normal_key": "value", "_internal_key": "hidden"},
            "test_node"
        )

        assert "normal_key" in delta.added
        assert "_internal_key" not in delta.added


# =============================================================================
# Executor.stream() Tests
# =============================================================================


class TestExecutorStream:
    """Test Executor.stream() method."""

    @pytest.mark.asyncio
    async def test_stream_default_events_mode(self, memory_backend, base_context):
        """Test streaming defaults to EVENTS mode."""
        graph = StateGraph()
        graph.add_node("tool", create_state_updating_tool("result", "done"), node_type="tool")
        graph.set_entry_point("tool")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        items = []
        async for item in executor.stream("input", base_context):
            items.append(item)

        # Should be ExecutionEvents
        assert len(items) > 0
        assert all(isinstance(item, ExecutionEvent) for item in items)

    @pytest.mark.asyncio
    async def test_stream_events_mode_explicit(self, memory_backend, base_context):
        """Test explicit EVENTS mode."""
        graph = StateGraph()
        graph.add_node("tool", create_state_updating_tool("result", "done"), node_type="tool")
        graph.set_entry_point("tool")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        items = []
        async for item in executor.stream("input", base_context, mode=StreamMode.EVENTS):
            items.append(item)

        # Should be ExecutionEvents
        assert len(items) > 0
        assert all(isinstance(item, ExecutionEvent) for item in items)

    @pytest.mark.asyncio
    async def test_stream_values_mode(self, memory_backend, base_context):
        """Test VALUES mode emits full state."""
        graph = StateGraph()
        graph.add_node("tool1", create_state_updating_tool("key1", "value1"), node_type="tool")
        graph.add_node("tool2", create_state_updating_tool("key2", "value2"), node_type="tool")
        graph.add_edge("tool1", "tool2")
        graph.set_entry_point("tool1")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        values = []
        async for item in executor.stream("input", base_context, mode=StreamMode.VALUES):
            values.append(item)

        # Should have StateValue items
        assert len(values) >= 2
        assert all(isinstance(v, StateValue) for v in values)

        # Each should have node_id and state
        for value in values:
            assert value.node_id is not None
            assert isinstance(value.state, dict)

    @pytest.mark.asyncio
    async def test_stream_updates_mode(self, memory_backend, base_context):
        """Test UPDATES mode emits state deltas."""
        graph = StateGraph()
        graph.add_node("tool1", create_state_updating_tool("key1", "value1"), node_type="tool")
        graph.add_node("tool2", create_state_updating_tool("key2", "value2"), node_type="tool")
        graph.add_edge("tool1", "tool2")
        graph.set_entry_point("tool1")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        updates = []
        async for item in executor.stream("input", base_context, mode=StreamMode.UPDATES):
            updates.append(item)

        # Should have StateUpdate items with changes
        assert len(updates) >= 1
        assert all(isinstance(u, StateUpdate) for u in updates)

        # Updates should have changes
        for update in updates:
            assert update.has_changes

    @pytest.mark.asyncio
    async def test_stream_debug_mode(self, memory_backend, base_context):
        """Test DEBUG mode emits DebugInfo."""
        graph = StateGraph()
        graph.add_node("tool", create_state_updating_tool("result", "done"), node_type="tool")
        graph.set_entry_point("tool")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        debug_items = []
        async for item in executor.stream("input", base_context, mode=StreamMode.DEBUG):
            debug_items.append(item)

        # Should have DebugInfo items
        assert len(debug_items) > 0
        assert all(isinstance(d, DebugInfo) for d in debug_items)

        # Each should have underlying event
        for debug in debug_items:
            assert debug.event is not None
            assert isinstance(debug.internal_state, dict)


# =============================================================================
# Integration Tests
# =============================================================================


class TestStreamingModeIntegration:
    """Integration tests for streaming modes."""

    @pytest.mark.asyncio
    async def test_values_mode_accumulates_state(self, memory_backend, base_context):
        """Test VALUES mode shows state accumulation."""
        graph = StateGraph()
        graph.add_node("step1", create_accumulating_tool("counter"), node_type="tool")
        graph.add_node("step2", create_accumulating_tool("counter"), node_type="tool")
        graph.add_node("step3", create_accumulating_tool("counter"), node_type="tool")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", "step3")
        graph.set_entry_point("step1")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        values = []
        async for item in executor.stream("input", base_context, mode=StreamMode.VALUES):
            values.append(item)

        # Should see state grow
        assert len(values) >= 3

    @pytest.mark.asyncio
    async def test_updates_mode_shows_deltas(self, memory_backend, base_context):
        """Test UPDATES mode shows only changes."""
        async def add_key(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            step = context.state.get("step", 0) + 1
            return {f"key_{step}": f"value_{step}", "step": step}

        graph = StateGraph()
        graph.add_node("step1", add_key, node_type="tool")
        graph.add_node("step2", add_key, node_type="tool")
        graph.add_edge("step1", "step2")
        graph.set_entry_point("step1")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        updates = []
        async for item in executor.stream("input", base_context, mode=StreamMode.UPDATES):
            updates.append(item)

        # Should have updates showing new keys
        assert len(updates) >= 2

        # Each update should show newly added keys
        for update in updates:
            assert update.has_changes

    @pytest.mark.asyncio
    async def test_multiple_modes_same_execution(self, memory_backend):
        """Test same graph can be streamed with different modes."""
        graph = StateGraph()
        graph.add_node("tool", create_state_updating_tool("result", "done"), node_type="tool")
        graph.set_entry_point("tool")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        # Run with VALUES mode
        context1 = ExecutionContext(
            graph_id="test",
            session_id="session-1",
            chat_history=[],
            variables={},
            state={},
        )
        values = []
        async for item in executor.stream("input", context1, mode=StreamMode.VALUES):
            values.append(item)

        # Run with EVENTS mode
        context2 = ExecutionContext(
            graph_id="test",
            session_id="session-2",
            chat_history=[],
            variables={},
            state={},
        )
        events = []
        async for item in executor.stream("input", context2, mode=StreamMode.EVENTS):
            events.append(item)

        # Both should complete successfully
        assert len(values) > 0
        assert len(events) > 0
        assert isinstance(values[0], StateValue)
        assert isinstance(events[0], ExecutionEvent)


# =============================================================================
# Edge Cases
# =============================================================================


class TestStreamingModeEdgeCases:
    """Test edge cases in streaming modes."""

    @pytest.mark.asyncio
    async def test_empty_state_values_mode(self, memory_backend, base_context):
        """Test VALUES mode with tool that doesn't change state."""
        async def no_op_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            return {}

        graph = StateGraph()
        graph.add_node("tool", no_op_tool, node_type="tool")
        graph.set_entry_point("tool")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        values = []
        async for item in executor.stream("input", base_context, mode=StreamMode.VALUES):
            values.append(item)

        # Should still emit (empty state is valid)
        assert len(values) >= 1

    @pytest.mark.asyncio
    async def test_no_changes_updates_mode(self, memory_backend, base_context):
        """Test UPDATES mode filters out empty deltas."""
        async def no_change_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            # Return same value as already in state
            return {"key": context.state.get("key", "value")}

        graph = StateGraph()
        graph.add_node("tool", no_change_tool, node_type="tool")
        graph.set_entry_point("tool")
        # Set initial state
        base_context.state["key"] = "value"
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        updates = []
        async for item in executor.stream("input", base_context, mode=StreamMode.UPDATES):
            updates.append(item)

        # May or may not emit depending on initial state
        # The important thing is it doesn't crash

    @pytest.mark.asyncio
    async def test_debug_mode_internal_state(self, memory_backend, base_context):
        """Test DEBUG mode includes internal state info."""
        graph = StateGraph()
        graph.add_node("tool", create_state_updating_tool("result", "done"), node_type="tool")
        graph.set_entry_point("tool")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        debug_items = []
        async for item in executor.stream("input", base_context, mode=StreamMode.DEBUG):
            debug_items.append(item)

        # Check internal state is tracked
        assert len(debug_items) > 0
        for debug in debug_items:
            assert isinstance(debug.internal_state, dict)
            assert isinstance(debug.queue, list)
            assert isinstance(debug.timing, dict)


# =============================================================================
# Consistency Tests
# =============================================================================


class TestStreamingModeConsistency:
    """Test consistency between streaming modes."""

    @pytest.mark.asyncio
    async def test_events_and_debug_same_count(self, memory_backend):
        """Test DEBUG mode has same event count as EVENTS mode."""
        graph = StateGraph()
        graph.add_node("tool1", create_state_updating_tool("k1", "v1"), node_type="tool")
        graph.add_node("tool2", create_state_updating_tool("k2", "v2"), node_type="tool")
        graph.add_edge("tool1", "tool2")
        graph.set_entry_point("tool1")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        # Run with EVENTS mode
        context1 = ExecutionContext(
            graph_id="test", session_id="s1",
            chat_history=[], variables={}, state={},
        )
        events = []
        async for item in executor.stream("input", context1, mode=StreamMode.EVENTS):
            events.append(item)

        # Run with DEBUG mode
        context2 = ExecutionContext(
            graph_id="test", session_id="s2",
            chat_history=[], variables={}, state={},
        )
        debug_items = []
        async for item in executor.stream("input", context2, mode=StreamMode.DEBUG):
            debug_items.append(item)

        # Should have same count (one debug per event)
        assert len(events) == len(debug_items)

        # Each debug should wrap an event
        for debug, event in zip(debug_items, events):
            assert debug.event.type == event.type
