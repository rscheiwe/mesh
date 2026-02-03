"""Tests for the interrupt system (human-in-the-loop).

Tests cover:
- InterruptState creation and serialization
- InterruptResume and InterruptReject
- StateGraph interrupt configuration
- Executor interrupt_before handling
- Executor interrupt_after handling
- Conditional interrupts
- Resume from interrupt
- Reject at interrupt
"""

import pytest
from datetime import datetime
from typing import Any, Dict

from mesh import (
    StateGraph,
    Executor,
    MemoryBackend,
    InterruptState,
    InterruptResume,
    InterruptReject,
    InterruptNotFoundError,
)
from mesh.core.state import ExecutionContext
from mesh.core.events import EventType
from mesh.nodes.base import BaseNode, NodeResult


# =============================================================================
# Test Fixtures
# =============================================================================


class MockToolNode(BaseNode):
    """Simple mock tool node for testing."""

    def __init__(self, id: str, return_value: Dict[str, Any] = None):
        super().__init__(id=id, config={})
        self.return_value = return_value or {"result": f"{id}_output"}

    async def _execute_impl(self, input: Any, context: "ExecutionContext") -> NodeResult:
        return NodeResult(
            output=self.return_value,
            state={f"{self.id}_executed": True},
        )


def create_test_graph_with_interrupts():
    """Create a simple graph with interrupt configuration."""
    graph = StateGraph()

    # Add nodes
    graph.add_node("tool1", MockToolNode("tool1"))
    graph.add_node("tool2", MockToolNode("tool2"))
    graph.add_node("tool3", MockToolNode("tool3"))

    # Add edges
    graph.add_edge("START", "tool1")
    graph.add_edge("tool1", "tool2")
    graph.add_edge("tool2", "tool3")

    graph.set_entry_point("tool1")

    return graph


def create_test_context():
    """Create a test execution context."""
    return ExecutionContext(
        graph_id="test-graph",
        session_id="test-session",
        chat_history=[],
        variables={},
        state={},
    )


# =============================================================================
# InterruptState Tests
# =============================================================================


class TestInterruptState:
    """Tests for InterruptState dataclass."""

    def test_create_interrupt_state(self):
        """Test creating an interrupt state."""
        state = InterruptState.create(
            node_id="test_node",
            position="before",
            state={"key": "value"},
            input_data={"input": "data"},
        )

        assert state.node_id == "test_node"
        assert state.position == "before"
        assert state.state == {"key": "value"}
        assert state.input_data == {"input": "data"}
        assert state.output_data is None
        assert state.interrupt_id is not None
        assert len(state.interrupt_id) == 36  # UUID format

    def test_create_interrupt_state_with_output(self):
        """Test creating interrupt state with output (after position)."""
        state = InterruptState.create(
            node_id="test_node",
            position="after",
            state={"key": "value"},
            input_data={"input": "data"},
            output_data={"output": "result"},
            metadata={"review_key": "review_value"},
        )

        assert state.position == "after"
        assert state.output_data == {"output": "result"}
        assert state.metadata == {"review_key": "review_value"}

    def test_interrupt_state_serialization(self):
        """Test serializing and deserializing interrupt state."""
        original = InterruptState.create(
            node_id="test_node",
            position="before",
            state={"key": "value"},
            input_data={"input": "data"},
            pending_queue=[("node1", {"x": 1}), ("node2", {"y": 2})],
            waiting_nodes={"node3": {"node_id": "node3", "received_inputs": {}, "expected_inputs": ["a", "b"]}},
            loop_counts={"edge1": 5},
        )

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = InterruptState.from_dict(data)

        assert restored.interrupt_id == original.interrupt_id
        assert restored.node_id == original.node_id
        assert restored.position == original.position
        assert restored.state == original.state
        assert restored.input_data == original.input_data
        assert restored.pending_queue == original.pending_queue
        assert restored.waiting_nodes == original.waiting_nodes
        assert restored.loop_counts == original.loop_counts

    def test_interrupt_state_created_at(self):
        """Test that created_at timestamp is set."""
        before = datetime.now()
        state = InterruptState.create(
            node_id="test_node",
            position="before",
            state={},
            input_data={},
        )
        after = datetime.now()

        assert before <= state.created_at <= after


class TestInterruptResume:
    """Tests for InterruptResume dataclass."""

    def test_basic_resume(self):
        """Test basic resume without modifications."""
        resume = InterruptResume()

        assert resume.modified_state is None
        assert resume.modified_input is None
        assert resume.skip_node is False

    def test_resume_with_modified_state(self):
        """Test resume with state modifications."""
        resume = InterruptResume(
            modified_state={"approved": True, "notes": "LGTM"},
        )

        assert resume.modified_state == {"approved": True, "notes": "LGTM"}

    def test_resume_with_modified_input(self):
        """Test resume with modified input."""
        resume = InterruptResume(
            modified_input={"updated_field": "new_value"},
        )

        assert resume.modified_input == {"updated_field": "new_value"}

    def test_resume_skip_node(self):
        """Test resume with skip_node flag."""
        resume = InterruptResume(skip_node=True)

        assert resume.skip_node is True


class TestInterruptReject:
    """Tests for InterruptReject dataclass."""

    def test_basic_reject(self):
        """Test basic rejection."""
        reject = InterruptReject(reason="Content policy violation")

        assert reject.reason == "Content policy violation"
        assert reject.metadata == {}

    def test_reject_with_metadata(self):
        """Test rejection with metadata."""
        reject = InterruptReject(
            reason="Requires manager approval",
            metadata={"escalation_required": True, "risk_level": "high"},
        )

        assert reject.reason == "Requires manager approval"
        assert reject.metadata["escalation_required"] is True


# =============================================================================
# StateGraph Interrupt Configuration Tests
# =============================================================================


class TestStateGraphInterrupts:
    """Tests for StateGraph interrupt configuration."""

    def test_set_interrupt_before(self):
        """Test setting interrupt_before on a node."""
        graph = create_test_graph_with_interrupts()

        graph.set_interrupt_before("tool2")

        compiled = graph.compile()
        assert "tool2" in compiled.interrupt_before
        assert compiled.interrupt_before["tool2"]["condition"] is None

    def test_set_interrupt_after(self):
        """Test setting interrupt_after on a node."""
        graph = create_test_graph_with_interrupts()

        graph.set_interrupt_after("tool1")

        compiled = graph.compile()
        assert "tool1" in compiled.interrupt_after
        assert compiled.interrupt_after["tool1"]["condition"] is None

    def test_set_interrupt_with_condition(self):
        """Test setting interrupt with a condition."""
        graph = create_test_graph_with_interrupts()

        def my_condition(state, input_data):
            return state.get("requires_review", False)

        graph.set_interrupt_before("tool2", condition=my_condition)

        compiled = graph.compile()
        assert compiled.interrupt_before["tool2"]["condition"] is my_condition

    def test_set_interrupt_with_metadata_extractor(self):
        """Test setting interrupt with metadata extractor."""
        graph = create_test_graph_with_interrupts()

        def extract_metadata(state, data):
            return {"summary": "Review needed", "data_size": len(str(data))}

        graph.set_interrupt_before("tool2", metadata_extractor=extract_metadata)

        compiled = graph.compile()
        assert compiled.interrupt_before["tool2"]["metadata_extractor"] is extract_metadata

    def test_set_interrupt_with_timeout(self):
        """Test setting interrupt with timeout."""
        graph = create_test_graph_with_interrupts()

        graph.set_interrupt_before("tool2", timeout=300.0)  # 5 minutes

        compiled = graph.compile()
        assert compiled.interrupt_before["tool2"]["timeout"] == 300.0

    def test_multiple_interrupts(self):
        """Test setting multiple interrupts."""
        graph = create_test_graph_with_interrupts()

        graph.set_interrupt_before("tool2")
        graph.set_interrupt_after("tool2")
        graph.set_interrupt_after("tool3")

        compiled = graph.compile()
        assert "tool2" in compiled.interrupt_before
        assert "tool2" in compiled.interrupt_after
        assert "tool3" in compiled.interrupt_after

    def test_interrupt_via_add_edge(self):
        """Test setting interrupt via add_edge parameters."""
        graph = StateGraph()
        graph.add_node("tool1", MockToolNode("tool1"))
        graph.add_node("tool2", MockToolNode("tool2"))

        graph.add_edge("START", "tool1")
        graph.add_edge("tool1", "tool2", interrupt_before=True)

        graph.set_entry_point("tool1")

        # Verify edge has interrupt_before flag
        compiled = graph.compile()
        edge = next(e for e in compiled.edges if e.source == "tool1" and e.target == "tool2")
        assert edge.interrupt_before is True

    def test_method_chaining(self):
        """Test that interrupt methods support chaining."""
        graph = create_test_graph_with_interrupts()

        result = (
            graph
            .set_interrupt_before("tool1")
            .set_interrupt_after("tool2")
            .set_interrupt_before("tool3", timeout=60.0)
        )

        assert result is graph  # Returns self for chaining


# =============================================================================
# Executor Interrupt Tests
# =============================================================================


class TestExecutorInterruptBefore:
    """Tests for executor interrupt_before handling."""

    @pytest.mark.asyncio
    async def test_interrupt_before_pauses_execution(self):
        """Test that interrupt_before pauses before node execution."""
        graph = create_test_graph_with_interrupts()
        graph.set_interrupt_before("tool2")

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        events = []
        async for event in executor.execute("test input", context):
            events.append(event)

        # Should have completed tool1, then paused before tool2
        event_types = [e.type for e in events]
        assert EventType.EXECUTION_START in event_types
        assert EventType.INTERRUPT in event_types
        assert EventType.EXECUTION_COMPLETE in event_types

        # Find the interrupt event
        interrupt_event = next(e for e in events if e.type == EventType.INTERRUPT)
        assert interrupt_event.node_id == "tool2"
        assert interrupt_event.metadata["position"] == "before"

        # Find completion event - should indicate waiting for interrupt
        complete_event = next(e for e in events if e.type == EventType.EXECUTION_COMPLETE)
        assert complete_event.metadata["status"] == "waiting_for_interrupt"

    @pytest.mark.asyncio
    async def test_interrupt_before_saves_state(self):
        """Test that interrupt_before saves state for resume."""
        graph = create_test_graph_with_interrupts()
        graph.set_interrupt_before("tool2")

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        async for _ in executor.execute("test input", context):
            pass

        # Check that interrupt state was saved
        assert "_interrupt_state" in context.state
        interrupt_data = context.state["_interrupt_state"]
        assert interrupt_data["node_id"] == "tool2"
        assert interrupt_data["position"] == "before"


class TestExecutorInterruptAfter:
    """Tests for executor interrupt_after handling."""

    @pytest.mark.asyncio
    async def test_interrupt_after_pauses_execution(self):
        """Test that interrupt_after pauses after node execution."""
        graph = create_test_graph_with_interrupts()
        graph.set_interrupt_after("tool1")

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        events = []
        async for event in executor.execute("test input", context):
            events.append(event)

        # Should have completed tool1, then paused after
        event_types = [e.type for e in events]
        assert EventType.NODE_COMPLETE in event_types
        assert EventType.INTERRUPT in event_types

        # Find the interrupt event
        interrupt_event = next(e for e in events if e.type == EventType.INTERRUPT)
        assert interrupt_event.node_id == "tool1"
        assert interrupt_event.metadata["position"] == "after"
        assert interrupt_event.output is not None  # Should have output for after

    @pytest.mark.asyncio
    async def test_interrupt_after_includes_output(self):
        """Test that interrupt_after includes node output."""
        graph = StateGraph()
        graph.add_node("tool1", MockToolNode("tool1", return_value={"important": "data"}))
        graph.add_node("tool2", MockToolNode("tool2"))
        graph.add_edge("START", "tool1")
        graph.add_edge("tool1", "tool2")
        graph.set_entry_point("tool1")
        graph.set_interrupt_after("tool1")

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        events = []
        async for event in executor.execute("test input", context):
            events.append(event)

        interrupt_event = next(e for e in events if e.type == EventType.INTERRUPT)
        assert interrupt_event.metadata["output_data"] == {"important": "data"}


class TestConditionalInterrupts:
    """Tests for conditional interrupt handling."""

    @pytest.mark.asyncio
    async def test_conditional_interrupt_triggers_when_true(self):
        """Test that conditional interrupt triggers when condition is True."""
        graph = create_test_graph_with_interrupts()

        def always_interrupt(state, data):
            return True

        graph.set_interrupt_before("tool2", condition=always_interrupt)

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        events = []
        async for event in executor.execute("test input", context):
            events.append(event)

        event_types = [e.type for e in events]
        assert EventType.INTERRUPT in event_types

    @pytest.mark.asyncio
    async def test_conditional_interrupt_skips_when_false(self):
        """Test that conditional interrupt skips when condition is False."""
        graph = create_test_graph_with_interrupts()

        def never_interrupt(state, data):
            return False

        graph.set_interrupt_before("tool2", condition=never_interrupt)

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        events = []
        async for event in executor.execute("test input", context):
            events.append(event)

        event_types = [e.type for e in events]
        # Should complete normally without interrupt
        assert EventType.INTERRUPT not in event_types
        assert EventType.EXECUTION_COMPLETE in event_types

        # Verify all nodes executed
        complete_events = [e for e in events if e.type == EventType.NODE_COMPLETE]
        node_ids = [e.node_id for e in complete_events]
        assert "tool1" in node_ids
        assert "tool2" in node_ids
        assert "tool3" in node_ids

    @pytest.mark.asyncio
    async def test_conditional_interrupt_uses_state(self):
        """Test that conditional interrupt can access state."""
        graph = create_test_graph_with_interrupts()

        def interrupt_if_flagged(state, data):
            return state.get("tool1_executed", False)

        graph.set_interrupt_before("tool2", condition=interrupt_if_flagged)

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        events = []
        async for event in executor.execute("test input", context):
            events.append(event)

        # Should trigger because tool1 sets tool1_executed=True
        event_types = [e.type for e in events]
        assert EventType.INTERRUPT in event_types

    @pytest.mark.asyncio
    async def test_conditional_interrupt_error_skips_interrupt(self):
        """Test that condition evaluation error skips interrupt (safe default)."""
        graph = create_test_graph_with_interrupts()

        def buggy_condition(state, data):
            raise ValueError("Oops!")

        graph.set_interrupt_before("tool2", condition=buggy_condition)

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        events = []
        async for event in executor.execute("test input", context):
            events.append(event)

        # Should continue execution when condition errors
        event_types = [e.type for e in events]
        assert EventType.INTERRUPT not in event_types
        assert EventType.EXECUTION_COMPLETE in event_types


class TestResumeFromInterrupt:
    """Tests for resuming execution from an interrupt."""

    @pytest.mark.asyncio
    async def test_basic_resume(self):
        """Test basic resume from interrupt."""
        graph = create_test_graph_with_interrupts()
        graph.set_interrupt_before("tool2")

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        # Execute until interrupt
        async for _ in executor.execute("test input", context):
            pass

        # Resume
        resume = InterruptResume()
        events = []
        async for event in executor.resume_from_interrupt(context, resume):
            events.append(event)

        # Should have resumed and completed
        event_types = [e.type for e in events]
        assert EventType.INTERRUPT_RESUMED in event_types
        assert EventType.EXECUTION_COMPLETE in event_types

        # Verify final status is completed
        complete_event = next(e for e in events if e.type == EventType.EXECUTION_COMPLETE)
        assert complete_event.metadata["status"] == "completed"

    @pytest.mark.asyncio
    async def test_resume_with_state_modification(self):
        """Test resume with state modifications applied."""
        graph = create_test_graph_with_interrupts()
        graph.set_interrupt_before("tool2")

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        # Execute until interrupt
        async for _ in executor.execute("test input", context):
            pass

        # Resume with state modification
        resume = InterruptResume(
            modified_state={"reviewer_approved": True, "reviewer_notes": "All good"}
        )
        async for _ in executor.resume_from_interrupt(context, resume):
            pass

        # Verify state was modified
        assert context.state.get("reviewer_approved") is True
        assert context.state.get("reviewer_notes") == "All good"

    @pytest.mark.asyncio
    async def test_resume_with_skip_node(self):
        """Test resume with skip_node flag."""
        graph = create_test_graph_with_interrupts()
        graph.set_interrupt_before("tool2")

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        # Execute until interrupt
        async for _ in executor.execute("test input", context):
            pass

        # Resume with skip_node
        resume = InterruptResume(skip_node=True)
        events = []
        async for event in executor.resume_from_interrupt(context, resume):
            events.append(event)

        # tool2 should be skipped - verify by checking executed nodes
        complete_events = [e for e in events if e.type == EventType.NODE_COMPLETE]
        node_ids = [e.node_id for e in complete_events]

        # tool2 should NOT be in the completed nodes during resume
        # (it was skipped)
        assert "tool2" not in node_ids

    @pytest.mark.asyncio
    async def test_resume_after_interrupt_after(self):
        """Test resume from interrupt_after position."""
        graph = create_test_graph_with_interrupts()
        graph.set_interrupt_after("tool1")

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        # Execute until interrupt
        async for _ in executor.execute("test input", context):
            pass

        # Resume
        resume = InterruptResume()
        events = []
        async for event in executor.resume_from_interrupt(context, resume):
            events.append(event)

        # Should have resumed and completed tool2 and tool3
        complete_events = [e for e in events if e.type == EventType.NODE_COMPLETE]
        node_ids = [e.node_id for e in complete_events]
        assert "tool2" in node_ids
        assert "tool3" in node_ids


class TestRejectAtInterrupt:
    """Tests for rejecting execution at an interrupt."""

    @pytest.mark.asyncio
    async def test_basic_reject(self):
        """Test basic rejection at interrupt."""
        graph = create_test_graph_with_interrupts()
        graph.set_interrupt_before("tool2")

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        # Execute until interrupt
        async for _ in executor.execute("test input", context):
            pass

        # Reject
        reject = InterruptReject(reason="Policy violation detected")
        events = []
        async for event in executor.resume_from_interrupt(context, reject):
            events.append(event)

        # Should have rejected and ended
        event_types = [e.type for e in events]
        assert EventType.INTERRUPT_REJECTED in event_types
        assert EventType.EXECUTION_COMPLETE in event_types

        # Verify rejection details
        reject_event = next(e for e in events if e.type == EventType.INTERRUPT_REJECTED)
        assert reject_event.metadata["reason"] == "Policy violation detected"

        # Verify final status is rejected
        complete_event = next(e for e in events if e.type == EventType.EXECUTION_COMPLETE)
        assert complete_event.metadata["status"] == "interrupt_rejected"

    @pytest.mark.asyncio
    async def test_reject_clears_interrupt_state(self):
        """Test that rejection clears interrupt state."""
        graph = create_test_graph_with_interrupts()
        graph.set_interrupt_before("tool2")

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        # Execute until interrupt
        async for _ in executor.execute("test input", context):
            pass

        assert "_interrupt_state" in context.state

        # Reject
        reject = InterruptReject(reason="Not allowed")
        async for _ in executor.resume_from_interrupt(context, reject):
            pass

        # Interrupt state should be cleared
        assert "_interrupt_state" not in context.state


class TestMultipleInterrupts:
    """Tests for multiple interrupts in a graph."""

    @pytest.mark.asyncio
    async def test_sequential_interrupts(self):
        """Test handling sequential interrupt points."""
        graph = create_test_graph_with_interrupts()
        graph.set_interrupt_before("tool2")
        graph.set_interrupt_before("tool3")

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        # First execution - pauses at tool2
        events1 = []
        async for event in executor.execute("test input", context):
            events1.append(event)

        interrupt1 = next(e for e in events1 if e.type == EventType.INTERRUPT)
        assert interrupt1.node_id == "tool2"

        # Resume - should pause at tool3
        resume = InterruptResume()
        events2 = []
        async for event in executor.resume_from_interrupt(context, resume):
            events2.append(event)

        interrupt2 = next(e for e in events2 if e.type == EventType.INTERRUPT)
        assert interrupt2.node_id == "tool3"

        # Resume again - should complete
        events3 = []
        async for event in executor.resume_from_interrupt(context, resume):
            events3.append(event)

        complete = next(e for e in events3 if e.type == EventType.EXECUTION_COMPLETE)
        assert complete.metadata["status"] == "completed"


class TestMetadataExtractor:
    """Tests for metadata extractor functionality."""

    @pytest.mark.asyncio
    async def test_metadata_extractor_called(self):
        """Test that metadata extractor is called and results included."""
        graph = create_test_graph_with_interrupts()

        def extract_review_data(state, input_data):
            return {
                "summary": "Review needed for tool2",
                "state_keys": list(state.keys()),
            }

        graph.set_interrupt_before("tool2", metadata_extractor=extract_review_data)

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        events = []
        async for event in executor.execute("test input", context):
            events.append(event)

        interrupt_event = next(e for e in events if e.type == EventType.INTERRUPT)
        review_metadata = interrupt_event.metadata.get("review_metadata", {})

        assert review_metadata.get("summary") == "Review needed for tool2"
        assert "state_keys" in review_metadata

    @pytest.mark.asyncio
    async def test_metadata_extractor_error_handled(self):
        """Test that metadata extractor errors are handled gracefully."""
        graph = create_test_graph_with_interrupts()

        def buggy_extractor(state, data):
            raise RuntimeError("Extraction failed!")

        graph.set_interrupt_before("tool2", metadata_extractor=buggy_extractor)

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        events = []
        async for event in executor.execute("test input", context):
            events.append(event)

        # Should still interrupt, just with empty metadata
        interrupt_event = next(e for e in events if e.type == EventType.INTERRUPT)
        assert interrupt_event is not None
        assert interrupt_event.metadata.get("review_metadata", {}) == {}


class TestInterruptWithNoBackend:
    """Tests for interrupt behavior without state backend."""

    @pytest.mark.asyncio
    async def test_interrupt_works_without_backend(self):
        """Test that interrupt works even without state backend."""
        graph = create_test_graph_with_interrupts()
        graph.set_interrupt_before("tool2")

        compiled = graph.compile()
        executor = Executor(compiled, state_backend=None)
        context = create_test_context()

        events = []
        async for event in executor.execute("test input", context):
            events.append(event)

        # Should still pause at interrupt
        event_types = [e.type for e in events]
        assert EventType.INTERRUPT in event_types

        # State should still be in context (just not persisted)
        assert "_interrupt_state" in context.state


class TestResumeErrors:
    """Tests for error handling during resume."""

    @pytest.mark.asyncio
    async def test_resume_without_interrupt_state_raises(self):
        """Test that resume without interrupt state raises error."""
        graph = create_test_graph_with_interrupts()

        compiled = graph.compile()
        backend = MemoryBackend()
        executor = Executor(compiled, backend)
        context = create_test_context()

        # Try to resume without prior interrupt
        resume = InterruptResume()

        with pytest.raises(ValueError, match="No interrupt state found"):
            async for _ in executor.resume_from_interrupt(context, resume):
                pass
