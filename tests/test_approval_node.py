"""Tests for ApprovalNode and approval workflow."""

import pytest
import asyncio
from typing import List

from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.nodes import ApprovalNode, ApprovalResult, approve, reject
from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.events import EventType, ExecutionEvent
from mesh.core.executor import ExecutionStatus


class SimpleNode(BaseNode):
    """Simple test node that passes input through."""

    def __init__(self, id: str, output_value: str = None):
        super().__init__(id)
        self.output_value = output_value

    async def _execute_impl(self, input, context):
        output = self.output_value if self.output_value else input
        return NodeResult(output=output)


@pytest.fixture
def approval_context():
    """Create execution context for approval tests."""
    return ExecutionContext(
        graph_id="test-approval-graph",
        session_id="test-approval-session",
        chat_history=[],
        variables={},
        state={},
    )


@pytest.fixture
def approval_graph():
    """Create a graph with an approval node."""
    graph = StateGraph()

    # Simple flow: start -> approval -> end
    graph.add_node("start", SimpleNode("start"))
    graph.add_node("approval", ApprovalNode(
        id="approval",
        approval_id="test_approval",
        approval_message="Please approve to continue",
    ))
    graph.add_node("end", SimpleNode("end", output_value="completed"))

    graph.add_edge("START", "start")
    graph.add_edge("start", "approval")
    graph.add_edge("approval", "end")
    graph.set_entry_point("start")

    return graph.compile()


class TestApprovalNode:
    """Tests for ApprovalNode basic functionality."""

    @pytest.mark.asyncio
    async def test_approval_node_creation(self):
        """Test ApprovalNode can be created with correct parameters."""
        node = ApprovalNode(
            id="test_approval",
            approval_id="plan_review",
            approval_message="Review the plan",
            timeout_seconds=300,
        )

        assert node.id == "test_approval"
        assert node.approval_id == "plan_review"
        assert node.approval_message == "Review the plan"
        assert node.timeout_seconds == 300

    @pytest.mark.asyncio
    async def test_approval_node_returns_pending(self, approval_context):
        """Test that ApprovalNode returns approval_pending=True."""
        node = ApprovalNode(
            id="test",
            approval_id="test_approval",
            approval_message="Approve please",
        )

        result = await node.execute(
            input={"plan": "test plan"},
            context=approval_context,
        )

        assert result.approval_pending is True
        assert result.approval_id == "test_approval"
        assert result.approval_data is not None
        assert result.approval_data["approval_message"] == "Approve please"

    @pytest.mark.asyncio
    async def test_approval_node_with_data_extractor(self, approval_context):
        """Test ApprovalNode with custom data extractor."""
        def extract_plan(input):
            return {"plan_title": input.get("title")}

        node = ApprovalNode(
            id="test",
            approval_id="plan_review",
            approval_message="Review plan",
            data_extractor=extract_plan,
        )

        result = await node.execute(
            input={"title": "Research Plan", "steps": [1, 2, 3]},
            context=approval_context,
        )

        assert result.approval_data["plan_title"] == "Research Plan"


class TestApprovalResult:
    """Tests for ApprovalResult class."""

    def test_approve_helper(self):
        """Test approve() helper function."""
        result = approve(
            modified_data={"updated": True},
            approver_id="user123",
        )

        assert result.approved is True
        assert result.modified_data == {"updated": True}
        assert result.approver_id == "user123"

    def test_reject_helper(self):
        """Test reject() helper function."""
        result = reject(
            reason="Plan needs more detail",
            approver_id="user456",
        )

        assert result.approved is False
        assert result.rejection_reason == "Plan needs more detail"
        assert result.approver_id == "user456"

    def test_approval_result_serialization(self):
        """Test ApprovalResult can be serialized and deserialized."""
        original = ApprovalResult(
            approved=True,
            modified_data={"key": "value"},
            approver_id="test_user",
            metadata={"timestamp": "2024-01-01"},
        )

        as_dict = original.to_dict()
        restored = ApprovalResult.from_dict(as_dict)

        assert restored.approved == original.approved
        assert restored.modified_data == original.modified_data
        assert restored.approver_id == original.approver_id
        assert restored.metadata == original.metadata


class TestApprovalWorkflow:
    """Tests for approval workflow with Executor."""

    @pytest.mark.asyncio
    async def test_execution_pauses_at_approval(self, approval_graph, approval_context):
        """Test that execution pauses when ApprovalNode is encountered."""
        backend = MemoryBackend()
        executor = Executor(approval_graph, backend)

        events: List[ExecutionEvent] = []
        async for event in executor.execute("test input", approval_context):
            events.append(event)

        # Check that execution stopped at approval
        complete_events = [e for e in events if e.type == EventType.EXECUTION_COMPLETE]
        assert len(complete_events) == 1

        complete_event = complete_events[0]
        assert complete_event.metadata.get("status") == ExecutionStatus.WAITING_FOR_APPROVAL
        assert complete_event.metadata.get("approval_id") == "test_approval"

    @pytest.mark.asyncio
    async def test_execution_resumes_after_approval(self, approval_graph, approval_context):
        """Test that execution resumes correctly after approval."""
        backend = MemoryBackend()
        executor = Executor(approval_graph, backend)

        # First run - will pause at approval
        events1: List[ExecutionEvent] = []
        async for event in executor.execute("test input", approval_context):
            events1.append(event)

        # Verify execution paused
        assert approval_context.state.get("_pending_execution") is not None

        # Resume with approval
        approval_result = approve()
        events2: List[ExecutionEvent] = []
        async for event in executor.resume(approval_context, approval_result):
            events2.append(event)

        # Check we got approval received event
        approval_events = [e for e in events2 if e.type == EventType.APPROVAL_RECEIVED]
        assert len(approval_events) == 1

        # Check execution completed
        complete_events = [e for e in events2 if e.type == EventType.EXECUTION_COMPLETE]
        assert len(complete_events) == 1
        assert complete_events[0].metadata.get("status") == ExecutionStatus.COMPLETED

        # Check pending state was cleared
        assert approval_context.state.get("_pending_execution") is None

    @pytest.mark.asyncio
    async def test_execution_ends_on_rejection(self, approval_graph, approval_context):
        """Test that execution ends when approval is rejected."""
        backend = MemoryBackend()
        executor = Executor(approval_graph, backend)

        # First run - will pause at approval
        async for event in executor.execute("test input", approval_context):
            pass

        # Resume with rejection
        rejection = reject(reason="Plan is invalid")
        events: List[ExecutionEvent] = []
        async for event in executor.resume(approval_context, rejection):
            events.append(event)

        # Check we got rejection event
        rejection_events = [e for e in events if e.type == EventType.APPROVAL_REJECTED]
        assert len(rejection_events) == 1
        assert rejection_events[0].metadata.get("rejection_reason") == "Plan is invalid"

        # Check execution ended with rejection status
        complete_events = [e for e in events if e.type == EventType.EXECUTION_COMPLETE]
        assert len(complete_events) == 1
        assert complete_events[0].metadata.get("status") == ExecutionStatus.APPROVAL_REJECTED

    @pytest.mark.asyncio
    async def test_resume_without_pending_state_raises_error(self, approval_graph, approval_context):
        """Test that resume raises error when no pending state exists."""
        backend = MemoryBackend()
        executor = Executor(approval_graph, backend)

        # Try to resume without first executing
        with pytest.raises(ValueError, match="No pending execution state found"):
            async for _ in executor.resume(approval_context, approve()):
                pass


class TestApprovalEventTypes:
    """Tests for approval-related event types."""

    def test_approval_event_types_exist(self):
        """Test that approval event types are defined."""
        assert hasattr(EventType, "APPROVAL_PENDING")
        assert hasattr(EventType, "APPROVAL_RECEIVED")
        assert hasattr(EventType, "APPROVAL_REJECTED")
        assert hasattr(EventType, "APPROVAL_TIMEOUT")

    def test_approval_events_are_data_prefixed(self):
        """Test that approval events follow data-* prefix pattern."""
        assert EventType.APPROVAL_PENDING.value.startswith("data-")
        assert EventType.APPROVAL_RECEIVED.value.startswith("data-")
        assert EventType.APPROVAL_REJECTED.value.startswith("data-")
        assert EventType.APPROVAL_TIMEOUT.value.startswith("data-")
