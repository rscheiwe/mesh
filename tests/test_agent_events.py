"""Test event coverage for AgentNode with new event types.

This test suite validates that AgentNode correctly handles and emits all event types,
including the newly added reasoning, metadata, source, file, and custom data events.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from mesh.nodes.agent import AgentNode
from mesh.core.state import ExecutionContext
from mesh.core.events import EventType, EventEmitter
from mesh.backends import MemoryBackend


class MockVelAgent:
    """Mock Vel agent that emits various event types."""

    def __init__(self, events_to_emit=None):
        """Initialize mock agent with predefined events to emit.

        Args:
            events_to_emit: List of event dicts to emit during streaming
        """
        self.id = "test-agent"
        self.events_to_emit = events_to_emit or []
        self.__module__ = "vel.agent"

    async def run_stream(self, input, session_id):
        """Stream predefined events."""
        for event in self.events_to_emit:
            yield event


@pytest.fixture
def execution_context():
    """Create execution context for testing."""
    emitter = EventEmitter()

    context = ExecutionContext(
        graph_id="test-graph",
        session_id="test-session",
        chat_history=[],
        variables={},
        state={},
    )
    context._event_emitter = emitter
    return context


@pytest.mark.asyncio
async def test_reasoning_events(execution_context):
    """Test reasoning event handling (o1/o3/Claude Extended Thinking)."""
    # Create mock agent that emits reasoning events
    events_to_emit = [
        {"type": "start"},
        {"type": "reasoning-start", "id": "r1"},
        {"type": "reasoning-delta", "id": "r1", "delta": "Let me think about this..."},
        {"type": "reasoning-delta", "id": "r1", "delta": " The answer is"},
        {"type": "reasoning-end", "id": "r1"},
        {"type": "text-start", "id": "t1"},
        {"type": "text-delta", "id": "t1", "delta": "The answer is 42."},
        {"type": "text-end", "id": "t1"},
        {"type": "finish-message", "finishReason": "stop"},
    ]

    agent = MockVelAgent(events_to_emit)
    node = AgentNode(id="test-node", agent=agent)

    # Collect emitted events
    events = []

    async def collector(event):
        events.append(event)

    execution_context._event_emitter.on(collector)

    # Execute node
    result = await node._execute_impl("test input", execution_context)

    # Verify reasoning events were emitted
    reasoning_starts = [e for e in events if e.type == EventType.REASONING_START]
    assert len(reasoning_starts) == 1
    assert reasoning_starts[0].metadata["block_id"] == "r1"

    reasoning_tokens = [e for e in events if e.type == EventType.REASONING_TOKEN]
    assert len(reasoning_tokens) == 2
    assert reasoning_tokens[0].content == "Let me think about this..."
    assert reasoning_tokens[1].content == " The answer is"
    assert reasoning_tokens[0].metadata["event_subtype"] == "reasoning"

    reasoning_ends = [e for e in events if e.type == EventType.REASONING_END]
    assert len(reasoning_ends) == 1
    assert reasoning_ends[0].metadata["block_id"] == "r1"

    # Verify text content also captured
    assert result.output["content"] == "The answer is 42."


@pytest.mark.asyncio
async def test_response_metadata_events(execution_context):
    """Test response metadata event handling (usage tracking)."""
    events_to_emit = [
        {"type": "start"},
        {"type": "text-start", "id": "t1"},
        {"type": "text-delta", "id": "t1", "delta": "Hello!"},
        {"type": "text-end", "id": "t1"},
        {
            "type": "response-metadata",
            "id": "resp-123",
            "modelId": "gpt-4",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "timestamp": "2025-10-20T12:00:00Z",
        },
        {"type": "finish-message", "finishReason": "stop"},
    ]

    agent = MockVelAgent(events_to_emit)
    node = AgentNode(id="test-node", agent=agent)

    events = []

    async def collector(event):
        events.append(event)

    execution_context._event_emitter.on(collector)

    result = await node._execute_impl("test input", execution_context)

    # Verify metadata event was emitted
    metadata_events = [e for e in events if e.type == EventType.RESPONSE_METADATA]
    assert len(metadata_events) == 1

    meta = metadata_events[0].metadata
    assert meta["id"] == "resp-123"
    assert meta["model_id"] == "gpt-4"
    assert meta["usage"]["prompt_tokens"] == 10
    assert meta["usage"]["completion_tokens"] == 5
    assert meta["usage"]["total_tokens"] == 15
    assert meta["timestamp"] == "2025-10-20T12:00:00Z"


@pytest.mark.asyncio
async def test_source_events(execution_context):
    """Test source/citation event handling (Gemini grounding)."""
    events_to_emit = [
        {"type": "start"},
        {"type": "text-start", "id": "t1"},
        {"type": "text-delta", "id": "t1", "delta": "According to the source"},
        {"type": "text-end", "id": "t1"},
        {
            "type": "source",
            "sources": [
                {"url": "https://example.com/doc1", "title": "Document 1"},
                {"url": "https://example.com/doc2", "title": "Document 2"},
            ],
        },
        {"type": "finish-message", "finishReason": "stop"},
    ]

    agent = MockVelAgent(events_to_emit)
    node = AgentNode(id="test-node", agent=agent)

    events = []

    async def collector(event):
        events.append(event)

    execution_context._event_emitter.on(collector)

    result = await node._execute_impl("test input", execution_context)

    # Verify source event was emitted
    source_events = [e for e in events if e.type == EventType.SOURCE]
    assert len(source_events) == 1

    sources = source_events[0].metadata["sources"]
    assert len(sources) == 2
    assert sources[0]["url"] == "https://example.com/doc1"
    assert sources[0]["title"] == "Document 1"
    assert sources[1]["url"] == "https://example.com/doc2"


@pytest.mark.asyncio
async def test_file_events(execution_context):
    """Test file attachment event handling (multi-modal)."""
    events_to_emit = [
        {"type": "start"},
        {"type": "text-start", "id": "t1"},
        {"type": "text-delta", "id": "t1", "delta": "Here is the image"},
        {"type": "text-end", "id": "t1"},
        {
            "type": "file",
            "name": "diagram.png",
            "mimeType": "image/png",
            "content": "base64encodedcontent==",
        },
        {"type": "finish-message", "finishReason": "stop"},
    ]

    agent = MockVelAgent(events_to_emit)
    node = AgentNode(id="test-node", agent=agent)

    events = []

    async def collector(event):
        events.append(event)

    execution_context._event_emitter.on(collector)

    result = await node._execute_impl("test input", execution_context)

    # Verify file event was emitted
    file_events = [e for e in events if e.type == EventType.FILE]
    assert len(file_events) == 1

    file_meta = file_events[0].metadata
    assert file_meta["name"] == "diagram.png"
    assert file_meta["mime_type"] == "image/png"
    assert file_meta["content"] == "base64encodedcontent=="


@pytest.mark.asyncio
async def test_custom_data_events(execution_context):
    """Test custom data-* event passthrough."""
    events_to_emit = [
        {"type": "start"},
        {"type": "data-progress", "data": {"percent": 25}, "transient": True},
        {"type": "text-start", "id": "t1"},
        {"type": "text-delta", "id": "t1", "delta": "Processing..."},
        {"type": "text-end", "id": "t1"},
        {"type": "data-progress", "data": {"percent": 100}, "transient": True},
        {"type": "finish-message", "finishReason": "stop"},
    ]

    agent = MockVelAgent(events_to_emit)
    node = AgentNode(id="test-node", agent=agent)

    events = []

    async def collector(event):
        events.append(event)

    execution_context._event_emitter.on(collector)

    result = await node._execute_impl("test input", execution_context)

    # Verify custom data events were emitted
    custom_events = [e for e in events if e.type == EventType.CUSTOM_DATA]
    assert len(custom_events) == 2

    # First progress event
    assert custom_events[0].content["percent"] == 25
    assert custom_events[0].metadata["data_type"] == "data-progress"
    assert custom_events[0].metadata["transient"] is True

    # Second progress event
    assert custom_events[1].content["percent"] == 100
    assert custom_events[1].metadata["data_type"] == "data-progress"


@pytest.mark.asyncio
async def test_rlm_events(execution_context):
    """Test RLM middleware events passthrough."""
    events_to_emit = [
        {"type": "start"},
        {"type": "data-rlm-start", "data": {"context_size": 1000}, "transient": False},
        {"type": "data-rlm-step-start", "data": {"step": 1}, "transient": True},
        {"type": "text-start", "id": "t1"},
        {"type": "text-delta", "id": "t1", "delta": "Analysis..."},
        {"type": "text-end", "id": "t1"},
        {"type": "data-rlm-step-finish", "data": {"step": 1, "tokens": 50}, "transient": True},
        {"type": "data-rlm-complete", "data": {"total_steps": 1}, "transient": False},
        {"type": "finish-message", "finishReason": "stop"},
    ]

    agent = MockVelAgent(events_to_emit)
    node = AgentNode(id="test-node", agent=agent)

    events = []

    async def collector(event):
        events.append(event)

    execution_context._event_emitter.on(collector)

    result = await node._execute_impl("test input", execution_context)

    # Verify RLM events were emitted
    rlm_events = [e for e in events if e.type == EventType.CUSTOM_DATA and e.metadata["data_type"].startswith("data-rlm")]
    assert len(rlm_events) == 4

    # Check event types
    rlm_event_types = [e.metadata["data_type"] for e in rlm_events]
    assert "data-rlm-start" in rlm_event_types
    assert "data-rlm-step-start" in rlm_event_types
    assert "data-rlm-step-finish" in rlm_event_types
    assert "data-rlm-complete" in rlm_event_types


@pytest.mark.asyncio
async def test_combined_events_scenario(execution_context):
    """Test realistic scenario with multiple event types combined."""
    events_to_emit = [
        {"type": "start"},
        # Reasoning phase
        {"type": "reasoning-start", "id": "r1"},
        {"type": "reasoning-delta", "id": "r1", "delta": "Analyzing the question..."},
        {"type": "reasoning-end", "id": "r1"},
        # Tool call
        {"type": "tool-input-start", "toolCallId": "tc1", "toolName": "search"},
        {"type": "tool-input-delta", "toolCallId": "tc1", "inputTextDelta": '{"query"'},
        {"type": "tool-input-available", "toolCallId": "tc1", "toolName": "search", "input": {"query": "test"}},
        {"type": "tool-output-available", "toolCallId": "tc1", "output": "result"},
        # Response with sources
        {"type": "text-start", "id": "t1"},
        {"type": "text-delta", "id": "t1", "delta": "Based on the search"},
        {"type": "text-end", "id": "t1"},
        {"type": "source", "sources": [{"url": "https://example.com"}]},
        # Metadata
        {"type": "response-metadata", "id": "resp-1", "modelId": "o1-preview", "usage": {"total_tokens": 100}},
        {"type": "finish-message", "finishReason": "stop"},
    ]

    agent = MockVelAgent(events_to_emit)
    node = AgentNode(id="test-node", agent=agent)

    events = []

    async def collector(event):
        events.append(event)

    execution_context._event_emitter.on(collector)

    result = await node._execute_impl("test input", execution_context)

    # Verify all event types present
    event_types = {e.type for e in events}
    assert EventType.REASONING_START in event_types
    assert EventType.REASONING_TOKEN in event_types
    assert EventType.REASONING_END in event_types
    assert EventType.TOOL_CALL_START in event_types
    assert EventType.TOOL_CALL_COMPLETE in event_types
    assert EventType.TOKEN in event_types
    assert EventType.SOURCE in event_types
    assert EventType.RESPONSE_METADATA in event_types

    # Verify order of events makes sense
    reasoning_start_idx = next(i for i, e in enumerate(events) if e.type == EventType.REASONING_START)
    tool_call_idx = next(i for i, e in enumerate(events) if e.type == EventType.TOOL_CALL_START)
    assert reasoning_start_idx < tool_call_idx, "Reasoning should come before tool calls"


@pytest.mark.asyncio
async def test_unknown_events_ignored_gracefully(execution_context):
    """Test that unknown event types are handled gracefully (no errors)."""
    events_to_emit = [
        {"type": "start"},
        {"type": "unknown-event-type", "data": "should be ignored"},
        {"type": "text-start", "id": "t1"},
        {"type": "text-delta", "id": "t1", "delta": "Hello"},
        {"type": "text-end", "id": "t1"},
        {"type": "another-unknown-type"},
        {"type": "finish-message", "finishReason": "stop"},
    ]

    agent = MockVelAgent(events_to_emit)
    node = AgentNode(id="test-node", agent=agent)

    events = []

    async def collector(event):
        events.append(event)

    execution_context._event_emitter.on(collector)

    # Should not raise any errors
    result = await node._execute_impl("test input", execution_context)

    # Verify known events were processed
    assert result.output["content"] == "Hello"

    # Unknown events should not appear in output (ignored gracefully)
    event_types = [e.type for e in events]
    assert "unknown-event-type" not in event_types
    assert "another-unknown-type" not in event_types
