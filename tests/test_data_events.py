"""Wire format of custom data-* events (ExecutionEvent.to_dict)."""

from mesh.core.events import (
    EventType,
    ExecutionEvent,
    create_mesh_node_start_event,
    transform_event_for_transient_mode,
)


def test_vel_data_event_keeps_its_own_type_and_payload():
    raw = {
        "type": "data-object-complete",
        "data": {"object": {"vertical": "Retail"}, "mode": "object"},
        "transient": False,
    }
    event = ExecutionEvent(
        type=EventType.CUSTOM_DATA,
        node_id="extract",
        content=raw["data"],
        metadata={"data_type": "data-object-complete", "transient": False},
        raw_event=raw,
    )

    assert event.to_dict() == raw


def test_vel_data_event_keeps_id_and_transient_flag():
    raw = {"type": "data-progress", "id": "p1", "data": {"pct": 50}, "transient": True}
    event = ExecutionEvent(type=EventType.CUSTOM_DATA, raw_event=raw)

    assert event.to_dict() == raw


def test_data_type_metadata_without_raw_event_uses_content_as_data():
    event = ExecutionEvent(
        type=EventType.CUSTOM_DATA,
        node_id="conv",
        content={"conversation_complete": True},
        metadata={"data_type": "data-conversation-complete", "conversation_id": "c1"},
    )

    assert event.to_dict() == {
        "type": "data-conversation-complete",
        "data": {"conversation_complete": True},
    }


def test_mesh_helper_events_serialize_under_their_data_event_type():
    event = create_mesh_node_start_event(
        node_id="explain", node_type="agent", is_final=True, is_intermediate=False
    )

    assert event.to_dict() == {
        "type": "data-mesh-node-start",
        "data": {
            "node_id": "explain",
            "node_type": "agent",
            "is_final": True,
            "is_intermediate": False,
        },
        "transient": True,
    }


def test_transient_mode_events_serialize_under_prefixed_type():
    original = ExecutionEvent(type=EventType.TEXT_DELTA, node_id="a", delta="hi")

    chunk = transform_event_for_transient_mode(original, "agent").to_dict()

    assert chunk["type"] == "data-agent-node-text-delta"
    assert chunk["data"]["delta"] == "hi"
    assert chunk["transient"] is True


def test_custom_data_without_a_data_type_stays_data_custom():
    event = ExecutionEvent(
        type=EventType.CUSTOM_DATA,
        node_id="rag",
        metadata={"type": "rag_retrieval_start", "query": "q"},
    )

    assert event.to_dict()["type"] == "data-custom"
