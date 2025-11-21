"""Test event_mode functionality with multi-agent workflow.

This test creates a simple two-agent workflow:
- Agent 1: event_mode='silent' - should NOT emit events
- Agent 2: event_mode='full' - should emit all events

Run: python examples/test_event_mode.py
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.utils import load_env

# Load environment variables
load_env()

# Import Vel SDK
try:
    from vel import Agent
    VEL_AVAILABLE = True
except ImportError as e:
    VEL_AVAILABLE = False
    print("⚠️  Vel SDK not available")
    print(f"    Error: {e}")


async def main():
    if not VEL_AVAILABLE:
        print("Exiting: Vel SDK required for this example")
        return

    print("=== Testing event_mode with multi-agent workflow ===\n")

    # Create two simple agents
    agent_1 = Agent(
        id="writer",
        model={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
        },
    )

    agent_2 = Agent(
        id="reviewer",
        model={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
        },
    )

    # Build graph: START -> agent_1 (silent) -> agent_2 (full)
    graph = StateGraph()

    # Add nodes with event_mode configuration
    graph.add_node(
        "writer_agent",
        agent_1,
        node_type="agent",
        system_prompt="Write 2 short tweets about the topic.",
        event_mode="silent",  # Should NOT emit any events
    )

    graph.add_node(
        "reviewer_agent",
        agent_2,
        node_type="agent",
        system_prompt="Review these tweets: {{writer_agent.output}}",
        event_mode="full",  # Should emit all events
    )

    # Connect nodes
    graph.add_edge("START", "writer_agent")
    graph.add_edge("writer_agent", "reviewer_agent")
    graph.set_entry_point("writer_agent")

    # Compile graph
    compiled_graph = graph.compile()

    print("=" * 80)
    print("GRAPH CONFIGURATION:")
    for node_id, node in compiled_graph.nodes.items():
        event_mode = getattr(node, 'event_mode', 'N/A')
        print(f"  {node_id}: {node.__class__.__name__} - event_mode={event_mode}")
    print("=" * 80)
    print()

    # Execute with event tracking
    context = ExecutionContext(
        graph_id="test-event-mode",
        session_id="test-session",
    )

    executor = Executor(compiled_graph, MemoryBackend())

    print("EXECUTING GRAPH...\n")
    print("Expected behavior:")
    print("  - NO events from 'writer_agent' (silent mode)")
    print("  - ALL events from 'reviewer_agent' (full mode)")
    print()
    print("=" * 80)
    print("EVENTS RECEIVED:")
    print("=" * 80)

    event_count = {"writer_agent": 0, "reviewer_agent": 0, "START": 0, "other": 0}

    async for event in executor.execute("dogs", context):
        node_id = event.node_id or "unknown"

        # Track events by node
        if node_id in event_count:
            event_count[node_id] += 1
        else:
            event_count["other"] += 1

        # Print event details
        print(f"[{event.type}] from '{node_id}' - {event.content if event.content else ''}")

    print()
    print("=" * 80)
    print("EVENT SUMMARY:")
    print("=" * 80)
    print(f"  START: {event_count['START']} events")
    print(f"  writer_agent (silent): {event_count['writer_agent']} events")
    print(f"  reviewer_agent (full): {event_count['reviewer_agent']} events")
    print(f"  other: {event_count['other']} events")
    print()

    # Validate
    if event_count['writer_agent'] == 0:
        print("✅ SUCCESS: writer_agent emitted 0 events (silent mode working)")
    else:
        print(f"❌ FAILURE: writer_agent emitted {event_count['writer_agent']} events (should be 0)")

    if event_count['reviewer_agent'] > 0:
        print("✅ SUCCESS: reviewer_agent emitted events (full mode working)")
    else:
        print("❌ FAILURE: reviewer_agent emitted 0 events (should emit events)")


if __name__ == "__main__":
    asyncio.run(main())
