"""Streaming Modes Example - Different views of execution.

Architecture:
============

    ┌─────────────────────────────────────────────────────────────┐
    │                    EXECUTION ENGINE                         │
    │                                                             │
    │   [Node 1] ──► [Node 2] ──► [Node 3] ──► [Node 4]          │
    │       │            │            │            │              │
    │       ▼            ▼            ▼            ▼              │
    │   ┌─────────────────────────────────────────────────────┐   │
    │   │              RAW EVENT STREAM                       │   │
    │   │  (node_start, token, node_complete, ...)           │   │
    │   └─────────────────────────────────────────────────────┘   │
    │                          │                                  │
    │              ┌───────────┴───────────┐                      │
    │              │   StreamModeAdapter   │                      │
    │              └───────────┬───────────┘                      │
    │                          │                                  │
    │   ┌──────────┬───────────┼───────────┬──────────┐          │
    │   ▼          ▼           ▼           ▼          ▼          │
    │ VALUES    UPDATES    MESSAGES    EVENTS     DEBUG          │
    │ (full     (deltas    (chat       (all       (internal      │
    │  state)    only)      msgs)      events)    state)         │
    └─────────────────────────────────────────────────────────────┘

    Mode Comparison:
    ┌──────────┬────────────────────────────────────────────────┐
    │  Mode    │  Output                                        │
    ├──────────┼────────────────────────────────────────────────┤
    │  VALUES  │  Full state snapshot after each node           │
    │  UPDATES │  Only what changed (added/modified/removed)    │
    │ MESSAGES │  Chat messages only (lowest volume)            │
    │  EVENTS  │  All ExecutionEvent objects (default)          │
    │  DEBUG   │  Events + internal queue/timing state          │
    └──────────┴────────────────────────────────────────────────┘

This example demonstrates streaming modes for:
- UI updates with full state snapshots (VALUES)
- Efficient state diff updates (UPDATES)
- Chat message streaming (MESSAGES)
- Full event streaming (EVENTS - default)
- Debugging and development (DEBUG)

Key APIs:
- executor.stream(input, context, mode=StreamMode.VALUES)
- StreamMode enum (VALUES, UPDATES, MESSAGES, EVENTS, DEBUG)
- StateValue, StateUpdate, StreamMessage, DebugInfo dataclasses
"""

import asyncio
from mesh import (
    StateGraph,
    Executor,
    ExecutionContext,
    MemoryBackend,
    StreamMode,
    StateValue,
    StateUpdate,
    StreamMessage,
    DebugInfo,
)
from mesh.core.events import EventType


# =============================================================================
# Tool Functions
# =============================================================================


async def step_one(input, context):
    """First step - adds initial data."""
    context.state["step1_data"] = "Hello"
    context.state["counter"] = 1
    return {"step": 1, "message": "Step 1 complete"}


async def step_two(input, context):
    """Second step - modifies data."""
    context.state["step1_data"] = "Hello World"  # Modified
    context.state["step2_data"] = "New data"  # Added
    context.state["counter"] = 2  # Modified
    return {"step": 2, "message": "Step 2 complete"}


async def step_three(input, context):
    """Third step - finalizes."""
    context.state["final_result"] = "Complete"
    context.state["counter"] = 3
    return {"step": 3, "message": "Step 3 complete"}


# =============================================================================
# Build Test Graph
# =============================================================================


def build_test_graph():
    """Build a simple 3-step graph for testing streaming modes."""
    graph = StateGraph()
    graph.add_node("step1", step_one, node_type="tool")
    graph.add_node("step2", step_two, node_type="tool")
    graph.add_node("step3", step_three, node_type="tool")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.set_entry_point("step1")
    return graph.compile()


# =============================================================================
# Example: VALUES Mode - Full State Snapshots
# =============================================================================


async def values_mode_example():
    """Demonstrate VALUES streaming mode - full state after each node."""
    print("\n" + "=" * 60)
    print("Example 1: VALUES Mode (Full State Snapshots)")
    print("=" * 60)

    compiled = build_test_graph()
    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    context = ExecutionContext(
        graph_id="values-demo",
        session_id="session-001",
        chat_history=[],
        variables={},
        state={},
    )

    print("\n[1] Streaming with VALUES mode...")
    print("    (Shows complete state after each node)\n")

    async for value in executor.stream("Start", context, mode=StreamMode.VALUES):
        if isinstance(value, StateValue):
            print(f"    Node: {value.node_id}")
            print(f"    Full State: {value.state}")
            print(f"    Timestamp: {value.timestamp}")
            print()

    print("[2] VALUES mode is ideal for:")
    print("    - UI state synchronization")
    print("    - State snapshots for debugging")
    print("    - When you need complete context at each step")


# =============================================================================
# Example: UPDATES Mode - State Deltas
# =============================================================================


async def updates_mode_example():
    """Demonstrate UPDATES streaming mode - state changes only."""
    print("\n" + "=" * 60)
    print("Example 2: UPDATES Mode (State Deltas)")
    print("=" * 60)

    compiled = build_test_graph()
    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    context = ExecutionContext(
        graph_id="updates-demo",
        session_id="session-002",
        chat_history=[],
        variables={},
        state={},
    )

    print("\n[1] Streaming with UPDATES mode...")
    print("    (Shows only what changed)\n")

    async for update in executor.stream("Start", context, mode=StreamMode.UPDATES):
        if isinstance(update, StateUpdate):
            print(f"    Node: {update.node_id}")
            if update.added:
                print(f"    Added: {update.added}")
            if update.modified:
                print(f"    Modified: {update.modified}")
            if update.removed:
                print(f"    Removed: {update.removed}")
            print()

    print("[2] UPDATES mode is ideal for:")
    print("    - Bandwidth-efficient updates")
    print("    - Real-time diff synchronization")
    print("    - Tracking what changed at each step")


# =============================================================================
# Example: EVENTS Mode - All Execution Events
# =============================================================================


async def events_mode_example():
    """Demonstrate EVENTS streaming mode - all execution events."""
    print("\n" + "=" * 60)
    print("Example 3: EVENTS Mode (All Execution Events)")
    print("=" * 60)

    compiled = build_test_graph()
    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    context = ExecutionContext(
        graph_id="events-demo",
        session_id="session-003",
        chat_history=[],
        variables={},
        state={},
    )

    print("\n[1] Streaming with EVENTS mode (default)...")
    print("    (Shows all ExecutionEvent objects)\n")

    event_counts = {}
    async for event in executor.stream("Start", context, mode=StreamMode.EVENTS):
        event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Show key events
        if event.type == EventType.NODE_START:
            print(f"    START: {event.node_id}")
        elif event.type == EventType.NODE_COMPLETE:
            print(f"    COMPLETE: {event.node_id}")
        elif event.type == EventType.EXECUTION_COMPLETE:
            print(f"    EXECUTION COMPLETE")

    print(f"\n[2] Event type counts:")
    for event_type, count in sorted(event_counts.items()):
        print(f"    {event_type}: {count}")

    print("\n[3] EVENTS mode is ideal for:")
    print("    - Detailed execution monitoring")
    print("    - Custom event handling")
    print("    - When you need access to all event metadata")


# =============================================================================
# Example: DEBUG Mode - Everything Including Internals
# =============================================================================


async def debug_mode_example():
    """Demonstrate DEBUG streaming mode - includes internal state."""
    print("\n" + "=" * 60)
    print("Example 4: DEBUG Mode (Internal State)")
    print("=" * 60)

    compiled = build_test_graph()
    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    context = ExecutionContext(
        graph_id="debug-demo",
        session_id="session-004",
        chat_history=[],
        variables={},
        state={},
    )

    print("\n[1] Streaming with DEBUG mode...")
    print("    (Includes internal execution state)\n")

    debug_count = 0
    async for debug_info in executor.stream("Start", context, mode=StreamMode.DEBUG):
        if isinstance(debug_info, DebugInfo):
            debug_count += 1
            event = debug_info.event

            # Show first few debug items
            if debug_count <= 5:
                print(f"    Event: {event.type}")
                print(f"    Internal: {debug_info.internal_state}")
                print(f"    Queue: {debug_info.queue}")
                print(f"    Timing: {debug_info.timing}")
                print()

    if debug_count > 5:
        print(f"    ... and {debug_count - 5} more debug items\n")

    print(f"[2] Total debug items: {debug_count}")
    print("\n[3] DEBUG mode is ideal for:")
    print("    - Debugging execution issues")
    print("    - Understanding internal queue state")
    print("    - Development and troubleshooting")


# =============================================================================
# Example: Comparing All Modes
# =============================================================================


async def comparison_example():
    """Compare output counts across all streaming modes."""
    print("\n" + "=" * 60)
    print("Example 5: Mode Comparison")
    print("=" * 60)

    compiled = build_test_graph()
    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    results = {}

    for mode in [StreamMode.VALUES, StreamMode.UPDATES, StreamMode.MESSAGES, StreamMode.EVENTS, StreamMode.DEBUG]:
        context = ExecutionContext(
            graph_id="comparison-demo",
            session_id=f"session-{mode.value}",
            chat_history=[],
            variables={},
            state={},
        )

        count = 0
        async for _ in executor.stream("Start", context, mode=mode):
            count += 1

        results[mode.value] = count

    print("\n[1] Items emitted by each mode:\n")
    for mode_name, count in results.items():
        bar = "#" * min(count, 50)
        print(f"    {mode_name:10} | {count:3} | {bar}")

    print("\n[2] Mode selection guide:")
    print("    VALUES  - Best for UI state sync, moderate volume")
    print("    UPDATES - Best for bandwidth efficiency, low volume")
    print("    MESSAGES - Best for chat UIs, lowest volume")
    print("    EVENTS  - Best for full control, moderate volume")
    print("    DEBUG   - Best for debugging, highest volume")


# =============================================================================
# Example: Practical Use Case - UI Updates
# =============================================================================


async def ui_update_example():
    """Demonstrate practical UI update pattern."""
    print("\n" + "=" * 60)
    print("Example 6: Practical UI Update Pattern")
    print("=" * 60)

    compiled = build_test_graph()
    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    context = ExecutionContext(
        graph_id="ui-demo",
        session_id="session-ui",
        chat_history=[],
        variables={},
        state={},
    )

    print("\n[1] Simulating UI updates with VALUES mode...\n")

    async for value in executor.stream("Start", context, mode=StreamMode.VALUES):
        if isinstance(value, StateValue):
            # Simulate sending to WebSocket/SSE
            ui_update = {
                "type": "state_update",
                "node": value.node_id,
                "state": value.state,
                "timestamp": value.timestamp.isoformat() if value.timestamp else None,
            }
            print(f"    -> UI Update: {ui_update}")

    print("\n[2] In production, these updates would be sent to:")
    print("    - WebSocket connections")
    print("    - Server-Sent Events (SSE)")
    print("    - Polling endpoints")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all streaming mode examples."""
    print("\n" + "=" * 60)
    print("MESH STREAMING MODES EXAMPLES")
    print("=" * 60)

    await values_mode_example()
    await updates_mode_example()
    await events_mode_example()
    await debug_mode_example()
    await comparison_example()
    await ui_update_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
