"""Checkpointing Example - Save and restore execution state.

Architecture:
============

    ┌─────────────────────────────────────────────────────────────┐
    │                      EXECUTION FLOW                         │
    │                                                             │
    │   [Node 1] ──► [Node 2] ──► [Node 3] ──► [Node 4]          │
    │       │            │            │            │              │
    │       ▼            ▼            ▼            ▼              │
    │   ┌────────────────────────────────────────────────┐        │
    │   │              STATE BACKEND                     │        │
    │   │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐          │        │
    │   │  │ CP1 │  │ CP2 │  │ CP3 │  │ CP4 │          │        │
    │   │  └─────┘  └─────┘  └─────┘  └─────┘          │        │
    │   │     ▲         ▲         ▲         ▲           │        │
    │   │     │         │         │         │           │        │
    │   │  checkpoint() calls save state snapshots      │        │
    │   └────────────────────────────────────────────────┘        │
    │                                                             │
    │   On crash: restore(checkpoint_id) ──► Resume from CP      │
    └─────────────────────────────────────────────────────────────┘

This example demonstrates how to checkpoint execution state for:
- Long-running workflows that may need to pause/resume
- Crash recovery
- Branching execution from a saved point

Key APIs:
- executor.checkpoint(context) - Save current state
- executor.restore(checkpoint_id) - Restore from checkpoint
- backend.list_checkpoints(session_id) - List available checkpoints
"""

import asyncio
from mesh import (
    StateGraph,
    Executor,
    ExecutionContext,
    MemoryBackend,
    Checkpoint,
)
from mesh.core.events import EventType


# =============================================================================
# Tool Functions
# =============================================================================


async def step_one(input, context):
    """First processing step."""
    context.state["step1_done"] = True
    context.state["data"] = "Initial data from step 1"
    return {"message": "Step 1 complete", "progress": 33}


async def step_two(input, context):
    """Second processing step - simulates long operation."""
    context.state["step2_done"] = True
    context.state["data"] += " -> Enhanced in step 2"
    return {"message": "Step 2 complete", "progress": 66}


async def step_three(input, context):
    """Final processing step."""
    context.state["step3_done"] = True
    context.state["data"] += " -> Finalized in step 3"
    return {"message": "Step 3 complete", "progress": 100}


# =============================================================================
# Example: Basic Checkpointing
# =============================================================================


async def basic_checkpoint_example():
    """Demonstrate basic checkpoint save and restore."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Checkpointing")
    print("=" * 60)

    # Build a simple sequential graph
    graph = StateGraph()
    graph.add_node("step1", step_one, node_type="tool")
    graph.add_node("step2", step_two, node_type="tool")
    graph.add_node("step3", step_three, node_type="tool")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.set_entry_point("step1")
    compiled = graph.compile()

    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    # Execute the workflow
    context = ExecutionContext(
        graph_id="checkpoint-demo",
        session_id="session-001",
        chat_history=[],
        variables={},
        state={},
    )

    print("\n[1] Executing workflow...")
    async for event in executor.execute("Start processing", context):
        if event.type == EventType.NODE_COMPLETE:
            print(f"    Completed: {event.node_id}")

    print(f"\n[2] Final state: {context.state}")

    # Create a checkpoint after execution
    checkpoint_id = await executor.checkpoint(context)
    print(f"\n[3] Created checkpoint: {checkpoint_id}")

    # List available checkpoints
    checkpoints = await backend.list_checkpoints(context.session_id)
    print(f"[4] Available checkpoints: {len(checkpoints)}")

    # Restore from checkpoint
    restored_context = await executor.restore(checkpoint_id)
    print(f"\n[5] Restored state: {restored_context.state}")
    print(f"    Session ID matches: {restored_context.session_id == context.session_id}")


# =============================================================================
# Example: Checkpoint for Crash Recovery
# =============================================================================


async def crash_recovery_example():
    """Demonstrate using checkpoints for crash recovery."""
    print("\n" + "=" * 60)
    print("Example 2: Crash Recovery Pattern")
    print("=" * 60)

    graph = StateGraph()
    graph.add_node("step1", step_one, node_type="tool")
    graph.add_node("step2", step_two, node_type="tool")
    graph.set_entry_point("step1")
    graph.add_edge("step1", "step2")
    compiled = graph.compile()

    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    context = ExecutionContext(
        graph_id="recovery-demo",
        session_id="session-002",
        chat_history=[],
        variables={},
        state={"retry_count": 0},
    )

    # Simulate: Execute first step and checkpoint
    print("\n[1] Executing step 1...")
    event_count = 0
    async for event in executor.execute("Start", context):
        event_count += 1
        if event.type == EventType.NODE_COMPLETE and event.node_id == "step1":
            # Checkpoint after step 1
            checkpoint_id = await executor.checkpoint(context)
            print(f"    Checkpointed after step1: {checkpoint_id[:8]}...")

    print(f"[2] Completed with {event_count} events")
    print(f"    State after execution: {context.state}")

    # Simulate: System crash and recovery
    print("\n[3] Simulating crash recovery...")
    print("    (In real scenario, checkpoint_id would be stored in database)")

    # Restore and continue
    recovered = await executor.restore(checkpoint_id)
    print(f"[4] Recovered state: {recovered.state}")


# =============================================================================
# Example: Branching from Checkpoint
# =============================================================================


async def branching_example():
    """Demonstrate creating branches from a checkpoint."""
    print("\n" + "=" * 60)
    print("Example 3: Branching from Checkpoint")
    print("=" * 60)

    graph = StateGraph()
    graph.add_node("init", step_one, node_type="tool")
    graph.set_entry_point("init")
    compiled = graph.compile()

    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    # Create initial context and execute
    main_context = ExecutionContext(
        graph_id="branch-demo",
        session_id="main-session",
        chat_history=[],
        variables={},
        state={"branch": "main"},
    )

    print("\n[1] Executing main branch...")
    async for event in executor.execute("Initialize", main_context):
        pass

    # Checkpoint the main branch
    main_checkpoint = await executor.checkpoint(main_context)
    print(f"[2] Main checkpoint: {main_checkpoint[:8]}...")

    # Create a branch from the checkpoint
    branch_context = await executor.restore(main_checkpoint)
    branch_context.session_id = "branch-session"  # New session for branch
    branch_context.state["branch"] = "experiment-A"

    print(f"\n[3] Created branch with state: {branch_context.state}")

    # Save the branch checkpoint
    branch_checkpoint = await executor.checkpoint(branch_context)
    print(f"[4] Branch checkpoint: {branch_checkpoint[:8]}...")

    # Verify both branches exist independently
    main_checkpoints = await backend.list_checkpoints("main-session")
    branch_checkpoints = await backend.list_checkpoints("branch-session")

    print(f"\n[5] Main session checkpoints: {len(main_checkpoints)}")
    print(f"    Branch session checkpoints: {len(branch_checkpoints)}")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all checkpointing examples."""
    print("\n" + "=" * 60)
    print("MESH CHECKPOINTING EXAMPLES")
    print("=" * 60)

    await basic_checkpoint_example()
    await crash_recovery_example()
    await branching_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
