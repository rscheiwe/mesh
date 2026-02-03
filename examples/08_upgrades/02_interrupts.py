"""Interrupts Example - Human-in-the-loop workflows.

Architecture:
============

    ┌─────────────────────────────────────────────────────────────┐
    │                   INTERRUPT FLOW                            │
    │                                                             │
    │   [Draft] ──► [Review] ──►  ⏸️  ──► [Publish] ──► [Notify]  │
    │                             │                               │
    │                    INTERRUPT_BEFORE                         │
    │                             │                               │
    │              ┌──────────────┴──────────────┐                │
    │              │                             │                │
    │              ▼                             ▼                │
    │      ┌─────────────┐              ┌─────────────┐           │
    │      │  APPROVE    │              │   REJECT    │           │
    │      │             │              │             │           │
    │      │ Resume()    │              │ Reject()    │           │
    │      └──────┬──────┘              └──────┬──────┘           │
    │             │                            │                  │
    │             ▼                            ▼                  │
    │      Continue to                   Abort workflow           │
    │      [Publish]                     (no publish)             │
    └─────────────────────────────────────────────────────────────┘

This example demonstrates interrupt patterns for:
- Approval workflows requiring human confirmation
- Content review before publishing
- Sensitive operations needing explicit authorization

Key APIs:
- graph.set_interrupt_before("node_id") - Pause before node execution
- graph.set_interrupt_after("node_id") - Pause after node execution
- executor.resume_from_interrupt(context, InterruptResume()) - Continue execution
- executor.resume_from_interrupt(context, InterruptReject(reason)) - Abort execution
"""

import asyncio
from datetime import datetime
from mesh import (
    StateGraph,
    Executor,
    ExecutionContext,
    MemoryBackend,
    InterruptResume,
    InterruptReject,
)
from mesh.core.events import EventType


# =============================================================================
# Tool Functions
# =============================================================================


async def draft_content(input, context):
    """Create draft content for review."""
    query = input if isinstance(input, str) else input.get("query", "default")
    draft = f"Draft content about: {query}\n\nThis is the generated content..."
    context.state["draft"] = draft
    return {"draft": draft, "word_count": len(draft.split())}


async def publish_content(input, context):
    """Publish content (sensitive operation)."""
    draft = context.state.get("draft", "No content")
    published_url = f"https://example.com/articles/{datetime.now().strftime('%Y%m%d%H%M%S')}"
    context.state["published"] = True
    context.state["url"] = published_url
    return {"published": True, "url": published_url}


async def send_notification(input, context):
    """Send notification about published content."""
    url = context.state.get("url", "unknown")
    return {"notified": True, "message": f"Content published at {url}"}


async def delete_data(input, context):
    """Delete data (destructive operation)."""
    context.state["deleted"] = True
    return {"deleted": True, "timestamp": datetime.now().isoformat()}


# =============================================================================
# Example: Approval Workflow
# =============================================================================


async def approval_workflow_example():
    """Demonstrate approval workflow with interrupt before sensitive operation."""
    print("\n" + "=" * 60)
    print("Example 1: Approval Workflow")
    print("=" * 60)

    # Build workflow with interrupt before publish
    graph = StateGraph()
    graph.add_node("draft", draft_content, node_type="tool")
    graph.add_node("publish", publish_content, node_type="tool")
    graph.add_node("notify", send_notification, node_type="tool")
    graph.add_edge("draft", "publish")
    graph.add_edge("publish", "notify")
    graph.set_entry_point("draft")

    # Set interrupt BEFORE the publish node
    graph.set_interrupt_before("publish")

    compiled = graph.compile()
    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    context = ExecutionContext(
        graph_id="approval-demo",
        session_id="session-001",
        chat_history=[],
        variables={},
        state={},
    )

    # Execute until interrupt
    print("\n[1] Executing workflow until approval needed...")
    interrupt_event = None

    async for event in executor.execute("Write about AI agents", context):
        if event.type == EventType.NODE_COMPLETE:
            print(f"    Completed: {event.node_id}")
        elif event.type == EventType.INTERRUPT:
            interrupt_event = event
            print(f"\n[2] INTERRUPT: Approval required before '{event.node_id}'")
            print(f"    Draft content: {context.state.get('draft', '')[:50]}...")

    # Simulate user approval
    print("\n[3] User approves the content...")
    resume = InterruptResume()

    # Resume execution
    print("[4] Resuming workflow...")
    async for event in executor.resume_from_interrupt(context, resume):
        if event.type == EventType.NODE_COMPLETE:
            print(f"    Completed: {event.node_id}")
        elif event.type == EventType.EXECUTION_COMPLETE:
            print("\n[5] Workflow completed!")

    print(f"\n[6] Final state:")
    print(f"    Published: {context.state.get('published')}")
    print(f"    URL: {context.state.get('url')}")


# =============================================================================
# Example: Rejection Flow
# =============================================================================


async def rejection_flow_example():
    """Demonstrate rejecting an interrupt to abort execution."""
    print("\n" + "=" * 60)
    print("Example 2: Rejection Flow")
    print("=" * 60)

    graph = StateGraph()
    graph.add_node("draft", draft_content, node_type="tool")
    graph.add_node("publish", publish_content, node_type="tool")
    graph.add_edge("draft", "publish")
    graph.set_entry_point("draft")
    graph.set_interrupt_before("publish")

    compiled = graph.compile()
    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    context = ExecutionContext(
        graph_id="rejection-demo",
        session_id="session-002",
        chat_history=[],
        variables={},
        state={},
    )

    # Execute until interrupt
    print("\n[1] Executing until approval needed...")
    async for event in executor.execute("Write controversial content", context):
        if event.type == EventType.INTERRUPT:
            print(f"[2] INTERRUPT: Content requires review")
            print(f"    Draft: {context.state.get('draft', '')[:50]}...")

    # Simulate user rejection
    print("\n[3] User rejects the content...")
    reject = InterruptReject(reason="Content does not meet guidelines")

    # Attempt to resume with rejection
    print("[4] Rejecting and aborting workflow...")
    try:
        async for event in executor.resume_from_interrupt(context, reject):
            if event.type == EventType.EXECUTION_COMPLETE:
                print("    Execution completed (aborted)")
    except Exception as e:
        print(f"    Workflow aborted: {e}")

    print(f"\n[5] Final state:")
    print(f"    Published: {context.state.get('published', False)}")
    print(f"    (Content was NOT published due to rejection)")


# =============================================================================
# Example: Multi-Step Approval
# =============================================================================


async def multi_step_approval_example():
    """Demonstrate multiple approval points in a workflow."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Step Approval")
    print("=" * 60)

    graph = StateGraph()
    graph.add_node("draft", draft_content, node_type="tool")
    graph.add_node("publish", publish_content, node_type="tool")
    graph.add_node("delete", delete_data, node_type="tool")
    graph.add_edge("draft", "publish")
    graph.add_edge("publish", "delete")
    graph.set_entry_point("draft")

    # Multiple interrupt points
    graph.set_interrupt_before("publish")
    graph.set_interrupt_before("delete")

    compiled = graph.compile()
    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    context = ExecutionContext(
        graph_id="multi-approval-demo",
        session_id="session-003",
        chat_history=[],
        variables={},
        state={},
    )

    # First execution phase
    print("\n[1] Phase 1: Execute until first approval...")
    async for event in executor.execute("Process data", context):
        if event.type == EventType.NODE_COMPLETE:
            print(f"    Completed: {event.node_id}")
        elif event.type == EventType.INTERRUPT:
            print(f"\n[2] INTERRUPT 1: Approval needed for '{event.node_id}'")

    # First approval
    print("[3] Approving publish...")
    async for event in executor.resume_from_interrupt(context, InterruptResume()):
        if event.type == EventType.NODE_COMPLETE:
            print(f"    Completed: {event.node_id}")
        elif event.type == EventType.INTERRUPT:
            print(f"\n[4] INTERRUPT 2: Approval needed for '{event.node_id}'")

    # Second approval
    print("[5] Approving delete...")
    async for event in executor.resume_from_interrupt(context, InterruptResume()):
        if event.type == EventType.NODE_COMPLETE:
            print(f"    Completed: {event.node_id}")
        elif event.type == EventType.EXECUTION_COMPLETE:
            print("\n[6] All approvals granted, workflow complete!")

    print(f"\n[7] Final state:")
    print(f"    Published: {context.state.get('published')}")
    print(f"    Deleted: {context.state.get('deleted')}")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all interrupt examples."""
    print("\n" + "=" * 60)
    print("MESH INTERRUPT EXAMPLES")
    print("=" * 60)

    await approval_workflow_example()
    await rejection_flow_example()
    await multi_step_approval_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
