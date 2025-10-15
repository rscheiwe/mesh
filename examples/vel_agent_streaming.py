"""Vel agent with streaming example.

This example demonstrates how to use a Vel agent in a Mesh graph
with token-by-token streaming output.
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.utils import load_env

# Load environment variables
load_env()

# Import Vel SDK
try:
    from vel import Agent as VelAgent
    VEL_AVAILABLE = True
except ImportError as e:
    VEL_AVAILABLE = False
    print("⚠️  Vel SDK not available or has import issues.")
    print(f"    Error: {e}")
    print("    Install with: pip install 'mesh[vel]'")
    print("\n    Note: Check https://github.com/rscheiwe/vel for installation")
    print("    For now, try the LLM example instead: python examples/simple_agent.py\n")


async def main():
    """Run a Vel agent workflow with streaming."""

    if not VEL_AVAILABLE:
        print("Exiting: Vel SDK required for this example")
        return

    print("=== Mesh Vel Agent Streaming Example ===\n")

    # Create Vel agent
    print("Creating Vel agent...")
    vel_agent = VelAgent(
        id="assistant",
        model={
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.7,
        },
    )
    print(f"✓ Vel agent created: {vel_agent.id}\n")

    # Build graph with Vel agent
    graph = StateGraph()

    # Add Vel agent as a node
    graph.add_node(
        "vel_agent",
        vel_agent,
        node_type="agent",
        system_prompt="You are helping the user with their questions.",
    )

    # Connect START -> vel_agent
    graph.add_edge("START", "vel_agent")
    graph.set_entry_point("vel_agent")

    # Compile graph
    compiled_graph = graph.compile()
    print(f"Graph compiled: {len(compiled_graph.nodes)} nodes\n")

    # Create executor with memory backend
    backend = MemoryBackend()
    executor = Executor(compiled_graph, backend)

    # Create execution context
    context = ExecutionContext(
        graph_id="vel-streaming-example",
        session_id="vel-session-1",
        chat_history=[],
        variables={},
        state={},
    )

    # Execute with streaming
    print("User: Tell me a short joke about Python programming\n")
    print("Agent: ", end="", flush=True)

    async for event in executor.execute(
        "Tell me a short joke about Python programming",
        context
    ):
        print(event, "\n")
        # if event.type == "token":
        #     # Check if it's tool input streaming
        #     if event.metadata.get("event_subtype") == "tool_input":
        #         # Tool argument streaming - optionally display
        #         pass
        #     else:
        #         # Print text tokens as they stream in
        #         print(event.content, end="", flush=True)
        # elif event.type == "message_start":
        #     # Message generation starting
        #     pass
        # elif event.type == "message_complete":
        #     # Message generation complete (text block or finish)
        #     if "finish_reason" in event.metadata:
        #         # Final message completion
        #         pass
        # elif event.type == "tool_call_start":
        #     # Tool execution starting
        #     tool_name = event.metadata.get('tool_name', 'unknown')
        #     if event.metadata.get("input"):
        #         # Tool input is ready
        #         print(f"\n[Calling: {tool_name}]", end="", flush=True)
        #     else:
        #         # Tool call just started (before args complete)
        #         pass
        # elif event.type == "tool_call_complete":
        #     # Tool execution complete with result
        #     print(f" ✓\n", end="", flush=True)
        # elif event.type == "node_start":
        #     # Node execution started
        #     pass
        # elif event.type == "node_complete":
        #     # Node execution completed
        #     print("\n")
        # elif event.type == "execution_complete":
        #     print(f"\n✓ Execution complete!")
        #     print(f"   Session ID: {context.session_id}")
        #     print(f"   Trace ID: {context.trace_id}")


async def multi_turn_conversation():
    """Example of multi-turn conversation with state persistence."""

    if not VEL_AVAILABLE:
        return

    print("\n\n=== Multi-Turn Conversation Example ===\n")

    # Create agent
    vel_agent = VelAgent(
        id="chatbot",
        model={
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.7,
        },
        session_persistence="persistent",
        session_storage="database",
    )

    # Build graph
    graph = StateGraph()
    graph.add_node("agent", vel_agent, node_type="agent")
    graph.add_edge("START", "agent")
    graph.set_entry_point("agent")
    compiled = graph.compile()

    # Use SQLite for state persistence
    from mesh.backends import SQLiteBackend
    backend = SQLiteBackend("vel_example_state.db")
    executor = Executor(compiled, backend)

    # Shared session for conversation continuity
    session_id = "conversation-1"

    # First message
    context1 = ExecutionContext(
        graph_id="conversation",
        session_id=session_id,
        chat_history=[],
        variables={},
        state={},
    )

    print("User: My favorite color is blue.\n")
    print("Agent: ", end="", flush=True)

    async for event in executor.execute("My favorite color is blue.", context1):
        if event.type == "token":
            print(event.content, end="", flush=True)
        elif event.type == "node_complete":
            print("\n")

    # Second message - agent should remember
    await asyncio.sleep(0.5)

    context2 = ExecutionContext(
        graph_id="conversation",
        session_id=session_id,
        chat_history=context1.chat_history,
        variables={},
        state=await backend.load(session_id) or {},
    )

    print("\nUser: What's my favorite color?\n")
    print("Agent: ", end="", flush=True)

    async for event in executor.execute("What's my favorite color?", context2):
        if event.type == "token":
            print(event.content, end="", flush=True)
        elif event.type == "execution_complete":
            print("\n\n✓ Conversation complete!")


if __name__ == "__main__":
    # Run basic streaming example
    asyncio.run(main())

    # Uncomment to run multi-turn conversation example:
    # asyncio.run(multi_turn_conversation())
