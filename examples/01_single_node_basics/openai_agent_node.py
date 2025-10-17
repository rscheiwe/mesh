"""OpenAI Agents SDK integration example.

This example demonstrates how to use an OpenAI Agents SDK agent in a Mesh graph
with token-by-token streaming output.
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.utils import load_env

# Load environment variables
load_env()

# Import OpenAI Agents SDK
try:
    from agents import Agent
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    print("⚠️  OpenAI Agents SDK not available.")
    print(f"    Error: {e}")
    print("    Install with: pip install openai-agents")
    print("\n    For now, try the Vel example instead: python examples/vel_agent_streaming.py\n")


async def main():
    """Run an OpenAI Agents SDK agent workflow with streaming."""

    if not AGENTS_AVAILABLE:
        print("Exiting: OpenAI Agents SDK required for this example")
        return

    print("=== Mesh OpenAI Agents SDK Streaming Example ===\n")

    # Create OpenAI agent
    print("Creating OpenAI agent...")
    openai_agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant that answers questions concisely."
    )
    print(f"✓ OpenAI agent created: {openai_agent.name}\n")

    # Build graph with OpenAI agent
    graph = StateGraph()

    # Add OpenAI agent as a node
    graph.add_node(
        "openai_agent",
        openai_agent,
        node_type="agent",
        system_prompt="You are helping the user with their questions.",
    )

    # Connect START -> openai_agent
    graph.add_edge("START", "openai_agent")
    graph.set_entry_point("openai_agent")

    # Compile graph
    compiled_graph = graph.compile()
    print(f"Graph compiled: {len(compiled_graph.nodes)} nodes\n")

    # Create executor with memory backend
    backend = MemoryBackend()
    executor = Executor(compiled_graph, backend)

    # Create execution context
    context = ExecutionContext(
        graph_id="openai-streaming-example",
        session_id="openai-session-1",
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
        #     # Print tokens as they stream in
        #     print(event.content, end="", flush=True)
        # elif event.type == "message_start":
        #     # Message generation starting
        #     pass
        # elif event.type == "message_complete":
        #     # Message generation complete
        #     pass
        # elif event.type == "tool_call_start":
        #     # Tool execution starting
        #     tool_name = event.metadata.get('tool_name', 'unknown')
        #     print(f"\n[Calling: {tool_name}]", end="", flush=True)
        # elif event.type == "tool_call_complete":
        #     # Tool execution complete
        #     result = event.output
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


if __name__ == "__main__":
    # Run basic streaming example
    asyncio.run(main())
