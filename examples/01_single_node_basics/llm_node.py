"""Simple programmatic usage example.

This example demonstrates basic usage of Mesh to build and execute
a simple agent workflow programmatically.
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.nodes import LLMNode
from mesh.utils import load_env

# Load environment variables from .env file
load_env()


async def main():
    """Run a simple LLM-based workflow."""
    print("=== Mesh Simple Example ===\n")

    # Build graph programmatically
    graph = StateGraph()

    # Add an LLM node
    graph.add_node(
        "llm",
        None,  # No agent instance needed for LLM node
        node_type="llm",
        model="gpt-4o",
        system_prompt="You are a helpful assistant. Keep responses concise.",
    )

    # Connect START -> llm
    graph.add_edge("START", "llm")
    graph.set_entry_point("llm")

    # Compile graph
    compiled_graph = graph.compile()
    print(f"Graph compiled: {len(compiled_graph.nodes)} nodes, {len(compiled_graph.edges)} edges\n")

    # Create executor
    backend = MemoryBackend()
    executor = Executor(compiled_graph, backend)

    # Create execution context
    context = ExecutionContext(
        graph_id="simple-example",
        session_id="test-session",
        chat_history=[],
        variables={},
        state={},
    )

    # Execute with streaming
    print("Executing graph...\n")
    # print("Response: ", end="", flush=True)

    async for event in executor.execute("what is the capital of France?", context):
        print(event.__dict__)


if __name__ == "__main__":
    asyncio.run(main())
