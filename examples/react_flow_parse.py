"""React Flow JSON parsing example.

This example demonstrates how to parse Flowise-compatible React Flow JSON
and execute the resulting graph.
"""

import asyncio
import json
from mesh import (
    ReactFlowParser,
    NodeRegistry,
    Executor,
    ExecutionContext,
    MemoryBackend,
)
from mesh.utils import load_env

# Load environment variables from .env file
load_env()


# Sample React Flow JSON (Flowise-compatible)
SAMPLE_FLOW = {
    "nodes": [
        {
            "id": "start_0",
            "type": "startAgentflow",
            "data": {"name": "startAgentflow", "label": "Start", "inputs": {}},
            "position": {"x": 100, "y": 100},
        },
        {
            "id": "llm_0",
            "type": "llmAgentflow",
            "data": {
                "name": "llmAgentflow",
                "label": "LLM",
                "inputs": {
                    "model": "gpt-4",
                    "systemPrompt": "You are a helpful assistant.",
                },
            },
            "position": {"x": 300, "y": 100},
        },
        {
            "id": "end_0",
            "type": "endAgentflow",
            "data": {"name": "endAgentflow", "label": "End", "inputs": {}},
            "position": {"x": 500, "y": 100},
        },
    ],
    "edges": [
        {"source": "start_0", "target": "llm_0"},
        {"source": "llm_0", "target": "end_0"},
    ],
}


async def main():
    """Parse and execute React Flow JSON."""
    print("=== Mesh React Flow Parser Example ===\n")

    # Create node registry
    registry = NodeRegistry()

    # Parse React Flow JSON
    parser = ReactFlowParser(registry)
    print("Parsing React Flow JSON...")
    graph = parser.parse(SAMPLE_FLOW)
    print(f"✓ Graph parsed: {len(graph.nodes)} nodes, {len(graph.edges)} edges\n")

    # Create executor
    backend = MemoryBackend()
    executor = Executor(graph, backend)

    # Create execution context
    context = ExecutionContext(
        graph_id="react-flow-example",
        session_id="test-session",
        chat_history=[],
        variables={},
        state={},
    )

    # Execute
    print("Executing graph...\n")
    print("Response: ", end="", flush=True)

    async for event in executor.execute("Hello! Tell me a joke.", context):
        if event.type == "token":
            print(event.content, end="", flush=True)
        elif event.type == "execution_complete":
            print(f"\n\nExecution complete!")


if __name__ == "__main__":
    asyncio.run(main())
