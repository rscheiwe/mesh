"""Multi-node agent chain example.

This example demonstrates connecting multiple nodes in sequence:
- Agent 1 (analyzer): Analyzes user input
- Agent 2 (responder): Responds based on analysis

This shows how output from one node flows to the next node.
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.nodes import LLMNode
from mesh.utils import load_env

# Load environment variables
load_env()


async def main():
    """Run a multi-node chain: analyzer -> responder."""

    print("=== Multi-Node Agent Chain Example ===\n")

    # Build graph with two LLM nodes in sequence
    graph = StateGraph()

    # Node 1: Analyzer - extracts key information
    graph.add_node(
        "analyzer",
        None,
        node_type="llm",
        model="gpt-4o",
        system_prompt=(
            "You are an analyzer. Extract the key information from the user's message. "
            "Identify: (1) the main topic, (2) the user's intent, (3) any specific questions. "
            "Be concise - respond in 2-3 sentences."
        ),
    )

    # Node 2: Responder - crafts response based on analysis
    graph.add_node(
        "responder",
        None,
        node_type="llm",
        model="gpt-4o",
        system_prompt=(
            "You are a helpful assistant. Based on the analysis provided, "
            "give a helpful and detailed response to the user. "
            "The analysis is: {{analyzer.content}}"
        ),
    )

    # Connect nodes: START -> analyzer -> responder
    graph.add_edge("START", "analyzer")
    graph.add_edge("analyzer", "responder")
    graph.set_entry_point("analyzer")

    # Compile graph
    compiled_graph = graph.compile()
    print(f"Graph compiled: {len(compiled_graph.nodes)} nodes, {len(compiled_graph.edges)} edges\n")

    # Create executor
    backend = MemoryBackend()
    executor = Executor(compiled_graph, backend)

    # Create execution context
    context = ExecutionContext(
        graph_id="agent-chain",
        session_id="test-session",
        chat_history=[],
        variables={},
        state={},
    )

    # Execute with streaming
    user_input = "I'm learning Python and confused about async/await. Can you explain it?"
    print(f"User: {user_input}\n")
    print("="*80)

    current_node = None

    async for event in executor.execute(user_input, context):
        if event.type == "node_start":
            current_node = event.node_id
            print(f"\n[{current_node.upper()}]")
            print("-"*80)

        elif event.type == "token":
            # Print tokens as they stream
            print(event.content, end="", flush=True)

        elif event.type == "node_complete":
            print("\n" + "-"*80)

        elif event.type == "execution_complete":
            print("\n" + "="*80)
            print("\n✓ Execution complete!")
            print(f"  Total nodes executed: {len(context.executed_data)}")


if __name__ == "__main__":
    asyncio.run(main())
