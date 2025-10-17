"""Tool node to LLM node example.

This example demonstrates:
- Using a Tool node to execute a Python function
- Passing the tool output to an LLM node
- Variable resolution ({{tool_id.output}})
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.nodes import ToolNode, LLMNode
from mesh.utils import load_env

# Load environment variables
load_env()


# Define a simple tool function
def calculate_stats(input_data):
    """Calculate statistics for a list of numbers."""
    # Extract numbers from input
    if isinstance(input_data, dict):
        numbers = input_data.get("numbers", [])
    elif isinstance(input_data, str):
        # Parse comma-separated numbers
        try:
            numbers = [float(x.strip()) for x in input_data.split(",")]
        except:
            return {"error": "Could not parse numbers"}
    else:
        numbers = input_data

    if not numbers:
        return {"error": "No numbers provided"}

    # Calculate stats
    return {
        "count": len(numbers),
        "sum": sum(numbers),
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


async def main():
    """Run a tool -> LLM workflow."""

    print("=== Tool to LLM Example ===\n")

    # Build graph
    graph = StateGraph()

    # Node 1: Tool node - calculates statistics
    graph.add_node(
        "calculator",
        None,
        node_type="tool",
        function=calculate_stats,
    )

    # Node 2: LLM node - interprets results
    graph.add_node(
        "interpreter",
        None,
        node_type="llm",
        model="gpt-4o",
        system_prompt=(
            "You are a data interpreter. The user provided numbers and we calculated statistics. "
            "Here are the results: {{calculator.output}}. "
            "Explain these statistics in a friendly, easy-to-understand way. "
            "Mention any interesting patterns you notice."
        ),
    )

    # Connect: START -> calculator -> interpreter
    graph.add_edge("START", "calculator")
    graph.add_edge("calculator", "interpreter")
    graph.set_entry_point("calculator")

    # Compile
    compiled_graph = graph.compile()
    print(f"Graph: {len(compiled_graph.nodes)} nodes\n")

    # Execute
    backend = MemoryBackend()
    executor = Executor(compiled_graph, backend)

    context = ExecutionContext(
        graph_id="tool-to-llm",
        session_id="test-session",
        chat_history=[],
        variables={},
        state={},
    )

    # Input: comma-separated numbers
    user_input = "10, 25, 15, 30, 20, 18, 22"
    print(f"Input numbers: {user_input}\n")
    print("="*80)

    current_node = None

    async for event in executor.execute(user_input, context):
        if event.type == "node_start":
            current_node = event.node_id
            print(f"\n[{current_node.upper()}]")
            print("-"*80)

        elif event.type == "node_complete":
            if current_node == "calculator":
                print(f"Output: {event.output}")
            print("-"*80)

        elif event.type == "token":
            # LLM tokens
            print(event.content, end="", flush=True)

        elif event.type == "execution_complete":
            print("\n" + "="*80)
            print("\n✓ Complete!")


if __name__ == "__main__":
    asyncio.run(main())
