"""Example: Cyclic Graph with Max Iterations

This demonstrates using max_iterations to control loop cycles.
The loop runs a fixed number of times before exiting.
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.utils import load_env

load_env()


def process_item(input: dict) -> dict:
    """Process an item in the loop."""
    count = input.get("count", 0)
    value = input.get("value", "")
    new_count = count + 1
    new_value = value + f"[{new_count}]"

    print(f"  Processing iteration {new_count}: {new_value}")

    return {"count": new_count, "value": new_value}


async def main():
    print("=" * 60)
    print("Cyclic Graph Example: Max Iterations")
    print("=" * 60)
    print()

    # Build graph with cycle limited by max_iterations
    graph = StateGraph()

    graph.add_node("process", process_item, node_type="tool")

    # Self-loop with max iterations
    graph.add_edge("START", "process")
    graph.add_edge(
        "process",
        "process",  # Loop back to itself!
        is_loop_edge=True,
        max_iterations=5,  # Run exactly 5 times
    )

    graph.set_entry_point("process")

    # Compile
    compiled = graph.compile()
    print("✓ Graph compiled with self-loop")
    print()

    # Execute
    executor = Executor(compiled, MemoryBackend())

    context = ExecutionContext(
        graph_id="max-iter-graph",
        session_id="test-1",
        chat_history=[],
        variables={},
        state={},
    )

    print("Running loop with max_iterations=5:")
    print("-" * 40)

    async for event in executor.execute({"count": 0, "value": "START"}, context):
        if event.type == "execution_complete":
            print()
            print(f"Final output: {event.output}")
            print(f"Loop iterations tracked: {context.loop_iterations}")

    print()
    print("=" * 60)
    print("Loop completed after 5 iterations!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
