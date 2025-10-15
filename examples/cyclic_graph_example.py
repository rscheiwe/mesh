"""Example: Cyclic Graph with Loop Until Condition

This example demonstrates Mesh's support for controlled cycles in graphs.
It implements a "loop until divisible by 5" pattern similar to your screenshot.

The graph repeatedly increments a number until it's divisible by 5.
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.utils import load_env

load_env()


def increment(input: dict) -> dict:
    """Increment the value by 1."""
    value = input.get("value", 0)
    new_value = value + 1
    print(f"  Increment: {value} -> {new_value}")
    return {"value": new_value}


def check_divisible_by_5(input: dict) -> dict:
    """Check if value is divisible by 5."""
    value = input.get("value", 0)
    is_divisible = (value % 5) == 0
    print(f"  Check: {value} % 5 = {value % 5} (divisible: {is_divisible})")
    return {"value": value, "divisible": is_divisible}


async def main():
    print("=" * 60)
    print("Cyclic Graph Example: Loop Until Divisible by 5")
    print("=" * 60)
    print()

    # Build cyclic graph
    graph = StateGraph()

    # Add nodes
    graph.add_node("check", check_divisible_by_5, node_type="tool")
    graph.add_node("increment", increment, node_type="tool")

    # Add edges
    graph.add_edge("START", "check")

    # Loop edge: check -> increment (when NOT divisible)
    # This creates a cycle: check -> increment -> check
    graph.add_edge(
        "increment",
        "check",
        is_loop_edge=True,  # Mark as controlled cycle
        loop_condition=lambda state, output: not output.get("divisible", False),
        max_iterations=20,  # Safety limit
    )

    # Exit edge: increment -> END (when divisible)
    graph.add_edge(
        "check",
        "increment",
        loop_condition=lambda state, output: not output.get("divisible", False),
    )

    # Set entry point
    graph.set_entry_point("check")

    # Compile
    compiled = graph.compile()
    print("✓ Graph compiled successfully (cycle detected and allowed!)")
    print()

    # Execute
    executor = Executor(compiled, MemoryBackend())

    # Test with different starting values
    test_values = [1, 3, 7, 10, 13]

    for start_value in test_values:
        print(f"\nStarting value: {start_value}")
        print("-" * 40)

        context = ExecutionContext(
            graph_id="fives-graph",
            session_id=f"test-{start_value}",
            chat_history=[],
            variables={},
            state={},
        )

        final_value = None
        iteration_count = 0

        async for event in executor.execute({"value": start_value}, context):
            if event.type == "node_complete":
                iteration_count += 1
                if event.output and "value" in event.output:
                    final_value = event.output["value"]

            elif event.type == "execution_complete":
                print(f"  Final value: {final_value}")
                print(f"  Iterations: {iteration_count}")
                print(f"  Loop iterations: {context.loop_iterations}")

    print()
    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
