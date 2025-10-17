"""Visual timing diagram showing when loop_condition is evaluated.

This shows the exact sequence of events during loop execution.
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.utils import load_env

load_env()


def increment(input: dict) -> dict:
    """Node that increments a counter."""
    value = input.get("value", 0)
    new_value = value + 1
    print(f"    │ [3] Node executes: value {value} → {new_value}")
    return {"value": new_value}


def loop_condition(state: dict, output: dict) -> bool:
    """Condition that determines if loop should continue."""
    value = output.get("value", 0)
    should_continue = value < 3
    print(f"    │ [4] Condition evaluated: value={value} < 3? {should_continue}")
    return should_continue


async def main():
    print()
    print("LOOP TIMING DIAGRAM")
    print("=" * 70)
    print()

    graph = StateGraph()
    graph.add_node("increment", increment, node_type="tool")
    graph.add_edge("START", "increment")
    graph.add_edge(
        "increment",
        "increment",
        is_loop_edge=True,
        loop_condition=loop_condition,
        max_iterations=10
    )
    graph.set_entry_point("increment")

    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())
    context = ExecutionContext(
        graph_id="timing",
        session_id="demo",
        chat_history=[],
        variables={},
        state={}
    )

    print("Timeline of events:")
    print()

    iteration = 0
    async for event in executor.execute({"value": 0}, context):
        if event.type == "node_start" and event.node_id == "increment":
            iteration += 1
            print(f"[Iteration {iteration}]")
            print(f"    │ [1] Node dequeued from execution queue")
            print(f"    │ [2] Node about to execute...")

        elif event.type == "node_complete" and event.node_id == "increment":
            # After condition prints, we'll see if it was queued
            pass

        elif event.type == "execution_complete":
            print(f"    │")
            print(f"    └─▶ [5] Condition returned False → LOOP EXITS")
            print()
            print("-" * 70)
            print(f"Final output: {event.output}")
            print(f"Total executions: {iteration}")
            print("-" * 70)

    print()
    print("KEY INSIGHTS:")
    print()
    print("1. loop_condition is evaluated AFTER each node execution")
    print("2. If condition returns True → node is queued for next iteration")
    print("3. If condition returns False → node is NOT queued (loop exits)")
    print("4. The loop exits cleanly without executing the node again")
    print()
    print("In this example:")
    print("  - Iteration 1: value=1 < 3? True  → Continue (queue again)")
    print("  - Iteration 2: value=2 < 3? True  → Continue (queue again)")
    print("  - Iteration 3: value=3 < 3? False → Exit (don't queue)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
