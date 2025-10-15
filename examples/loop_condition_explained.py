"""Loop Condition Explained: A detailed walkthrough

This example demonstrates exactly how loop_condition works with detailed logging
to show when the condition is evaluated and how the loop exits.
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.utils import load_env

load_env()


def increment(input: dict) -> dict:
    """Increment the value by 1."""
    value = input.get("value", 0)
    new_value = value + 1
    print(f"  [NODE EXECUTED] increment: {value} -> {new_value}")
    return {"value": new_value}


def should_continue_loop(state: dict, output: dict) -> bool:
    """
    Loop condition function.

    This function is called AFTER the node executes, BEFORE deciding whether
    to queue it again for another iteration.

    Args:
        state: The shared state dictionary (context.state)
        output: The output from the node that just executed

    Returns:
        True: Continue looping (queue the node again)
        False: Exit the loop (don't queue the node again)
    """
    value = output.get("value", 0)
    should_continue = value < 5

    print(f"  [CONDITION CHECK] value={value}, continue_loop={should_continue}")

    return should_continue


async def main():
    print("=" * 70)
    print("LOOP CONDITION MECHANISM EXPLAINED")
    print("=" * 70)
    print()
    print("Graph Structure: START -> increment -> increment (loop)")
    print("Loop Condition: Continue while value < 5")
    print("=" * 70)
    print()

    # Build graph with loop
    graph = StateGraph()
    graph.add_node("increment", increment, node_type="tool")
    graph.add_edge("START", "increment")
    graph.add_edge(
        "increment",
        "increment",  # Loop back to itself
        is_loop_edge=True,
        loop_condition=should_continue_loop,  # Your custom condition
        max_iterations=100  # Safety limit
    )
    graph.set_entry_point("increment")

    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())

    context = ExecutionContext(
        graph_id="loop-explained",
        session_id="demo",
        chat_history=[],
        variables={},
        state={},
    )

    print("EXECUTION FLOW:")
    print("-" * 70)
    print()

    iteration = 0
    async for event in executor.execute({"value": 0}, context):
        if event.type == "node_start":
            iteration += 1
            print(f"[ITERATION {iteration}] Starting node: {event.node_id}")

        elif event.type == "node_complete":
            print(f"  [NODE COMPLETE] Output: {event.output}")
            # After this, the executor will check the loop condition

        elif event.type == "execution_complete":
            print()
            print("-" * 70)
            print(f"[EXECUTION COMPLETE] Final output: {event.output}")
            print(f"Total iterations: {iteration}")
            print(f"Loop iterations tracked: {context.loop_iterations}")

    print()
    print("=" * 70)
    print("SUMMARY OF WHAT HAPPENED:")
    print("=" * 70)
    print()
    print("1. Node 'increment' executes (value: 0 -> 1)")
    print("   -> Condition checked: value=1 < 5? YES -> Continue loop")
    print()
    print("2. Node 'increment' executes (value: 1 -> 2)")
    print("   -> Condition checked: value=2 < 5? YES -> Continue loop")
    print()
    print("3. Node 'increment' executes (value: 2 -> 3)")
    print("   -> Condition checked: value=3 < 5? YES -> Continue loop")
    print()
    print("4. Node 'increment' executes (value: 3 -> 4)")
    print("   -> Condition checked: value=4 < 5? YES -> Continue loop")
    print()
    print("5. Node 'increment' executes (value: 4 -> 5)")
    print("   -> Condition checked: value=5 < 5? NO -> EXIT LOOP")
    print()
    print("The loop exits because loop_condition returned False.")
    print("The node is NOT queued for another iteration.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
