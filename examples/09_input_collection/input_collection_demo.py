"""Demo: Auto input collection for upstream tools.

This example demonstrates how mesh can automatically detect Tool nodes
upstream of LLM nodes that need user input, and inject interrupt points
to collect those inputs via chat.

Graph structure:
    Calculator (needs a, b) ──┐
                              ├──> LLM (analyzes results)
    DataFetcher (needs query) ┘

When executed:
1. Executor hits Calculator → INTERRUPT → frontend shows form for (a, b)
2. User provides values → resume → Calculator executes
3. Executor hits DataFetcher → INTERRUPT → frontend shows form for (query)
4. User provides value → resume → DataFetcher executes
5. LLM receives both outputs and generates analysis
"""

import asyncio
from typing import Dict, Any

from mesh import (
    StateGraph,
    Executor,
    ExecutionContext,
    MemoryBackend,
    InterruptResume,
    setup_input_collection,
)
from mesh.core.events import EventType


# ============================================================================
# Tool functions (upstream of LLM)
# ============================================================================

def calculator(a: int, b: int, operation: str = "add") -> Dict[str, Any]:
    """Perform arithmetic operations.

    Args:
        a: First number
        b: Second number
        operation: Operation to perform (add, subtract, multiply, divide)

    Returns:
        Result of the operation
    """
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: division by zero",
    }
    result = operations.get(operation, "Unknown operation")
    return {
        "result": result,
        "operation": operation,
        "a": a,
        "b": b,
    }


def data_fetcher(query: str, limit: int = 10) -> Dict[str, Any]:
    """Fetch data based on a search query.

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        Simulated search results
    """
    # Simulated data fetch
    return {
        "query": query,
        "results": [f"Result {i+1} for '{query}'" for i in range(min(limit, 3))],
        "total": limit,
    }


# ============================================================================
# Build and analyze graph
# ============================================================================

def build_graph():
    """Build the example graph with input collection enabled."""
    graph = StateGraph()

    # Add tool nodes (upstream of LLM)
    graph.add_node("calculator", calculator, node_type="tool")
    graph.add_node("data_fetcher", data_fetcher, node_type="tool")

    # Add LLM node (downstream - receives tool outputs)
    graph.add_node(
        "analyzer",
        None,
        node_type="llm",
        model="gpt-4o-mini",
        system_prompt="Analyze the calculator result and data fetcher results. Provide a summary.",
    )

    # Connect: both tools feed into LLM
    graph.add_edge("START", "calculator")
    graph.add_edge("START", "data_fetcher")
    graph.add_edge("calculator", "analyzer")
    graph.add_edge("data_fetcher", "analyzer")

    graph.set_entry_point("calculator")

    # Analyze and inject input collection interrupts
    requirements = setup_input_collection(graph)

    print("=" * 60)
    print("INPUT COLLECTION ANALYSIS")
    print("=" * 60)
    for req in requirements:
        print(f"\nTool: {req.tool_name} (node: {req.node_id})")
        print(f"Description: {req.tool_description or 'N/A'}")
        print("Required params:")
        for param in req.required_params:
            print(f"  - {param.name}: {param.param_type}")
            if param.description:
                print(f"    {param.description}")
    print("=" * 60)

    return graph, requirements


# ============================================================================
# Execute with simulated user input
# ============================================================================

async def execute_with_input_collection():
    """Execute the graph, handling input collection interrupts."""
    graph_builder, requirements = build_graph()

    # Compile the graph (now includes interrupt_before for tools needing input)
    compiled = graph_builder.compile()

    # Set up executor
    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    context = ExecutionContext(
        graph_id="input-collection-demo",
        session_id="demo-session-1",
        chat_history=[],
        variables={},
        state={},
    )

    print("\n" + "=" * 60)
    print("EXECUTION START")
    print("=" * 60)

    # First execution - will hit first interrupt
    async for event in executor.execute("Analyze the data", context):
        print(f"\nEvent: {event.type.value}")

        if event.type == EventType.INTERRUPT:
            metadata = event.metadata
            print(f"  Node: {event.node_id}")
            print(f"  Interrupt type: {metadata.get('review_metadata', {}).get('interrupt_type')}")

            # Extract param requirements
            review_meta = metadata.get("review_metadata", {})
            if review_meta.get("interrupt_type") == "input_collection":
                print(f"  Tool: {review_meta.get('tool_name')}")
                print("  Params needed:")
                for param in review_meta.get("required_params", []):
                    print(f"    - {param['name']}: {param['type']}")

        elif event.type == EventType.EXECUTION_COMPLETE:
            status = event.metadata.get("status")
            print(f"  Status: {status}")

            if status == "waiting_for_interrupt":
                # Simulate user providing input
                interrupt_id = event.metadata.get("interrupt_id")
                node_id = event.metadata.get("node_id")

                print(f"\n  [SIMULATING USER INPUT for {node_id}]")

                # Provide different inputs based on which tool
                if node_id == "calculator":
                    user_input = {"a": 10, "b": 5, "operation": "multiply"}
                    print(f"  User provides: {user_input}")
                elif node_id == "data_fetcher":
                    user_input = {"query": "machine learning trends"}
                    print(f"  User provides: {user_input}")
                else:
                    user_input = {}

                # Resume execution
                resume = InterruptResume(modified_input=user_input)

                print("\n  [RESUMING EXECUTION]")
                async for resume_event in executor.resume_from_interrupt(context, resume):
                    print(f"\n  Resume Event: {resume_event.type.value}")

                    if resume_event.type == EventType.NODE_COMPLETE:
                        print(f"    Node: {resume_event.node_id}")
                        print(f"    Output: {resume_event.output}")

                    elif resume_event.type == EventType.INTERRUPT:
                        # Another tool needs input
                        print(f"    Another interrupt at: {resume_event.node_id}")
                        # In real app, would handle this recursively

                    elif resume_event.type == EventType.EXECUTION_COMPLETE:
                        final_status = resume_event.metadata.get("status")
                        print(f"    Final status: {final_status}")
                        if final_status == "completed":
                            print(f"    Final output: {resume_event.output}")

        elif event.type == EventType.NODE_COMPLETE:
            print(f"  Node: {event.node_id}")
            if event.output:
                print(f"  Output: {event.output}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    asyncio.run(execute_with_input_collection())
