"""Test DataHandler → LLM variable resolution.

This test replicates the issue where LLM can't see DataHandler output.
"""

import asyncio
from mesh import StateGraph
from mesh.nodes import LLMNode, ToolNode
from mesh.core.state import ExecutionContext
from mesh.backends.memory import MemoryBackend


# Mock tool that returns data like DataHandlerNode would
async def mock_data_handler(input, context):
    """Simulates DataHandler returning database rows.

    Note: ToolNode will wrap this in {"output": ...} automatically.
    """
    return {
        "rows": [
            {"id": 1, "name": "Agent A", "disabled": False},
            {"id": 2, "name": "Agent B", "disabled": False},
            {"id": 3, "name": "Agent C", "disabled": False},
        ],
        "count": 3,
        "query": "SELECT * FROM agents WHERE disabled = :val",
        "params": {"val": False},
    }


async def main():
    """Run test graph."""
    print("🧪 Testing DataHandler → LLM variable resolution\n")

    # Create graph
    graph = StateGraph()

    # Add mock DataHandler (using ToolNode to replicate the issue)
    data_handler = ToolNode(
        id="data_handler_0",
        tool_fn=mock_data_handler,
    )
    graph.add_node("data_handler_0", data_handler, node_type="tool")

    # Add a second tool that will print what it receives
    async def check_variables(input, context):
        """Tool to check if variables are resolved."""
        from mesh.utils.variables import VariableResolver

        resolver = VariableResolver(context)

        # Debug: Check what's in executed_data
        print(f"\n🔍 Debug: executed_data contents:")
        for item in context.executed_data:
            print(f"   - {item['node_id']}: {item['output']}")

        # Debug: Try direct lookup
        direct_lookup = context.get_node_output("data_handler_0")
        print(f"\n🔍 Debug: Direct lookup of data_handler_0:")
        print(f"   Result: {direct_lookup}")

        # Try to resolve the variable
        try:
            # Try WITH .output (what user tried)
            count_with_output = await resolver.resolve("{{data_handler_0.output.count}}")
            full_with_output = await resolver.resolve("{{data_handler_0.output}}")

            # Try WITHOUT .output (direct access)
            count_direct = await resolver.resolve("{{data_handler_0.count}}")
            full_direct = await resolver.resolve("{{data_handler_0}}")

            print(f"\n🔍 Variable Resolution Test:")
            print(f"   WITH .output:")
            print(f"     {{{{data_handler_0.output.count}}}} = '{count_with_output}'")
            print(f"     {{{{data_handler_0.output}}}} = '{full_with_output}'")
            print(f"   WITHOUT .output (direct):")
            print(f"     {{{{data_handler_0.count}}}} = '{count_direct}'")
            print(f"     {{{{data_handler_0}}}} = '{full_direct}'")

            return {
                "resolved_count_with_output": count_with_output,
                "resolved_full_with_output": full_with_output,
                "resolved_count_direct": count_direct,
                "resolved_full_direct": full_direct,
                "success": True,
            }
        except Exception as e:
            print(f"\n❌ Variable resolution failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "success": False,
            }

    checker = ToolNode(
        id="variable_checker",
        tool_fn=check_variables,
    )
    graph.add_node("variable_checker", checker, node_type="tool")

    # Connect nodes
    graph.add_edge("START", "data_handler_0")
    graph.add_edge("data_handler_0", "variable_checker")

    # Set entry point
    graph.set_entry_point("data_handler_0")

    # Compile graph
    compiled_graph = graph.compile()

    # Execute
    print("📊 Graph structure:")
    print(f"  START → data_handler_0 → variable_checker\n")

    from mesh.core.executor import Executor

    executor = Executor(compiled_graph, MemoryBackend())
    context = ExecutionContext(
        graph_id="test",
        session_id="test-session",
    )

    print("▶️  Executing graph...\n")

    async for event in executor.execute("how many rows?", context):
        # Show all node_start events to check for duplicates
        if event.type == "node_start":
            print(f"🟢 NODE_START: {event.node_id} - metadata: {getattr(event, 'metadata', {})}")

        if event.type == "node_complete":
            print(f"✅ {event.node_id} completed")
            if event.node_id == "data_handler_0":
                print(f"   Output: {event.output}\n")
            elif event.node_id == "variable_checker":
                print(f"   Result: {event.output}\n")

    print("\n📝 Executed data in context:")
    for item in context.executed_data:
        print(f"  - {item['node_id']}: {item['output']}")

    print("\n✨ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
