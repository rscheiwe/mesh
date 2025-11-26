"""Test ToolNode output wrapping and variable resolution.

This test verifies that generic ToolNodes properly wrap their outputs
in {"output": ...} so they can be accessed via {{tool_id.output}}.
"""

import asyncio
from mesh import StateGraph
from mesh.nodes import ToolNode
from mesh.core.state import ExecutionContext
from mesh.backends.memory import MemoryBackend


# Test 1: Tool that returns a simple string
async def simple_string_tool(input, context):
    """Returns a simple string."""
    return "Hello, World!"


# Test 2: Tool that returns a dict
async def dict_tool(input, context):
    """Returns a dictionary."""
    return {
        "message": "Success",
        "count": 42,
        "items": ["a", "b", "c"]
    }


# Test 3: Tool that returns a number
def sync_number_tool(input, context):
    """Returns a number (sync function)."""
    return 12345


async def main():
    """Run test graph."""
    print("🧪 Testing ToolNode output wrapping\n")

    # Create graph
    graph = StateGraph()

    # Add string tool
    string_tool = ToolNode(
        id="string_tool",
        tool_fn=simple_string_tool,
    )
    graph.add_node("string_tool", string_tool, node_type="tool")

    # Add dict tool
    dict_tool_node = ToolNode(
        id="dict_tool",
        tool_fn=dict_tool,
    )
    graph.add_node("dict_tool", dict_tool_node, node_type="tool")

    # Add sync number tool
    number_tool = ToolNode(
        id="number_tool",
        tool_fn=sync_number_tool,
    )
    graph.add_node("number_tool", number_tool, node_type="tool")

    # Add checker tool that resolves variables from all previous tools
    async def check_all_outputs(input, context):
        """Check if all tool outputs can be resolved."""
        from mesh.utils.variables import VariableResolver
        resolver = VariableResolver(context)

        results = {}

        # Test string tool
        try:
            string_output = await resolver.resolve("{{string_tool.output}}")
            results["string_tool.output"] = string_output
        except Exception as e:
            results["string_tool.output"] = f"ERROR: {e}"

        # Test dict tool
        try:
            dict_output = await resolver.resolve("{{dict_tool.output}}")
            dict_message = await resolver.resolve("{{dict_tool.output.message}}")
            dict_count = await resolver.resolve("{{dict_tool.output.count}}")
            results["dict_tool.output"] = dict_output
            results["dict_tool.output.message"] = dict_message
            results["dict_tool.output.count"] = dict_count
        except Exception as e:
            results["dict_tool"] = f"ERROR: {e}"

        # Test number tool
        try:
            number_output = await resolver.resolve("{{number_tool.output}}")
            results["number_tool.output"] = number_output
        except Exception as e:
            results["number_tool.output"] = f"ERROR: {e}"

        return results

    checker = ToolNode(
        id="checker",
        tool_fn=check_all_outputs,
    )
    graph.add_node("checker", checker, node_type="tool")

    # Connect nodes in sequence
    graph.add_edge("START", "string_tool")
    graph.add_edge("string_tool", "dict_tool")
    graph.add_edge("dict_tool", "number_tool")
    graph.add_edge("number_tool", "checker")

    # Set entry point
    graph.set_entry_point("string_tool")

    # Compile graph
    compiled_graph = graph.compile()

    print("📊 Graph structure:")
    print("  START → string_tool → dict_tool → number_tool → checker\n")

    from mesh.core.executor import Executor

    executor = Executor(compiled_graph, MemoryBackend())
    context = ExecutionContext(
        graph_id="test-toolnode",
        session_id="test-session",
    )

    print("▶️  Executing graph...\n")

    async for event in executor.execute("test input", context):
        if event.type == "node_start":
            metadata = getattr(event, 'metadata', {})
            tool_name = metadata.get('tool_name', 'N/A')
            print(f"🟢 START: {event.node_id} (tool: {tool_name})")

        if event.type == "node_complete":
            print(f"✅ COMPLETE: {event.node_id}")
            if event.node_id != "START":
                print(f"   Output: {event.output}\n")

    print("\n" + "=" * 80)
    print("📝 VARIABLE RESOLUTION RESULTS:")
    print("=" * 80)

    checker_output = context.get_node_output("checker")
    if checker_output and "output" in checker_output:
        for key, value in checker_output["output"].items():
            status = "✅" if "ERROR" not in str(value) else "❌"
            print(f"{status} {{{{{key}}}}} = {value}")
    else:
        print("❌ Checker output not found")

    print("\n" + "=" * 80)
    print("📊 RAW OUTPUTS IN CONTEXT:")
    print("=" * 80)
    for item in context.executed_data:
        if item['node_id'] != 'START':
            print(f"\n{item['node_id']}:")
            print(f"  {item['output']}")

    print("\n✨ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
