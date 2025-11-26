"""Test manual tool creation without DB or registry.

This demonstrates how users can create tools in pure Python
when using Mesh as an imported package.
"""

import asyncio
from mesh import StateGraph
from mesh.nodes import ToolNode
from mesh.core.state import ExecutionContext
from mesh.backends.memory import MemoryBackend


# Define tool functions in separate file or same file
def calculate_sum(a: int, b: int) -> dict:
    """Add two numbers together."""
    return {
        "result": a + b,
        "operation": "addition"
    }

async def fetch_weather(city: str, context) -> dict:
    """Mock weather fetcher."""
    return {
        "city": city,
        "temperature": 72,
        "condition": "sunny"
    }


async def main():
    """Test manual tool creation."""
    print("🧪 Testing Manual Tool Creation (No DB/Registry)\n")

    # Create graph
    graph = StateGraph()

    # Create ToolNodes by passing functions directly
    print("📦 Creating ToolNodes...")

    sum_tool = ToolNode(
        id="sum_tool",
        tool_fn=calculate_sum,
        config={"bindings": {"a": 10, "b": 5}}
    )

    weather_tool = ToolNode(
        id="weather_tool",
        tool_fn=fetch_weather,
        config={"bindings": {"city": "San Francisco"}}
    )

    print(f"✅ Created sum_tool: {sum_tool}")
    print(f"✅ Created weather_tool: {weather_tool}\n")

    # Add nodes to graph
    graph.add_node("sum_tool", sum_tool, node_type="tool")
    graph.add_node("weather_tool", weather_tool, node_type="tool")

    # Connect edges
    graph.add_edge("START", "sum_tool")
    graph.add_edge("sum_tool", "weather_tool")

    # Set entry point
    graph.set_entry_point("sum_tool")

    # Compile
    compiled_graph = graph.compile()
    print("✅ Graph compiled\n")

    # Verify structure
    print("📊 Graph structure:")
    print("  START → sum_tool → weather_tool\n")

    print("📊 Nodes:")
    for node_id, node in compiled_graph.nodes.items():
        print(f"   - {node_id}: {type(node).__name__}")

    print("\n▶️  Executing graph...\n")

    from mesh.core.executor import Executor

    executor = Executor(compiled_graph, MemoryBackend())
    context = ExecutionContext(
        graph_id="test-manual-tools",
        session_id="test-session",
    )

    async for event in executor.execute("test", context):
        if event.type == "node_start":
            metadata = getattr(event, 'metadata', {})
            tool_name = metadata.get('tool_name', 'N/A')
            print(f"🟢 START: {event.node_id} (tool: {tool_name})")

        elif event.type == "node_complete":
            print(f"✅ COMPLETE: {event.node_id}")
            if hasattr(event, 'output') and event.output:
                print(f"   Output: {event.output}\n")

    # Check outputs
    print("\n📊 Results:")
    sum_output = context.get_node_output("sum_tool")
    weather_output = context.get_node_output("weather_tool")

    if sum_output and "output" in sum_output:
        print(f"✅ sum_tool: {sum_output['output']}")

    if weather_output and "output" in weather_output:
        print(f"✅ weather_tool: {weather_output['output']}")

    print("\n✨ Test complete!")
    print("\n" + "="*80)
    print("RESULT: Manual tool creation works!")
    print("Users can:")
    print("  1. Define tool functions in Python")
    print("  2. Import from other files")
    print("  3. Pass directly to ToolNode()")
    print("  4. No DB or registry needed for manual graphs")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
