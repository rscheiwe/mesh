"""Test AgentNode with dynamic tools (programmatic API).

This example demonstrates using Mesh's programmatic API (StateGraph) with
Vel's new dynamic tools pattern (v0.3.0+).

Key pattern: Tools are defined and attached to the Agent BEFORE wrapping in AgentNode.
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend


async def main():
    """Test programmatic graph with dynamic tools."""
    print("🧪 Testing Mesh StateGraph + Vel Dynamic Tools\n")

    # Import Vel SDK
    try:
        from vel import Agent as VelAgent
        from vel.tools import ToolSpec
    except ImportError:
        print("❌ Vel SDK not available. Install with: pip install vel")
        return

    # Step 1: Define tool functions
    def get_weather(city: str) -> dict:
        """Get the current weather for a city."""
        # Mock implementation
        weather_data = {
            "San Francisco": {"temp": 65, "condition": "Foggy"},
            "New York": {"temp": 72, "condition": "Sunny"},
            "London": {"temp": 55, "condition": "Rainy"},
        }
        return weather_data.get(city, {"temp": 70, "condition": "Unknown"})

    def calculate_sum(a: int, b: int) -> dict:
        """Add two numbers together."""
        return {"result": a + b, "operation": "addition"}

    # Step 2: Wrap functions in ToolSpec (NEW dynamic tools API)
    weather_tool = ToolSpec.from_function(get_weather)
    calculator_tool = ToolSpec.from_function(calculate_sum)

    print("✅ Created ToolSpec instances:")
    print(f"   - {weather_tool.name}")
    print(f"   - {calculator_tool.name}\n")

    # Step 3: Create Vel Agent WITH tools
    # Tools are part of the agent configuration (not separate nodes!)
    agent = VelAgent(
        id="assistant",
        model={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
        },
        tools=[weather_tool, calculator_tool],  # ✅ Pass ToolSpec instances directly
    )

    print("✅ Created VelAgent with tools")
    print(f"   Agent ID: {agent.id}")
    print(f"   Instance tools: {list(agent._instance_tools.keys())}\n")

    # Step 4: Build Mesh graph
    # AgentNode wraps the pre-configured agent
    graph = StateGraph()
    graph.add_node(
        "agent",
        agent,  # Pre-configured agent with tools
        node_type="agent",
        system_prompt="You are a helpful assistant with access to weather and calculator tools. Use them when needed.",
    )
    graph.add_edge("START", "agent")
    graph.set_entry_point("agent")

    compiled_graph = graph.compile()
    print(f"✅ Graph compiled: {len(compiled_graph.nodes)} nodes\n")

    # Step 5: Execute
    backend = MemoryBackend()
    executor = Executor(compiled_graph, backend)

    context = ExecutionContext(
        graph_id="dynamic-tools-example",
        session_id="session-1",
        chat_history=[],
        variables={},
        state={},
    )

    # Test 1: Weather query
    print("="*80)
    print("Test 1: Tool that returns structured data")
    print("="*80)
    user_input = "What's the weather in San Francisco?"
    print(f"User: {user_input}\n")
    print("Agent: ", end="", flush=True)

    full_response = ""
    async for event in executor.execute(user_input, context):
        if event.type == "text-delta":
            delta = event.delta
            if delta:
                print(delta, end="", flush=True)
                full_response += delta
        elif event.type == "tool-input-available":
            tool_name = event.metadata.get("tool_name", "unknown")
            tool_input = event.metadata.get("input", {})
            print(f"\n\n[Tool Call: {tool_name}({tool_input})]", end="", flush=True)
        elif event.type == "tool-output-available":
            tool_output = event.output
            print(f"\n[Tool Result: {tool_output}]\n\nAgent: ", end="", flush=True)

    print("\n")

    # Test 2: Calculator query
    print("\n" + "="*80)
    print("Test 2: Different tool, same agent")
    print("="*80)
    context.chat_history.append({"role": "user", "content": user_input})
    context.chat_history.append({"role": "assistant", "content": full_response})

    user_input2 = "What is 456 + 789?"
    print(f"User: {user_input2}\n")
    print("Agent: ", end="", flush=True)

    async for event in executor.execute(user_input2, context):
        if event.type == "text-delta":
            delta = event.delta
            if delta:
                print(delta, end="", flush=True)
        elif event.type == "tool-input-available":
            tool_name = event.metadata.get("tool_name", "unknown")
            tool_input = event.metadata.get("input", {})
            print(f"\n\n[Tool Call: {tool_name}({tool_input})]", end="", flush=True)
        elif event.type == "tool-output-available":
            tool_output = event.output
            print(f"\n[Tool Result: {tool_output}]\n\nAgent: ", end="", flush=True)

    print("\n")

    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print("✅ Tools defined as Python functions")
    print("✅ Wrapped in ToolSpec using ToolSpec.from_function()")
    print("✅ Passed directly to VelAgent constructor (no global registry!)")
    print("✅ AgentNode wraps pre-configured agent")
    print("✅ Tools execute successfully via Vel's runtime")
    print("\nThis is the correct pattern for Vel v0.3.0+ dynamic tools!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
