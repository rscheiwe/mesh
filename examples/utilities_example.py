"""Example demonstrating the message handler and tool schematization utilities."""

import asyncio
import json
from typing import List, Optional

from mesh import Edge, Graph
from mesh.compilation import GraphExecutor, StaticCompiler
from mesh.nodes import AgentNode
from mesh.nodes.agent import AgentConfig
from mesh.nodes.llm import LLMProvider, Message
from mesh.utils import (
    MessageFormat,
    MessageHandler,
    convert_to_openai_format,
    create_tool_from_function,
    function_to_tool_schema,
)


# Example functions to convert to tools
def get_weather(location: str, units: str = "celsius") -> str:
    """Get the current weather for a location.

    Args:
        location: City or location name
        units: Temperature units (celsius or fahrenheit)

    Returns:
        Weather information string
    """
    # Mock implementation
    return f"The weather in {location} is 22°{units[0].upper()} and sunny"


def calculate_tip(
    bill_amount: float, tip_percentage: float = 15.0, split_ways: Optional[int] = None
) -> dict:
    """Calculate tip amount and total.

    Args:
        bill_amount: The bill amount before tip
        tip_percentage: Tip percentage (default 15%)
        split_ways: Optional number of people to split the bill

    Returns:
        Dict with tip amount, total, and per person amounts
    """
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip

    result = {"bill": bill_amount, "tip": tip, "total": total}

    if split_ways and split_ways > 1:
        result["per_person"] = total / split_ways

    return result


async def message_handler_example():
    """Demonstrate message format conversion."""
    print("=== Message Handler Example ===\n")

    handler = MessageHandler()

    # Example messages in different formats

    # OpenAI format messages
    openai_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {
            "role": "assistant",
            "content": "Python is a high-level programming language.",
        },
        {"role": "user", "content": "Tell me more"},
        {"role": "function", "name": "get_info", "content": "Additional info here"},
    ]

    print("Original OpenAI messages:")
    print(json.dumps(openai_messages, indent=2))

    # Convert to mesh format
    mesh_messages = handler.convert(
        openai_messages, to_format=MessageFormat.MESH, from_format=MessageFormat.OPENAI
    )

    print("\n\nConverted to Mesh format:")
    for msg in mesh_messages:
        print(f"  {msg.role}: {msg.content[:50]}...")
        if msg.metadata:
            print(f"    metadata: {msg.metadata}")

    # Convert to Anthropic format
    anthropic_result = handler.convert(
        mesh_messages, to_format=MessageFormat.ANTHROPIC, from_format=MessageFormat.MESH
    )

    print("\n\nConverted to Anthropic format:")
    print(json.dumps(anthropic_result, indent=2))

    # Auto-detect format
    print("\n\nAuto-detection test:")
    detected_format = handler._detect_format(openai_messages)
    print(f"Detected format: {detected_format.value}")


async def tool_schematization_example():
    """Demonstrate tool schema generation."""
    print("\n\n=== Tool Schematization Example ===\n")

    # Generate schema for weather function
    weather_schema = function_to_tool_schema(get_weather)
    print("Weather function schema:")
    print(json.dumps(weather_schema, indent=2))

    # Generate schema for calculate_tip function
    tip_schema = function_to_tool_schema(
        calculate_tip, description="Calculate restaurant tip and split the bill"
    )
    print("\n\nTip calculator schema:")
    print(json.dumps(tip_schema, indent=2))

    # Create FunctionTool instances
    weather_tool = create_tool_from_function(get_weather)
    tip_tool = create_tool_from_function(calculate_tip)

    print("\n\nCreated FunctionTools:")
    print(f"1. {weather_tool.name}: {weather_tool.description}")
    print(f"2. {tip_tool.name}: {tip_tool.description}")


async def integrated_example():
    """Example using both utilities in a real scenario."""
    print("\n\n=== Integrated Example with Agent ===\n")

    # Create graph with agent that has tools
    graph = Graph()

    # Create tools from functions
    tools = [
        create_tool_from_function(get_weather),
        create_tool_from_function(calculate_tip),
    ]

    agent = AgentNode(
        config=AgentConfig(
            name="Assistant",
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key="your-api-key",
            tools=tools,
            system_prompt="You are a helpful assistant with weather and tip calculation tools.",
        )
    )

    graph.add_node(agent)
    # Agent node has no incoming edges, so it becomes a starting node automatically

    # Compile
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)
    executor = GraphExecutor()

    # Prepare messages with message handler
    handler = MessageHandler()

    # Simulate conversation from frontend
    frontend_messages = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": "I'll check the weather in Paris for you."},
        {
            "role": "function",
            "name": "get_weather",
            "content": "The weather in Paris is 22°C and sunny",
        },
        {
            "role": "assistant",
            "content": "The weather in Paris is currently 22°C and sunny.",
        },
        {
            "role": "user",
            "content": "Great! I had lunch for €45. How much should I tip?",
        },
    ]

    # Convert to mesh format for processing
    mesh_messages = handler.convert(
        frontend_messages,
        to_format=MessageFormat.MESH,
        from_format=MessageFormat.OPENAI,
    )

    print("Converted messages for agent:")
    for msg in mesh_messages:
        print(f"  {msg.role}: {msg.content[:60]}...")

    # The agent would process these messages
    # (Not executing due to API key requirement)
    print("\n[Agent would process messages with available tools]")

    # Show tool schemas that would be sent to the LLM
    print("\nTool schemas provided to LLM:")
    for tool in tools:
        schema = tool.get_schema()
        print(f"\n{schema['name']}:")
        print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    # Run examples
    asyncio.run(message_handler_example())
    asyncio.run(tool_schematization_example())
    asyncio.run(integrated_example())
