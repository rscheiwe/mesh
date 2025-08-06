"""Example demonstrating agent nodes with tools."""

import asyncio
from typing import Any, Dict, Optional

from mesh import Edge, Graph
from mesh.compilation import GraphExecutor, StaticCompiler
from mesh.nodes import AgentNode
from mesh.nodes.agent import AgentConfig, Tool, ToolResult
from mesh.nodes.llm import LLMProvider
from mesh.state import GraphState


# Define some example tools
class CalculatorTool(Tool):
    """A simple calculator tool."""

    def __init__(self):
        super().__init__(
            name="calculator", description="Perform basic mathematical calculations"
        )

    async def execute(self, expression: str) -> ToolResult:
        """Evaluate a mathematical expression."""
        try:
            # In production, use a safer evaluation method
            result = eval(expression, {"__builtins__": {}}, {})
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        }


class WeatherTool(Tool):
    """Mock weather tool."""

    def __init__(self):
        super().__init__(
            name="weather", description="Get current weather for a location"
        )

    async def execute(self, location: str) -> ToolResult:
        """Get weather for a location."""
        # Mock implementation
        weather_data = {
            "location": location,
            "temperature": 72,
            "conditions": "Partly cloudy",
            "humidity": 65,
        }
        return ToolResult(success=True, output=weather_data)

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City or location name"}
            },
            "required": ["location"],
        }


async def main():
    """Run an agent graph example."""

    # Create graph
    graph = Graph()

    # Create agent with tools
    agent = AgentNode(
        config=AgentConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            tools=[CalculatorTool(), WeatherTool()],
            max_iterations=3,
        )
    )

    # Add nodes to graph
    graph.add_node(agent)
    # Agent node has no incoming edges, so it becomes a starting node automatically

    # Compile the graph
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)

    print(f"Compiled graph: {compiled.metadata}")

    # Execute with different queries
    executor = GraphExecutor()

    # Test 1: Math question
    print("\n--- Test 1: Math calculation ---")
    result1 = await executor.execute(
        compiled, initial_input={"prompt": "What is 15 * 7 + 23?"}
    )
    print(f"Success: {result1.success}")
    print(f"Agent response: {result1.outputs[agent.id].data}")

    # Test 2: Weather query
    print("\n--- Test 2: Weather query ---")
    result2 = await executor.execute(
        compiled, initial_input={"prompt": "What's the weather like in San Francisco?"}
    )
    print(f"Success: {result2.success}")
    print(f"Agent response: {result2.outputs[agent.id].data}")

    # Test 3: Complex query requiring multiple tools
    print("\n--- Test 3: Complex query ---")
    result3 = await executor.execute(
        compiled,
        initial_input={
            "prompt": "If the temperature in New York is 68°F, what is that in Celsius? Use the formula (F - 32) * 5/9"
        },
    )
    print(f"Success: {result3.success}")
    print(f"Agent response: {result3.outputs[agent.id].data}")


if __name__ == "__main__":
    asyncio.run(main())
