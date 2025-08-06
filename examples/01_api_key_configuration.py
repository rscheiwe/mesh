"""Examples of different ways to configure API keys in mesh."""

import asyncio
import os

from mesh import Edge, Graph
from mesh.compilation import EventHandler, EventCollector, GraphExecutor, StaticCompiler
from mesh.nodes import LLMNode
from mesh.nodes.llm import LLMConfig, LLMProvider
from mesh.utils import print_event

async def main():
    """Example 2: Explicit API key (for production/multi-tenant scenarios)."""
    print("\n=== Example 2: Explicit API Key ===")

    graph = Graph()

    llm = LLMNode(
        config=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )

    graph.add_node(llm)
    # LLM node has no incoming edges, so it becomes a starting node automatically

    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)

    # Create event handler with the utility print function
    event_handler = EventHandler()
    event_handler.add_listener(print_event)
    
    # Execute with event handler
    executor = GraphExecutor(event_handler=event_handler)

    result = await executor.execute(compiled, initial_input={"prompt": "Say hello!"})
    # print(result)  # Uncomment to see full ExecutionResult object
    print(f"\nSuccess: {result.success}")
    print(f"Response: {result.outputs[llm.id].data}")


if __name__ == "__main__":
    asyncio.run(main())
