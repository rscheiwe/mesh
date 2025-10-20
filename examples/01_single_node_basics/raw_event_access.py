"""Raw Event Access Example

This example demonstrates how to access the raw/vanilla event alongside
the standardized Mesh ExecutionEvent.

The raw_event field contains:
- For Vel agents: The Vel event dict (already standardized)
- For AgentNode with use_native_events=False: The Vel-translated event dict
- For AgentNode with use_native_events=True: The native provider event object
- For LLMNode: The Vel-format event dict (standardized)

Key principle: raw_event matches the event format being used, not always
the original provider event.

This is useful for:
- Debugging event translation
- Accessing additional fields in the standardized format
- Understanding the event flow
- Consistent event structure regardless of source
"""

import asyncio
import os
from dotenv import load_dotenv
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.nodes import LLMNode
from mesh.core.events import EventType

# Load environment variables
load_dotenv()


async def main():
    print("\n" + "=" * 70)
    print("RAW EVENT ACCESS DEMO")
    print("=" * 70)

    # Build simple graph with LLM node
    graph = StateGraph()
    graph.add_node("llm", None, node_type="llm", model="gpt-4", temperature=0.7)
    graph.add_edge("START", "llm")
    graph.set_entry_point("llm")

    # Compile and execute
    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())
    context = ExecutionContext(
        graph_id="raw-event-demo",
        session_id="demo-session",
        chat_history=[],
        variables={},
        state={},
    )

    print("\nExecuting: 'What is 2+2?'\n")
    print("Showing both standardized Mesh event AND raw provider event:\n")

    token_count = 0
    async for event in executor.execute("What is 2+2?", context):
        if event.type == EventType.TOKEN:
            token_count += 1

            # Access standardized Mesh event fields
            print(f"Token {token_count}:")
            print(f"  Content: '{event.content}'")
            print(f"  Node ID: {event.node_id}")
            print(f"  Provider: {event.metadata.get('provider')}")

            # Access raw event (Vel-format for LLMNode)
            if event.raw_event:
                if isinstance(event.raw_event, dict):
                    # Vel-format event dict
                    print(f"  Raw Event (Vel format):")
                    print(f"    Type: {event.raw_event.get('type')}")
                    print(f"    Delta: '{event.raw_event.get('delta')}'")
                    print(f"    ID: {event.raw_event.get('id')}")
                else:
                    # Native provider event (only when use_native_events=True)
                    print(f"  Raw Event Type: {type(event.raw_event).__name__}")

            print()

        elif event.type == EventType.EXECUTION_COMPLETE:
            print(f"\n{'=' * 70}")
            print(f"EXECUTION COMPLETE")
            print(f"{'=' * 70}")
            print(f"Total tokens: {token_count}")
            print(f"Final output: {event.output}")

    print("\n" + "=" * 70)
    print("KEY BENEFITS")
    print("=" * 70)
    print("✅ Standardized Mesh events for consistency across providers")
    print("✅ Raw events preserve the format being used (Vel or native)")
    print("✅ use_native_events=False → raw_event = Vel-format dict")
    print("✅ use_native_events=True → raw_event = Native provider object")
    print("✅ No information loss - both formats accessible")
    print("✅ Useful for debugging event translation and custom processing")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=sk-...")
        exit(1)

    asyncio.run(main())
