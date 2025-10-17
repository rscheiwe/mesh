"""
Event Translation Comparison Example

This example demonstrates the difference between:
1. Vel-translated events (default) - Standardized stream protocol events
2. Native provider events - Provider-specific event structure

By default, Mesh uses Vel's translation for consistent event handling across providers.
You can opt into native events with use_native_events=True.
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.utils import load_env

# Load environment variables
load_env()


async def vel_translated_example():
    """Example using Vel-translated events (default)."""
    print("=== Example 1: Vel-Translated Events (Default) ===\n")

    # This requires both Vel and OpenAI Agents SDK
    try:
        from agents import Agent
        from vel import get_translator
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e}")
        print("Install with: pip install mesh[all]")
        return

    # Create agent
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant."
    )

    # Build graph with default settings (use_native_events=False)
    graph = StateGraph()
    graph.add_node(
        "agent",
        agent,
        node_type="agent",
        # use_native_events=False is the default
        config={"model": "gpt-4"}
    )
    graph.add_edge("START", "agent")
    graph.set_entry_point("agent")

    # Execute
    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())
    context = ExecutionContext(
        graph_id="vel-translation",
        session_id="session-1",
        chat_history=[],
        variables={},
        state={},
    )

    print("Using Vel-translated events (standardized):\n")
    print("Agent: ", end="", flush=True)

    async for event in executor.execute("What is 2+2?", context):
        # With Vel translation, you get consistent event types
        if event.type == "token":
            print(event.content, end="", flush=True)
        elif event.type == "message_complete":
            if "finish_reason" in event.metadata:
                print(f"\n[Finished: {event.metadata['finish_reason']}]")

    print("\n")


async def native_events_example():
    """Example using native provider events."""
    print("\n=== Example 2: Native Provider Events ===\n")

    try:
        from agents import Agent
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e}")
        return

    # Create agent
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant."
    )

    # Build graph with native events enabled
    graph = StateGraph()
    graph.add_node(
        "agent",
        agent,
        node_type="agent",
        use_native_events=True,  # Use provider's native events
        config={"model": "gpt-4"}
    )
    graph.add_edge("START", "agent")
    graph.set_entry_point("agent")

    # Execute
    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())
    context = ExecutionContext(
        graph_id="native-events",
        session_id="session-2",
        chat_history=[],
        variables={},
        state={},
    )

    print("Using native OpenAI Agents SDK events (provider-specific):\n")
    print("Agent: ", end="", flush=True)

    async for event in executor.execute("What is 2+2?", context):
        # With native events, you get OpenAI Agents SDK specific event types
        if event.type == "token":
            print(event.content, end="", flush=True)
        elif event.type == "message_complete":
            print("\n[Message complete]")

    print("\n")


async def comparison_summary():
    """Print comparison summary."""
    print("\n=== Event Translation Comparison ===\n")

    print("Vel-Translated Events (Default):")
    print("  ✅ Consistent across all providers (OpenAI, Anthropic, Google)")
    print("  ✅ Follows Vel stream protocol (text-start, text-delta, etc.)")
    print("  ✅ Same code works with any provider")
    print("  ✅ Battle-tested in production")
    print("  ⚠️  Requires Vel package")
    print()

    print("Native Provider Events:")
    print("  ✅ Direct access to provider-specific features")
    print("  ✅ Works without Vel dependency")
    print("  ✅ Lower latency (no translation layer)")
    print("  ⚠️  Different event structure per provider")
    print("  ⚠️  Requires provider-specific code")
    print()

    print("When to use each:")
    print("  • Vel-translated (default): Multi-provider support, consistency")
    print("  • Native events: Single provider, performance-critical, provider-specific features")
    print()


async def main():
    """Run all examples."""
    await vel_translated_example()
    await native_events_example()
    await comparison_summary()


if __name__ == "__main__":
    asyncio.run(main())
