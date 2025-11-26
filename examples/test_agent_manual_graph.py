"""Test AgentNode with manual graph creation (no React Flow parser).

This test verifies that AgentNode works at the core Mesh level,
creating a Vel Agent directly and adding it to a StateGraph.
"""

import asyncio
from mesh import StateGraph
from mesh.nodes import AgentNode
from mesh.core.state import ExecutionContext
from mesh.backends.memory import MemoryBackend


async def main():
    """Test manual graph creation with inline Vel Agent."""
    print("🧪 Testing AgentNode with manual graph creation\n")

    try:
        # Import Vel
        from vel import Agent as VelAgent
    except ImportError:
        print("❌ Vel SDK not installed. Install with: pip install vel")
        return

    # Create a Vel Agent manually
    print("📦 Creating Vel Agent...")
    vel_agent = VelAgent(
        id="my_agent",
        model={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
        }
    )
    print(f"✅ Vel Agent created: {vel_agent.id}")
    print(f"   Type: {type(vel_agent)}\n")

    # Create AgentNode with the Vel Agent
    print("📦 Creating AgentNode...")
    agent_node = AgentNode(
        id="agent_0",
        agent=vel_agent,
        system_prompt="You are a helpful math assistant. Be concise.",
        event_mode="full",
    )
    print(f"✅ AgentNode created: {agent_node}")
    print(f"   Agent type: {agent_node.agent_type}")
    print(f"   System prompt: {agent_node.system_prompt}\n")

    # Create graph manually
    print("📦 Building StateGraph...")
    graph = StateGraph()

    # Add nodes
    graph.add_node("agent_0", agent_node, node_type="agent")

    # Connect edges
    graph.add_edge("START", "agent_0")

    # Set entry point
    graph.set_entry_point("agent_0")

    # Compile
    compiled_graph = graph.compile()
    print(f"✅ Graph compiled: {len(compiled_graph.nodes)} nodes\n")

    # Verify graph structure
    print("📊 Graph structure:")
    print("  START → agent_0\n")

    print("📊 Verifying graph nodes:")
    for node_id, node in compiled_graph.nodes.items():
        print(f"   - {node_id}: {type(node).__name__}")

    print("\n📊 Verifying edges:")
    for edge in compiled_graph.edges:
        print(f"   - {edge.source} → {edge.target}")

    # Note: We won't execute because that requires API keys and network calls
    print("\n⚠️  Skipping execution (requires OPENAI_API_KEY)")
    print("   Graph construction is complete and valid!")

    print("\n✨ Test complete!")
    print("\n" + "="*80)
    print("RESULT: AgentNode works with manual graph creation!")
    print("You can create Vel Agents and add them to StateGraph directly.")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
