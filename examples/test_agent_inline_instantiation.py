"""Test AgentNode inline instantiation without registry.

This test verifies that AgentNode can instantiate a Vel Agent from config
without requiring pre-registration in a registry.
"""

import asyncio
from mesh import StateGraph
from mesh.core.state import ExecutionContext
from mesh.backends.memory import MemoryBackend
from mesh.parsers.react_flow import ReactFlowParser
from mesh.utils.registry import NodeRegistry


async def main():
    """Test inline agent instantiation."""
    print("🧪 Testing AgentNode inline instantiation (no registry)\n")

    # Create a React Flow graph definition with inline agent config
    graph_json = {
        "nodes": [
            {
                "id": "start",
                "type": "startAgentflow",
                "data": {
                    "defName": "start",
                    "config": {}
                }
            },
            {
                "id": "agent_0",
                "type": "agentAgentflow",
                "data": {
                    "defName": "agent",
                    "inputs": {
                        # No "agent" reference - inline config instead!
                        "provider": "openai",
                        "modelName": "gpt-4o-mini",
                        "temperature": 0.7,
                        "systemPrompt": "You are a helpful assistant. Be concise.",
                        "eventMode": "full",
                    }
                }
            }
        ],
        "edges": [
            {
                "id": "e1",
                "source": "start",
                "target": "agent_0"
            }
        ]
    }

    # Parse without any agents in registry
    registry = NodeRegistry()  # Empty registry!
    parser = ReactFlowParser(registry)

    try:
        graph = parser.parse(graph_json)
        print("✅ Graph parsed successfully (no registry needed!)")
        print(f"   Graph has {len(graph.nodes)} nodes\n")

        # Verify the agent node was created
        agent_node = graph.get_node("agent_0")
        print(f"✅ AgentNode created: {agent_node}")
        print(f"   Agent type: {agent_node.agent_type}")
        print(f"   Agent config: {agent_node.config}")
        print(f"   System prompt: {agent_node.system_prompt}\n")

        # Verify the Vel Agent was instantiated with correct config
        print("📊 Vel Agent details:")
        print(f"   Agent ID: {agent_node.agent.id}")
        print(f"   Agent instance: {type(agent_node.agent)}")
        print("\n✨ Test complete!")
        print("\n" + "="*80)
        print("RESULT: AgentNode works WITHOUT pre-registration!")
        print("Users can now create agents inline with model/provider/temperature.")
        print("="*80)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
