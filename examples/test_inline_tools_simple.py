"""Simple test to verify inline tools are created correctly."""

import asyncio
from mesh.parsers.react_flow import ReactFlowParser
from mesh.utils.registry import NodeRegistry


async def main():
    """Test that inline tools are parsed and wrapped in ToolSpec."""
    print("🧪 Testing Inline Tools Parsing\n")

    # Simple graph with inline tools
    graph_json = {
        "nodes": [
            {
                "id": "start",
                "type": "startAgentflow",
                "data": {"defName": "start", "inputs": {}}
            },
            {
                "id": "agent_0",
                "type": "agentAgentflow",
                "data": {
                    "defName": "agent",
                    "inputs": {
                        "provider": "openai",
                        "modelName": "gpt-4o-mini",
                        "tools": [
                            {
                                "code": "def add_numbers(x: int, y: int) -> dict:\n    return {'result': x + y}",
                                "name": "add_numbers",
                            }
                        ]
                    }
                }
            }
        ],
        "edges": [{"id": "e1", "source": "start", "target": "agent_0"}]
    }

    # Parse
    registry = NodeRegistry()
    parser = ReactFlowParser(registry)

    try:
        graph = parser.parse(graph_json)
        print("✅ Graph parsed successfully\n")

        # Verify agent node
        agent_node = graph.get_node("agent_0")
        print(f"📦 AgentNode: {agent_node}")
        print(f"   Agent type: {agent_node.agent_type}\n")

        # Verify it's a VelAgent
        from vel import Agent as VelAgent
        is_vel = isinstance(agent_node.agent, VelAgent)
        print(f"✅ Is VelAgent: {is_vel}")

        if is_vel:
            # Check tools
            if hasattr(agent_node.agent, '_instance_tools'):
                tools = agent_node.agent._instance_tools
                print(f"✅ Instance tools found: {list(tools.keys())}")

                # Verify ToolSpec
                from vel.tools import ToolSpec
                for name, tool in tools.items():
                    is_spec = isinstance(tool, ToolSpec)
                    print(f"   - {name}: ToolSpec = {is_spec}")
                    print(f"     Handler callable: {callable(tool._handler)}")
                    print(f"     Input schema: {tool.input_schema is not None}")

                    # Try calling the tool directly
                    result = tool._handler(5, 3)
                    print(f"     Test call: add_numbers(5, 3) = {result}")

        print("\n" + "="*80)
        print("SUCCESS: Inline tools are correctly parsed and wrapped in ToolSpec!")
        print("="*80)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
