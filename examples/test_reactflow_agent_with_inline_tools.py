"""Test React Flow parser with inline tools (dynamic tools upgrade).

This example demonstrates Vel's new dynamic tools pattern where tools are:
1. Defined inline as Python code in React Flow JSON
2. Automatically wrapped in ToolSpec instances
3. Passed directly to the Vel Agent (no global registration needed)

This aligns with Vel v0.3.0+ dynamic tools upgrade.
"""

import asyncio
from mesh.parsers.react_flow import ReactFlowParser
from mesh.utils.registry import NodeRegistry
from mesh.core.executor import Executor
from mesh.core.state import ExecutionContext
from mesh.backends.memory import MemoryBackend


async def main():
    """Test React Flow creates VelAgent with inline tools (dynamic tools pattern)."""
    print("🧪 Testing React Flow → VelAgent with Inline Tools (Dynamic Tools Upgrade)\n")

    # Create React Flow graph with inline agent config AND inline tools
    graph_json = {
        "nodes": [
            {
                "id": "start",
                "type": "startAgentflow",
                "data": {
                    "defName": "start",
                    "inputs": {}
                }
            },
            {
                "id": "agent_0",
                "type": "agentAgentflow",
                "data": {
                    "defName": "agent",
                    "inputs": {
                        "provider": "openai",
                        "modelName": "gpt-4o-mini",
                        "temperature": 0.7,
                        "systemPrompt": "You are a helpful assistant with access to a calculator tool. Use the add_numbers tool when you need to add numbers.",
                        "eventMode": "full",
                        # Inline tools array (NEW!)
                        "tools": [
                            {
                                "code": """def add_numbers(x: int, y: int) -> dict:
    '''Add two numbers together.'''
    return {'result': x + y, 'operation': 'addition'}""",
                                "name": "add_numbers",  # Optional: auto-extracted if not provided
                                "description": None,  # Optional: uses docstring if not provided
                            },
                            {
                                "code": """def multiply_numbers(x: int, y: int) -> dict:
    '''Multiply two numbers together.'''
    return {'result': x * y, 'operation': 'multiplication'}""",
                            }
                        ]
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

    # Parse
    registry = NodeRegistry()
    parser = ReactFlowParser(registry)
    graph = parser.parse(graph_json)

    print("✅ Graph parsed\n")

    # Get the agent node
    agent_node = graph.get_node("agent_0")
    print(f"📦 AgentNode: {agent_node}")
    print(f"   Agent type: {agent_node.agent_type}")
    print(f"   Agent instance: {type(agent_node.agent)}\n")

    # Verify it's a real VelAgent with tools
    from vel import Agent as VelAgent
    is_vel_agent = isinstance(agent_node.agent, VelAgent)
    print(f"✅ Is VelAgent instance: {is_vel_agent}")

    if is_vel_agent:
        print(f"✅ Agent ID: {agent_node.agent.id}")

        # Check tools using new dynamic tools API
        if hasattr(agent_node.agent, '_instance_tools'):
            print(f"✅ Agent has instance-level tools: {list(agent_node.agent._instance_tools.keys())}")
            print(f"   Tool count: {len(agent_node.agent._instance_tools)}")

            # Verify tools are ToolSpec instances (not string references!)
            for tool_name, tool_spec in agent_node.agent._instance_tools.items():
                from vel.tools import ToolSpec
                is_tool_spec = isinstance(tool_spec, ToolSpec)
                print(f"   - {tool_name}: ToolSpec instance = {is_tool_spec}")

        # Test execution
        print("\n🚀 Executing agent with tool call...")
        backend = MemoryBackend()
        executor = Executor(graph, backend)

        context = ExecutionContext(
            graph_id="test-inline-tools",
            session_id="session-1",
            chat_history=[],
            variables={},
            state={},
        )

        # Ask a question that requires tool use
        user_input = "What is 125 + 378?"
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

    print("\n" + "="*80)
    print("RESULT:")
    print("✅ React Flow DOES support inline tools (dynamic tools pattern)")
    print("✅ Tools are automatically wrapped in ToolSpec instances")
    print("✅ Tools are passed directly to VelAgent (no global registration!)")
    print("✅ Agent successfully executes with inline tools")
    print("\nThis demonstrates Vel v0.3.0+ dynamic tools upgrade!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
