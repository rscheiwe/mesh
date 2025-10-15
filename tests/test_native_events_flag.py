"""
Quick test to verify use_native_events flag functionality
"""
from mesh import StateGraph
from mesh.nodes.agent import AgentNode


class MockAgent:
    """Mock agent for testing - mimics OpenAI Agents SDK structure"""
    name = "Test Agent"
    instructions = "Test"

    # Override __module__ to simulate it's from 'agents' package
    __module__ = "agents.test"


def test_vel_translation_default():
    """Test that Vel translation is used by default"""
    print("Test 1: Vel translation (default)")

    agent = MockAgent()
    agent_node = AgentNode(
        id="test",
        agent=agent,
        use_native_events=False  # Default
    )

    # Check if Vel SDK translator was set up
    if agent_node.vel_sdk_translator is not None:
        print("  ✅ Vel SDK translator initialized")
    else:
        print("  ⚠️  Vel SDK translator not available (expected if Vel not installed)")

    print(f"  use_native_events: {agent_node.use_native_events}")
    print(f"  agent_type: {agent_node.agent_type}")
    print()


def test_native_events_enabled():
    """Test that native events can be enabled"""
    print("Test 2: Native events enabled")

    agent = MockAgent()
    agent_node = AgentNode(
        id="test",
        agent=agent,
        use_native_events=True
    )

    assert agent_node.use_native_events == True, "use_native_events should be True"
    assert agent_node.vel_sdk_translator is None, "Vel SDK translator should not be initialized"

    print("  ✅ Native events enabled")
    print(f"  use_native_events: {agent_node.use_native_events}")
    print(f"  vel_sdk_translator: {agent_node.vel_sdk_translator}")
    print()


def test_state_graph_integration():
    """Test StateGraph passes the flag correctly"""
    print("Test 3: StateGraph integration")

    agent = MockAgent()
    graph = StateGraph()

    # Add with default (Vel translation)
    graph.add_node("agent1", agent, node_type="agent")
    node1 = graph._nodes["agent1"]
    print(f"  agent1 use_native_events: {node1.use_native_events} (expected: False)")

    # Add with native events
    graph.add_node("agent2", agent, node_type="agent", use_native_events=True)
    node2 = graph._nodes["agent2"]
    print(f"  agent2 use_native_events: {node2.use_native_events} (expected: True)")

    assert node1.use_native_events == False, "agent1 should use Vel translation"
    assert node2.use_native_events == True, "agent2 should use native events"

    print("  ✅ StateGraph correctly passes use_native_events flag")
    print()


def test_vel_import():
    """Test Vel SDK translator import"""
    print("Test 4: Vel SDK translator API")

    try:
        from vel import get_openai_agents_translator

        translator = get_openai_agents_translator()
        print(f"  ✅ Vel SDK translator API available")
        print(f"  OpenAI Agents SDK translator: {type(translator).__name__}")

        # Test that translate method exists
        assert hasattr(translator, 'translate'), "Translator should have translate method"
        assert hasattr(translator, 'reset'), "Translator should have reset method"
        print(f"  ✅ Translator has required methods: translate, reset")
    except ImportError as e:
        print(f"  ⚠️  Vel not available: {e}")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing use_native_events Flag Implementation")
    print("=" * 60)
    print()

    test_vel_translation_default()
    test_native_events_enabled()
    test_state_graph_integration()
    test_vel_import()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
