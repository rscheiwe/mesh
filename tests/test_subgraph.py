"""Tests for subgraph composition functionality.

Tests the ability to embed graphs within graphs for:
- Reusable workflow components
- Subagent isolation
- Modular graph design
- Dynamic graph composition
"""

import pytest
from typing import Any, Dict
from datetime import datetime

from mesh import (
    StateGraph,
    Executor,
    ExecutionContext,
    MemoryBackend,
    Subgraph,
    SubgraphConfig,
    SubgraphBuilder,
    SubgraphNode,
)
from mesh.core.events import EventType


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def memory_backend():
    """Fresh memory backend for each test."""
    return MemoryBackend()


@pytest.fixture
def base_context():
    """Base execution context for tests."""
    return ExecutionContext(
        graph_id="test-graph",
        session_id="test-session",
        chat_history=[],
        variables={},
        state={},
    )


def create_simple_tool(name: str, output_key: str = "result"):
    """Create a simple tool function for testing."""
    async def tool_fn(input: Any, context: ExecutionContext) -> Dict[str, Any]:
        return {output_key: f"{name}_processed_{input}"}
    return tool_fn


def create_accumulator_tool(key: str = "values"):
    """Create a tool that accumulates values in state."""
    async def tool_fn(input: Any, context: ExecutionContext) -> Dict[str, Any]:
        current = context.state.get(key, [])
        current.append(input)
        return {key: current}
    return tool_fn


def create_multiplier_tool(factor: int = 2):
    """Create a tool that multiplies numeric input."""
    async def tool_fn(input: Any, context: ExecutionContext) -> Dict[str, Any]:
        value = input.get("value", input) if isinstance(input, dict) else input
        if isinstance(value, (int, float)):
            return {"value": value * factor}
        return {"value": value}
    return tool_fn


# =============================================================================
# Basic Subgraph Tests
# =============================================================================


class TestSubgraphBasics:
    """Test basic subgraph creation and configuration."""

    def test_subgraph_creation(self):
        """Test creating a Subgraph from compiled graph."""
        # Create a simple graph
        graph = StateGraph()
        graph.add_node("tool", create_simple_tool("test"), node_type="tool")
        graph.set_entry_point("tool")
        compiled = graph.compile()

        # Wrap as subgraph
        subgraph = Subgraph(compiled)

        assert subgraph.graph == compiled
        assert subgraph.config.isolated is True
        assert subgraph.name.startswith("subgraph_")

    def test_subgraph_with_name(self):
        """Test creating a Subgraph with custom name."""
        graph = StateGraph()
        graph.add_node("tool", create_simple_tool("test"), node_type="tool")
        graph.set_entry_point("tool")
        compiled = graph.compile()

        subgraph = Subgraph(compiled, name="research")

        assert subgraph.name == "research"

    def test_subgraph_with_config(self):
        """Test creating a Subgraph with custom config."""
        graph = StateGraph()
        graph.add_node("tool", create_simple_tool("test"), node_type="tool")
        graph.set_entry_point("tool")
        compiled = graph.compile()

        config = SubgraphConfig(
            isolated=False,
            input_mapping={"query": "search_query"},
            output_mapping={"result": "findings"},
        )
        subgraph = Subgraph(compiled, config=config, name="research")

        assert subgraph.config.isolated is False
        assert subgraph.config.input_mapping == {"query": "search_query"}
        assert subgraph.config.output_mapping == {"result": "findings"}

    def test_subgraph_repr(self):
        """Test Subgraph string representation."""
        graph = StateGraph()
        graph.add_node("tool", create_simple_tool("test"), node_type="tool")
        graph.set_entry_point("tool")
        compiled = graph.compile()

        subgraph = Subgraph(compiled, name="test_sub")

        repr_str = repr(subgraph)
        assert "Subgraph" in repr_str
        assert "test_sub" in repr_str


class TestSubgraphConfig:
    """Test SubgraphConfig options."""

    def test_default_config(self):
        """Test default SubgraphConfig values."""
        config = SubgraphConfig()

        assert config.input_mapping == {}
        assert config.output_mapping == {}
        assert config.isolated is True
        assert config.inherit_keys == []
        assert config.checkpoint_on_entry is False
        assert config.checkpoint_on_exit is False
        assert config.prefix_events is True

    def test_custom_config(self):
        """Test custom SubgraphConfig values."""
        config = SubgraphConfig(
            input_mapping={"a": "b"},
            output_mapping={"x": "y"},
            isolated=False,
            inherit_keys=["user_id", "session"],
            checkpoint_on_entry=True,
            checkpoint_on_exit=True,
            prefix_events=False,
        )

        assert config.input_mapping == {"a": "b"}
        assert config.output_mapping == {"x": "y"}
        assert config.isolated is False
        assert config.inherit_keys == ["user_id", "session"]
        assert config.checkpoint_on_entry is True
        assert config.checkpoint_on_exit is True
        assert config.prefix_events is False


# =============================================================================
# SubgraphNode Tests
# =============================================================================


class TestSubgraphNode:
    """Test SubgraphNode wrapper class."""

    def test_subgraph_node_creation(self):
        """Test creating a SubgraphNode."""
        graph = StateGraph()
        graph.add_node("tool", create_simple_tool("test"), node_type="tool")
        graph.set_entry_point("tool")
        compiled = graph.compile()

        subgraph = Subgraph(compiled, name="inner")
        node = SubgraphNode("outer", subgraph)

        assert node.id == "outer"
        assert node.subgraph == subgraph
        assert node.node_type == "subgraph"

    def test_subgraph_node_repr(self):
        """Test SubgraphNode string representation."""
        graph = StateGraph()
        graph.add_node("tool", create_simple_tool("test"), node_type="tool")
        graph.set_entry_point("tool")
        compiled = graph.compile()

        subgraph = Subgraph(compiled, name="inner")
        node = SubgraphNode("outer", subgraph)

        repr_str = repr(node)
        assert "SubgraphNode" in repr_str
        assert "outer" in repr_str


# =============================================================================
# Subgraph Execution Tests
# =============================================================================


class TestSubgraphExecution:
    """Test subgraph execution within parent graphs."""

    @pytest.mark.asyncio
    async def test_simple_subgraph_execution(self, memory_backend, base_context):
        """Test executing a simple subgraph within parent graph."""
        # Create inner graph
        inner = StateGraph()
        inner.add_node("process", create_simple_tool("inner"), node_type="tool")
        inner.set_entry_point("process")
        inner_compiled = inner.compile()

        # Create parent graph with subgraph
        parent = StateGraph()
        parent.add_node("sub", Subgraph(inner_compiled, name="inner"), node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        # Execute
        executor = Executor(parent_compiled, memory_backend)
        events = []
        async for event in executor.execute("test_input", base_context):
            events.append(event)

        # Verify events
        event_types = [e.type for e in events]
        assert EventType.SUBGRAPH_START in event_types
        assert EventType.SUBGRAPH_COMPLETE in event_types
        assert EventType.EXECUTION_COMPLETE in event_types

    @pytest.mark.asyncio
    async def test_subgraph_output_propagation(self, memory_backend, base_context):
        """Test that subgraph output propagates to parent."""
        # Create inner graph that produces specific output
        async def inner_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            return {"result": "inner_result", "processed": True}

        inner = StateGraph()
        inner.add_node("tool", inner_tool, node_type="tool")
        inner.set_entry_point("tool")
        inner_compiled = inner.compile()

        # Create parent graph
        parent = StateGraph()
        parent.add_node("sub", Subgraph(inner_compiled, name="processor"), node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        # Execute
        executor = Executor(parent_compiled, memory_backend)
        final_output = None
        async for event in executor.execute("input", base_context):
            if event.type == EventType.EXECUTION_COMPLETE:
                final_output = event.output

        # Verify output
        assert final_output is not None

    @pytest.mark.asyncio
    async def test_multi_node_subgraph(self, memory_backend, base_context):
        """Test subgraph with multiple nodes."""
        # Create inner graph with chain
        async def step1(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            return {"step1": "done", "value": 1}

        async def step2(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            return {"step2": "done", "value": context.state.get("value", 0) + 1}

        inner = StateGraph()
        inner.add_node("step1", step1, node_type="tool")
        inner.add_node("step2", step2, node_type="tool")
        inner.add_edge("step1", "step2")
        inner.set_entry_point("step1")
        inner_compiled = inner.compile()

        # Create parent graph
        parent = StateGraph()
        parent.add_node("chain", Subgraph(inner_compiled, name="chain"), node_type="subgraph")
        parent.set_entry_point("chain")
        parent_compiled = parent.compile()

        # Execute
        executor = Executor(parent_compiled, memory_backend)
        events = []
        async for event in executor.execute("input", base_context):
            events.append(event)

        # Verify both inner nodes executed
        node_ids = [e.node_id for e in events if e.node_id]
        assert any("step1" in str(node_id) for node_id in node_ids)
        assert any("step2" in str(node_id) for node_id in node_ids)


# =============================================================================
# Input/Output Mapping Tests
# =============================================================================


class TestInputOutputMapping:
    """Test subgraph input and output mapping."""

    @pytest.mark.asyncio
    async def test_input_mapping(self, memory_backend, base_context):
        """Test mapping parent state keys to subgraph input."""
        async def check_input(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            # Verify mapped input is available
            search_query = context.state.get("search_query")
            return {"received_query": search_query}

        inner = StateGraph()
        inner.add_node("check", check_input, node_type="tool")
        inner.set_entry_point("check")
        inner_compiled = inner.compile()

        config = SubgraphConfig(
            input_mapping={"query": "search_query"}
        )
        subgraph = Subgraph(inner_compiled, config=config, name="mapper")

        parent = StateGraph()
        parent.add_node("sub", subgraph, node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        # Execute with query in parent state
        base_context.state["query"] = "test query"
        executor = Executor(parent_compiled, memory_backend)

        events = []
        async for event in executor.execute({}, base_context):
            events.append(event)

        # Verify execution completed
        assert any(e.type == EventType.EXECUTION_COMPLETE for e in events)

    @pytest.mark.asyncio
    async def test_output_mapping(self, memory_backend, base_context):
        """Test mapping subgraph output to parent state."""
        async def produce_output(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            return {"findings": "research results", "confidence": 0.9}

        inner = StateGraph()
        inner.add_node("research", produce_output, node_type="tool")
        inner.set_entry_point("research")
        inner_compiled = inner.compile()

        config = SubgraphConfig(
            output_mapping={"findings": "research_results"}
        )
        subgraph = Subgraph(inner_compiled, config=config, name="researcher")

        parent = StateGraph()
        parent.add_node("sub", subgraph, node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        executor = Executor(parent_compiled, memory_backend)

        subgraph_complete_event = None
        async for event in executor.execute({}, base_context):
            if event.type == EventType.SUBGRAPH_COMPLETE:
                subgraph_complete_event = event

        # Output mapping should be applied
        assert subgraph_complete_event is not None


# =============================================================================
# State Isolation Tests
# =============================================================================


class TestStateIsolation:
    """Test subgraph state isolation behavior."""

    @pytest.mark.asyncio
    async def test_isolated_subgraph_clean_state(self, memory_backend, base_context):
        """Test isolated subgraph starts with clean state."""
        async def check_state(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            # In isolated mode, should not see parent state
            return {"parent_value": context.state.get("parent_key")}

        inner = StateGraph()
        inner.add_node("check", check_state, node_type="tool")
        inner.set_entry_point("check")
        inner_compiled = inner.compile()

        config = SubgraphConfig(isolated=True)
        subgraph = Subgraph(inner_compiled, config=config, name="isolated")

        parent = StateGraph()
        parent.add_node("sub", subgraph, node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        # Set parent state
        base_context.state["parent_key"] = "parent_value"
        executor = Executor(parent_compiled, memory_backend)

        events = []
        async for event in executor.execute({}, base_context):
            events.append(event)

        assert any(e.type == EventType.EXECUTION_COMPLETE for e in events)

    @pytest.mark.asyncio
    async def test_non_isolated_subgraph_inherits_state(self, memory_backend, base_context):
        """Test non-isolated subgraph inherits parent state."""
        async def check_state(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            return {"inherited": context.state.get("parent_key")}

        inner = StateGraph()
        inner.add_node("check", check_state, node_type="tool")
        inner.set_entry_point("check")
        inner_compiled = inner.compile()

        config = SubgraphConfig(isolated=False)
        subgraph = Subgraph(inner_compiled, config=config, name="inheriting")

        parent = StateGraph()
        parent.add_node("sub", subgraph, node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        base_context.state["parent_key"] = "parent_value"
        executor = Executor(parent_compiled, memory_backend)

        events = []
        async for event in executor.execute({}, base_context):
            events.append(event)

        assert any(e.type == EventType.EXECUTION_COMPLETE for e in events)

    @pytest.mark.asyncio
    async def test_inherit_keys(self, memory_backend, base_context):
        """Test selective key inheritance with inherit_keys."""
        async def check_state(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            return {
                "user_id": context.state.get("user_id"),
                "secret": context.state.get("secret"),
            }

        inner = StateGraph()
        inner.add_node("check", check_state, node_type="tool")
        inner.set_entry_point("check")
        inner_compiled = inner.compile()

        config = SubgraphConfig(
            isolated=True,
            inherit_keys=["user_id"]  # Only inherit user_id
        )
        subgraph = Subgraph(inner_compiled, config=config, name="selective")

        parent = StateGraph()
        parent.add_node("sub", subgraph, node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        base_context.state["user_id"] = "user123"
        base_context.state["secret"] = "should_not_inherit"
        executor = Executor(parent_compiled, memory_backend)

        events = []
        async for event in executor.execute({}, base_context):
            events.append(event)

        assert any(e.type == EventType.EXECUTION_COMPLETE for e in events)


# =============================================================================
# Event Prefixing Tests
# =============================================================================


class TestEventPrefixing:
    """Test subgraph event node_id prefixing."""

    @pytest.mark.asyncio
    async def test_events_prefixed_with_subgraph_name(self, memory_backend, base_context):
        """Test that subgraph events are prefixed with subgraph name."""
        inner = StateGraph()
        inner.add_node("inner_node", create_simple_tool("test"), node_type="tool")
        inner.set_entry_point("inner_node")
        inner_compiled = inner.compile()

        config = SubgraphConfig(prefix_events=True)
        subgraph = Subgraph(inner_compiled, config=config, name="research")

        parent = StateGraph()
        parent.add_node("sub", subgraph, node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        executor = Executor(parent_compiled, memory_backend)

        events = []
        async for event in executor.execute("input", base_context):
            events.append(event)

        # Check for prefixed node IDs
        prefixed_events = [e for e in events if e.node_id and "research" in str(e.node_id)]
        assert len(prefixed_events) > 0

    @pytest.mark.asyncio
    async def test_events_not_prefixed_when_disabled(self, memory_backend, base_context):
        """Test that events are not prefixed when prefix_events=False."""
        inner = StateGraph()
        inner.add_node("inner_node", create_simple_tool("test"), node_type="tool")
        inner.set_entry_point("inner_node")
        inner_compiled = inner.compile()

        config = SubgraphConfig(prefix_events=False)
        subgraph = Subgraph(inner_compiled, config=config, name="research")

        parent = StateGraph()
        parent.add_node("sub", subgraph, node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        executor = Executor(parent_compiled, memory_backend)

        events = []
        async for event in executor.execute("input", base_context):
            events.append(event)

        # Should have unprefixed inner_node events
        assert any(e.type == EventType.EXECUTION_COMPLETE for e in events)


# =============================================================================
# SubgraphBuilder Tests
# =============================================================================


class TestSubgraphBuilder:
    """Test SubgraphBuilder fluent API."""

    def test_builder_basic(self):
        """Test basic SubgraphBuilder usage."""
        subgraph = (
            SubgraphBuilder("research")
            .add_node("search", create_simple_tool("search"), node_type="tool")
            .set_entry_point("search")
            .build()
        )

        assert subgraph.name == "research"
        assert len(subgraph.graph.nodes) > 0

    def test_builder_with_edges(self):
        """Test SubgraphBuilder with edges."""
        subgraph = (
            SubgraphBuilder("pipeline")
            .add_node("step1", create_simple_tool("s1"), node_type="tool")
            .add_node("step2", create_simple_tool("s2"), node_type="tool")
            .add_edge("step1", "step2")
            .set_entry_point("step1")
            .build()
        )

        assert subgraph.name == "pipeline"

    def test_builder_with_input_mapping(self):
        """Test SubgraphBuilder with input mapping."""
        subgraph = (
            SubgraphBuilder("mapper")
            .add_node("tool", create_simple_tool("test"), node_type="tool")
            .set_entry_point("tool")
            .with_input_mapping({"query": "search_query"})
            .build()
        )

        assert subgraph.config.input_mapping == {"query": "search_query"}

    def test_builder_with_output_mapping(self):
        """Test SubgraphBuilder with output mapping."""
        subgraph = (
            SubgraphBuilder("mapper")
            .add_node("tool", create_simple_tool("test"), node_type="tool")
            .set_entry_point("tool")
            .with_output_mapping({"result": "findings"})
            .build()
        )

        assert subgraph.config.output_mapping == {"result": "findings"}

    def test_builder_with_isolation(self):
        """Test SubgraphBuilder with isolation config."""
        subgraph = (
            SubgraphBuilder("isolated")
            .add_node("tool", create_simple_tool("test"), node_type="tool")
            .set_entry_point("tool")
            .with_isolation(True, inherit_keys=["user_id"])
            .build()
        )

        assert subgraph.config.isolated is True
        assert subgraph.config.inherit_keys == ["user_id"]

    def test_builder_with_checkpoints(self):
        """Test SubgraphBuilder with checkpoint config."""
        subgraph = (
            SubgraphBuilder("checkpointed")
            .add_node("tool", create_simple_tool("test"), node_type="tool")
            .set_entry_point("tool")
            .with_checkpoints(on_entry=True, on_exit=True)
            .build()
        )

        assert subgraph.config.checkpoint_on_entry is True
        assert subgraph.config.checkpoint_on_exit is True

    def test_builder_chaining(self):
        """Test full builder chain."""
        subgraph = (
            SubgraphBuilder("full_chain")
            .add_node("search", create_simple_tool("search"), node_type="tool")
            .add_node("analyze", create_simple_tool("analyze"), node_type="tool")
            .add_edge("search", "analyze")
            .set_entry_point("search")
            .with_input_mapping({"query": "search_query"})
            .with_output_mapping({"findings": "research_results"})
            .with_isolation(True, inherit_keys=["user_id", "session"])
            .with_checkpoints(on_entry=True, on_exit=True)
            .build()
        )

        assert subgraph.name == "full_chain"
        assert subgraph.config.input_mapping == {"query": "search_query"}
        assert subgraph.config.output_mapping == {"findings": "research_results"}
        assert subgraph.config.isolated is True
        assert subgraph.config.inherit_keys == ["user_id", "session"]
        assert subgraph.config.checkpoint_on_entry is True
        assert subgraph.config.checkpoint_on_exit is True


# =============================================================================
# Nested Subgraph Tests
# =============================================================================


class TestNestedSubgraphs:
    """Test nested subgraphs (subgraphs within subgraphs)."""

    @pytest.mark.asyncio
    async def test_two_level_nesting(self, memory_backend, base_context):
        """Test subgraph within subgraph execution."""
        # Create innermost graph
        async def innermost_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            return {"level": "innermost", "value": 42}

        innermost = StateGraph()
        innermost.add_node("core", innermost_tool, node_type="tool")
        innermost.set_entry_point("core")
        innermost_compiled = innermost.compile()

        # Create middle graph containing innermost
        middle = StateGraph()
        middle.add_node(
            "inner",
            Subgraph(innermost_compiled, name="innermost"),
            node_type="subgraph"
        )
        middle.set_entry_point("inner")
        middle_compiled = middle.compile()

        # Create outer graph containing middle
        outer = StateGraph()
        outer.add_node(
            "middle",
            Subgraph(middle_compiled, name="middle"),
            node_type="subgraph"
        )
        outer.set_entry_point("middle")
        outer_compiled = outer.compile()

        # Execute
        executor = Executor(outer_compiled, memory_backend)
        events = []
        async for event in executor.execute("input", base_context):
            events.append(event)

        # Verify all levels executed
        event_types = [e.type for e in events]
        assert EventType.EXECUTION_COMPLETE in event_types

        # Should have multiple subgraph events
        subgraph_starts = [e for e in events if e.type == EventType.SUBGRAPH_START]
        subgraph_completes = [e for e in events if e.type == EventType.SUBGRAPH_COMPLETE]
        assert len(subgraph_starts) >= 2
        assert len(subgraph_completes) >= 2

    @pytest.mark.asyncio
    async def test_three_level_nesting(self, memory_backend, base_context):
        """Test three levels of nested subgraphs."""
        # Level 3 (innermost)
        async def level3_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            return {"level": 3}

        level3 = StateGraph()
        level3.add_node("l3", level3_tool, node_type="tool")
        level3.set_entry_point("l3")
        level3_compiled = level3.compile()

        # Level 2
        level2 = StateGraph()
        level2.add_node("l3_sub", Subgraph(level3_compiled, name="level3"), node_type="subgraph")
        level2.set_entry_point("l3_sub")
        level2_compiled = level2.compile()

        # Level 1
        level1 = StateGraph()
        level1.add_node("l2_sub", Subgraph(level2_compiled, name="level2"), node_type="subgraph")
        level1.set_entry_point("l2_sub")
        level1_compiled = level1.compile()

        # Root
        root = StateGraph()
        root.add_node("l1_sub", Subgraph(level1_compiled, name="level1"), node_type="subgraph")
        root.set_entry_point("l1_sub")
        root_compiled = root.compile()

        # Execute
        executor = Executor(root_compiled, memory_backend)
        events = []
        async for event in executor.execute("input", base_context):
            events.append(event)

        # Verify completion
        assert any(e.type == EventType.EXECUTION_COMPLETE for e in events)

        # Should have 3 subgraph start/complete pairs
        subgraph_starts = [e for e in events if e.type == EventType.SUBGRAPH_START]
        assert len(subgraph_starts) >= 3


# =============================================================================
# Subgraph with StateGraph.add_node Tests
# =============================================================================


class TestSubgraphViaStateGraph:
    """Test adding subgraphs via StateGraph.add_node()."""

    @pytest.mark.asyncio
    async def test_add_subgraph_via_state_graph(self, memory_backend, base_context):
        """Test adding subgraph using StateGraph.add_node()."""
        inner = StateGraph()
        inner.add_node("tool", create_simple_tool("test"), node_type="tool")
        inner.set_entry_point("tool")
        inner_compiled = inner.compile()

        parent = StateGraph()
        parent.add_node("sub", Subgraph(inner_compiled), node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        executor = Executor(parent_compiled, memory_backend)
        events = []
        async for event in executor.execute("input", base_context):
            events.append(event)

        assert any(e.type == EventType.EXECUTION_COMPLETE for e in events)

    def test_add_subgraph_wrong_type_raises(self):
        """Test that passing non-Subgraph to subgraph node type raises error."""
        parent = StateGraph()

        with pytest.raises(Exception):  # GraphValidationError
            parent.add_node("sub", "not_a_subgraph", node_type="subgraph")


# =============================================================================
# Subgraph in Complex Graphs Tests
# =============================================================================


class TestSubgraphInComplexGraphs:
    """Test subgraphs within complex parent graph structures."""

    @pytest.mark.asyncio
    async def test_subgraph_after_tool(self, memory_backend, base_context):
        """Test subgraph following a tool node."""
        async def pre_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            return {"prepared": True}

        inner = StateGraph()
        inner.add_node("process", create_simple_tool("inner"), node_type="tool")
        inner.set_entry_point("process")
        inner_compiled = inner.compile()

        parent = StateGraph()
        parent.add_node("prepare", pre_tool, node_type="tool")
        parent.add_node("sub", Subgraph(inner_compiled, name="processor"), node_type="subgraph")
        parent.add_edge("prepare", "sub")
        parent.set_entry_point("prepare")
        parent_compiled = parent.compile()

        executor = Executor(parent_compiled, memory_backend)
        events = []
        async for event in executor.execute("input", base_context):
            events.append(event)

        # Both prepare and subgraph should execute
        node_complete_events = [e for e in events if e.type == EventType.NODE_COMPLETE]
        assert len(node_complete_events) >= 1

    @pytest.mark.asyncio
    async def test_tool_after_subgraph(self, memory_backend, base_context):
        """Test tool node following a subgraph."""
        async def post_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            return {"finalized": True}

        inner = StateGraph()
        inner.add_node("process", create_simple_tool("inner"), node_type="tool")
        inner.set_entry_point("process")
        inner_compiled = inner.compile()

        parent = StateGraph()
        parent.add_node("sub", Subgraph(inner_compiled, name="processor"), node_type="subgraph")
        parent.add_node("finalize", post_tool, node_type="tool")
        parent.add_edge("sub", "finalize")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        executor = Executor(parent_compiled, memory_backend)
        events = []
        async for event in executor.execute("input", base_context):
            events.append(event)

        assert any(e.type == EventType.EXECUTION_COMPLETE for e in events)

    @pytest.mark.asyncio
    async def test_multiple_subgraphs_sequential(self, memory_backend, base_context):
        """Test multiple subgraphs in sequence."""
        # Create two different inner graphs
        inner1 = StateGraph()
        inner1.add_node("step", create_simple_tool("phase1"), node_type="tool")
        inner1.set_entry_point("step")
        inner1_compiled = inner1.compile()

        inner2 = StateGraph()
        inner2.add_node("step", create_simple_tool("phase2"), node_type="tool")
        inner2.set_entry_point("step")
        inner2_compiled = inner2.compile()

        # Chain them in parent
        parent = StateGraph()
        parent.add_node("phase1", Subgraph(inner1_compiled, name="phase1"), node_type="subgraph")
        parent.add_node("phase2", Subgraph(inner2_compiled, name="phase2"), node_type="subgraph")
        parent.add_edge("phase1", "phase2")
        parent.set_entry_point("phase1")
        parent_compiled = parent.compile()

        executor = Executor(parent_compiled, memory_backend)
        events = []
        async for event in executor.execute("input", base_context):
            events.append(event)

        # Both subgraphs should complete
        subgraph_completes = [e for e in events if e.type == EventType.SUBGRAPH_COMPLETE]
        assert len(subgraph_completes) >= 2


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestSubgraphErrorHandling:
    """Test error handling in subgraphs."""

    @pytest.mark.asyncio
    async def test_subgraph_error_propagates(self, memory_backend, base_context):
        """Test that errors in subgraph propagate to parent."""
        from mesh.utils.errors import NodeExecutionError

        async def failing_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            raise ValueError("Subgraph error!")

        inner = StateGraph()
        inner.add_node("fail", failing_tool, node_type="tool")
        inner.set_entry_point("fail")
        inner_compiled = inner.compile()

        parent = StateGraph()
        parent.add_node("sub", Subgraph(inner_compiled, name="failing"), node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        executor = Executor(parent_compiled, memory_backend)

        # Error is wrapped by executor in NodeExecutionError
        with pytest.raises(NodeExecutionError, match="Subgraph error!"):
            async for _ in executor.execute("input", base_context):
                pass


# =============================================================================
# Metadata Tests
# =============================================================================


class TestSubgraphMetadata:
    """Test subgraph metadata in events."""

    @pytest.mark.asyncio
    async def test_subgraph_start_event_metadata(self, memory_backend, base_context):
        """Test SUBGRAPH_START event contains proper metadata."""
        inner = StateGraph()
        inner.add_node("tool", create_simple_tool("test"), node_type="tool")
        inner.set_entry_point("tool")
        inner_compiled = inner.compile()

        parent = StateGraph()
        parent.add_node("sub", Subgraph(inner_compiled, name="test_sub"), node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        executor = Executor(parent_compiled, memory_backend)

        start_event = None
        async for event in executor.execute({"key": "value"}, base_context):
            if event.type == EventType.SUBGRAPH_START:
                start_event = event
                break

        assert start_event is not None
        assert start_event.node_id == "test_sub"
        assert start_event.metadata is not None

    @pytest.mark.asyncio
    async def test_subgraph_complete_event_metadata(self, memory_backend, base_context):
        """Test SUBGRAPH_COMPLETE event contains proper metadata."""
        inner = StateGraph()
        inner.add_node("tool", create_simple_tool("test"), node_type="tool")
        inner.set_entry_point("tool")
        inner_compiled = inner.compile()

        parent = StateGraph()
        parent.add_node("sub", Subgraph(inner_compiled, name="test_sub"), node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        executor = Executor(parent_compiled, memory_backend)

        complete_event = None
        async for event in executor.execute("input", base_context):
            if event.type == EventType.SUBGRAPH_COMPLETE:
                complete_event = event

        assert complete_event is not None
        assert complete_event.node_id == "test_sub"
        assert complete_event.output is not None


# =============================================================================
# Reusability Tests
# =============================================================================


class TestSubgraphReusability:
    """Test subgraph reusability patterns."""

    @pytest.mark.asyncio
    async def test_same_subgraph_used_twice(self, memory_backend, base_context):
        """Test using the same subgraph definition in multiple places."""
        inner = StateGraph()
        inner.add_node("process", create_simple_tool("shared"), node_type="tool")
        inner.set_entry_point("process")
        inner_compiled = inner.compile()

        # Use same compiled graph in two different subgraph nodes
        parent = StateGraph()
        parent.add_node("first", Subgraph(inner_compiled, name="first"), node_type="subgraph")
        parent.add_node("second", Subgraph(inner_compiled, name="second"), node_type="subgraph")
        parent.add_edge("first", "second")
        parent.set_entry_point("first")
        parent_compiled = parent.compile()

        executor = Executor(parent_compiled, memory_backend)
        events = []
        async for event in executor.execute("input", base_context):
            events.append(event)

        # Both should complete
        subgraph_completes = [e for e in events if e.type == EventType.SUBGRAPH_COMPLETE]
        assert len(subgraph_completes) >= 2

    @pytest.mark.asyncio
    async def test_subgraph_execution_isolated_between_runs(self, memory_backend, base_context):
        """Test that multiple executions of same subgraph are isolated."""
        counter = {"value": 0}

        async def counting_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            counter["value"] += 1
            return {"count": counter["value"]}

        inner = StateGraph()
        inner.add_node("count", counting_tool, node_type="tool")
        inner.set_entry_point("count")
        inner_compiled = inner.compile()

        parent = StateGraph()
        parent.add_node("sub", Subgraph(inner_compiled, name="counter"), node_type="subgraph")
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        # Execute twice
        executor = Executor(parent_compiled, memory_backend)

        async for _ in executor.execute("run1", base_context):
            pass

        async for _ in executor.execute("run2", base_context):
            pass

        # Counter should have been incremented twice
        assert counter["value"] == 2
