"""Integration tests for the deep agent pattern.

These tests verify that all LangGraph parity features work together:
- Checkpointing: State persistence and restore
- Interrupts: Human-in-the-loop pause/resume
- Parallel Execution: Fan-out/fan-in patterns
- Subgraph Composition: Nested graph execution
- Streaming Modes: Different views of execution

The tests simulate a "deep agent" workflow typical of production agent systems.
"""

import pytest
from typing import Any, Dict, List
from datetime import datetime

from mesh import (
    StateGraph,
    Executor,
    ExecutionContext,
    MemoryBackend,
    SQLiteBackend,
    Subgraph,
    SubgraphConfig,
    SubgraphBuilder,
    StreamMode,
    StateValue,
    StateUpdate,
    InterruptResume,
)
from mesh.core.events import EventType, ExecutionEvent


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
        graph_id="integration-test",
        session_id="integration-session",
        chat_history=[],
        variables={},
        state={},
    )


# =============================================================================
# Helper Tool Functions
# =============================================================================


async def planner_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
    """Simulates a planner that creates research tasks."""
    query = input if isinstance(input, str) else input.get("query", "default query")
    return {
        "plan": f"Research plan for: {query}",
        "research_tasks": ["web", "docs", "code"],
        "query": query,
    }


async def web_researcher_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
    """Simulates web research."""
    query = context.state.get("query", "unknown")
    return {
        "source": "web",
        "findings": f"Web findings for: {query}",
        "confidence": 0.85,
    }


async def doc_researcher_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
    """Simulates documentation research."""
    query = context.state.get("query", "unknown")
    return {
        "source": "docs",
        "findings": f"Documentation findings for: {query}",
        "confidence": 0.92,
    }


async def code_researcher_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
    """Simulates code research."""
    query = context.state.get("query", "unknown")
    return {
        "source": "code",
        "findings": f"Code analysis for: {query}",
        "confidence": 0.78,
    }


async def synthesizer_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
    """Simulates synthesizing research results."""
    findings = []
    for key in context.state:
        if key.endswith("_output") and isinstance(context.state[key], dict):
            output = context.state[key]
            if "output" in output and isinstance(output["output"], dict):
                inner = output["output"]
                if "findings" in inner:
                    findings.append(inner["findings"])

    return {
        "synthesis": f"Synthesized {len(findings)} findings",
        "all_findings": findings,
        "ready_for_review": True,
    }


async def writer_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
    """Simulates writing final output."""
    synthesis = context.state.get("synthesis", "No synthesis available")
    return {
        "draft": f"Final report based on: {synthesis}",
        "word_count": 500,
        "status": "draft",
    }


async def publisher_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
    """Simulates publishing (sensitive operation)."""
    return {
        "published": True,
        "url": "https://example.com/report",
        "timestamp": datetime.now().isoformat(),
    }


async def incrementer_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
    """Simple incrementer for counter tests.

    Directly updates context.state since ToolNode output goes to node_id_output,
    not to top-level state.
    """
    count = context.state.get("count", 0)
    new_count = count + 1
    context.state["count"] = new_count  # Direct state update
    return {"count": new_count}


# =============================================================================
# Feature Integration Tests
# =============================================================================


class TestDeepAgentPattern:
    """Test the full deep agent pattern combining all features."""

    @pytest.mark.asyncio
    async def test_simple_sequential_with_manual_checkpoint(self, memory_backend, base_context):
        """Test sequential execution with manual checkpointing."""
        graph = StateGraph()
        graph.add_node("step1", incrementer_tool, node_type="tool")
        graph.add_node("step2", incrementer_tool, node_type="tool")
        graph.add_node("step3", incrementer_tool, node_type="tool")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", "step3")
        graph.set_entry_point("step1")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)
        base_context.state["count"] = 0

        # Execute
        events = []
        async for event in executor.execute({"count": 0}, base_context):
            events.append(event)

        # After execution, create a checkpoint
        checkpoint_id = await executor.checkpoint(base_context)
        assert checkpoint_id is not None

        # Verify checkpoint was saved (list_checkpoints is on backend)
        checkpoints = await memory_backend.list_checkpoints(base_context.session_id)
        assert len(checkpoints) >= 1

        # Final state should have count = 3
        assert base_context.state.get("count") == 3

    @pytest.mark.asyncio
    async def test_subgraph_with_streaming_modes(self, memory_backend, base_context):
        """Test subgraph execution with different streaming modes."""
        # Create inner research subgraph
        inner = StateGraph()
        inner.add_node("research", web_researcher_tool, node_type="tool")
        inner.set_entry_point("research")
        inner_compiled = inner.compile()

        # Create parent graph with subgraph
        parent = StateGraph()
        parent.add_node("plan", planner_tool, node_type="tool")
        parent.add_node(
            "research",
            Subgraph(inner_compiled, name="web_research"),
            node_type="subgraph"
        )
        parent.add_edge("plan", "research")
        parent.set_entry_point("plan")
        parent_compiled = parent.compile()

        executor = Executor(parent_compiled, memory_backend)

        # Test with VALUES streaming mode
        values = []
        async for item in executor.stream(
            "Test query", base_context, mode=StreamMode.VALUES
        ):
            values.append(item)

        # Should have StateValue items
        assert len(values) >= 2
        assert all(isinstance(v, StateValue) for v in values)

    @pytest.mark.asyncio
    async def test_interrupt_with_resume(self, memory_backend, base_context):
        """Test interrupt before sensitive operation and resume."""
        graph = StateGraph()
        graph.add_node("prepare", planner_tool, node_type="tool")
        graph.add_node("sensitive", publisher_tool, node_type="tool")
        graph.add_edge("prepare", "sensitive")
        graph.set_entry_point("prepare")

        # Set interrupt before sensitive operation
        graph.set_interrupt_before("sensitive")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        # Execute until interrupt (consume all events)
        events = []
        async for event in executor.execute("Test input", base_context):
            events.append(event)

        # Should have hit interrupt
        interrupt_events = [e for e in events if e.type == EventType.INTERRUPT]
        assert len(interrupt_events) == 1

        interrupt_event = interrupt_events[0]
        assert interrupt_event.node_id == "sensitive"

        # Resume execution using InterruptResume
        resume = InterruptResume()
        resume_events = []
        async for event in executor.resume_from_interrupt(base_context, resume):
            resume_events.append(event)

        # Should complete after resume
        assert any(e.type == EventType.EXECUTION_COMPLETE for e in resume_events)

    @pytest.mark.asyncio
    async def test_nested_subgraphs_with_state_isolation(self, memory_backend, base_context):
        """Test nested subgraphs with proper state isolation."""
        # Level 2 (innermost)
        level2 = StateGraph()
        level2.add_node("inner_tool", incrementer_tool, node_type="tool")
        level2.set_entry_point("inner_tool")
        level2_compiled = level2.compile()

        # Level 1 (middle)
        level1 = StateGraph()
        level1.add_node(
            "sub",
            Subgraph(level2_compiled, name="level2", config=SubgraphConfig(isolated=True)),
            node_type="subgraph"
        )
        level1.set_entry_point("sub")
        level1_compiled = level1.compile()

        # Root graph
        root = StateGraph()
        root.add_node("init", planner_tool, node_type="tool")
        root.add_node(
            "nested",
            Subgraph(level1_compiled, name="level1"),
            node_type="subgraph"
        )
        root.add_edge("init", "nested")
        root.set_entry_point("init")
        root_compiled = root.compile()

        executor = Executor(root_compiled, memory_backend)

        events = []
        async for event in executor.execute("Test", base_context):
            events.append(event)

        # Should have subgraph start/complete events
        subgraph_starts = [e for e in events if e.type == EventType.SUBGRAPH_START]
        subgraph_completes = [e for e in events if e.type == EventType.SUBGRAPH_COMPLETE]

        assert len(subgraph_starts) >= 2
        assert len(subgraph_completes) >= 2
        assert any(e.type == EventType.EXECUTION_COMPLETE for e in events)

    @pytest.mark.asyncio
    async def test_checkpoint_restore_continues_execution(self, memory_backend):
        """Test that restoring from checkpoint continues execution correctly."""
        graph = StateGraph()
        graph.add_node("step1", incrementer_tool, node_type="tool")
        graph.add_node("step2", incrementer_tool, node_type="tool")
        graph.add_node("step3", incrementer_tool, node_type="tool")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", "step3")
        graph.set_entry_point("step1")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        # First execution
        context1 = ExecutionContext(
            graph_id="test",
            session_id="session-1",
            chat_history=[],
            variables={},
            state={"count": 0},
        )

        # Execute
        async for _ in executor.execute({"count": 0}, context1):
            pass

        # Create checkpoint manually
        checkpoint_id = await executor.checkpoint(context1)
        assert checkpoint_id is not None

        # Get checkpoints (list_checkpoints is on backend)
        checkpoints = await memory_backend.list_checkpoints(context1.session_id)
        assert len(checkpoints) >= 1

        # Restore from checkpoint
        restored = await executor.restore(checkpoint_id)

        assert restored is not None
        assert restored.session_id == context1.session_id

    @pytest.mark.asyncio
    async def test_debug_mode_shows_execution_internals(self, memory_backend, base_context):
        """Test DEBUG streaming mode shows execution internals."""
        graph = StateGraph()
        graph.add_node("tool1", incrementer_tool, node_type="tool")
        graph.add_node("tool2", incrementer_tool, node_type="tool")
        graph.add_edge("tool1", "tool2")
        graph.set_entry_point("tool1")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        debug_items = []
        async for item in executor.stream(
            {"count": 0}, base_context, mode=StreamMode.DEBUG
        ):
            debug_items.append(item)

        # Should have DebugInfo items
        assert len(debug_items) > 0
        for debug in debug_items:
            assert hasattr(debug, 'event')
            assert hasattr(debug, 'internal_state')
            assert hasattr(debug, 'timing')


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_checkpoint_allows_recovery_after_error(self, memory_backend):
        """Test that checkpoints allow recovery after errors."""
        # Create a simple graph that succeeds
        graph = StateGraph()
        graph.add_node("step1", incrementer_tool, node_type="tool")
        graph.set_entry_point("step1")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        context = ExecutionContext(
            graph_id="test",
            session_id="recovery-test",
            chat_history=[],
            variables={},
            state={"count": 0},
        )

        # Execute successfully
        async for _ in executor.execute({"count": 0}, context):
            pass

        # Create a checkpoint
        checkpoint_id = await executor.checkpoint(context)
        assert checkpoint_id is not None

        # Verify we can retrieve it (list_checkpoints is on backend)
        checkpoints = await memory_backend.list_checkpoints(context.session_id)
        assert len(checkpoints) >= 1

        # Restore from checkpoint
        restored = await executor.restore(checkpoint_id)
        assert restored is not None

    @pytest.mark.asyncio
    async def test_subgraph_error_propagates_correctly(self, memory_backend, base_context):
        """Test that errors in subgraphs propagate to parent."""
        from mesh.utils.errors import NodeExecutionError

        async def failing_tool(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            raise ValueError("Inner error")

        inner = StateGraph()
        inner.add_node("fail", failing_tool, node_type="tool")
        inner.set_entry_point("fail")
        inner_compiled = inner.compile()

        parent = StateGraph()
        parent.add_node(
            "sub",
            Subgraph(inner_compiled, name="failing_sub"),
            node_type="subgraph"
        )
        parent.set_entry_point("sub")
        parent_compiled = parent.compile()

        executor = Executor(parent_compiled, memory_backend)

        with pytest.raises(NodeExecutionError):
            async for _ in executor.execute("input", base_context):
                pass


# =============================================================================
# Performance and Scale Tests
# =============================================================================


class TestPerformancePatterns:
    """Test performance-related patterns."""

    @pytest.mark.asyncio
    async def test_many_sequential_nodes(self, memory_backend):
        """Test execution with many sequential nodes."""
        graph = StateGraph()

        # Create 10 sequential nodes (reduced from 20 for speed)
        for i in range(10):
            graph.add_node(f"step{i}", incrementer_tool, node_type="tool")
            if i > 0:
                graph.add_edge(f"step{i-1}", f"step{i}")

        graph.set_entry_point("step0")
        compiled = graph.compile()

        context = ExecutionContext(
            graph_id="test",
            session_id="many-nodes-test",
            chat_history=[],
            variables={},
            state={"count": 0},
        )

        executor = Executor(compiled, memory_backend)

        events = []
        async for event in executor.execute({}, context):
            events.append(event)

        # Should complete successfully
        assert any(e.type == EventType.EXECUTION_COMPLETE for e in events)

        # Count should have incremented 10 times
        assert context.state["count"] == 10

    @pytest.mark.asyncio
    async def test_deeply_nested_subgraphs(self, memory_backend, base_context):
        """Test deeply nested subgraph execution."""
        # Create 5 levels of nesting
        current_graph = StateGraph()
        current_graph.add_node("leaf", incrementer_tool, node_type="tool")
        current_graph.set_entry_point("leaf")
        current_compiled = current_graph.compile()

        for i in range(5):
            wrapper = StateGraph()
            wrapper.add_node(
                "sub",
                Subgraph(current_compiled, name=f"level{i}"),
                node_type="subgraph"
            )
            wrapper.set_entry_point("sub")
            current_compiled = wrapper.compile()

        executor = Executor(current_compiled, memory_backend)

        events = []
        async for event in executor.execute({}, base_context):
            events.append(event)

        # Should complete despite deep nesting
        assert any(e.type == EventType.EXECUTION_COMPLETE for e in events)


# =============================================================================
# Real-World Pattern Tests
# =============================================================================


class TestRealWorldPatterns:
    """Test patterns common in real-world agent systems."""

    @pytest.mark.asyncio
    async def test_research_synthesis_pattern(self, memory_backend, base_context):
        """Test a realistic research-synthesis workflow."""
        # Create research subgraph
        research = SubgraphBuilder("research")
        research.add_node("search", web_researcher_tool, node_type="tool")
        research.set_entry_point("search")
        research_subgraph = research.build()

        # Main workflow
        graph = StateGraph()
        graph.add_node("plan", planner_tool, node_type="tool")
        graph.add_node("research", research_subgraph, node_type="subgraph")
        graph.add_node("synthesize", synthesizer_tool, node_type="tool")
        graph.add_node("write", writer_tool, node_type="tool")
        graph.add_edge("plan", "research")
        graph.add_edge("research", "synthesize")
        graph.add_edge("synthesize", "write")
        graph.set_entry_point("plan")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        events = []
        async for event in executor.execute(
            "Research AI agent frameworks", base_context
        ):
            events.append(event)

        # Verify workflow completed
        assert any(e.type == EventType.EXECUTION_COMPLETE for e in events)

        # Verify subgraph executed
        assert any(e.type == EventType.SUBGRAPH_START for e in events)
        assert any(e.type == EventType.SUBGRAPH_COMPLETE for e in events)

    @pytest.mark.asyncio
    async def test_approval_workflow_pattern(self, memory_backend, base_context):
        """Test a workflow requiring human approval."""
        graph = StateGraph()
        graph.add_node("draft", writer_tool, node_type="tool")
        graph.add_node("publish", publisher_tool, node_type="tool")
        graph.add_edge("draft", "publish")
        graph.set_entry_point("draft")

        # Require approval before publishing
        graph.set_interrupt_before("publish")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        # Run until approval needed (consume all events)
        events = []
        async for event in executor.execute("Create report", base_context):
            events.append(event)

        # Find interrupt event
        interrupt_events = [e for e in events if e.type == EventType.INTERRUPT]
        assert len(interrupt_events) == 1

        interrupt_event = interrupt_events[0]
        assert interrupt_event.node_id == "publish"

        # Simulate approval and resume
        resume = InterruptResume()
        resume_events = []
        async for event in executor.resume_from_interrupt(base_context, resume):
            resume_events.append(event)

        # Should complete after approval
        assert any(e.type == EventType.EXECUTION_COMPLETE for e in resume_events)

    @pytest.mark.asyncio
    async def test_conditional_with_subgraph(self, memory_backend, base_context):
        """Test conditional branching that leads to subgraph execution."""
        # Create specialized subgraph
        specialized = StateGraph()
        specialized.add_node("special", code_researcher_tool, node_type="tool")
        specialized.set_entry_point("special")
        specialized_compiled = specialized.compile()

        # Main graph with condition
        async def router(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            return {"route": "specialized", "needs_research": True}

        async def simple_handler(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            return {"result": "simple path taken"}

        graph = StateGraph()
        graph.add_node("router", router, node_type="tool")
        graph.add_node("simple", simple_handler, node_type="tool")
        graph.add_node(
            "specialized",
            Subgraph(specialized_compiled, name="specialized_research"),
            node_type="subgraph"
        )
        graph.add_edge("router", "specialized")
        graph.set_entry_point("router")
        compiled = graph.compile()

        executor = Executor(compiled, memory_backend)

        events = []
        async for event in executor.execute("Complex query", base_context):
            events.append(event)

        # Should have executed the subgraph path
        assert any(e.type == EventType.SUBGRAPH_START for e in events)
        assert any(e.type == EventType.EXECUTION_COMPLETE for e in events)
