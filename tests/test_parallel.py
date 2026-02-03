"""Tests for the parallel execution system.

Tests cover:
- Send dataclass
- ParallelBranch dataclass
- ParallelConfig
- ParallelExecutor with various strategies
- StateGraph parallel edge methods
- Aggregator functions
"""

import pytest
import asyncio
from typing import Any, Dict

from mesh import (
    StateGraph,
    MemoryBackend,
    Send,
    ParallelBranch,
    ParallelConfig,
    ParallelExecutor,
    ParallelResult,
    ParallelErrorStrategy,
    ParallelExecutionError,
    default_aggregator,
    list_aggregator,
    keyed_aggregator,
)
from mesh.core.state import ExecutionContext
from mesh.nodes.base import BaseNode, NodeResult


# =============================================================================
# Test Fixtures
# =============================================================================


class MockWorkerNode(BaseNode):
    """Mock worker node that simulates async work."""

    def __init__(self, id: str, delay: float = 0.01, fail: bool = False):
        super().__init__(id=id, config={})
        self.delay = delay
        self.fail = fail
        self.executed = False

    async def _execute_impl(self, input: Any, context: "ExecutionContext") -> NodeResult:
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError(f"Node {self.id} failed intentionally")
        self.executed = True
        return NodeResult(
            output={"worker_id": self.id, "input": input, "status": "completed"},
            state={f"{self.id}_executed": True},
        )


def create_test_context():
    """Create a test execution context."""
    return ExecutionContext(
        graph_id="test-graph",
        session_id="test-session",
        chat_history=[],
        variables={},
        state={},
    )


# =============================================================================
# Send Tests
# =============================================================================


class TestSend:
    """Tests for Send dataclass."""

    def test_basic_send(self):
        """Test creating a basic Send."""
        send = Send(node="worker", input={"task": "process"})

        assert send.node == "worker"
        assert send.input == {"task": "process"}

    def test_send_requires_string_node(self):
        """Test that Send.node must be a string."""
        with pytest.raises(ValueError, match="must be a string"):
            Send(node=123, input={})

    def test_send_requires_dict_input(self):
        """Test that Send.input must be a dict."""
        with pytest.raises(ValueError, match="must be a dict"):
            Send(node="worker", input="not a dict")

    def test_send_with_complex_input(self):
        """Test Send with complex nested input."""
        send = Send(
            node="worker",
            input={
                "nested": {"deep": {"value": [1, 2, 3]}},
                "items": ["a", "b", "c"],
            },
        )

        assert send.input["nested"]["deep"]["value"] == [1, 2, 3]


class TestParallelBranch:
    """Tests for ParallelBranch dataclass."""

    def test_basic_branch(self):
        """Test creating a basic parallel branch."""
        branch = ParallelBranch(source="start", targets=["worker_1", "worker_2"])

        assert branch.source == "start"
        assert branch.targets == ["worker_1", "worker_2"]
        assert branch.is_dynamic is False

    def test_branch_requires_two_targets(self):
        """Test that static branches require at least 2 targets."""
        with pytest.raises(ValueError, match="at least 2 targets"):
            ParallelBranch(source="start", targets=["worker_1"])

    def test_dynamic_branch_allows_single_target(self):
        """Test that dynamic branches allow single target (for Send routing)."""
        # Dynamic branches are determined at runtime
        branch = ParallelBranch(source="router", targets=["worker"], is_dynamic=True)
        assert branch.is_dynamic is True


class TestParallelConfig:
    """Tests for ParallelConfig."""

    def test_default_config(self):
        """Test default parallel configuration."""
        config = ParallelConfig()

        assert config.max_concurrency == 10
        assert config.error_strategy == ParallelErrorStrategy.CONTINUE_ALL
        assert config.timeout is None
        assert config.preserve_order is True

    def test_custom_config(self):
        """Test custom parallel configuration."""
        config = ParallelConfig(
            max_concurrency=5,
            error_strategy=ParallelErrorStrategy.FAIL_FAST,
            timeout=30.0,
            preserve_order=False,
        )

        assert config.max_concurrency == 5
        assert config.error_strategy == ParallelErrorStrategy.FAIL_FAST
        assert config.timeout == 30.0

    def test_invalid_concurrency(self):
        """Test that max_concurrency must be positive."""
        with pytest.raises(ValueError, match="at least 1"):
            ParallelConfig(max_concurrency=0)

    def test_invalid_timeout(self):
        """Test that timeout must be positive."""
        with pytest.raises(ValueError, match="positive"):
            ParallelConfig(timeout=-1.0)


# =============================================================================
# ParallelResult Tests
# =============================================================================


class TestParallelResult:
    """Tests for ParallelResult dataclass."""

    def test_empty_result(self):
        """Test empty parallel result."""
        result = ParallelResult()

        assert result.results == {}
        assert result.errors == []
        assert result.completed == []
        assert result.failed == []
        assert not result.has_errors
        assert not result.all_succeeded

    def test_successful_result(self):
        """Test successful parallel result."""
        result = ParallelResult(
            results={"worker_1": "output1", "worker_2": "output2"},
            completed=["worker_1", "worker_2"],
        )

        assert result.all_succeeded
        assert not result.has_errors
        assert not result.partial_success

    def test_partial_result(self):
        """Test partial success result."""
        result = ParallelResult(
            results={"worker_1": "output1"},
            completed=["worker_1"],
            errors=[("worker_2", RuntimeError("failed"))],
            failed=["worker_2"],
        )

        assert result.has_errors
        assert result.partial_success
        assert not result.all_succeeded


# =============================================================================
# ParallelExecutor Tests
# =============================================================================


class TestParallelExecutor:
    """Tests for ParallelExecutor."""

    @pytest.mark.asyncio
    async def test_execute_parallel_basic(self):
        """Test basic parallel execution."""
        executor = ParallelExecutor()
        context = create_test_context()

        async def mock_node_executor(node_id: str, input_data: Dict, ctx):
            await asyncio.sleep(0.01)
            return {"node_id": node_id, "processed": True}

        branches = [
            ("worker_1", {"data": 1}),
            ("worker_2", {"data": 2}),
            ("worker_3", {"data": 3}),
        ]

        result = await executor.execute_parallel(branches, mock_node_executor, context)

        assert result.all_succeeded
        assert len(result.completed) == 3
        assert "worker_1" in result.results
        assert "worker_2" in result.results
        assert "worker_3" in result.results

    @pytest.mark.asyncio
    async def test_execute_parallel_empty(self):
        """Test parallel execution with empty branches."""
        executor = ParallelExecutor()
        context = create_test_context()

        async def mock_executor(node_id, input_data, ctx):
            return {}

        result = await executor.execute_parallel([], mock_executor, context)

        assert result.results == {}
        assert result.completed == []

    @pytest.mark.asyncio
    async def test_execute_parallel_concurrency_limit(self):
        """Test that concurrency limit is respected."""
        config = ParallelConfig(max_concurrency=2)
        executor = ParallelExecutor(config)
        context = create_test_context()

        concurrent_count = 0
        max_concurrent = 0

        async def counting_executor(node_id: str, input_data: Dict, ctx):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return {"node_id": node_id}

        branches = [
            ("worker_1", {}),
            ("worker_2", {}),
            ("worker_3", {}),
            ("worker_4", {}),
        ]

        await executor.execute_parallel(branches, counting_executor, context)

        # Max concurrent should be limited to 2
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_execute_parallel_continue_all_strategy(self):
        """Test CONTINUE_ALL error strategy."""
        config = ParallelConfig(error_strategy=ParallelErrorStrategy.CONTINUE_ALL)
        executor = ParallelExecutor(config)
        context = create_test_context()

        async def failing_executor(node_id: str, input_data: Dict, ctx):
            if node_id == "worker_2":
                raise RuntimeError("Worker 2 failed")
            return {"node_id": node_id}

        branches = [
            ("worker_1", {}),
            ("worker_2", {}),
            ("worker_3", {}),
        ]

        result = await executor.execute_parallel(branches, failing_executor, context)

        # Should continue and collect results from successful branches
        assert result.has_errors
        assert len(result.errors) == 1
        assert "worker_1" in result.results
        assert "worker_3" in result.results
        assert "worker_2" not in result.results

    @pytest.mark.asyncio
    async def test_execute_parallel_fail_fast_strategy(self):
        """Test FAIL_FAST error strategy."""
        config = ParallelConfig(error_strategy=ParallelErrorStrategy.FAIL_FAST)
        executor = ParallelExecutor(config)
        context = create_test_context()

        async def failing_executor(node_id: str, input_data: Dict, ctx):
            if node_id == "worker_2":
                raise RuntimeError("Worker 2 failed")
            await asyncio.sleep(0.1)  # Longer delay for other workers
            return {"node_id": node_id}

        branches = [
            ("worker_1", {}),
            ("worker_2", {}),
            ("worker_3", {}),
        ]

        with pytest.raises(ParallelExecutionError, match="Worker 2 failed|worker_2"):
            await executor.execute_parallel(branches, failing_executor, context)

    @pytest.mark.asyncio
    async def test_execute_sends_basic(self):
        """Test executing dynamic sends."""
        executor = ParallelExecutor()
        context = create_test_context()

        async def mock_executor(node_id: str, input_data: Dict, ctx):
            return {"node_id": node_id, "item": input_data.get("item")}

        sends = [
            Send("worker", {"item": "task_1"}),
            Send("worker", {"item": "task_2"}),
            Send("worker", {"item": "task_3"}),
        ]

        result = await executor.execute_sends(sends, mock_executor, context)

        assert result.all_succeeded
        # Results should be ordered
        ordered = result.results.get("_ordered", [])
        assert len(ordered) == 3
        assert ordered[0]["item"] == "task_1"
        assert ordered[1]["item"] == "task_2"
        assert ordered[2]["item"] == "task_3"

    @pytest.mark.asyncio
    async def test_execute_sends_preserves_order(self):
        """Test that sends preserve order even with different completion times."""
        config = ParallelConfig(preserve_order=True)
        executor = ParallelExecutor(config)
        context = create_test_context()

        async def varying_delay_executor(node_id: str, input_data: Dict, ctx):
            # First item takes longest, last takes shortest
            delay = 0.05 - (input_data.get("index", 0) * 0.01)
            await asyncio.sleep(max(0.01, delay))
            return {"index": input_data.get("index")}

        sends = [
            Send("worker", {"index": 0}),
            Send("worker", {"index": 1}),
            Send("worker", {"index": 2}),
        ]

        result = await executor.execute_sends(sends, varying_delay_executor, context)

        ordered = result.results.get("_ordered", [])
        assert ordered[0]["index"] == 0
        assert ordered[1]["index"] == 1
        assert ordered[2]["index"] == 2


# =============================================================================
# StateGraph Parallel Methods Tests
# =============================================================================


class TestStateGraphParallelMethods:
    """Tests for StateGraph parallel methods."""

    def test_add_parallel_edges(self):
        """Test adding parallel edges."""
        graph = StateGraph()
        graph.add_node("start", MockWorkerNode("start"))
        graph.add_node("worker_1", MockWorkerNode("worker_1"))
        graph.add_node("worker_2", MockWorkerNode("worker_2"))
        graph.add_node("worker_3", MockWorkerNode("worker_3"))

        result = graph.add_parallel_edges("start", ["worker_1", "worker_2", "worker_3"])

        assert result is graph  # Returns self for chaining

        # Check that edges were created
        assert len(graph._edges) == 3
        targets = [e.target for e in graph._edges if e.source == "start"]
        assert "worker_1" in targets
        assert "worker_2" in targets
        assert "worker_3" in targets

        # Check parallel branch was recorded
        assert len(graph._parallel_branches) == 1
        assert graph._parallel_branches[0]["source"] == "start"
        assert graph._parallel_branches[0]["targets"] == ["worker_1", "worker_2", "worker_3"]

    def test_add_parallel_edges_requires_two_targets(self):
        """Test that add_parallel_edges requires at least 2 targets."""
        graph = StateGraph()
        graph.add_node("start", MockWorkerNode("start"))
        graph.add_node("worker_1", MockWorkerNode("worker_1"))

        from mesh.utils.errors import GraphValidationError

        with pytest.raises(GraphValidationError, match="at least 2 targets"):
            graph.add_parallel_edges("start", ["worker_1"])

    def test_add_fan_in_edge(self):
        """Test adding fan-in edge."""
        graph = StateGraph()
        graph.add_node("worker_1", MockWorkerNode("worker_1"))
        graph.add_node("worker_2", MockWorkerNode("worker_2"))
        graph.add_node("worker_3", MockWorkerNode("worker_3"))
        graph.add_node("aggregator", MockWorkerNode("aggregator"))

        result = graph.add_fan_in_edge(
            ["worker_1", "worker_2", "worker_3"],
            "aggregator",
        )

        assert result is graph  # Returns self for chaining

        # Check that edges were created
        sources = [e.source for e in graph._edges if e.target == "aggregator"]
        assert "worker_1" in sources
        assert "worker_2" in sources
        assert "worker_3" in sources

        # Check fan-in was recorded
        assert "aggregator" in graph._fan_in_nodes
        assert graph._fan_in_nodes["aggregator"] == ["worker_1", "worker_2", "worker_3"]

    def test_add_fan_in_edge_with_aggregator(self):
        """Test adding fan-in edge with custom aggregator."""
        graph = StateGraph()
        graph.add_node("worker_1", MockWorkerNode("worker_1"))
        graph.add_node("worker_2", MockWorkerNode("worker_2"))
        graph.add_node("aggregator", MockWorkerNode("aggregator"))

        def custom_agg(results):
            return {"combined": list(results.values())}

        graph.add_fan_in_edge(
            ["worker_1", "worker_2"],
            "aggregator",
            aggregator=custom_agg,
        )

        assert "aggregator" in graph._fan_in_aggregators
        assert graph._fan_in_aggregators["aggregator"] is custom_agg

    def test_add_fan_in_edge_requires_two_sources(self):
        """Test that add_fan_in_edge requires at least 2 sources."""
        graph = StateGraph()
        graph.add_node("worker_1", MockWorkerNode("worker_1"))
        graph.add_node("aggregator", MockWorkerNode("aggregator"))

        from mesh.utils.errors import GraphValidationError

        with pytest.raises(GraphValidationError, match="at least 2 sources"):
            graph.add_fan_in_edge(["worker_1"], "aggregator")

    def test_parallel_config_in_compiled_graph(self):
        """Test that parallel config is passed to compiled graph."""
        graph = StateGraph()
        graph.add_node("start", MockWorkerNode("start"))
        graph.add_node("worker_1", MockWorkerNode("worker_1"))
        graph.add_node("worker_2", MockWorkerNode("worker_2"))
        graph.add_node("aggregator", MockWorkerNode("aggregator"))

        graph.add_edge("START", "start")
        graph.add_parallel_edges("start", ["worker_1", "worker_2"])
        graph.add_fan_in_edge(["worker_1", "worker_2"], "aggregator")

        graph.set_entry_point("start")

        compiled = graph.compile()

        assert len(compiled.parallel_branches) == 1
        assert "aggregator" in compiled.fan_in_nodes


# =============================================================================
# Aggregator Function Tests
# =============================================================================


class TestAggregators:
    """Tests for aggregator functions."""

    def test_default_aggregator_merges_dicts(self):
        """Test that default aggregator merges dict results."""
        results = {
            "worker_1": {"finding_1": "a"},
            "worker_2": {"finding_2": "b"},
            "worker_3": {"finding_3": "c"},
        }

        aggregated = default_aggregator(results)

        assert aggregated == {
            "finding_1": "a",
            "finding_2": "b",
            "finding_3": "c",
        }

    def test_default_aggregator_with_non_dicts(self):
        """Test default aggregator with non-dict results."""
        results = {
            "worker_1": "string_result",
            "worker_2": 123,
            "worker_3": ["list", "result"],
        }

        aggregated = default_aggregator(results)

        assert "parallel_results" in aggregated
        assert aggregated["parallel_results"] == results

    def test_default_aggregator_empty(self):
        """Test default aggregator with empty input."""
        assert default_aggregator({}) == {}

    def test_list_aggregator(self):
        """Test list aggregator."""
        results = {
            "worker_1": {"data": 1},
            "worker_2": {"data": 2},
        }

        aggregated = list_aggregator(results)

        assert "results" in aggregated
        assert len(aggregated["results"]) == 2

    def test_keyed_aggregator(self):
        """Test keyed aggregator."""
        results = {
            "researcher_1": {"findings": ["a"]},
            "researcher_2": {"findings": ["b"]},
        }

        aggregator = keyed_aggregator("research_results")
        aggregated = aggregator(results)

        assert "research_results" in aggregated
        assert aggregated["research_results"] == results

    def test_keyed_aggregator_custom_key(self):
        """Test keyed aggregator with custom key."""
        results = {"a": 1, "b": 2}

        aggregator = keyed_aggregator("custom_key")
        aggregated = aggregator(results)

        assert "custom_key" in aggregated


# =============================================================================
# Integration Tests
# =============================================================================


class TestParallelIntegration:
    """Integration tests for parallel execution."""

    def test_fan_out_fan_in_graph_structure(self):
        """Test that fan-out/fan-in creates correct graph structure."""
        graph = StateGraph()

        # Add nodes
        graph.add_node("dispatcher", MockWorkerNode("dispatcher"))
        graph.add_node("worker_1", MockWorkerNode("worker_1"))
        graph.add_node("worker_2", MockWorkerNode("worker_2"))
        graph.add_node("worker_3", MockWorkerNode("worker_3"))
        graph.add_node("collector", MockWorkerNode("collector"))

        # Build fan-out/fan-in pattern
        graph.add_edge("START", "dispatcher")
        graph.add_parallel_edges("dispatcher", ["worker_1", "worker_2", "worker_3"])
        graph.add_fan_in_edge(["worker_1", "worker_2", "worker_3"], "collector")

        graph.set_entry_point("dispatcher")

        # Compile and verify structure
        compiled = graph.compile()

        # Dispatcher should have 3 children
        children = compiled.get_children("dispatcher")
        assert len(children) == 3

        # Collector should have 3 parents
        parents = compiled.get_parents("collector")
        assert len(parents) == 3

    def test_parallel_error_handling_strategies(self):
        """Test different error handling strategies."""
        # Just verify configs are valid
        configs = [
            ParallelConfig(error_strategy=ParallelErrorStrategy.FAIL_FAST),
            ParallelConfig(error_strategy=ParallelErrorStrategy.CONTINUE_ALL),
            ParallelConfig(error_strategy=ParallelErrorStrategy.CONTINUE_PARTIAL),
        ]

        for config in configs:
            executor = ParallelExecutor(config)
            assert executor.config.error_strategy is not None

    @pytest.mark.asyncio
    async def test_parallel_continue_partial_all_fail(self):
        """Test CONTINUE_PARTIAL raises when all branches fail."""
        config = ParallelConfig(error_strategy=ParallelErrorStrategy.CONTINUE_PARTIAL)
        executor = ParallelExecutor(config)
        context = create_test_context()

        async def always_fail(node_id: str, input_data: Dict, ctx):
            raise RuntimeError(f"{node_id} failed")

        branches = [
            ("worker_1", {}),
            ("worker_2", {}),
        ]

        with pytest.raises(ParallelExecutionError, match="All parallel branches failed"):
            await executor.execute_parallel(branches, always_fail, context)
