"""Tests for graph execution."""

import asyncio

import pytest

from mesh import Edge, Graph
from mesh.compilation import GraphExecutor, StaticCompiler
from mesh.core.edge import EdgeType
from mesh.nodes import ConditionalNode, CustomFunctionNode
from mesh.state import GraphState


class TestExecution:
    """Test graph execution functionality."""

    @pytest.mark.asyncio
    async def test_basic_execution(self):
        """Test basic graph execution."""
        graph = Graph()

        # Use ToolNode instead of CustomFunctionNode for terminal node
        from mesh.nodes import ToolNode
        from mesh.nodes.tool import ToolNodeConfig
        
        def double_value(value):
            return value * 2
        
        process = ToolNode(
            tool_func=double_value,
            config=ToolNodeConfig(tool_name="doubler", output_key="result"),
            extract_args=lambda data: {"value": data.get("value", 0)}
        )

        graph.add_node(process)

        compiler = StaticCompiler()
        compiled = await compiler.compile(graph)

        executor = GraphExecutor()
        result = await executor.execute(compiled, initial_input={"value": 21})

        assert result.success
        assert result.get_final_output()["result"] == 42

    @pytest.mark.asyncio
    async def test_terminal_node_detection(self):
        """Test that nodes without outgoing edges are terminal."""
        graph = Graph()

        # Use ToolNode for terminal branches
        from mesh.nodes import ToolNode
        from mesh.nodes.tool import ToolNodeConfig
        
        branch1 = ToolNode(
            tool_func=lambda value: {"branch": 1, "value": value},
            config=ToolNodeConfig(tool_name="branch1", output_key="result"),
            extract_args=lambda d: {"value": d.get("value")}
        )
        branch2 = ToolNode(
            tool_func=lambda value: {"branch": 2, "value": value},
            config=ToolNodeConfig(tool_name="branch2", output_key="result"),
            extract_args=lambda d: {"value": d.get("value")}
        )

        graph.add_node(branch1)
        graph.add_node(branch2)

        compiler = StaticCompiler()
        compiled = await compiler.compile(graph)

        executor = GraphExecutor()
        result = await executor.execute(compiled, initial_input={"value": "test"})

        # Both terminal nodes should have outputs
        assert len(result.terminal_outputs) == 2
        # ToolNode stores the result in the 'result' key
        outputs = list(result.terminal_outputs.values())
        # Check that both branches produced output
        assert any(out.get("result", {}).get("branch") == 1 for out in outputs)
        assert any(out.get("result", {}).get("branch") == 2 for out in outputs)

    @pytest.mark.asyncio
    async def test_max_loops_protection(self):
        """Test that max_loops prevents infinite loops."""
        graph = Graph()

        # Create an initial node and a counter with self-loop
        init = CustomFunctionNode(lambda d, s: {"count": 0})
        counter = CustomFunctionNode(lambda d, s: {"count": d.get("count", 0) + 1})

        graph.add_node(init)
        graph.add_node(counter)

        # Connect init to counter
        graph.add_edge(Edge(init.id, counter.id))

        # Add self-loop on counter
        graph.add_edge(Edge(counter.id, counter.id, edge_type=EdgeType.LOOP))

        compiler = StaticCompiler()
        compiled = await compiler.compile(graph)

        # Execute with low max_loops
        executor = GraphExecutor(max_loops=5)
        result = await executor.execute(compiled, initial_input={"count": 0})

        # Should have executed exactly 5 times
        counter_output = result.outputs[counter.id]
        assert counter_output.data["count"] == 5

    @pytest.mark.asyncio
    async def test_state_persistence(self):
        """Test state persistence across node executions."""
        graph = Graph()

        async def increment_state(data, state):
            if state:
                count = await state.get("total", 0)
                await state.set("total", count + 1)
            return {"incremented": True}

        # Use ToolNode for the terminal node
        from mesh.nodes import ToolNode
        from mesh.nodes.tool import ToolNodeConfig
        
        inc1 = CustomFunctionNode(increment_state)
        inc2 = CustomFunctionNode(increment_state)
        # Only the last one needs to be a ToolNode (terminal)
        inc3 = ToolNode(
            tool_func=lambda: {"incremented": True},
            config=ToolNodeConfig(tool_name="final_increment", store_in_state=False),
            extract_args=lambda data: {}
        )
        # Still call increment_state for inc3
        async def inc3_wrapper(data, state):
            await increment_state(data, state)
            return {"incremented": True}
        inc3.tool_func = inc3_wrapper

        # Chain nodes
        for node in [inc1, inc2, inc3]:
            graph.add_node(node)

        graph.add_edge(Edge(inc1.id, inc2.id))
        graph.add_edge(Edge(inc2.id, inc3.id))

        compiler = StaticCompiler()
        compiled = await compiler.compile(graph)

        state = GraphState()
        executor = GraphExecutor()
        result = await executor.execute(compiled, state=state)

        # Should have incremented 3 times
        assert await state.get("total") == 3

    @pytest.mark.asyncio
    async def test_conditional_execution(self):
        """Test conditional edge execution."""
        graph = Graph()

        condition = ConditionalNode(
            condition=lambda d, s: d.get("value", 0) > 10,
            true_output={"path": "high"},
            false_output={"path": "low"},
        )

        # Use ToolNode for terminal nodes
        from mesh.nodes import ToolNode
        from mesh.nodes.tool import ToolNodeConfig
        
        high_node = ToolNode(
            tool_func=lambda: {"result": "high path"},
            config=ToolNodeConfig(tool_name="high_path"),
            extract_args=lambda d: {}
        )
        low_node = ToolNode(
            tool_func=lambda: {"result": "low path"},
            config=ToolNodeConfig(tool_name="low_path"),
            extract_args=lambda d: {}
        )

        graph.add_node(condition)
        graph.add_node(high_node)
        graph.add_node(low_node)

        # Conditional edges
        graph.add_edge(
            Edge(
                condition.id,
                high_node.id,
                edge_type=EdgeType.CONDITIONAL,
                condition=lambda d: d.get("branch") == "true",
            )
        )

        graph.add_edge(
            Edge(
                condition.id,
                low_node.id,
                edge_type=EdgeType.CONDITIONAL,
                condition=lambda d: d.get("branch") == "false",
            )
        )

        compiler = StaticCompiler()
        compiled = await compiler.compile(graph)
        executor = GraphExecutor()

        # Test high path
        result_high = await executor.execute(compiled, initial_input={"value": 15})
        high_output = result_high.outputs.get(high_node.id)
        assert high_output and high_output.data["result"] == "high path"

        # Test low path
        result_low = await executor.execute(compiled, initial_input={"value": 5})
        low_output = result_low.outputs.get(low_node.id)
        assert low_output and low_output.data["result"] == "low path"

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test that independent nodes execute in parallel."""
        execution_order = []

        async def slow_node(name, delay):
            async def process(data, state):
                execution_order.append(f"{name}_start")
                await asyncio.sleep(delay)
                execution_order.append(f"{name}_end")
                return {"node": name}

            return process

        graph = Graph()

        # Use ToolNode for terminal nodes that can run in parallel
        from mesh.nodes import ToolNode
        from mesh.nodes.tool import ToolNodeConfig
        
        node1 = ToolNode(
            tool_func=await slow_node("node1", 0.1),
            config=ToolNodeConfig(tool_name="node1"),
            extract_args=lambda d: {}
        )
        node2 = ToolNode(
            tool_func=await slow_node("node2", 0.1),
            config=ToolNodeConfig(tool_name="node2"),
            extract_args=lambda d: {}
        )

        graph.add_node(node1)
        graph.add_node(node2)

        compiler = StaticCompiler()
        compiled = await compiler.compile(graph)

        # Verify they're in the same execution group (first group now since no start node)
        assert len(compiled.execution_plan[0]) == 2  # Both in first group

        executor = GraphExecutor()
        result = await executor.execute(compiled)

        # If parallel, starts should be before any ends
        assert execution_order.index("node1_start") < execution_order.index("node1_end")
        assert execution_order.index("node2_start") < execution_order.index("node2_end")
        # Parallel execution: both starts before any end
        assert execution_order.index("node1_start") < execution_order.index("node2_end")
        assert execution_order.index("node2_start") < execution_order.index("node1_end")
