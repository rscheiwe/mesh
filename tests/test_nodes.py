"""Tests for node implementations."""

import asyncio

import pytest

from mesh.nodes import (
    ConditionalNode,
    CustomFunctionNode,
    LLMNode,
    MultiToolNode,
    ToolNode,
)
from mesh.nodes.llm import LLMConfig, LLMProvider
from mesh.nodes.tool import ToolNodeConfig
from mesh.state import GraphState


class TestNodes:
    """Test various node types."""

    @pytest.mark.asyncio
    async def test_custom_function_node(self):
        """Test CustomFunctionNode."""

        def double_value(data, state):
            return {"result": data.get("value", 0) * 2}

        node = CustomFunctionNode(double_value)
        result = await node.run({"value": 21})

        assert result.data["result"] == 42

    @pytest.mark.asyncio
    async def test_async_custom_function(self):
        """Test CustomFunctionNode with async function."""

        async def async_process(data, state):
            await asyncio.sleep(0.01)  # Simulate async work
            return {"processed": True, "value": data.get("value")}

        node = CustomFunctionNode(async_process)
        result = await node.run({"value": "test"})

        assert result.data["processed"] is True
        assert result.data["value"] == "test"

    @pytest.mark.asyncio
    async def test_conditional_node(self):
        """Test ConditionalNode branching."""
        node = ConditionalNode(
            condition=lambda data, state: data.get("value", 0) > 10,
            true_output={"branch": "high"},
            false_output={"branch": "low"},
        )

        # Test true condition
        result_true = await node.run({"value": 15})
        assert result_true.data["branch"] == "true"
        assert result_true.data["result"] == {"branch": "high"}

        # Test false condition
        result_false = await node.run({"value": 5})
        assert result_false.data["branch"] == "false"
        assert result_false.data["result"] == {"branch": "low"}

    @pytest.mark.asyncio
    async def test_tool_node(self):
        """Test ToolNode execution."""

        def add_numbers(a: int, b: int) -> int:
            return a + b

        node = ToolNode(
            tool_func=add_numbers,
            config=ToolNodeConfig(tool_name="adder", output_key="sum"),
            extract_args=lambda data: {"a": data.get("x", 0), "b": data.get("y", 0)},
        )

        result = await node.run({"x": 10, "y": 32})

        assert result.data["sum"] == 42
        assert result.data["tool_name"] == "adder"
        assert result.data["success"] is True

    @pytest.mark.asyncio
    async def test_tool_node_error_handling(self):
        """Test ToolNode error handling."""

        def failing_tool():
            raise ValueError("Tool failed")

        node = ToolNode(
            tool_func=failing_tool, config=ToolNodeConfig(tool_name="failing")
        )

        result = await node.run({})

        assert result.data["success"] is False
        assert "Tool failed" in result.data["error"]
        assert result.data["tool_result"] is None

    @pytest.mark.asyncio
    async def test_multi_tool_node_parallel(self):
        """Test MultiToolNode parallel execution."""

        async def tool1(value):
            await asyncio.sleep(0.01)
            return value * 2

        async def tool2(value):
            await asyncio.sleep(0.01)
            return value + 10

        node = MultiToolNode(
            tools=[("double", tool1), ("add_ten", tool2)], parallel=True
        )

        result = await node.run({"tool_args": {"value": 5}})

        assert result.data["tool_results"]["double"] == 10
        assert result.data["tool_results"]["add_ten"] == 15
        assert len(result.data["tools_executed"]) == 2

    @pytest.mark.asyncio
    async def test_multi_tool_node_sequential(self):
        """Test MultiToolNode sequential execution."""
        call_order = []

        def tool1():
            call_order.append(1)
            return "first"

        def tool2():
            call_order.append(2)
            return "second"

        node = MultiToolNode(tools=[tool1, tool2], parallel=False)

        result = await node.run({"tool_args": {}})

        # Verify sequential execution
        assert call_order == [1, 2]
        assert result.data["tool_results"]["tool_0"] == "first"
        assert result.data["tool_results"]["tool_1"] == "second"

    def test_llm_node_mock(self):
        """Test LLMNode with mock."""
        # We'll test the structure, not actual API calls
        node = LLMNode(
            config=LLMConfig(
                provider=LLMProvider.OPENAI, model="gpt-3.5-turbo", api_key="mock-key"
            )
        )

        assert node.llm_config.provider == LLMProvider.OPENAI
        assert node.llm_config.model == "gpt-3.5-turbo"
        assert node.llm_config.stream is False  # Default
        assert node.llm_config.use_async is True  # Default


class TestNodeWithState:
    """Test nodes that interact with state."""

    @pytest.mark.asyncio
    async def test_node_state_interaction(self):
        """Test node reading and writing to state."""

        async def stateful_node(data, state):
            if state:
                counter = await state.get("counter", 0)
                await state.set("counter", counter + 1)
                return {"counter": counter + 1}
            return {"counter": 1}

        node = CustomFunctionNode(stateful_node)
        state = GraphState()

        # First execution
        result1 = await node.run({}, state)
        assert result1.data["counter"] == 1

        # Second execution
        result2 = await node.run({}, state)
        assert result2.data["counter"] == 2

        # Verify state
        assert await state.get("counter") == 2

    @pytest.mark.asyncio
    async def test_tool_node_state_storage(self):
        """Test ToolNode storing results in state."""

        def compute_value(x):
            return x * x

        node = ToolNode(
            tool_func=compute_value,
            config=ToolNodeConfig(
                tool_name="square",
                output_key="squared",
                store_in_state=True,
                state_key="last_square",
            ),
            extract_args=lambda data: {"x": data.get("number", 0)},
        )

        state = GraphState()
        result = await node.run({"number": 7}, state)

        assert result.data["squared"] == 49
        assert await state.get("last_square") == 49
