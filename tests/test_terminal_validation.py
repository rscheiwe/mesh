"""Tests for terminal node validation."""

import pytest

from mesh import Edge, Graph
from mesh.compilation import StaticCompiler
from mesh.nodes import (
    ConditionalNode,
    CustomFunctionNode,
    LLMNode,
    ToolNode,
)
from mesh.nodes.llm import LLMConfig, LLMProvider
from mesh.nodes.tool import ToolNodeConfig


class TestTerminalNodeValidation:
    """Test terminal node validation."""

    def test_llm_node_can_be_terminal(self):
        """Test that LLM nodes can be terminal."""
        graph = Graph()
        
        llm = LLMNode(
            config=LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-3.5-turbo",
                api_key="test-key",
            )
        )
        
        graph.add_node(llm)
        
        # Should validate successfully
        errors = graph.validate()
        assert len(errors) == 0
        
        # Check the property
        assert llm.can_be_terminal is True

    def test_tool_node_can_be_terminal(self):
        """Test that Tool nodes can be terminal."""
        graph = Graph()
        
        def dummy_tool():
            return "result"
        
        tool = ToolNode(
            tool_func=dummy_tool,
            config=ToolNodeConfig(tool_name="test_tool"),
        )
        
        graph.add_node(tool)
        
        # Should validate successfully
        errors = graph.validate()
        assert len(errors) == 0
        
        # Check the property
        assert tool.can_be_terminal is True

    def test_custom_function_node_cannot_be_terminal(self):
        """Test that CustomFunctionNode cannot be terminal."""
        graph = Graph()
        
        custom = CustomFunctionNode(lambda data, state: {"result": "test"})
        
        graph.add_node(custom)
        
        # Should fail validation
        errors = graph.validate()
        assert len(errors) == 1
        assert "cannot be a terminal node" in errors[0]
        assert "CustomFunctionNode" in errors[0]
        
        # Check the property
        assert custom.can_be_terminal is False

    def test_conditional_node_cannot_be_terminal(self):
        """Test that ConditionalNode cannot be terminal."""
        graph = Graph()
        
        conditional = ConditionalNode(
            condition=lambda data, state: data.get("value", 0) > 10,
            true_output={"result": "high"},
            false_output={"result": "low"},
        )
        
        graph.add_node(conditional)
        
        # Should fail validation
        errors = graph.validate()
        assert len(errors) == 1
        assert "cannot be a terminal node" in errors[0]
        
        # Check the property
        assert conditional.can_be_terminal is False

    def test_mixed_graph_validates_correctly(self):
        """Test a graph with both valid and invalid terminal nodes."""
        graph = Graph()
        
        # Starting node (CustomFunctionNode - ok as starting node)
        start = CustomFunctionNode(lambda data, state: {"processed": data})
        
        # Terminal nodes
        llm = LLMNode(
            config=LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-3.5-turbo",
                api_key="test-key",
            )
        )
        
        # Another CustomFunctionNode as terminal (should fail)
        end_custom = CustomFunctionNode(lambda data, state: {"final": data})
        
        graph.add_node(start)
        graph.add_node(llm)
        graph.add_node(end_custom)
        
        # Create edges: start -> llm (branch 1), start -> end_custom (branch 2)
        graph.add_edge(Edge(start.id, llm.id))
        graph.add_edge(Edge(start.id, end_custom.id))
        
        # Should fail validation because end_custom is terminal but not allowed
        errors = graph.validate()
        assert len(errors) == 1
        assert "cannot be a terminal node" in errors[0]
        assert end_custom.id in errors[0] or "CustomFunctionNode" in errors[0]

    @pytest.mark.asyncio
    async def test_compiler_rejects_invalid_terminal_nodes(self):
        """Test that the compiler rejects graphs with invalid terminal nodes."""
        graph = Graph()
        
        # Create a simple graph with CustomFunctionNode as terminal
        process = CustomFunctionNode(lambda data, state: {"result": data})
        graph.add_node(process)
        
        compiler = StaticCompiler()
        
        # Should raise ValueError during compilation
        with pytest.raises(ValueError, match="cannot be a terminal node"):
            await compiler.compile(graph)

    def test_valid_terminal_nodes_in_chain(self):
        """Test that intermediate nodes don't need can_be_terminal=True."""
        graph = Graph()
        
        # Chain: custom -> tool (only tool needs to be terminal-capable)
        custom = CustomFunctionNode(lambda data, state: {"processed": data})
        tool = ToolNode(
            tool_func=lambda x: x,
            config=ToolNodeConfig(tool_name="final_tool"),
        )
        
        graph.add_node(custom)
        graph.add_node(tool)
        graph.add_edge(Edge(custom.id, tool.id))
        
        # Should validate successfully - custom is not terminal
        errors = graph.validate()
        assert len(errors) == 0
        
        # Verify properties
        assert custom.can_be_terminal is False
        assert tool.can_be_terminal is True