"""Tests for event handler system."""

import asyncio
from typing import List

import pytest

from mesh import Edge, Graph
from mesh.compilation import (
    EventCollector,
    EventHandler,
    GraphExecutor,
    StaticCompiler,
    StreamChunk,
    StreamingGraphExecutor,
)
from mesh.core.events import EventType, GraphEndEvent, GraphStartEvent, NodeEndEvent, NodeStartEvent
from mesh.nodes import LLMNode, ToolNode
from mesh.nodes.llm import LLMConfig, LLMProvider
from mesh.nodes.tool import ToolNodeConfig


class TestEventHandler:
    """Test event handler functionality."""

    @pytest.mark.asyncio
    async def test_event_handler_with_graph_executor(self):
        """Test that GraphExecutor emits events when handler is provided."""
        # Create a simple graph
        graph = Graph()
        
        def dummy_tool(value):
            return {"result": value * 2}
        
        tool = ToolNode(
            tool_func=dummy_tool,
            config=ToolNodeConfig(tool_name="doubler", output_key="result"),
            extract_args=lambda d: {"value": d.get("value", 0)}
        )
        
        graph.add_node(tool)
        
        # Compile the graph
        compiler = StaticCompiler()
        compiled = await compiler.compile(graph)
        
        # Create event handler and collector
        event_handler = EventHandler()
        collector = EventCollector()
        event_handler.add_listener(collector)
        
        # Execute with event handler
        executor = GraphExecutor(event_handler=event_handler)
        result = await executor.execute(compiled, initial_input={"value": 21})
        
        # Verify execution succeeded
        assert result.success
        output = result.get_final_output()
        # ToolNode wraps the result
        assert output.get("result").get("result") == 42
        
        # Verify events were emitted
        assert len(collector.events) > 0
        
        # Check for specific event types
        graph_starts = collector.get_events_by_type(EventType.GRAPH_START)
        assert len(graph_starts) == 1
        
        graph_ends = collector.get_events_by_type(EventType.GRAPH_END)
        assert len(graph_ends) == 1
        
        node_starts = collector.get_events_by_type(EventType.NODE_START)
        assert len(node_starts) == 1
        
        node_ends = collector.get_events_by_type(EventType.NODE_END)
        assert len(node_ends) == 1

    @pytest.mark.asyncio
    async def test_event_handler_with_streaming_executor(self):
        """Test that StreamingGraphExecutor emits events and yields chunks."""
        # Create a simple graph
        graph = Graph()
        
        def dummy_tool(value):
            return {"result": value * 2}
        
        tool = ToolNode(
            tool_func=dummy_tool,
            config=ToolNodeConfig(tool_name="doubler", output_key="result"),
            extract_args=lambda d: {"value": d.get("value", 0)}
        )
        
        graph.add_node(tool)
        
        # Compile the graph
        compiler = StaticCompiler()
        compiled = await compiler.compile(graph)
        
        # Create event handler and collector
        event_handler = EventHandler()
        collector = EventCollector()
        event_handler.add_listener(collector)
        
        # Execute with streaming executor
        executor = StreamingGraphExecutor(event_handler=event_handler)
        
        chunks = []
        execution_result = None
        
        async for item in executor.execute_streaming(compiled, initial_input={"value": 21}):
            if isinstance(item, StreamChunk):
                chunks.append(item)
            else:
                # Should be ExecutionResult at the end
                execution_result = item
        
        # Verify execution succeeded
        assert execution_result is not None
        assert execution_result.success
        output = execution_result.get_final_output()
        assert output.get("result").get("result") == 42
        
        # Verify events were emitted (not yielded as chunks)
        assert len(collector.events) > 0
        
        # Check for specific event types
        graph_starts = collector.get_events_by_type(EventType.GRAPH_START)
        assert len(graph_starts) == 1
        
        graph_ends = collector.get_events_by_type(EventType.GRAPH_END)
        assert len(graph_ends) == 1

    @pytest.mark.asyncio
    async def test_events_parity_between_executors(self):
        """Test that both executors emit the same events."""
        # Create a simple graph
        graph = Graph()
        
        def dummy_tool(value):
            return {"result": value * 2}
        
        tool = ToolNode(
            tool_func=dummy_tool,
            config=ToolNodeConfig(tool_name="doubler", output_key="result"),
            extract_args=lambda d: {"value": d.get("value", 0)}
        )
        
        graph.add_node(tool)
        
        # Compile the graph
        compiler = StaticCompiler()
        compiled = await compiler.compile(graph)
        
        # Collect events from GraphExecutor
        handler1 = EventHandler()
        collector1 = EventCollector()
        handler1.add_listener(collector1)
        
        executor1 = GraphExecutor(event_handler=handler1)
        result1 = await executor1.execute(compiled, initial_input={"value": 21})
        
        # Collect events from StreamingGraphExecutor
        handler2 = EventHandler()
        collector2 = EventCollector()
        handler2.add_listener(collector2)
        
        executor2 = StreamingGraphExecutor(event_handler=handler2)
        async for _ in executor2.execute_streaming(compiled, initial_input={"value": 21}):
            pass  # Just consume the stream
        
        # Both should have emitted the same types of events
        types1 = {e.type for e in collector1.events}
        types2 = {e.type for e in collector2.events}
        
        # Core events should be the same
        assert EventType.GRAPH_START in types1
        assert EventType.GRAPH_START in types2
        assert EventType.GRAPH_END in types1
        assert EventType.GRAPH_END in types2
        assert EventType.NODE_START in types1
        assert EventType.NODE_START in types2
        assert EventType.NODE_END in types1
        assert EventType.NODE_END in types2

    @pytest.mark.asyncio
    async def test_no_events_without_handler(self):
        """Test that executors work without event handlers."""
        # Create a simple graph
        graph = Graph()
        
        def dummy_tool(value):
            return {"result": value * 2}
        
        tool = ToolNode(
            tool_func=dummy_tool,
            config=ToolNodeConfig(tool_name="doubler", output_key="result"),
            extract_args=lambda d: {"value": d.get("value", 0)}
        )
        
        graph.add_node(tool)
        
        # Compile the graph
        compiler = StaticCompiler()
        compiled = await compiler.compile(graph)
        
        # Execute without event handler - should work fine
        executor1 = GraphExecutor()  # No event handler
        result1 = await executor1.execute(compiled, initial_input={"value": 21})
        assert result1.success
        output1 = result1.get_final_output()
        assert output1.get("result").get("result") == 42
        
        # Same for streaming executor
        executor2 = StreamingGraphExecutor()  # No event handler
        execution_result = None
        async for item in executor2.execute_streaming(compiled, initial_input={"value": 21}):
            if not isinstance(item, StreamChunk):
                execution_result = item
        
        assert execution_result is not None
        assert execution_result.success
        output2 = execution_result.get_final_output()
        assert output2.get("result").get("result") == 42

    @pytest.mark.asyncio
    async def test_multiple_event_listeners(self):
        """Test that multiple listeners receive events."""
        # Create a simple graph
        graph = Graph()
        
        def dummy_tool(value):
            return {"result": value * 2}
        
        tool = ToolNode(
            tool_func=dummy_tool,
            config=ToolNodeConfig(tool_name="doubler", output_key="result"),
            extract_args=lambda d: {"value": d.get("value", 0)}
        )
        
        graph.add_node(tool)
        
        # Compile the graph
        compiler = StaticCompiler()
        compiled = await compiler.compile(graph)
        
        # Create event handler with multiple listeners
        event_handler = EventHandler()
        
        collector1 = EventCollector()
        collector2 = EventCollector()
        custom_events = []
        
        event_handler.add_listener(collector1)
        event_handler.add_listener(collector2)
        event_handler.add_listener(lambda e: custom_events.append(e.type))
        
        # Execute with event handler
        executor = GraphExecutor(event_handler=event_handler)
        result = await executor.execute(compiled, initial_input={"value": 21})
        
        # All listeners should have received events
        assert len(collector1.events) > 0
        assert len(collector2.events) > 0
        assert len(custom_events) > 0
        
        # They should have received the same events
        assert len(collector1.events) == len(collector2.events)
        assert len(collector1.events) == len(custom_events)