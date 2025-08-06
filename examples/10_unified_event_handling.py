#!/usr/bin/env python3
"""
Example showing unified event handling across different execution modes.

This demonstrates how the new EventHandler provides consistent observability
regardless of whether you use GraphExecutor or StreamingGraphExecutor.
"""

import asyncio
from typing import Any, Dict

from mesh import Edge, Graph
from mesh.compilation import (
    EventCollector,
    EventHandler,
    GraphExecutor,
    StaticCompiler,
    StreamChunk,
    StreamingGraphExecutor,
)
from mesh.core.events import EventType
from mesh.utils import print_event, EventPrinter, format_event_summary
from mesh.nodes import ConditionalNode, CustomFunctionNode, LLMNode, ToolNode
from mesh.nodes.llm import LLMConfig, LLMProvider
from mesh.nodes.tool import ToolNodeConfig


async def demonstrate_unified_events():
    """Show how both executors emit the same events."""
    
    # Create a simple graph
    graph = Graph()
    
    # Data preprocessing node
    def preprocess(data, state):
        value = data.get("value", 0)
        return {"processed_value": value * 2}
    
    process = CustomFunctionNode(preprocess)
    
    # Tool node for calculation (terminal node)
    def calculate(processed_value):
        return {"result": processed_value + 10}
    
    calc = ToolNode(
        tool_func=calculate,
        config=ToolNodeConfig(tool_name="calculator", output_key="calculation"),
        extract_args=lambda d: {"processed_value": d.get("processed_value", 0)}
    )
    
    # Add nodes and edges
    graph.add_node(process)
    graph.add_node(calc)
    graph.add_edge(Edge(process.id, calc.id))
    
    # Compile the graph
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)
    
    print("=" * 60)
    print("UNIFIED EVENT HANDLING DEMONSTRATION")
    print("=" * 60)
    
    # ========================================
    # Option 1: GraphExecutor with Events
    # ========================================
    print("\n1. GraphExecutor with EventHandler:")
    print("-" * 40)
    
    # Create event handler with utility print function
    event_handler1 = EventHandler()
    event_handler1.add_listener(print_event)
    
    # Also collect events for analysis
    collector1 = EventCollector()
    event_handler1.add_listener(collector1)
    
    # Execute with event handler
    executor1 = GraphExecutor(event_handler=event_handler1)
    result1 = await executor1.execute(compiled, initial_input={"value": 5})
    
    print(f"\nResult: {result1.get_final_output()}")
    print(f"Total events collected: {len(collector1.events)}")
    
    # Show event summary using utility
    print("\nEvent Summary:")
    print(format_event_summary(collector1.events))
    
    # ========================================
    # Option 2: StreamingGraphExecutor with Events
    # ========================================
    print("\n2. StreamingGraphExecutor with EventHandler:")
    print("-" * 40)
    
    # Create event handler using EventPrinter utility class
    event_handler2 = EventHandler()
    printer = EventPrinter(prefix="  ")  # Use the utility class
    event_handler2.add_listener(printer)
    
    collector2 = EventCollector()
    event_handler2.add_listener(collector2)
    
    # Execute with streaming executor
    executor2 = StreamingGraphExecutor(event_handler=event_handler2)
    
    print("  Streaming execution:")
    execution_result = None
    chunks = []
    
    async for item in executor2.execute_streaming(compiled, initial_input={"value": 5}):
        if isinstance(item, StreamChunk):
            chunks.append(item)
            print(f"  📝 Stream chunk from {item.node_name}: {item.content}")
        else:
            # ExecutionResult at the end
            execution_result = item
    
    print(f"\nResult: {execution_result.get_final_output()}")
    print(f"Total events collected: {len(collector2.events)}")
    print(f"Stream chunks received: {len(chunks)}")
    
    # ========================================
    # Option 3: No Event Handler (Still Works!)
    # ========================================
    print("\n3. Executors without EventHandler (no events):")
    print("-" * 40)
    
    # Both executors work fine without event handlers
    executor3 = GraphExecutor()  # No event handler
    result3 = await executor3.execute(compiled, initial_input={"value": 5})
    print(f"  GraphExecutor result: {result3.get_final_output()}")
    
    executor4 = StreamingGraphExecutor()  # No event handler
    async for item in executor4.execute_streaming(compiled, initial_input={"value": 5}):
        if not isinstance(item, StreamChunk):
            print(f"  StreamingGraphExecutor result: {item.get_final_output()}")
    
    # ========================================
    # Compare Events from Both Executors
    # ========================================
    print("\n4. Event Comparison:")
    print("-" * 40)
    
    event_types1 = [e.type for e in collector1.events]
    event_types2 = [e.type for e in collector2.events]
    
    print(f"  GraphExecutor events: {event_types1}")
    print(f"  StreamingGraphExecutor events: {event_types2}")
    
    # Core events should be the same
    core_events1 = {e.type for e in collector1.events}
    core_events2 = {e.type for e in collector2.events}
    
    if core_events1 == core_events2:
        print("  ✅ Both executors emit the same event types!")
    else:
        print(f"  ⚠️ Event types differ:")
        print(f"    Only in GraphExecutor: {core_events1 - core_events2}")
        print(f"    Only in StreamingGraphExecutor: {core_events2 - core_events1}")
    
    # ========================================
    # Advanced: Multiple Event Listeners
    # ========================================
    print("\n5. Multiple Event Listeners:")
    print("-" * 40)
    
    event_handler = EventHandler()
    
    # Add multiple listeners for different purposes
    metrics = {"node_count": 0, "total_time": 0}
    
    def count_nodes(event):
        if event.type == EventType.NODE_END:
            metrics["node_count"] += 1
            metrics["total_time"] += event.data.get("execution_time", 0)
    
    async def log_errors(event):
        if event.type == EventType.NODE_ERROR:
            print(f"  ❌ ERROR in {event.node_name}: {event.data.get('error')}")
    
    event_handler.add_listener(count_nodes)
    event_handler.add_listener(log_errors)
    event_handler.add_listener(print_event)
    
    executor = GraphExecutor(event_handler=event_handler)
    await executor.execute(compiled, initial_input={"value": 5})
    
    print(f"  Metrics: {metrics['node_count']} nodes executed in {metrics['total_time']:.3f}s total")


async def demonstrate_streaming_with_llm():
    """Show how streaming LLM content works with events."""
    
    print("\n" + "=" * 60)
    print("STREAMING LLM WITH EVENTS")
    print("=" * 60)
    
    # Note: This requires an actual LLM API key
    # For demonstration, we'll use a mock that simulates streaming
    
    graph = Graph()
    
    # Mock LLM node that simulates streaming
    class MockStreamingLLM(LLMNode):
        async def run(self, input_data: Dict[str, Any], state=None):
            # Simulate streaming by yielding chunks
            response = "Hello! This is a simulated streaming response."
            
            # In a real LLM node with streaming enabled, chunks would be 
            # yielded as StreamChunk objects automatically
            return {"response": response}
    
    llm = MockStreamingLLM(
        config=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            api_key="mock-key",
            stream=True,  # Enable streaming
        )
    )
    
    graph.add_node(llm)
    
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)
    
    # Create event handler
    event_handler = EventHandler()
    
    # Track both events and chunks
    events = []
    chunks = []
    
    event_handler.add_listener(lambda e: events.append(e.type))
    
    executor = StreamingGraphExecutor(event_handler=event_handler)
    
    print("\nStreaming execution with LLM:")
    print("-" * 40)
    
    async for item in executor.execute_streaming(compiled, initial_input={"prompt": "Hello!"}):
        if isinstance(item, StreamChunk):
            chunks.append(item)
            print(f"Chunk: '{item.content}'", end=" ")
        else:
            print(f"\n\nFinal result received")
    
    print(f"\nEvents emitted: {events}")
    print(f"Stream chunks: {len(chunks)}")


if __name__ == "__main__":
    asyncio.run(demonstrate_unified_events())
    # Uncomment to see LLM streaming example
    # asyncio.run(demonstrate_streaming_with_llm())