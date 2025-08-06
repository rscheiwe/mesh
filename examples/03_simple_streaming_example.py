"""Simple streaming example with LLM node using the new event handler system."""

import asyncio

from mesh import Edge, Graph
from mesh.compilation import (
    EventHandler,
    StaticCompiler,
    StreamChunk,
    StreamingGraphExecutor,
    ExecutionResult,
)
from mesh.core.events import EventType
from mesh.nodes import LLMNode
from mesh.nodes.llm import LLMConfig, LLMProvider
from mesh.utils import print_event, EventPrinter


async def main():
    """Example of streaming LLM responses with new event handler."""

    print("=== Streaming Example with Event Handler ===\n")

    # Create graph
    graph = Graph()

    # LLM node configured for streaming
    llm = LLMNode(
        config=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            stream=True,  # Enable streaming
        )
    )

    # Build graph
    graph.add_node(llm)
    # LLM node has no incoming edges, so it becomes a starting node automatically

    # Compile graph
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)

    # Create event handler for monitoring (events are separate from chunks now)
    event_handler = EventHandler()
    
    # Track execution events (optional - only if you want to monitor events)
    events_received = []
    def track_event(event):
        events_received.append(event.type)
        # You could also print specific events here if desired
        # For example, to see when nodes start/end:
        if event.type in [EventType.NODE_START, EventType.NODE_END]:
            print(f"\n[Event: {event.type.value} - {event.node_name}]", end="\n\n", flush=True)
    
    event_handler.add_listener(track_event)

    # Create streaming executor with event handler
    executor = StreamingGraphExecutor(event_handler=event_handler)

    # Execute with streaming
    print("User: Write a short story about a robot learning to paint\n")
    print("Bot: ", end="", flush=True)

    # Stream the response - now yields StreamChunk objects for content
    execution_result = None
    async for item in executor.execute_streaming(
        compiled,
        initial_input={
            "prompt": "Write a 50-words-or-less short story about a robot learning to paint"
        },
    ):
        if isinstance(item, StreamChunk):
            # This is actual streaming content
            print(item.content, end="", flush=True)
        elif isinstance(item, ExecutionResult):
            # This is the final result at the end
            execution_result = item
            print("\n")  # New line after completion

    # Show what events were emitted (separate from streaming)
    print(f"\n[Events emitted during execution: {events_received}]")
    
    if execution_result:
        print(f"Execution successful: {execution_result.success}")

    print("\n=== Streaming Complete ===")


async def advanced_streaming_example():
    """Example with metrics tracking using the new system."""

    print("\n\n=== Advanced Streaming Example ===\n")

    # Create graph
    graph = Graph()
    llm = LLMNode(
        config=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            stream=True,
            max_tokens=200,  # Limit response length
        )
    )

    graph.add_node(llm)
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)
    
    # Create event handler for detailed monitoring
    event_handler = EventHandler()
    
    # Use the EventPrinter utility for clean event display
    printer = EventPrinter(prefix="  [Event] ", verbose=False)  # Only show errors
    event_handler.add_listener(printer)
    
    executor = StreamingGraphExecutor(event_handler=event_handler)

    # Track streaming metrics
    chunk_count = 0
    start_time = asyncio.get_event_loop().time()

    print("User: Explain quantum computing in simple terms\n")
    print("Bot: ", end="", flush=True)

    # Collect all chunks for analysis
    full_response = ""
    execution_result = None

    async for item in executor.execute_streaming(
        compiled, initial_input={"prompt": "Explain quantum computing in simple terms"}
    ):
        if isinstance(item, StreamChunk):
            # Handle streaming content
            print(item.content, end="", flush=True)
            full_response += item.content
            chunk_count += 1
            
        elif isinstance(item, ExecutionResult):
            # Handle final result
            execution_result = item
            elapsed = asyncio.get_event_loop().time() - start_time
            
            print("\n\nStreaming Stats:")
            print(f"- Total chunks: {chunk_count}")
            print(f"- Time elapsed: {elapsed:.2f}s")
            if elapsed > 0:
                print(f"- Chunks/second: {chunk_count / elapsed:.1f}")
            print(f"- Response length: {len(full_response)} characters")
            print(f"- Execution successful: {execution_result.success}")


async def comparison_example():
    """Show the difference between streaming and non-streaming execution."""
    
    print("\n\n=== Comparison: Streaming vs Non-Streaming ===\n")
    
    # Create graph with streaming disabled
    graph_no_stream = Graph()
    llm_no_stream = LLMNode(
        config=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            stream=False,  # Streaming disabled
        )
    )
    graph_no_stream.add_node(llm_no_stream)
    
    # Create graph with streaming enabled
    graph_stream = Graph()
    llm_stream = LLMNode(
        config=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            stream=True,  # Streaming enabled
        )
    )
    graph_stream.add_node(llm_stream)
    
    compiler = StaticCompiler()
    compiled_no_stream = await compiler.compile(graph_no_stream)
    compiled_stream = await compiler.compile(graph_stream)
    
    prompt = {"prompt": "What is 2+2?"}
    
    # Non-streaming execution (no chunks will be yielded)
    print("1. Non-streaming execution:")
    print("-" * 40)
    
    executor_no_stream = StreamingGraphExecutor()
    chunk_count = 0
    async for item in executor_no_stream.execute_streaming(compiled_no_stream, initial_input=prompt):
        if isinstance(item, StreamChunk):
            chunk_count += 1
        elif isinstance(item, ExecutionResult):
            response = item.outputs[llm_no_stream.id].data.get("response", "")
            print(f"Response: {response}")
            print(f"Chunks received: {chunk_count} (no streaming)")
    
    print("\n2. Streaming execution:")
    print("-" * 40)
    print("Response: ", end="", flush=True)
    
    executor_stream = StreamingGraphExecutor()
    chunk_count = 0
    async for item in executor_stream.execute_streaming(compiled_stream, initial_input=prompt):
        if isinstance(item, StreamChunk):
            print(item.content, end="", flush=True)
            chunk_count += 1
        elif isinstance(item, ExecutionResult):
            print(f"\nChunks received: {chunk_count} (with streaming)")


if __name__ == "__main__":
    # Run basic streaming example
    asyncio.run(main())

    # Run advanced example with metrics
    # asyncio.run(advanced_streaming_example())
    
    # Run comparison example
    # asyncio.run(comparison_example())