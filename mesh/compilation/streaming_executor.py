"""Streaming graph execution with event support."""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from mesh.compilation.compiler import CompiledGraph
from mesh.compilation.event_handler import EventHandler
from mesh.compilation.executor import ExecutionResult
from mesh.core.events import (
    ErrorEvent,
    Event,
    EventType,
    GraphEndEvent,
    GraphStartEvent,
    NodeEndEvent,
    NodeStartEvent,
    StreamChunkEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from mesh.core.node import Node, NodeOutput, NodeStatus
from mesh.nodes.llm import LLMNode
from mesh.nodes.tool import MultiToolNode, ToolNode
from mesh.state.state import GraphState


@dataclass
class StreamChunk:
    """Represents a streaming data chunk."""
    content: str
    node_id: str
    node_name: str


class StreamingGraphExecutor:
    """Graph executor with streaming support.
    
    This executor yields data chunks during execution while emitting 
    events to the event handler for observability.
    """

    def __init__(
        self, 
        max_parallel_nodes: int = 10,
        event_handler: Optional[EventHandler] = None
    ):
        self.max_parallel_nodes = max_parallel_nodes
        self.event_handler = event_handler

    async def execute_streaming(
        self,
        compiled_graph: CompiledGraph,
        initial_input: Optional[Dict[str, Any]] = None,
        state: Optional[GraphState] = None,
    ) -> AsyncIterator[Union[StreamChunk, ExecutionResult]]:
        """Execute graph and yield streaming data chunks.

        Args:
            compiled_graph: Compiled graph to execute
            initial_input: Initial input data
            state: Optional shared state

        Yields:
            StreamChunk objects for streaming content
            ExecutionResult at the end of execution
        """
        start_time = datetime.utcnow()
        graph = compiled_graph.graph

        # Initialize state if not provided
        if state is None:
            state = GraphState()

        # Emit graph start event (to handler, not yielded)
        if self.event_handler:
            await self.event_handler.emit(
                GraphStartEvent(
                    graph_id=graph.id,
                    graph_name=graph.name,
                    metadata={"total_nodes": len(graph._nodes)},
                )
            )

        # Initialize tracking
        outputs: Dict[str, NodeOutput] = {}
        node_inputs: Dict[str, Any] = {}

        # Set initial input for start nodes
        if initial_input:
            for node_id in compiled_graph.execution_plan[0]:
                node_inputs[node_id] = initial_input

        try:
            # Execute the graph
            for parallel_group in compiled_graph.execution_plan:
                # Execute nodes in parallel group
                tasks = []

                for node_id in parallel_group:
                    node = graph.get_node(node_id)
                    if not node:
                        continue

                    # Get input for this node
                    input_data = node_inputs.get(node_id, {})

                    # Create execution task with event streaming
                    task = self._execute_node_streaming(node, input_data, state)
                    tasks.append((node_id, node, task))

                # Execute all tasks in parallel and stream events
                if tasks:
                    # Create async generators for each node
                    generators = []
                    for node_id, node, task in tasks:
                        gen = self._wrap_node_execution(node_id, node, task)
                        generators.append((node_id, gen))

                    # Merge all streams - yield chunks, emit events
                    async for item, node_id, output in self._merge_streams(generators):
                        # If it's a chunk, yield it to the caller
                        if isinstance(item, StreamChunk):
                            yield item
                        # If it's an event (including StreamChunkEvent), emit it to handler
                        elif isinstance(item, Event):
                            if self.event_handler:
                                await self.event_handler.emit(item)
                            # Don't yield events (they go to handler only)

                        if output:
                            outputs[node_id] = output

                            # Prepare inputs for successor nodes
                            for edge, successor in graph.get_successors(node_id):
                                if edge.evaluate_condition(output.data):
                                    node_inputs[successor.id] = output.data

            # Calculate execution time
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()

            # Emit graph end event
            success = all(
                output.status == NodeStatus.COMPLETED for output in outputs.values()
            )

            # Emit graph end event (to handler, not yielded)
            if self.event_handler:
                await self.event_handler.emit(
                    GraphEndEvent(
                        graph_id=graph.id,
                        success=success,
                        execution_time=execution_time,
                        metadata={
                            "nodes_executed": len(outputs),
                            "nodes_succeeded": sum(
                                1 for o in outputs.values() if o.status == NodeStatus.COMPLETED
                            ),
                        },
                    )
                )
            
            # Yield final ExecutionResult
            terminal_nodes = graph.get_terminal_nodes()
            terminal_outputs = {
                node_id: outputs[node_id].data 
                for node_id in terminal_nodes 
                if node_id in outputs
            }
            
            yield ExecutionResult(
                success=success,
                outputs=outputs,
                state=state,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
                metadata=compiled_graph.metadata,
                terminal_outputs=terminal_outputs,
            )

        except Exception as e:
            # Emit error event (to handler, not yielded)
            if self.event_handler:
                await self.event_handler.emit(
                    ErrorEvent(
                        error=str(e),
                        error_type=type(e).__name__,
                        metadata={"graph_id": graph.id},
                    )
                )
            raise  # Re-raise the exception

    async def _wrap_node_execution(
        self, node_id: str, node: Node, execution_task
    ) -> AsyncIterator[Tuple[Union[Event, StreamChunk], str, Optional[NodeOutput]]]:
        """Wrap node execution to yield events and chunks."""
        node_start_time = time.time()

        # Emit node start event
        yield (
            NodeStartEvent(
                node_id=node_id, node_name=node.name, node_type=type(node).__name__
            ),
            node_id,
            None,
        )

        try:
            # Execute the node
            output = None
            async for item in execution_task:
                if isinstance(item, NodeOutput):
                    output = item
                elif isinstance(item, StreamChunk):
                    # Forward streaming chunks (data)
                    yield (item, node_id, None)
                elif isinstance(item, Event):
                    # Forward events (for monitoring)
                    yield (item, node_id, None)
                else:
                    # Forward other items
                    yield (item, node_id, None)

            # Calculate execution time
            execution_time = time.time() - node_start_time

            # Emit node end event
            yield (
                NodeEndEvent(
                    node_id=node_id,
                    node_name=node.name,
                    success=output.status == NodeStatus.COMPLETED if output else False,
                    execution_time=execution_time,
                ),
                node_id,
                output,
            )

        except Exception as e:
            # Emit error event
            yield (
                ErrorEvent(
                    error=str(e),
                    error_type=type(e).__name__,
                    node_id=node_id,
                    node_name=node.name,
                ),
                node_id,
                NodeOutput(node_id=node_id, status=NodeStatus.FAILED, error=str(e)),
            )

    async def _execute_node_streaming(
        self, node: Node, input_data: Dict[str, Any], state: GraphState
    ) -> AsyncIterator[Any]:
        """Execute a node with streaming support."""
        # Special handling for different node types

        if isinstance(node, ToolNode):
            # Emit tool events
            tool_config = node.tool_config
            tool_args = node.extract_args(input_data)

            yield ToolStartEvent(
                node_id=node.id,
                node_name=node.name,
                tool_name=tool_config.tool_name,
                tool_args=tool_args,
            )

            try:
                output = await node.run(input_data, state)

                yield ToolEndEvent(
                    node_id=node.id,
                    node_name=node.name,
                    tool_name=tool_config.tool_name,
                    success=output.status == NodeStatus.COMPLETED,
                    result=output.data.get(tool_config.output_key or "tool_result"),
                )

                yield output

            except Exception as e:
                yield ErrorEvent(
                    error=str(e),
                    error_type="ToolError",
                    node_id=node.id,
                    node_name=node.name,
                )
                raise

        elif isinstance(node, LLMNode) and node.llm_config.stream:
            # Handle streaming LLM node
            output = await node.run(input_data, state)

            # Check if the node has a stream iterator
            if hasattr(node, "_stream_iterator") and node._stream_iterator:
                # Stream the chunks
                full_response = []
                async for chunk in node._stream_iterator:
                    chunk_content = (
                        chunk.content if hasattr(chunk, "content") else str(chunk)
                    )
                    full_response.append(chunk_content)

                    # Yield both an event (for monitoring) and a StreamChunk (for data)
                    # The event will be emitted to the handler, the chunk will be yielded to caller
                    from mesh.core.events import StreamChunkEvent
                    
                    # First yield the event for the handler
                    yield StreamChunkEvent(
                        node_id=node.id,
                        node_name=node.name,
                        content=chunk_content
                    )
                    
                    # Also yield StreamChunk for data
                    yield StreamChunk(
                        content=chunk_content,
                        node_id=node.id,
                        node_name=node.name
                    )

                # Update the output with the full response
                output.data["response"] = "".join(full_response)

                # Clean up the stream iterator
                delattr(node, "_stream_iterator")

            yield output

        else:
            # Regular node execution
            output = await node.run(input_data, state)
            yield output

    async def _merge_streams(
        self, generators: List[Tuple[str, AsyncIterator]]
    ) -> AsyncIterator[Tuple[Union[Event, StreamChunk], str, Optional[NodeOutput]]]:
        """Merge multiple async generators into a single stream."""
        # Create tasks for each generator
        tasks = {}
        for node_id, gen in generators:
            task = asyncio.create_task(self._get_next(gen))
            tasks[task] = (node_id, gen)

        while tasks:
            # Wait for the first task to complete
            done, pending = await asyncio.wait(
                tasks.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                node_id, gen = tasks.pop(task)

                try:
                    result = task.result()
                    if result is not None:
                        yield result

                        # Schedule next item from this generator
                        next_task = asyncio.create_task(self._get_next(gen))
                        tasks[next_task] = (node_id, gen)
                except StopAsyncIteration:
                    # This generator is exhausted
                    pass
                except Exception as e:
                    # Handle errors
                    yield (
                        ErrorEvent(
                            error=str(e), error_type=type(e).__name__, node_id=node_id
                        ),
                        node_id,
                        None,
                    )

    async def _get_next(self, gen: AsyncIterator) -> Optional[Any]:
        """Get the next item from an async generator."""
        try:
            return await gen.__anext__()
        except StopAsyncIteration:
            return None

