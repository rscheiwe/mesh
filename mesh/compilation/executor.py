"""Graph execution engine."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from mesh.compilation.compiler import CompilationStrategy, CompiledGraph
from mesh.compilation.dynamic_compiler import DynamicCompiler
from mesh.compilation.event_handler import EventHandler
from mesh.core.events import (
    ErrorEvent,
    Event,
    GraphEndEvent,
    GraphStartEvent,
    NodeEndEvent,
    NodeStartEvent,
    StreamChunkEvent,
)
from mesh.core.node import NodeOutput, NodeStatus
from mesh.state.state import GraphState


@dataclass
class ExecutionResult:
    """Result of graph execution."""

    success: bool
    outputs: Dict[str, NodeOutput]
    state: Optional[GraphState]
    start_time: datetime
    end_time: datetime
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    terminal_outputs: Dict[str, Any] = field(default_factory=dict)

    def get_final_output(self) -> Optional[Any]:
        """Get the final output from terminal nodes.

        Returns:
            Final output data or None
        """
        # Return terminal outputs if available
        if self.terminal_outputs:
            # If single terminal node, return its output directly
            if len(self.terminal_outputs) == 1:
                return list(self.terminal_outputs.values())[0]
            # If multiple terminal nodes, return all outputs
            return self.terminal_outputs

        # Fallback: return last completed node's output
        final_outputs = []
        for node_id, output in self.outputs.items():
            if output.status == NodeStatus.COMPLETED:
                final_outputs.append((node_id, output.data))

        if final_outputs:
            return final_outputs[-1][1]

        return None


class GraphExecutor:
    """Executes compiled graphs."""

    def __init__(
        self,
        max_parallel_nodes: int = 10,
        max_loops: int = 100,
        event_handler: Optional[EventHandler] = None,
    ):
        self.max_parallel_nodes = max_parallel_nodes
        self.max_loops = max_loops
        self.dynamic_compiler = DynamicCompiler()
        self.event_handler = event_handler

    async def execute(
        self,
        compiled_graph: CompiledGraph,
        initial_input: Optional[Dict[str, Any]] = None,
        state: Optional[GraphState] = None,
        stream_events: bool = False,
        max_loops: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute a compiled graph.

        Args:
            compiled_graph: Compiled graph to execute
            initial_input: Initial input data
            state: Optional shared state (creates new if None)
            max_loops: Maximum number of loops to allow (overrides default)

        Returns:
            ExecutionResult with execution details
        """
        start_time = datetime.utcnow()
        graph = compiled_graph.graph

        # Emit graph start event
        if self.event_handler:
            await self.event_handler.emit(
                GraphStartEvent(
                    graph_id=graph.id,
                    graph_name=graph.name,
                    metadata={"total_nodes": len(graph._nodes)},
                )
            )

        # Initialize state if not provided
        if state is None:
            state = GraphState()

        # Initialize tracking
        outputs: Dict[str, NodeOutput] = {}
        node_inputs: Dict[str, Any] = {}
        terminal_outputs: Dict[str, Any] = {}

        # Track loop iterations per node
        node_iterations: Dict[str, int] = {}
        effective_max_loops = max_loops or self.max_loops

        # Identify terminal nodes (nodes with no outgoing edges)
        terminal_nodes = self._identify_terminal_nodes(graph)

        # Set initial input for start nodes
        if initial_input:
            for node_id in compiled_graph.execution_plan[0]:
                node_inputs[node_id] = initial_input

        # Determine execution strategy
        is_dynamic = compiled_graph.metadata.get("strategy") == "dynamic"

        try:
            if is_dynamic:
                await self._execute_dynamic(
                    compiled_graph,
                    outputs,
                    node_inputs,
                    state,
                    node_iterations,
                    effective_max_loops,
                    terminal_nodes,
                    terminal_outputs,
                )
            else:
                await self._execute_static(
                    compiled_graph,
                    outputs,
                    node_inputs,
                    state,
                    node_iterations,
                    effective_max_loops,
                    terminal_nodes,
                    terminal_outputs,
                )

            success = all(
                output.status == NodeStatus.COMPLETED for output in outputs.values()
            )

        except Exception as e:
            success = False
            # Add error to metadata
            compiled_graph.metadata["execution_error"] = str(e)
            
            # Emit error event
            if self.event_handler:
                await self.event_handler.emit(
                    ErrorEvent(
                        error=str(e),
                        error_type=type(e).__name__,
                        metadata={"graph_id": graph.id},
                    )
                )

        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        # Emit graph end event
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

        return ExecutionResult(
            success=success,
            outputs=outputs,
            state=state,
            start_time=start_time,
            end_time=end_time,
            execution_time=execution_time,
            metadata=compiled_graph.metadata,
            terminal_outputs=terminal_outputs,
        )

    async def _execute_static(
        self,
        compiled_graph: CompiledGraph,
        outputs: Dict[str, NodeOutput],
        node_inputs: Dict[str, Any],
        state: GraphState,
        node_iterations: Dict[str, int],
        max_loops: int,
        terminal_nodes: Set[str],
        terminal_outputs: Dict[str, Any],
    ) -> None:
        """Execute graph with static execution plan.

        Args:
            compiled_graph: Compiled graph
            outputs: Output tracking dict
            node_inputs: Input tracking dict
            state: Shared state
            node_iterations: Loop iteration tracking
            max_loops: Maximum allowed loops
            terminal_nodes: Set of terminal node IDs
            terminal_outputs: Terminal node outputs
        """
        graph = compiled_graph.graph

        for parallel_group in compiled_graph.execution_plan:
            # Execute nodes in parallel group
            tasks = []

            for node_id in parallel_group:
                # Check loop limit
                if node_id in node_iterations and node_iterations[node_id] >= max_loops:
                    continue  # Skip node if loop limit reached

                node = graph.get_node(node_id)
                if not node:
                    continue

                # Get input for this node
                input_data = node_inputs.get(node_id, {})

                # Create execution task
                task = self._execute_node(node, input_data, state)
                tasks.append((node_id, task))

            # Wait for all tasks in group
            if tasks:
                results = await asyncio.gather(
                    *[task for _, task in tasks], return_exceptions=True
                )

                # Process results
                for i, (node_id, _) in enumerate(tasks):
                    result = results[i]

                    # Track iterations
                    node_iterations[node_id] = node_iterations.get(node_id, 0) + 1

                    if isinstance(result, Exception):
                        # Create error output
                        outputs[node_id] = NodeOutput(
                            node_id=node_id, status=NodeStatus.FAILED, error=str(result)
                        )
                    else:
                        outputs[node_id] = result

                        # Store terminal node outputs
                        if node_id in terminal_nodes:
                            terminal_outputs[node_id] = result.data

                        # Prepare inputs for successor nodes
                        for edge, successor in graph.get_successors(node_id):
                            if edge.evaluate_condition(result.data):
                                node_inputs[successor.id] = result.data

    async def _execute_dynamic(
        self,
        compiled_graph: CompiledGraph,
        outputs: Dict[str, NodeOutput],
        node_inputs: Dict[str, Any],
        state: GraphState,
        node_iterations: Dict[str, int],
        max_loops: int,
        terminal_nodes: Set[str],
        terminal_outputs: Dict[str, Any],
    ) -> None:
        """Execute graph with dynamic execution plan.

        Args:
            compiled_graph: Compiled graph
            outputs: Output tracking dict
            node_inputs: Input tracking dict
            state: Shared state
            node_iterations: Loop iteration tracking
            max_loops: Maximum allowed loops
            terminal_nodes: Set of terminal node IDs
            terminal_outputs: Terminal node outputs
        """
        graph = compiled_graph.graph

        # Start with initial nodes
        pending_groups = [compiled_graph.execution_plan[0]]

        while pending_groups:
            # Execute next group
            current_group = pending_groups.pop(0)

            # Execute nodes in parallel
            tasks = []
            for node_id in current_group:
                # Check loop limit
                if node_id in node_iterations and node_iterations[node_id] >= max_loops:
                    continue  # Skip node if loop limit reached

                node = graph.get_node(node_id)
                if not node:
                    continue

                input_data = node_inputs.get(node_id, {})
                task = self._execute_node(node, input_data, state)
                tasks.append((node_id, task))

            # Wait for completion
            if tasks:
                results = await asyncio.gather(
                    *[task for _, task in tasks], return_exceptions=True
                )

                # Process results and prepare next nodes
                for i, (node_id, _) in enumerate(tasks):
                    result = results[i]

                    # Track iterations
                    node_iterations[node_id] = node_iterations.get(node_id, 0) + 1

                    if isinstance(result, Exception):
                        outputs[node_id] = NodeOutput(
                            node_id=node_id, status=NodeStatus.FAILED, error=str(result)
                        )
                    else:
                        outputs[node_id] = result

                        # Store terminal node outputs
                        if node_id in terminal_nodes:
                            terminal_outputs[node_id] = result.data

                        # Prepare inputs for successors
                        for edge, successor in graph.get_successors(node_id):
                            if edge.evaluate_condition(result.data):
                                node_inputs[successor.id] = result.data

            # Determine next nodes to execute
            next_groups = await self.dynamic_compiler.get_next_nodes(
                graph, outputs, state
            )

            if next_groups:
                pending_groups.extend(next_groups)

    def _identify_terminal_nodes(self, graph: Any) -> Set[str]:
        """Identify terminal nodes (nodes with no outgoing edges).

        Args:
            graph: The graph to analyze

        Returns:
            Set of terminal node IDs
        """
        return graph.get_terminal_nodes()

    async def _execute_node(
        self, node: Any, input_data: Dict[str, Any], state: GraphState
    ) -> NodeOutput:
        """Execute a single node.

        Args:
            node: Node to execute
            input_data: Input data for the node
            state: Shared state

        Returns:
            NodeOutput from execution
        """
        import time
        node_start_time = time.time()
        
        # Emit node start event
        if self.event_handler:
            await self.event_handler.emit(
                NodeStartEvent(
                    node_id=node.id,
                    node_name=node.name,
                    node_type=type(node).__name__,
                )
            )
        
        try:
            # Execute the node
            output = await node.run(input_data, state)
            
            # Calculate execution time
            execution_time = time.time() - node_start_time
            
            # Emit node end event
            if self.event_handler:
                await self.event_handler.emit(
                    NodeEndEvent(
                        node_id=node.id,
                        node_name=node.name,
                        success=output.status == NodeStatus.COMPLETED,
                        execution_time=execution_time,
                    )
                )
            
            return output
            
        except Exception as e:
            # Calculate execution time
            execution_time = time.time() - node_start_time
            
            # Emit error event
            if self.event_handler:
                await self.event_handler.emit(
                    ErrorEvent(
                        error=str(e),
                        error_type=type(e).__name__,
                        node_id=node.id,
                        node_name=node.name,
                    )
                )
            
            # Re-raise the exception
            raise
