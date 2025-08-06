"""Executor that supports smart edges with built-in routing logic."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ..core.edge import Edge, EdgeType
from ..core.node import Node, NodeOutput
from ..core.smart_edge import FeedbackLoopEdge, SmartEdge, SmartEdgeType
from ..state import GraphState
from .compiler import CompiledGraph
from .executor import ExecutionResult, GraphExecutor


class SmartGraphExecutor(GraphExecutor):
    """Extended executor that handles smart edges with built-in logic.

    This executor can process graphs with feedback loops and complex
    routing without requiring intermediate decision nodes.
    """

    def __init__(self, max_parallel_nodes: int = 10, max_loops: int = 100):
        super().__init__(max_parallel_nodes, max_loops)
        self._loop_counters: Dict[str, int] = {}

    async def execute(
        self,
        graph: CompiledGraph,
        initial_input: Optional[Any] = None,
        state: Optional[GraphState] = None,
        max_loops: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute graph with smart edge support."""

        start_time = datetime.now()

        if max_loops is not None:
            self.max_loops = max_loops

        # Initialize execution state
        self._loop_counters.clear()
        ready_nodes = set(graph.graph._start_nodes)  # Get start nodes from the graph
        completed_nodes: Set[str] = set()
        node_outputs: Dict[str, NodeOutput] = {}

        # Track feedback loop states
        feedback_states: Dict[str, Dict[str, Any]] = {}
        # Track loop iterations per edge
        loop_iterations: Dict[str, int] = {}

        # Use initial input for start nodes
        current_input = initial_input

        while ready_nodes:
            # Check for infinite loops
            for node_id in ready_nodes:
                self._loop_counters[node_id] = self._loop_counters.get(node_id, 0) + 1
                if self._loop_counters[node_id] > self.max_loops:
                    raise RuntimeError(
                        f"Node {node_id} exceeded max loops ({self.max_loops})"
                    )

            # Execute ready nodes
            tasks = []
            for node_id in ready_nodes:
                # Remove from completed if re-executing (for loops)
                if node_id in completed_nodes:
                    completed_nodes.remove(node_id)

                node = graph.graph.get_node(node_id)

                # Prepare input
                if hasattr(node, "merge_inputs") and callable(node.merge_inputs):
                    input_data = node.merge_inputs(node_outputs)
                else:
                    if node_id in graph.graph._start_nodes:
                        input_data = current_input
                    else:
                        # Get inputs from predecessors
                        predecessors = graph.graph.get_predecessors(node_id)
                        if len(predecessors) == 1:
                            pred_edge, pred_node = predecessors[0]
                            pred_id = pred_node.id
                            pred_output = node_outputs.get(pred_id)
                            input_data = pred_output.data if pred_output else {}
                        else:
                            # Merge multiple inputs including previous outputs
                            input_data = {}

                            # First, check if this node has previous output (for loops)
                            if node_id in node_outputs:
                                # Include previous output for re-execution
                                prev_data = node_outputs[node_id].data
                                if isinstance(prev_data, dict):
                                    input_data.update(prev_data)

                            # Then merge inputs from predecessors
                            for pred_edge, pred_node in predecessors:
                                pred_id = pred_node.id
                                if pred_id in node_outputs:
                                    pred_data = node_outputs[pred_id].data
                                    # For loop edges, ensure fresh data is used
                                    if pred_edge.edge_type == EdgeType.LOOP:
                                        # Always use the latest output from loop predecessor
                                        if isinstance(pred_data, dict):
                                            input_data.update(pred_data)
                                        else:
                                            input_data[pred_id] = pred_data
                                    else:
                                        # Normal edge behavior
                                        if isinstance(pred_data, dict):
                                            input_data.update(pred_data)
                                        else:
                                            input_data[pred_id] = pred_data

                # Execute node
                task = asyncio.create_task(self._execute_node(node, input_data, state))
                tasks.append((node_id, task))

            # Wait for completion
            just_completed = set()
            for node_id, task in tasks:
                output = await task
                node_outputs[node_id] = output
                completed_nodes.add(node_id)
                just_completed.add(node_id)

            # Clear ready nodes for next iteration
            ready_nodes.clear()

            # Determine next nodes using smart edge logic
            next_nodes = await self._get_next_nodes_smart(
                graph,
                completed_nodes,
                node_outputs,
                state,
                feedback_states,
                loop_iterations,
                just_completed,
            )
            ready_nodes.update(next_nodes)

        # Calculate execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Identify terminal outputs
        terminal_outputs = {}
        terminal_nodes = graph.graph.get_terminal_nodes()

        # For feedback loops, the feedback node becomes terminal when loop completes
        # Check which nodes have no outgoing edges OR whose loop edges didn't activate
        for node_id in completed_nodes:
            successors = graph.graph.get_successors(node_id)
            is_terminal = True

            for edge, _ in successors:
                # If it has a non-loop edge, it's not terminal
                if edge.edge_type != EdgeType.LOOP:
                    is_terminal = False
                    break

            if is_terminal and node_id in node_outputs:
                terminal_outputs[node_id] = node_outputs[node_id]

        # Create result
        result = ExecutionResult(
            success=True,
            outputs=node_outputs,
            state=state,
            start_time=start_time,
            end_time=end_time,
            execution_time=execution_time,
            terminal_outputs=terminal_outputs,
        )

        return result

    async def _get_next_nodes_smart(
        self,
        graph: CompiledGraph,
        completed_nodes: Set[str],
        node_outputs: Dict[str, NodeOutput],
        state: Optional[GraphState],
        feedback_states: Dict[str, Dict[str, Any]],
        loop_iterations: Dict[str, int],
        just_completed: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Get next nodes to execute, handling smart edges."""

        ready = set()

        # If just_completed is provided, only check those nodes
        # Otherwise check all completed nodes
        nodes_to_check = just_completed if just_completed else completed_nodes

        # Check each node's outgoing edges
        for node_id in nodes_to_check:
            node_output = node_outputs.get(node_id)
            if not node_output:
                continue

            # Get outgoing edges
            successors = graph.graph.get_successors(node_id)

            for edge, successor_node in successors:
                successor_id = successor_node.id
                # For loop edges, allow re-execution even if already completed
                if successor_id in completed_nodes:
                    if not (
                        isinstance(edge, SmartEdge) and edge.edge_type == EdgeType.LOOP
                    ):
                        # Skip if not a loop edge and already completed
                        # unless source was just executed
                        if node_id not in (just_completed or set()):
                            continue

                if not edge:
                    continue

                # Handle smart edges
                if isinstance(edge, SmartEdge):
                    # Special handling for feedback loops
                    if (
                        isinstance(edge, FeedbackLoopEdge)
                        or edge.edge_type == EdgeType.LOOP
                    ):
                        # Track iterations for this edge
                        edge_key = f"{node_id}->{successor_id}"
                        current_iter = loop_iterations.get(edge_key, 0)

                        # Check if we should continue the loop
                        should_continue = edge.should_continue_loop(
                            node_output.data, state, current_iter
                        )
                        if should_continue:
                            loop_iterations[edge_key] = current_iter + 1
                            ready.add(successor_id)
                        else:
                            # Loop complete - find alternative paths
                            # Look for non-loop edges from the same source
                            for (
                                alt_edge,
                                alt_successor_node,
                            ) in graph.graph.get_successors(node_id):
                                alt_successor_id = alt_successor_node.id
                                if alt_successor_id != successor_id:
                                    if alt_edge and alt_edge.edge_type != EdgeType.LOOP:
                                        if alt_edge.evaluate_condition(
                                            node_output.data
                                        ):
                                            ready.add(alt_successor_id)

                    elif edge.edge_type == SmartEdgeType.MULTI_CONDITIONAL:
                        # Use router to find target
                        target = edge.get_next_target(node_output.data, state)
                        if target and target not in completed_nodes:
                            ready.add(target)

                    else:
                        # Standard smart edge
                        target = edge.get_next_target(node_output.data, state)
                        if target and target not in completed_nodes:
                            ready.add(target)

                else:
                    # Regular edge
                    if edge.evaluate_condition(node_output.data):
                        # Check if all predecessors are complete (excluding loop edges)
                        predecessors = list(graph.graph.get_predecessors(successor_id))
                        # Filter out loop edges when checking readiness
                        non_loop_predecessors = [
                            (pred_edge, pred_node)
                            for pred_edge, pred_node in predecessors
                            if pred_edge.edge_type != EdgeType.LOOP
                        ]

                        # Check if this is part of a loop - if the predecessor was just re-executed
                        # due to a loop, we should also re-execute this node
                        is_in_loop = any(
                            pred_node.id not in completed_nodes
                            for pred_edge, pred_node in non_loop_predecessors
                        )

                        all_preds_complete = all(
                            pred_node.id in completed_nodes
                            for pred_edge, pred_node in non_loop_predecessors
                        )

                        # Special case: if the source node was just executed, allow the successor
                        # even if it was already completed (for re-execution in loops)
                        if node_id in (just_completed or set()):
                            # Remove from completed to allow re-execution
                            if successor_id in completed_nodes:
                                completed_nodes.remove(successor_id)
                            ready.add(successor_id)
                        elif all_preds_complete or is_in_loop:
                            ready.add(successor_id)

        return ready
