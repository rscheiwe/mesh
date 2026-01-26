"""Queue-based graph executor with streaming support.

This module implements the core execution engine that orchestrates node execution,
manages dependencies, handles conditional branching, and emits streaming events.

Based on Flowise's execution loop pattern from buildAgentflow.ts:1628-1816.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Set, List, Optional, Union, AsyncIterator
from datetime import datetime
import asyncio

from mesh.core.graph import ExecutionGraph
from mesh.core.events import ExecutionEvent, EventType, EventEmitter
from mesh.core.state import ExecutionContext
from mesh.utils.errors import NodeExecutionError


class ExecutionStatus:
    """Execution status constants."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    APPROVAL_REJECTED = "approval_rejected"


@dataclass
class NodeQueueItem:
    """Item in the execution queue.

    Attributes:
        node_id: ID of the node to execute
        inputs: Combined inputs from parent nodes
    """

    node_id: str
    inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WaitingNode:
    """Node waiting for multiple parent inputs.

    When a node has multiple parents, it must wait for all parent outputs
    before it can execute. This class tracks which inputs have been received.

    Attributes:
        node_id: ID of the waiting node
        received_inputs: Mapping of parent node IDs to their outputs
        expected_inputs: Set of parent node IDs we're waiting for
        is_conditional: Whether this node is after a conditional branch
        conditional_groups: Grouping of conditional branches
    """

    node_id: str
    received_inputs: Dict[str, Any] = field(default_factory=dict)
    expected_inputs: Set[str] = field(default_factory=set)
    is_conditional: bool = False
    conditional_groups: Dict[str, List[str]] = field(default_factory=dict)


class Executor:
    """Queue-based graph executor with streaming support.

    The executor implements a breadth-first execution strategy with dependency
    tracking. It maintains a queue of nodes ready to execute and a waiting list
    for nodes that need multiple inputs.

    Key features:
    - Dependency-based execution ordering
    - Multi-parent input combining
    - Conditional branching support
    - Loop detection and limiting
    - Streaming event emission
    - State persistence coordination
    """

    def __init__(
        self,
        graph: ExecutionGraph,
        state_backend: Optional[Any] = None,
        event_emitter: Optional[EventEmitter] = None,
        max_iterations: int = 1000,
    ):
        """Initialize executor.

        Args:
            graph: ExecutionGraph to execute
            state_backend: Optional state backend for persistence
            event_emitter: Optional event emitter for streaming
            max_iterations: Maximum execution iterations (safety limit)
        """
        self.graph = graph
        self.state_backend = state_backend
        self.events = event_emitter or EventEmitter()
        self.max_iterations = max_iterations

    async def execute(
        self,
        input_data: Union[str, Dict[str, Any]],
        context: ExecutionContext,
    ) -> AsyncIterator[ExecutionEvent]:
        """Execute graph with streaming events.

        This is the main execution method that orchestrates the entire workflow.
        It implements the queue-based execution pattern from Flowise:

        1. Initialize queue with starting nodes
        2. While queue not empty:
           a. Dequeue node
           b. Execute node with context
           c. Emit progress events
           d. Process outputs and queue children
           e. Handle conditional branching
        3. Persist final state

        Args:
            input_data: Input to the graph (string or dict)
            context: Execution context with state and variables

        Yields:
            ExecutionEvent: Streaming execution events

        Raises:
            NodeExecutionError: If a node execution fails
        """
        # Set event emitter in context
        context._event_emitter = self.events

        # Initialize execution state
        queue: List[NodeQueueItem] = []
        waiting_nodes: Dict[str, WaitingNode] = {}
        loop_counts: Dict[str, int] = {}

        # Emit execution start event
        yield ExecutionEvent(
            type=EventType.EXECUTION_START,
            metadata={"graph_id": context.graph_id, "trace_id": context.trace_id},
        )

        # Initialize queue with starting nodes
        for node_id in self.graph.starting_nodes:
            queue.append(NodeQueueItem(node_id=node_id, inputs={"input": input_data}))

        iteration = 0
        while queue and iteration < self.max_iterations:
            iteration += 1
            current = queue.pop(0)

            try:
                # Get node instance
                node = self.graph.get_node(current.node_id)

                # Import node types for checks
                from mesh.nodes.agent import AgentNode
                from mesh.nodes.llm import LLMNode
                from mesh.nodes.tool import ToolNode
                from mesh.nodes.start import StartNode
                from mesh.nodes.dynamic_tool_selector import DynamicToolSelectorNode

                # Skip Tool nodes that only feed into DynamicToolSelector
                # These nodes provide metadata only, not execution
                if isinstance(node, ToolNode):
                    # Check if ALL children are DynamicToolSelector nodes
                    children = self.graph.children.get(current.node_id, [])
                    all_children_are_selectors = (
                        len(children) > 0 and
                        all(
                            isinstance(self.graph.get_node(child_id), DynamicToolSelectorNode)
                            for child_id in children
                        )
                    )
                    if all_children_are_selectors:
                        # Skip execution - the DynamicToolSelector already has tool info from parsing
                        # Queue children with empty output (they don't need it from tool execution)
                        for child_id in children:
                            child_deps = self.graph.dependencies.get(child_id, set())
                            if len(child_deps) <= 1:
                                queue.append(NodeQueueItem(node_id=child_id, inputs={}))
                            else:
                                # Multi-parent handling for DynamicToolSelector with multiple tool inputs
                                if child_id not in waiting_nodes:
                                    waiting_nodes[child_id] = WaitingNode(
                                        node_id=child_id,
                                        expected_inputs=child_deps.copy(),
                                    )
                                waiting_nodes[child_id].received_inputs[current.node_id] = {}
                                if waiting_nodes[child_id].expected_inputs <= set(
                                    waiting_nodes[child_id].received_inputs.keys()
                                ):
                                    queue.append(NodeQueueItem(
                                        node_id=child_id,
                                        inputs=waiting_nodes[child_id].received_inputs,
                                    ))
                                    del waiting_nodes[child_id]
                        continue  # Skip normal execution for this tool node

                # Emit node start event (skip for nodes that emit their own with metadata)
                if not isinstance(node, (AgentNode, LLMNode, ToolNode, StartNode)):
                    yield ExecutionEvent(
                        type=EventType.NODE_START,
                        node_id=current.node_id,
                        timestamp=datetime.now(),
                    )

                # Set up event queue for streaming during execution
                event_queue: asyncio.Queue[Optional[ExecutionEvent]] = asyncio.Queue()

                async def event_listener(event: ExecutionEvent):
                    """Capture events emitted during node execution."""
                    await event_queue.put(event)

                # Register listener
                self.events.on(event_listener)

                # Execute node in background task
                execute_task = asyncio.create_task(
                    node.execute(input=current.inputs, context=context)
                )

                # Yield events as they stream in
                result = None
                while True:
                    # Check if execution is complete
                    if execute_task.done():
                        # Get result
                        result = await execute_task
                        # Drain any remaining events
                        while not event_queue.empty():
                            try:
                                evt = event_queue.get_nowait()
                                if evt:
                                    yield evt
                            except asyncio.QueueEmpty:
                                break
                        break

                    # Wait for next event or task completion
                    try:
                        evt = await asyncio.wait_for(event_queue.get(), timeout=0.01)
                        if evt:
                            yield evt
                    except asyncio.TimeoutError:
                        # No events yet, continue checking
                        continue

                # Remove listener
                self.events.off(event_listener)

                # Update state
                if result.state:
                    context.state.update(result.state)
                    if self.state_backend:
                        await self.state_backend.save(context.session_id, context.state)

                # Update chat history
                if result.chat_history:
                    context.chat_history.extend(result.chat_history)

                # Store execution data
                context.add_executed_node(current.node_id, result.output)

                # Emit node complete event
                yield ExecutionEvent(
                    type=EventType.NODE_COMPLETE,
                    node_id=current.node_id,
                    output=result.output,
                    timestamp=datetime.now(),
                    metadata=result.metadata,
                )

                # Check for approval pending - pause execution
                if result.approval_pending:
                    # Save execution state for resume
                    pending_state = {
                        "queue": [(item.node_id, item.inputs) for item in queue],
                        "waiting_nodes": {
                            k: {
                                "node_id": v.node_id,
                                "received_inputs": v.received_inputs,
                                "expected_inputs": list(v.expected_inputs),
                            }
                            for k, v in waiting_nodes.items()
                        },
                        "loop_counts": loop_counts,
                        "current_node_id": current.node_id,
                        "approval_id": result.approval_id,
                        "approval_data": result.approval_data,
                        "iteration": iteration,
                    }
                    context.state["_pending_execution"] = pending_state

                    # Persist state
                    if self.state_backend:
                        await self.state_backend.save(context.session_id, context.state)

                    # Yield completion event with approval status
                    yield ExecutionEvent(
                        type=EventType.EXECUTION_COMPLETE,
                        output=result.output,
                        timestamp=datetime.now(),
                        metadata={
                            "status": ExecutionStatus.WAITING_FOR_APPROVAL,
                            "approval_id": result.approval_id,
                            "approval_data": result.approval_data,
                            "iterations": iteration,
                            "trace_id": context.trace_id,
                        },
                    )
                    return  # Exit generator - execution paused

                # Process outputs and queue children
                await self._process_node_outputs(
                    node_id=current.node_id,
                    result=result,
                    queue=queue,
                    waiting_nodes=waiting_nodes,
                    loop_counts=loop_counts,
                    context=context,
                )

            except Exception as e:
                error_msg = str(e)
                yield ExecutionEvent(
                    type=EventType.NODE_ERROR,
                    node_id=current.node_id,
                    error=error_msg,
                    timestamp=datetime.now(),
                )
                raise NodeExecutionError(current.node_id, error_msg, e)

        # Check for iteration limit
        if iteration >= self.max_iterations:
            yield ExecutionEvent(
                type=EventType.EXECUTION_ERROR,
                error=f"Execution exceeded maximum iterations ({self.max_iterations})",
            )

        # Get final output
        final_output = None
        if context.executed_data:
            final_output = context.executed_data[-1].get("output")

        # Emit execution complete event
        yield ExecutionEvent(
            type=EventType.EXECUTION_COMPLETE,
            output=final_output,
            timestamp=datetime.now(),
            metadata={"iterations": iteration, "trace_id": context.trace_id},
        )

    async def _process_node_outputs(
        self,
        node_id: str,
        result: Any,
        queue: List[NodeQueueItem],
        waiting_nodes: Dict[str, WaitingNode],
        loop_counts: Dict[str, int],
        context: ExecutionContext,
    ) -> None:
        """Process node outputs and determine next nodes to execute.

        This method implements the complex logic of:
        - Determining which child nodes to execute
        - Handling conditional branching (ignoring unfulfilled branches)
        - Managing multi-parent dependencies
        - Combining inputs from multiple parents
        - Handling loop nodes

        Reference: Flowise's processNodeOutputs() in buildAgentflow.ts:598-693

        Args:
            node_id: ID of the node that just executed
            result: NodeResult from execution
            queue: Execution queue to add ready nodes
            waiting_nodes: Nodes waiting for multiple inputs
            loop_counts: Loop iteration counts
            context: Execution context
        """
        child_node_ids = self.graph.get_children(node_id)

        # Handle conditional branching - determine nodes to ignore
        ignore_nodes = await self._determine_nodes_to_ignore(node_id, result)

        for child_id in child_node_ids:
            if child_id in ignore_nodes:
                continue

            # Check if this edge is a loop edge with conditions
            edge = self._get_edge(node_id, child_id)
            if edge and edge.is_loop_edge:
                # Check loop conditions
                should_continue = await self._should_continue_loop(
                    edge=edge,
                    node_id=node_id,
                    child_id=child_id,
                    result=result,
                    context=context,
                )
                if not should_continue:
                    # Loop condition not met or max iterations reached - skip this edge
                    continue

            # Get parent dependencies for this child
            parent_ids = self.graph.get_parents(child_id)

            # Single parent - queue immediately
            if len(parent_ids) == 1:
                queue.append(
                    NodeQueueItem(
                        node_id=child_id,
                        inputs=result.output,
                    )
                )
                continue

            # Multiple parents - need to wait for all inputs
            if child_id not in waiting_nodes:
                # Initialize waiting node
                waiting_nodes[child_id] = WaitingNode(
                    node_id=child_id,
                    received_inputs={node_id: result.output},
                    expected_inputs=parent_ids,
                )
            else:
                # Add this parent's output
                waiting_nodes[child_id].received_inputs[node_id] = result.output

            # Check if all inputs received
            if self._has_all_inputs(waiting_nodes[child_id]):
                # Combine inputs and queue
                combined = self._combine_inputs(waiting_nodes[child_id].received_inputs)
                queue.append(NodeQueueItem(node_id=child_id, inputs=combined))
                # Remove from waiting
                del waiting_nodes[child_id]

        # Handle loop nodes
        if hasattr(result, "loop_to_node") and result.loop_to_node:
            count = loop_counts.get(node_id, 0) + 1
            max_loops = getattr(result, "max_loops", 10)

            if count < max_loops:
                loop_counts[node_id] = count
                queue.append(
                    NodeQueueItem(
                        node_id=result.loop_to_node,
                        inputs=result.output,
                    )
                )

    async def _determine_nodes_to_ignore(
        self, node_id: str, result: Any
    ) -> Set[str]:
        """Determine which child nodes to ignore based on conditional logic.

        When a ConditionNode executes, some branches are fulfilled and others
        are not. This method identifies which downstream nodes should be skipped.

        Reference: Flowise's determineNodesToIgnore() in buildAgentflow.ts:558-593

        Args:
            node_id: ID of the node that just executed
            result: NodeResult from execution

        Returns:
            Set of node IDs to ignore
        """
        ignore_nodes: Set[str] = set()

        # Check if this node has condition metadata
        if not hasattr(result, "metadata") or "conditions" not in result.metadata:
            return ignore_nodes

        conditions = result.metadata["conditions"]
        child_ids = self.graph.get_children(node_id)

        # For each condition, if not fulfilled, ignore its target nodes
        for condition in conditions:
            if not condition.get("fulfilled", False):
                target = condition.get("target")
                if target and target in child_ids:
                    ignore_nodes.add(target)

        return ignore_nodes

    def _get_edge(self, source: str, target: str):
        """Get edge between two nodes.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Edge if found, None otherwise
        """
        for edge in self.graph.edges:
            if edge.source == source and edge.target == target:
                return edge
        return None

    async def _should_continue_loop(
        self,
        edge,
        node_id: str,
        child_id: str,
        result: Any,
        context: ExecutionContext,
    ) -> bool:
        """Check if a loop edge should be followed.

        Evaluates loop conditions and max iteration limits.

        Args:
            edge: The loop edge
            node_id: Source node ID
            child_id: Target node ID
            result: Node result
            context: Execution context

        Returns:
            True if loop should continue, False otherwise
        """
        edge_key = f"{node_id}->{child_id}"

        # Check max iterations first
        if edge.max_iterations is not None:
            current_iteration = context.get_loop_iteration(edge_key)
            if current_iteration >= edge.max_iterations:
                # Max iterations reached
                return False

        # Check loop condition if provided
        if edge.loop_condition is not None:
            try:
                # Call condition: condition(state, output) -> bool
                should_continue = edge.loop_condition(context.state, result.output)
                if not should_continue:
                    # Condition returned False - exit loop
                    return False
            except Exception as e:
                # Condition evaluation failed - exit loop for safety
                return False

        # Increment iteration count
        context.increment_loop_iteration(edge_key)
        return True

    def _has_all_inputs(self, waiting_node: WaitingNode) -> bool:
        """Check if a waiting node has received all expected inputs.

        Args:
            waiting_node: WaitingNode to check

        Returns:
            True if all inputs received
        """
        received = set(waiting_node.received_inputs.keys())
        return received >= waiting_node.expected_inputs

    def _combine_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Combine inputs from multiple parent nodes.

        The combination strategy is:
        1. If all inputs are dicts, merge them
        2. If mixed types, create a dict with parent IDs as keys
        3. Special handling for common patterns (e.g., content aggregation)

        Args:
            inputs: Mapping of parent node IDs to their outputs

        Returns:
            Combined input dictionary
        """
        if not inputs:
            return {}

        # Check if all inputs are dictionaries
        all_dicts = all(isinstance(v, dict) for v in inputs.values())

        if all_dicts:
            # Merge all dicts
            combined = {}
            for parent_id, output in inputs.items():
                combined.update(output)
            return combined
        else:
            # Return as-is with parent IDs as keys
            return inputs

    async def resume(
        self,
        context: ExecutionContext,
        approval_result: "ApprovalResult",
    ) -> AsyncIterator[ExecutionEvent]:
        """Resume execution after an approval node pause.

        This method continues execution from where it was paused by an ApprovalNode.
        It loads the saved execution state from context and processes the approval
        result to determine how to continue.

        Args:
            context: Execution context (should have been reloaded from state_backend)
            approval_result: Result of the approval decision

        Yields:
            ExecutionEvent: Streaming execution events

        Raises:
            ValueError: If no pending execution state found
            NodeExecutionError: If a node execution fails

        Example:
            >>> # After approval node paused execution:
            >>> approval_result = ApprovalResult(approved=True)
            >>> async for event in executor.resume(context, approval_result):
            ...     print(event.type)
        """
        from mesh.nodes.approval import ApprovalResult as ApprovalResultClass

        # Set event emitter in context
        context._event_emitter = self.events

        # Get pending execution state
        pending_state = context.state.get("_pending_execution")
        if not pending_state:
            raise ValueError(
                "No pending execution state found. Cannot resume without a prior approval pause."
            )

        # Emit approval result event
        if approval_result.approved:
            yield ExecutionEvent(
                type=EventType.APPROVAL_RECEIVED,
                metadata={
                    "approval_id": pending_state.get("approval_id"),
                    "approver_id": approval_result.approver_id,
                    "modified_data": approval_result.modified_data is not None,
                },
            )
        else:
            yield ExecutionEvent(
                type=EventType.APPROVAL_REJECTED,
                metadata={
                    "approval_id": pending_state.get("approval_id"),
                    "rejection_reason": approval_result.rejection_reason,
                    "approver_id": approval_result.approver_id,
                },
            )

            # Emit execution complete with rejection status
            yield ExecutionEvent(
                type=EventType.EXECUTION_COMPLETE,
                output=None,
                timestamp=datetime.now(),
                metadata={
                    "status": ExecutionStatus.APPROVAL_REJECTED,
                    "rejection_reason": approval_result.rejection_reason,
                    "trace_id": context.trace_id,
                },
            )
            return  # End execution on rejection

        # Restore execution state
        queue: List[NodeQueueItem] = []
        for node_id, inputs in pending_state.get("queue", []):
            queue.append(NodeQueueItem(node_id=node_id, inputs=inputs))

        waiting_nodes: Dict[str, WaitingNode] = {}
        for node_id, data in pending_state.get("waiting_nodes", {}).items():
            waiting_nodes[node_id] = WaitingNode(
                node_id=data["node_id"],
                received_inputs=data["received_inputs"],
                expected_inputs=set(data["expected_inputs"]),
            )

        loop_counts = pending_state.get("loop_counts", {})
        iteration = pending_state.get("iteration", 0)

        # Determine the input for the next node
        # If approval modified data, use that; otherwise use the original approval output
        if approval_result.modified_data is not None:
            resume_input = approval_result.modified_data
        else:
            resume_input = pending_state.get("approval_data", {}).get("input", {})

        # Get the node that was waiting for approval
        approval_node_id = pending_state.get("current_node_id")

        # Check if this is a conversation node that needs to continue (not move to children)
        # Conversation nodes set conversation_pending=True in approval_data
        is_conversation_continue = pending_state.get("approval_data", {}).get("conversation_pending", False)

        if approval_node_id:
            if is_conversation_continue:
                # Conversation node - re-execute the same node with full resume_input
                # resume_input contains: message, user_message, and messages array
                # The ConversationNode will use the messages array for history (stateless pattern)
                queue.append(NodeQueueItem(node_id=approval_node_id, inputs=resume_input))
            else:
                # Regular approval node - proceed to children
                child_node_ids = self.graph.get_children(approval_node_id)
                for child_id in child_node_ids:
                    # Get parent dependencies for this child
                    parent_ids = self.graph.get_parents(child_id)

                    # Single parent - queue immediately
                    if len(parent_ids) == 1:
                        queue.append(NodeQueueItem(node_id=child_id, inputs=resume_input))

        # Clear pending execution state
        del context.state["_pending_execution"]
        if self.state_backend:
            await self.state_backend.save(context.session_id, context.state)

        # Continue execution loop
        while queue and iteration < self.max_iterations:
            iteration += 1
            current = queue.pop(0)

            try:
                # Get node instance
                node = self.graph.get_node(current.node_id)

                # Emit node start event (skip for nodes that emit their own with metadata)
                from mesh.nodes.agent import AgentNode
                from mesh.nodes.llm import LLMNode
                from mesh.nodes.tool import ToolNode
                from mesh.nodes.start import StartNode

                if not isinstance(node, (AgentNode, LLMNode, ToolNode, StartNode)):
                    yield ExecutionEvent(
                        type=EventType.NODE_START,
                        node_id=current.node_id,
                        timestamp=datetime.now(),
                    )

                # Set up event queue for streaming during execution
                event_queue: asyncio.Queue[Optional[ExecutionEvent]] = asyncio.Queue()

                async def event_listener(event: ExecutionEvent):
                    """Capture events emitted during node execution."""
                    await event_queue.put(event)

                # Register listener
                self.events.on(event_listener)

                # Execute node in background task
                execute_task = asyncio.create_task(
                    node.execute(input=current.inputs, context=context)
                )

                # Yield events as they stream in
                result = None
                while True:
                    # Check if execution is complete
                    if execute_task.done():
                        # Get result
                        result = await execute_task
                        # Drain any remaining events
                        while not event_queue.empty():
                            try:
                                evt = event_queue.get_nowait()
                                if evt:
                                    yield evt
                            except asyncio.QueueEmpty:
                                break
                        break

                    # Wait for next event or task completion
                    try:
                        evt = await asyncio.wait_for(event_queue.get(), timeout=0.01)
                        if evt:
                            yield evt
                    except asyncio.TimeoutError:
                        continue

                # Remove listener
                self.events.off(event_listener)

                # Update state
                if result.state:
                    context.state.update(result.state)
                    if self.state_backend:
                        await self.state_backend.save(context.session_id, context.state)

                # Update chat history
                if result.chat_history:
                    context.chat_history.extend(result.chat_history)

                # Store execution data
                context.add_executed_node(current.node_id, result.output)

                # Emit node complete event
                yield ExecutionEvent(
                    type=EventType.NODE_COMPLETE,
                    node_id=current.node_id,
                    output=result.output,
                    timestamp=datetime.now(),
                    metadata=result.metadata,
                )

                # Check for approval pending - pause execution again
                if result.approval_pending:
                    # Save execution state for resume
                    pending_state = {
                        "queue": [(item.node_id, item.inputs) for item in queue],
                        "waiting_nodes": {
                            k: {
                                "node_id": v.node_id,
                                "received_inputs": v.received_inputs,
                                "expected_inputs": list(v.expected_inputs),
                            }
                            for k, v in waiting_nodes.items()
                        },
                        "loop_counts": loop_counts,
                        "current_node_id": current.node_id,
                        "approval_id": result.approval_id,
                        "approval_data": result.approval_data,
                        "iteration": iteration,
                    }
                    context.state["_pending_execution"] = pending_state

                    if self.state_backend:
                        await self.state_backend.save(context.session_id, context.state)

                    yield ExecutionEvent(
                        type=EventType.EXECUTION_COMPLETE,
                        output=result.output,
                        timestamp=datetime.now(),
                        metadata={
                            "status": ExecutionStatus.WAITING_FOR_APPROVAL,
                            "approval_id": result.approval_id,
                            "approval_data": result.approval_data,
                            "iterations": iteration,
                            "trace_id": context.trace_id,
                        },
                    )
                    return

                # Process outputs and queue children
                await self._process_node_outputs(
                    node_id=current.node_id,
                    result=result,
                    queue=queue,
                    waiting_nodes=waiting_nodes,
                    loop_counts=loop_counts,
                    context=context,
                )

            except Exception as e:
                error_msg = str(e)
                yield ExecutionEvent(
                    type=EventType.NODE_ERROR,
                    node_id=current.node_id,
                    error=error_msg,
                    timestamp=datetime.now(),
                )
                raise NodeExecutionError(current.node_id, error_msg, e)

        # Check for iteration limit
        if iteration >= self.max_iterations:
            yield ExecutionEvent(
                type=EventType.EXECUTION_ERROR,
                error=f"Execution exceeded maximum iterations ({self.max_iterations})",
            )

        # Get final output
        final_output = None
        if context.executed_data:
            final_output = context.executed_data[-1].get("output")

        # Emit execution complete event
        yield ExecutionEvent(
            type=EventType.EXECUTION_COMPLETE,
            output=final_output,
            timestamp=datetime.now(),
            metadata={
                "status": ExecutionStatus.COMPLETED,
                "iterations": iteration,
                "trace_id": context.trace_id,
            },
        )
