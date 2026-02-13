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
from mesh.interrupts import InterruptState, InterruptResume, InterruptReject


class ExecutionStatus:
    """Execution status constants."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    APPROVAL_REJECTED = "approval_rejected"
    WAITING_FOR_INTERRUPT = "waiting_for_interrupt"
    INTERRUPT_REJECTED = "interrupt_rejected"


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
                from mesh.nodes.conversation import ConversationNode

                # Skip Tool nodes that only feed into DynamicToolSelector or ConversationNode
                # These nodes provide metadata only, not execution
                if isinstance(node, ToolNode):
                    # Check if ALL children are tool-consumer nodes (DynamicToolSelector or ConversationNode)
                    children = self.graph.children.get(current.node_id, [])
                    all_children_are_tool_consumers = (
                        len(children) > 0 and
                        all(
                            isinstance(self.graph.get_node(child_id), (DynamicToolSelectorNode, ConversationNode))
                            for child_id in children
                        )
                    )
                    if all_children_are_tool_consumers:
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

                # Check for interrupt_before
                interrupt_before_config = self.graph.interrupt_before.get(current.node_id)
                if interrupt_before_config:
                    should_interrupt = True
                    condition = interrupt_before_config.get("condition")
                    if condition:
                        try:
                            should_interrupt = condition(context.state, current.inputs)
                        except Exception:
                            should_interrupt = False  # Condition error = don't interrupt

                    if should_interrupt:
                        # Extract metadata if extractor provided
                        metadata = {}
                        extractor = interrupt_before_config.get("metadata_extractor")
                        if extractor:
                            try:
                                metadata = extractor(context.state, current.inputs)
                            except Exception:
                                pass

                        # Create interrupt state
                        interrupt_state = InterruptState.create(
                            node_id=current.node_id,
                            position="before",
                            state=context.state,
                            input_data=current.inputs,
                            metadata=metadata,
                            pending_queue=[(item.node_id, item.inputs) for item in queue],
                            waiting_nodes={
                                k: {
                                    "node_id": v.node_id,
                                    "received_inputs": v.received_inputs,
                                    "expected_inputs": list(v.expected_inputs),
                                }
                                for k, v in waiting_nodes.items()
                            },
                            loop_counts=loop_counts,
                        )

                        # Save interrupt state to context
                        context.state["_interrupt_state"] = interrupt_state.to_dict()
                        if self.state_backend:
                            await self.state_backend.save(context.session_id, context.state)

                        # Emit interrupt event
                        yield ExecutionEvent(
                            type=EventType.INTERRUPT,
                            node_id=current.node_id,
                            metadata={
                                "interrupt_id": interrupt_state.interrupt_id,
                                "position": "before",
                                "state": context.state,
                                "input_data": current.inputs,
                                "review_metadata": metadata,
                            },
                            timestamp=datetime.now(),
                        )

                        # Emit execution complete with interrupt status
                        yield ExecutionEvent(
                            type=EventType.EXECUTION_COMPLETE,
                            output=None,
                            timestamp=datetime.now(),
                            metadata={
                                "status": ExecutionStatus.WAITING_FOR_INTERRUPT,
                                "interrupt_id": interrupt_state.interrupt_id,
                                "node_id": current.node_id,
                                "position": "before",
                                "iterations": iteration,
                                "trace_id": context.trace_id,
                            },
                        )
                        return  # Exit generator - execution paused

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
                context.add_executed_node(current.node_id, result.output, node_type=node.__class__.__name__)

                # Emit node complete event
                yield ExecutionEvent(
                    type=EventType.NODE_COMPLETE,
                    node_id=current.node_id,
                    output=result.output,
                    timestamp=datetime.now(),
                    metadata=result.metadata,
                )

                # Check for interrupt_after
                interrupt_after_config = self.graph.interrupt_after.get(current.node_id)
                if interrupt_after_config:
                    should_interrupt = True
                    condition = interrupt_after_config.get("condition")
                    if condition:
                        try:
                            should_interrupt = condition(context.state, result.output)
                        except Exception:
                            should_interrupt = False  # Condition error = don't interrupt

                    if should_interrupt:
                        # Extract metadata if extractor provided
                        metadata = {}
                        extractor = interrupt_after_config.get("metadata_extractor")
                        if extractor:
                            try:
                                metadata = extractor(context.state, result.output)
                            except Exception:
                                pass

                        # Create interrupt state
                        interrupt_state = InterruptState.create(
                            node_id=current.node_id,
                            position="after",
                            state=context.state,
                            input_data=current.inputs,
                            output_data=result.output,
                            metadata=metadata,
                            pending_queue=[(item.node_id, item.inputs) for item in queue],
                            waiting_nodes={
                                k: {
                                    "node_id": v.node_id,
                                    "received_inputs": v.received_inputs,
                                    "expected_inputs": list(v.expected_inputs),
                                }
                                for k, v in waiting_nodes.items()
                            },
                            loop_counts=loop_counts,
                        )

                        # Save interrupt state to context
                        context.state["_interrupt_state"] = interrupt_state.to_dict()
                        if self.state_backend:
                            await self.state_backend.save(context.session_id, context.state)

                        # Emit interrupt event
                        yield ExecutionEvent(
                            type=EventType.INTERRUPT,
                            node_id=current.node_id,
                            output=result.output,
                            metadata={
                                "interrupt_id": interrupt_state.interrupt_id,
                                "position": "after",
                                "state": context.state,
                                "output_data": result.output,
                                "review_metadata": metadata,
                            },
                            timestamp=datetime.now(),
                        )

                        # Emit execution complete with interrupt status
                        yield ExecutionEvent(
                            type=EventType.EXECUTION_COMPLETE,
                            output=result.output,
                            timestamp=datetime.now(),
                            metadata={
                                "status": ExecutionStatus.WAITING_FOR_INTERRUPT,
                                "interrupt_id": interrupt_state.interrupt_id,
                                "node_id": current.node_id,
                                "position": "after",
                                "iterations": iteration,
                                "trace_id": context.trace_id,
                            },
                        )
                        return  # Exit generator - execution paused

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
                # Insert at front so conversation runs before any deferred sibling nodes
                queue.insert(0, NodeQueueItem(node_id=approval_node_id, inputs=resume_input))
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
                context.add_executed_node(current.node_id, result.output, node_type=node.__class__.__name__)

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

    # Checkpoint methods

    async def checkpoint(
        self,
        context: ExecutionContext,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        queue: Optional[List[NodeQueueItem]] = None,
        waiting_nodes: Optional[Dict[str, WaitingNode]] = None,
    ) -> str:
        """Create a checkpoint of the current execution state.

        Captures full state including context, queue, and waiting nodes.
        Useful for branching workflows or recovering from failures.

        Args:
            context: Current execution context
            tags: Optional tags for filtering/searching
            metadata: Optional metadata (label, description, etc.)
            queue: Current execution queue (if mid-execution)
            waiting_nodes: Nodes waiting for inputs (if mid-execution)

        Returns:
            checkpoint_id: Unique identifier for the checkpoint

        Raises:
            ValueError: If no state backend configured

        Example:
            >>> checkpoint_id = await executor.checkpoint(context, tags=["review"])
        """
        from mesh.checkpoints import Checkpoint

        if not self.state_backend:
            raise ValueError("Cannot create checkpoint: no state_backend configured")

        # Serialize queue and waiting_nodes if provided
        pending_queue = []
        if queue:
            pending_queue = [(item.node_id, item.inputs) for item in queue]

        waiting_dict = {}
        if waiting_nodes:
            waiting_dict = {
                k: {
                    "node_id": v.node_id,
                    "received_inputs": v.received_inputs,
                    "expected_inputs": list(v.expected_inputs),
                }
                for k, v in waiting_nodes.items()
            }

        # Create checkpoint
        checkpoint = Checkpoint.create(
            session_id=context.session_id,
            graph_id=context.graph_id,
            state=context.state,
            chat_history=context.chat_history,
            variables=context.variables,
            executed_data=context.executed_data,
            loop_iterations=context.loop_iterations,
            pending_queue=pending_queue,
            waiting_nodes=waiting_dict,
            tags=tags,
            metadata=metadata,
        )

        # Save checkpoint
        await self.state_backend.save_checkpoint(checkpoint)

        return checkpoint.checkpoint_id

    async def restore(
        self,
        checkpoint_id: str,
    ) -> ExecutionContext:
        """Restore execution context from a checkpoint.

        Loads full state including context variables, state, and chat history.
        Does NOT restore queue position - use `replay()` for that.

        Args:
            checkpoint_id: ID of checkpoint to restore from

        Returns:
            ExecutionContext: Restored context ready for execution

        Raises:
            ValueError: If no state backend configured
            CheckpointNotFoundError: If checkpoint not found
            CheckpointIntegrityError: If checkpoint fails integrity check

        Example:
            >>> context = await executor.restore(checkpoint_id)
            >>> async for event in executor.execute("new input", context):
            ...     print(event)
        """
        from mesh.checkpoints import (
            Checkpoint,
            CheckpointNotFoundError,
            CheckpointIntegrityError,
        )

        if not self.state_backend:
            raise ValueError("Cannot restore: no state_backend configured")

        # Load checkpoint
        checkpoint = await self.state_backend.load_checkpoint(checkpoint_id)
        if not checkpoint:
            raise CheckpointNotFoundError(checkpoint_id)

        # Verify integrity
        if not checkpoint.verify_integrity():
            raise CheckpointIntegrityError(
                checkpoint_id,
                checkpoint.state_hash,
                checkpoint._compute_hash(),
            )

        # Create context from checkpoint
        context = ExecutionContext(
            graph_id=checkpoint.graph_id,
            session_id=checkpoint.session_id,
            chat_history=list(checkpoint.chat_history),
            variables=dict(checkpoint.variables),
            state=dict(checkpoint.state),
            executed_data=list(checkpoint.executed_data),
            loop_iterations=dict(checkpoint.loop_iterations),
        )

        return context

    async def branch(
        self,
        checkpoint_id: str,
        new_session_id: str,
        branch_name: Optional[str] = None,
    ) -> ExecutionContext:
        """Create a new branch from a checkpoint.

        Creates a new session with state copied from the checkpoint.
        The original checkpoint is preserved as the parent.

        Args:
            checkpoint_id: ID of checkpoint to branch from
            new_session_id: Session ID for the new branch
            branch_name: Optional name for the branch

        Returns:
            ExecutionContext: New context for the branch

        Raises:
            ValueError: If no state backend configured
            CheckpointNotFoundError: If checkpoint not found

        Example:
            >>> branch_ctx = await executor.branch(checkpoint_id, "exploration-1")
            >>> async for event in executor.execute("try different approach", branch_ctx):
            ...     print(event)
        """
        from mesh.checkpoints import Checkpoint, CheckpointNotFoundError

        if not self.state_backend:
            raise ValueError("Cannot branch: no state_backend configured")

        # Load parent checkpoint
        parent_checkpoint = await self.state_backend.load_checkpoint(checkpoint_id)
        if not parent_checkpoint:
            raise CheckpointNotFoundError(checkpoint_id)

        # Create context from parent
        context = ExecutionContext(
            graph_id=parent_checkpoint.graph_id,
            session_id=new_session_id,
            chat_history=list(parent_checkpoint.chat_history),
            variables=dict(parent_checkpoint.variables),
            state=dict(parent_checkpoint.state),
            executed_data=list(parent_checkpoint.executed_data),
            loop_iterations=dict(parent_checkpoint.loop_iterations),
        )

        # Create branch checkpoint
        branch_checkpoint = Checkpoint.create(
            session_id=new_session_id,
            graph_id=parent_checkpoint.graph_id,
            state=context.state,
            chat_history=context.chat_history,
            variables=context.variables,
            executed_data=context.executed_data,
            loop_iterations=context.loop_iterations,
            parent_checkpoint_id=checkpoint_id,
            tags=["branch"],
            metadata={"branch_name": branch_name} if branch_name else {},
        )

        await self.state_backend.save_checkpoint(branch_checkpoint)

        # Save initial state to new session
        await self.state_backend.save(new_session_id, context.state)

        return context

    async def replay(
        self,
        checkpoint_id: str,
        until_node: Optional[str] = None,
    ) -> AsyncIterator[ExecutionEvent]:
        """Replay execution from a checkpoint.

        Re-executes the graph from checkpoint state, optionally stopping
        at a specific node. Useful for debugging and reviewing execution.

        Args:
            checkpoint_id: ID of checkpoint to replay from
            until_node: Optional node ID to stop at (inclusive)

        Yields:
            ExecutionEvent: Streaming execution events

        Raises:
            ValueError: If no state backend configured
            CheckpointNotFoundError: If checkpoint not found

        Example:
            >>> async for event in executor.replay(checkpoint_id, until_node="reviewer"):
            ...     print(f"{event.node_id}: {event.type}")
        """
        from mesh.checkpoints import CheckpointNotFoundError

        if not self.state_backend:
            raise ValueError("Cannot replay: no state_backend configured")

        # Emit replay start event
        yield ExecutionEvent(
            type=EventType.REPLAY_START,
            metadata={"checkpoint_id": checkpoint_id, "until_node": until_node},
        )

        # Restore context
        context = await self.restore(checkpoint_id)
        context._event_emitter = self.events

        # Check if we have a pending queue to resume from
        checkpoint = await self.state_backend.load_checkpoint(checkpoint_id)
        if not checkpoint:
            raise CheckpointNotFoundError(checkpoint_id)

        # If checkpoint has pending queue, resume from there
        if checkpoint.pending_queue:
            # Restore queue and waiting nodes
            queue: List[NodeQueueItem] = []
            for node_id, inputs in checkpoint.pending_queue:
                queue.append(NodeQueueItem(node_id=node_id, inputs=inputs))

            waiting_nodes: Dict[str, WaitingNode] = {}
            for node_id, data in checkpoint.waiting_nodes.items():
                waiting_nodes[node_id] = WaitingNode(
                    node_id=data["node_id"],
                    received_inputs=data["received_inputs"],
                    expected_inputs=set(data["expected_inputs"]),
                )

            loop_counts: Dict[str, int] = dict(checkpoint.loop_iterations)
            iteration = 0

            # Execute from queue
            while queue and iteration < self.max_iterations:
                iteration += 1
                current = queue.pop(0)

                # Check for stop condition
                if until_node and current.node_id == until_node:
                    # Execute this node then stop
                    node = self.graph.get_node(current.node_id)
                    result = await node.execute(input=current.inputs, context=context)

                    yield ExecutionEvent(
                        type=EventType.NODE_COMPLETE,
                        node_id=current.node_id,
                        output=result.output,
                        timestamp=datetime.now(),
                    )

                    yield ExecutionEvent(
                        type=EventType.REPLAY_COMPLETE,
                        metadata={
                            "checkpoint_id": checkpoint_id,
                            "stopped_at": until_node,
                        },
                    )
                    return

                # Normal execution
                node = self.graph.get_node(current.node_id)

                yield ExecutionEvent(
                    type=EventType.NODE_START,
                    node_id=current.node_id,
                    timestamp=datetime.now(),
                )

                result = await node.execute(input=current.inputs, context=context)

                # Update context
                if result.state:
                    context.state.update(result.state)
                if result.chat_history:
                    context.chat_history.extend(result.chat_history)
                context.add_executed_node(current.node_id, result.output, node_type=node.__class__.__name__)

                yield ExecutionEvent(
                    type=EventType.NODE_COMPLETE,
                    node_id=current.node_id,
                    output=result.output,
                    timestamp=datetime.now(),
                )

                # Queue children
                await self._process_node_outputs(
                    node_id=current.node_id,
                    result=result,
                    queue=queue,
                    waiting_nodes=waiting_nodes,
                    loop_counts=loop_counts,
                    context=context,
                )

            yield ExecutionEvent(
                type=EventType.REPLAY_COMPLETE,
                metadata={
                    "checkpoint_id": checkpoint_id,
                    "iterations": iteration,
                },
            )
        else:
            # No pending queue - just restore and emit complete
            yield ExecutionEvent(
                type=EventType.REPLAY_COMPLETE,
                metadata={
                    "checkpoint_id": checkpoint_id,
                    "note": "Checkpoint has no pending execution state",
                },
            )

    async def resume_from_interrupt(
        self,
        context: ExecutionContext,
        resume_or_reject: Union[InterruptResume, InterruptReject],
    ) -> AsyncIterator[ExecutionEvent]:
        """Resume execution after an interrupt pause.

        This method continues execution from where it was paused by an interrupt.
        It loads the saved interrupt state and processes the resume/reject decision.

        Args:
            context: Execution context (should have been reloaded from state_backend)
            resume_or_reject: Either InterruptResume or InterruptReject

        Yields:
            ExecutionEvent: Streaming execution events

        Raises:
            ValueError: If no interrupt state found
            NodeExecutionError: If a node execution fails

        Example:
            >>> # After interrupt paused execution:
            >>> resume = InterruptResume(modified_state={"approved": True})
            >>> async for event in executor.resume_from_interrupt(context, resume):
            ...     print(event.type)
        """
        # Set event emitter in context
        context._event_emitter = self.events

        # Get interrupt state
        interrupt_data = context.state.get("_interrupt_state")
        if not interrupt_data:
            raise ValueError(
                "No interrupt state found. Cannot resume without a prior interrupt."
            )

        interrupt_state = InterruptState.from_dict(interrupt_data)

        # Handle rejection
        if isinstance(resume_or_reject, InterruptReject):
            yield ExecutionEvent(
                type=EventType.INTERRUPT_REJECTED,
                node_id=interrupt_state.node_id,
                metadata={
                    "interrupt_id": interrupt_state.interrupt_id,
                    "reason": resume_or_reject.reason,
                    "position": interrupt_state.position,
                },
                timestamp=datetime.now(),
            )

            # Clear interrupt state
            del context.state["_interrupt_state"]
            if self.state_backend:
                await self.state_backend.save(context.session_id, context.state)

            # Emit execution complete with rejection
            yield ExecutionEvent(
                type=EventType.EXECUTION_COMPLETE,
                output=None,
                timestamp=datetime.now(),
                metadata={
                    "status": ExecutionStatus.INTERRUPT_REJECTED,
                    "interrupt_id": interrupt_state.interrupt_id,
                    "rejection_reason": resume_or_reject.reason,
                    "trace_id": context.trace_id,
                },
            )
            return

        # Handle resume
        resume = resume_or_reject

        # Emit resume event
        yield ExecutionEvent(
            type=EventType.INTERRUPT_RESUMED,
            node_id=interrupt_state.node_id,
            metadata={
                "interrupt_id": interrupt_state.interrupt_id,
                "position": interrupt_state.position,
                "modified_state": resume.modified_state is not None,
                "modified_input": resume.modified_input is not None,
                "skip_node": resume.skip_node,
            },
            timestamp=datetime.now(),
        )

        # Apply state modifications if provided
        if resume.modified_state:
            context.state.update(resume.modified_state)

        # Restore queue and waiting nodes
        queue: List[NodeQueueItem] = []
        for node_id, inputs in interrupt_state.pending_queue:
            queue.append(NodeQueueItem(node_id=node_id, inputs=inputs))

        waiting_nodes: Dict[str, WaitingNode] = {}
        for node_id, data in interrupt_state.waiting_nodes.items():
            waiting_nodes[node_id] = WaitingNode(
                node_id=data["node_id"],
                received_inputs=data["received_inputs"],
                expected_inputs=set(data["expected_inputs"]),
            )

        loop_counts: Dict[str, int] = dict(interrupt_state.loop_counts)
        iteration = 0

        # Track which node we're resuming from to skip re-interrupting it
        resumed_from_node_id = interrupt_state.node_id
        resumed_from_position = interrupt_state.position

        # Determine what to do based on position and skip_node
        if interrupt_state.position == "before":
            if resume.skip_node:
                # Skip the node entirely - just continue with the queue
                resumed_from_node_id = None  # Don't skip interrupt checks for other nodes
            else:
                # Execute the node that was interrupted
                # Use modified input if provided, otherwise original
                node_input = resume.modified_input if resume.modified_input is not None else interrupt_state.input_data
                queue.insert(0, NodeQueueItem(node_id=interrupt_state.node_id, inputs=node_input))
        else:
            # "after" position - node already executed
            # Need to process its outputs and queue children
            # Get the node result from interrupt state
            node = self.graph.get_node(interrupt_state.node_id)

            # Create a mock result to queue children
            from mesh.nodes.base import NodeResult
            mock_result = NodeResult(output=interrupt_state.output_data)

            # Queue children of the interrupted node
            await self._process_node_outputs(
                node_id=interrupt_state.node_id,
                result=mock_result,
                queue=queue,
                waiting_nodes=waiting_nodes,
                loop_counts=loop_counts,
                context=context,
            )

        # Clear interrupt state
        del context.state["_interrupt_state"]
        if self.state_backend:
            await self.state_backend.save(context.session_id, context.state)

        # Continue execution loop
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

                if not isinstance(node, (AgentNode, LLMNode, ToolNode, StartNode)):
                    yield ExecutionEvent(
                        type=EventType.NODE_START,
                        node_id=current.node_id,
                        timestamp=datetime.now(),
                    )

                # Check for interrupt_before (skip if this is the node we just resumed from)
                # We skip the interrupt check for the resumed node to avoid re-triggering
                skip_interrupt_before = (
                    resumed_from_node_id == current.node_id and
                    resumed_from_position == "before"
                )
                interrupt_before_config = self.graph.interrupt_before.get(current.node_id)
                if interrupt_before_config and not skip_interrupt_before:
                    should_interrupt = True
                    condition = interrupt_before_config.get("condition")
                    if condition:
                        try:
                            should_interrupt = condition(context.state, current.inputs)
                        except Exception:
                            should_interrupt = False

                    if should_interrupt:
                        metadata = {}
                        extractor = interrupt_before_config.get("metadata_extractor")
                        if extractor:
                            try:
                                metadata = extractor(context.state, current.inputs)
                            except Exception:
                                pass

                        new_interrupt_state = InterruptState.create(
                            node_id=current.node_id,
                            position="before",
                            state=context.state,
                            input_data=current.inputs,
                            metadata=metadata,
                            pending_queue=[(item.node_id, item.inputs) for item in queue],
                            waiting_nodes={
                                k: {
                                    "node_id": v.node_id,
                                    "received_inputs": v.received_inputs,
                                    "expected_inputs": list(v.expected_inputs),
                                }
                                for k, v in waiting_nodes.items()
                            },
                            loop_counts=loop_counts,
                        )

                        context.state["_interrupt_state"] = new_interrupt_state.to_dict()
                        if self.state_backend:
                            await self.state_backend.save(context.session_id, context.state)

                        yield ExecutionEvent(
                            type=EventType.INTERRUPT,
                            node_id=current.node_id,
                            metadata={
                                "interrupt_id": new_interrupt_state.interrupt_id,
                                "position": "before",
                                "state": context.state,
                                "input_data": current.inputs,
                                "review_metadata": metadata,
                            },
                            timestamp=datetime.now(),
                        )

                        yield ExecutionEvent(
                            type=EventType.EXECUTION_COMPLETE,
                            output=None,
                            timestamp=datetime.now(),
                            metadata={
                                "status": ExecutionStatus.WAITING_FOR_INTERRUPT,
                                "interrupt_id": new_interrupt_state.interrupt_id,
                                "node_id": current.node_id,
                                "position": "before",
                                "iterations": iteration,
                                "trace_id": context.trace_id,
                            },
                        )
                        return

                # Clear the resumed_from tracking after first node - subsequent nodes should
                # still trigger their own interrupts
                if resumed_from_node_id == current.node_id:
                    resumed_from_node_id = None

                # Set up event queue for streaming
                event_queue: asyncio.Queue[Optional[ExecutionEvent]] = asyncio.Queue()

                async def event_listener(event: ExecutionEvent):
                    await event_queue.put(event)

                self.events.on(event_listener)

                # Execute node
                execute_task = asyncio.create_task(
                    node.execute(input=current.inputs, context=context)
                )

                result = None
                while True:
                    if execute_task.done():
                        result = await execute_task
                        while not event_queue.empty():
                            try:
                                evt = event_queue.get_nowait()
                                if evt:
                                    yield evt
                            except asyncio.QueueEmpty:
                                break
                        break

                    try:
                        evt = await asyncio.wait_for(event_queue.get(), timeout=0.01)
                        if evt:
                            yield evt
                    except asyncio.TimeoutError:
                        continue

                self.events.off(event_listener)

                # Update state
                if result.state:
                    context.state.update(result.state)
                    if self.state_backend:
                        await self.state_backend.save(context.session_id, context.state)

                if result.chat_history:
                    context.chat_history.extend(result.chat_history)

                context.add_executed_node(current.node_id, result.output, node_type=node.__class__.__name__)

                yield ExecutionEvent(
                    type=EventType.NODE_COMPLETE,
                    node_id=current.node_id,
                    output=result.output,
                    timestamp=datetime.now(),
                    metadata=result.metadata,
                )

                # Check for interrupt_after
                interrupt_after_config = self.graph.interrupt_after.get(current.node_id)
                if interrupt_after_config:
                    should_interrupt = True
                    condition = interrupt_after_config.get("condition")
                    if condition:
                        try:
                            should_interrupt = condition(context.state, result.output)
                        except Exception:
                            should_interrupt = False

                    if should_interrupt:
                        metadata = {}
                        extractor = interrupt_after_config.get("metadata_extractor")
                        if extractor:
                            try:
                                metadata = extractor(context.state, result.output)
                            except Exception:
                                pass

                        new_interrupt_state = InterruptState.create(
                            node_id=current.node_id,
                            position="after",
                            state=context.state,
                            input_data=current.inputs,
                            output_data=result.output,
                            metadata=metadata,
                            pending_queue=[(item.node_id, item.inputs) for item in queue],
                            waiting_nodes={
                                k: {
                                    "node_id": v.node_id,
                                    "received_inputs": v.received_inputs,
                                    "expected_inputs": list(v.expected_inputs),
                                }
                                for k, v in waiting_nodes.items()
                            },
                            loop_counts=loop_counts,
                        )

                        context.state["_interrupt_state"] = new_interrupt_state.to_dict()
                        if self.state_backend:
                            await self.state_backend.save(context.session_id, context.state)

                        yield ExecutionEvent(
                            type=EventType.INTERRUPT,
                            node_id=current.node_id,
                            output=result.output,
                            metadata={
                                "interrupt_id": new_interrupt_state.interrupt_id,
                                "position": "after",
                                "state": context.state,
                                "output_data": result.output,
                                "review_metadata": metadata,
                            },
                            timestamp=datetime.now(),
                        )

                        yield ExecutionEvent(
                            type=EventType.EXECUTION_COMPLETE,
                            output=result.output,
                            timestamp=datetime.now(),
                            metadata={
                                "status": ExecutionStatus.WAITING_FOR_INTERRUPT,
                                "interrupt_id": new_interrupt_state.interrupt_id,
                                "node_id": current.node_id,
                                "position": "after",
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

    async def stream(
        self,
        input_data: Union[str, Dict[str, Any]],
        context: ExecutionContext,
        mode: "StreamMode" = None,
    ) -> AsyncIterator[Any]:
        """Execute graph with specified streaming mode.

        This method provides different views of execution based on the
        streaming mode:
        - VALUES: Full state after each node
        - UPDATES: Only state changes (deltas)
        - MESSAGES: Accumulated chat messages
        - EVENTS: All execution events (default)
        - DEBUG: Everything including internal state

        Args:
            input_data: Input to the graph
            context: Execution context
            mode: Streaming mode (defaults to EVENTS)

        Yields:
            Items appropriate to the streaming mode:
            - StateValue for VALUES mode
            - StateUpdate for UPDATES mode
            - StreamMessage for MESSAGES mode
            - ExecutionEvent for EVENTS mode
            - DebugInfo for DEBUG mode

        Example:
            >>> from mesh.streaming.modes import StreamMode
            >>> async for state in executor.stream(input, context, mode=StreamMode.VALUES):
            ...     print(f"State: {state}")
        """
        from mesh.streaming.modes import StreamMode, StreamModeAdapter

        # Default to EVENTS mode
        if mode is None:
            mode = StreamMode.EVENTS

        # Create adapter for the specified mode
        adapter = StreamModeAdapter(mode)

        # Get the underlying execution event stream
        events = self.execute(input_data, context)

        # Adapt events to the specified mode
        async for item in adapter.adapt(events, context):
            yield item
