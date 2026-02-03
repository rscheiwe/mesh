"""Parallel execution support for Mesh.

This module provides fan-out/fan-in patterns for concurrent node execution:
- Send: Dynamic dispatch to nodes with specific inputs
- ParallelBranch: Static parallel branch configuration
- ParallelExecutor: Manages concurrent execution with semaphore-based limiting
- ParallelConfig: Configuration for parallel execution behavior

Example:
    >>> from mesh import StateGraph, Send
    >>>
    >>> # Static fan-out
    >>> graph.add_parallel_edges("START", ["worker_1", "worker_2", "worker_3"])
    >>> graph.add_fan_in_edge(["worker_1", "worker_2", "worker_3"], "aggregator")
    >>>
    >>> # Dynamic fan-out with Send
    >>> def dispatch(state):
    ...     return [Send("worker", {"item": i}) for i in state["items"]]
    >>> graph.add_conditional_edges("router", dispatch)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from mesh.core.state import ExecutionContext


class ParallelErrorStrategy(str, Enum):
    """Strategy for handling errors in parallel branches.

    - FAIL_FAST: Stop all branches on first error
    - CONTINUE_ALL: Continue execution, collect all errors
    - CONTINUE_PARTIAL: Continue with successful branches, fail if all fail
    """
    FAIL_FAST = "fail_fast"
    CONTINUE_ALL = "continue_all"
    CONTINUE_PARTIAL = "continue_partial"


@dataclass
class ParallelConfig:
    """Configuration for parallel execution.

    Attributes:
        max_concurrency: Maximum concurrent branches (default: 10)
        error_strategy: How to handle errors in parallel branches
        timeout: Optional timeout in seconds for all branches
        preserve_order: Whether to preserve result order for dynamic sends
    """
    max_concurrency: int = 10
    error_strategy: ParallelErrorStrategy = ParallelErrorStrategy.CONTINUE_ALL
    timeout: Optional[float] = None
    preserve_order: bool = True

    def __post_init__(self):
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass
class Send:
    """Represents a dynamic dispatch to a node with specific input.

    Used for fan-out patterns where parallelism is determined at runtime.

    Attributes:
        node: Target node ID to send to
        input: Input data for the node

    Example:
        >>> def dispatch_tasks(state):
        ...     items = state.get("items", [])
        ...     return [Send("worker", {"item": item}) for item in items]
    """
    node: str
    input: Dict[str, Any]

    def __post_init__(self):
        if not isinstance(self.node, str):
            raise ValueError("Send.node must be a string node name")
        if not isinstance(self.input, dict):
            raise ValueError("Send.input must be a dict")


@dataclass
class ParallelBranch:
    """Represents a static parallel branch configuration.

    Used for fan-out patterns where branches are known at graph definition time.

    Attributes:
        source: Source node ID
        targets: List of target node IDs to execute in parallel
        is_dynamic: Whether targets are determined dynamically via Send
    """
    source: str
    targets: List[str]
    is_dynamic: bool = False

    def __post_init__(self):
        if not self.is_dynamic and len(self.targets) < 2:
            raise ValueError("ParallelBranch requires at least 2 targets for static branches")


@dataclass
class FanInConfig:
    """Configuration for fan-in (aggregation) node.

    Attributes:
        target: Target node ID that receives aggregated results
        sources: List of source node IDs to wait for
        aggregator: Optional function to aggregate results
        wait_for_all: If True, wait for all sources; if False, proceed when any completes
    """
    target: str
    sources: List[str]
    aggregator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    wait_for_all: bool = True


@dataclass
class ParallelResult:
    """Result from parallel execution.

    Attributes:
        results: Mapping of node/branch ID to result
        errors: List of errors that occurred (if error_strategy allows continuation)
        completed: List of successfully completed branch IDs
        failed: List of failed branch IDs
    """
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Tuple[str, Exception]] = field(default_factory=list)
    completed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def all_succeeded(self) -> bool:
        return len(self.errors) == 0 and len(self.completed) > 0

    @property
    def partial_success(self) -> bool:
        return len(self.completed) > 0 and len(self.failed) > 0


class ParallelExecutionError(Exception):
    """Error during parallel execution."""

    def __init__(self, message: str, errors: List[Tuple[str, Exception]]):
        super().__init__(message)
        self.errors = errors


class ParallelExecutor:
    """Manages concurrent execution of parallel branches.

    Uses asyncio.Semaphore to limit concurrency and asyncio.gather
    for concurrent execution with configurable error handling.

    Example:
        >>> executor = ParallelExecutor(max_concurrency=10)
        >>> results = await executor.execute_parallel(
        ...     branches=[("node1", {"x": 1}), ("node2", {"x": 2})],
        ...     node_executor=my_executor_fn,
        ...     context=ctx,
        ... )
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize parallel executor.

        Args:
            config: Parallel execution configuration
        """
        self.config = config or ParallelConfig()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrency)

    async def execute_parallel(
        self,
        branches: List[Tuple[str, Dict[str, Any]]],
        node_executor: Callable,
        context: "ExecutionContext",
    ) -> ParallelResult:
        """Execute multiple branches concurrently.

        Args:
            branches: List of (node_id, input_data) tuples
            node_executor: Async callable (node_id, input, context) -> result
            context: Execution context

        Returns:
            ParallelResult with results and any errors
        """
        if not branches:
            return ParallelResult()

        async def run_branch(node_id: str, input_data: Dict[str, Any]) -> Tuple[str, Any]:
            async with self.semaphore:
                result = await node_executor(node_id, input_data, context)
                return node_id, result

        # Create tasks
        tasks = [
            asyncio.create_task(run_branch(node_id, input_data))
            for node_id, input_data in branches
        ]

        # Execute with optional timeout
        if self.config.timeout:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.config.timeout,
                )
            except asyncio.TimeoutError:
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                raise ParallelExecutionError(
                    f"Parallel execution timed out after {self.config.timeout}s",
                    [],
                )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        parallel_result = ParallelResult()

        for i, result in enumerate(results):
            node_id = branches[i][0]

            if isinstance(result, Exception):
                parallel_result.errors.append((node_id, result))
                parallel_result.failed.append(node_id)

                # Handle based on error strategy
                if self.config.error_strategy == ParallelErrorStrategy.FAIL_FAST:
                    raise ParallelExecutionError(
                        f"Parallel execution failed at branch '{node_id}'",
                        [(node_id, result)],
                    )
            else:
                _, node_result = result
                parallel_result.results[node_id] = node_result
                parallel_result.completed.append(node_id)

        # Check for total failure with CONTINUE_PARTIAL
        if (
            self.config.error_strategy == ParallelErrorStrategy.CONTINUE_PARTIAL
            and len(parallel_result.completed) == 0
            and len(parallel_result.failed) > 0
        ):
            raise ParallelExecutionError(
                "All parallel branches failed",
                parallel_result.errors,
            )

        return parallel_result

    async def execute_sends(
        self,
        sends: List[Send],
        node_executor: Callable,
        context: "ExecutionContext",
    ) -> ParallelResult:
        """Execute dynamic Send dispatches concurrently.

        Args:
            sends: List of Send objects
            node_executor: Async callable (node_id, input, context) -> result
            context: Execution context

        Returns:
            ParallelResult with results in order if preserve_order is True
        """
        if not sends:
            return ParallelResult()

        async def run_send(send: Send, index: int) -> Tuple[int, str, Any]:
            async with self.semaphore:
                result = await node_executor(send.node, send.input, context)
                return index, send.node, result

        # Create tasks
        tasks = [
            asyncio.create_task(run_send(send, i))
            for i, send in enumerate(sends)
        ]

        # Execute with optional timeout
        if self.config.timeout:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.config.timeout,
                )
            except asyncio.TimeoutError:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                raise ParallelExecutionError(
                    f"Send execution timed out after {self.config.timeout}s",
                    [],
                )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        parallel_result = ParallelResult()

        if self.config.preserve_order:
            # Use ordered list for Send results
            ordered_results = [None] * len(sends)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    send = sends[i]
                    parallel_result.errors.append((f"send_{i}_{send.node}", result))
                    parallel_result.failed.append(f"send_{i}_{send.node}")

                    if self.config.error_strategy == ParallelErrorStrategy.FAIL_FAST:
                        raise ParallelExecutionError(
                            f"Send execution failed at index {i}",
                            [(f"send_{i}_{send.node}", result)],
                        )
                else:
                    index, node_id, node_result = result
                    ordered_results[index] = node_result
                    parallel_result.completed.append(f"send_{index}_{node_id}")

            # Store ordered results as list
            parallel_result.results["_ordered"] = ordered_results
        else:
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    send = sends[i]
                    parallel_result.errors.append((f"send_{i}_{send.node}", result))
                    parallel_result.failed.append(f"send_{i}_{send.node}")

                    if self.config.error_strategy == ParallelErrorStrategy.FAIL_FAST:
                        raise ParallelExecutionError(
                            f"Send execution failed at index {i}",
                            [(f"send_{i}_{send.node}", result)],
                        )
                else:
                    index, node_id, node_result = result
                    key = f"send_{index}_{node_id}"
                    parallel_result.results[key] = node_result
                    parallel_result.completed.append(key)

        # Check for total failure with CONTINUE_PARTIAL
        if (
            self.config.error_strategy == ParallelErrorStrategy.CONTINUE_PARTIAL
            and len(parallel_result.completed) == 0
            and len(parallel_result.failed) > 0
        ):
            raise ParallelExecutionError(
                "All Send dispatches failed",
                parallel_result.errors,
            )

        return parallel_result


def default_aggregator(results: Dict[str, Any]) -> Dict[str, Any]:
    """Default aggregator that merges all results into a single dict.

    If all results are dicts, merges them.
    Otherwise, returns results as-is with source keys.

    Args:
        results: Mapping of source node ID to result

    Returns:
        Aggregated result dict
    """
    if not results:
        return {}

    # Check if all results are dicts
    all_dicts = all(isinstance(v, dict) for v in results.values())

    if all_dicts:
        # Merge all dicts, later entries overwrite earlier
        merged = {}
        for result in results.values():
            merged.update(result)
        return merged
    else:
        # Return with source keys
        return {"parallel_results": results}


def list_aggregator(results: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregator that collects results into a list.

    Args:
        results: Mapping of source node ID to result

    Returns:
        Dict with 'results' key containing list of all results
    """
    return {"results": list(results.values())}


def keyed_aggregator(key: str = "findings") -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create an aggregator that stores results under a specific key.

    Args:
        key: Key name for the aggregated results

    Returns:
        Aggregator function
    """
    def aggregator(results: Dict[str, Any]) -> Dict[str, Any]:
        return {key: results}

    return aggregator
