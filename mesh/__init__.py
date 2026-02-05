"""
Mesh: Python Agent Graph Orchestration Engine

A lightweight library for orchestrating multi-agent workflows as executable graphs.
Provides both programmatic (LangGraph-style) and declarative (React Flow JSON) interfaces
for building complex agent systems with token-by-token streaming.

Example:
    >>> from mesh import StateGraph, ExecutionContext, Executor
    >>> from mesh.nodes import AgentNode
    >>> from mesh.backends import MemoryBackend
    >>>
    >>> # Build graph programmatically
    >>> graph = StateGraph()
    >>> graph.add_node("agent", agent_instance, node_type="agent")
    >>> graph.add_edge("START", "agent")
    >>> graph.set_entry_point("agent")
    >>> compiled = graph.compile()
    >>>
    >>> # Execute with streaming
    >>> executor = Executor(compiled, MemoryBackend())
    >>> context = ExecutionContext(
    ...     graph_id="my-graph",
    ...     session_id="user-123",
    ...     chat_history=[],
    ...     variables={},
    ...     state={}
    ... )
    >>> async for event in executor.execute("Hello!", context):
    ...     print(event)
"""

__version__ = "0.1.6"

# Core components
from mesh.core.graph import ExecutionGraph, Edge, NodeConfig
from mesh.core.executor import Executor, ExecutionContext, NodeQueueItem
from mesh.core.events import ExecutionEvent
from mesh.core.state import StateManager

# Builders
from mesh.builders.state_graph import StateGraph

# Node registry
from mesh.utils.registry import NodeRegistry

# Backends
from mesh.backends.base import StateBackend
from mesh.backends.memory import MemoryBackend
from mesh.backends.sqlite import SQLiteBackend

# Parsers
from mesh.parsers.react_flow import ReactFlowParser

# Streaming
from mesh.streaming.iterator import StreamIterator
from mesh.streaming.sse import SSEAdapter

# Checkpointing
from mesh.checkpoints import (
    Checkpoint,
    CheckpointConfig,
    CheckpointNotFoundError,
    CheckpointIntegrityError,
)

# Interrupts (human-in-the-loop)
from mesh.interrupts import (
    InterruptConfig,
    InterruptState,
    InterruptResume,
    InterruptReject,
    InterruptError,
    InterruptNotFoundError,
    InterruptTimeoutError,
)

# Parallel execution (fan-out/fan-in)
from mesh.parallel import (
    Send,
    ParallelBranch,
    ParallelConfig,
    ParallelExecutor,
    ParallelResult,
    ParallelErrorStrategy,
    ParallelExecutionError,
    default_aggregator,
    list_aggregator,
    keyed_aggregator,
)

# Subgraph composition (nested graphs)
from mesh.subgraph import (
    Subgraph,
    SubgraphConfig,
    SubgraphBuilder,
)
from mesh.nodes.subgraph import SubgraphNode

# Streaming modes
from mesh.streaming.modes import (
    StreamMode,
    StreamModeAdapter,
    StateValue,
    StateUpdate,
    StreamMessage,
    DebugInfo,
)

# Input collection (auto-detect tools needing user input)
from mesh.input_collection import (
    RequiredParam,
    ToolInputRequirement,
    inspect_tool_required_params,
    inspect_data_handler_required_params,
    analyze_graph_input_requirements,
    inject_input_collection_interrupts,
    setup_input_collection,
    enable_input_collection_on_graph,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "ExecutionGraph",
    "Edge",
    "NodeConfig",
    "Executor",
    "ExecutionContext",
    "NodeQueueItem",
    "ExecutionEvent",
    "StateManager",
    # Builders
    "StateGraph",
    # Registry
    "NodeRegistry",
    # Backends
    "StateBackend",
    "MemoryBackend",
    "SQLiteBackend",
    # Parsers
    "ReactFlowParser",
    # Streaming
    "StreamIterator",
    "SSEAdapter",
    # Checkpointing
    "Checkpoint",
    "CheckpointConfig",
    "CheckpointNotFoundError",
    "CheckpointIntegrityError",
    # Interrupts
    "InterruptConfig",
    "InterruptState",
    "InterruptResume",
    "InterruptReject",
    "InterruptError",
    "InterruptNotFoundError",
    "InterruptTimeoutError",
    # Parallel execution
    "Send",
    "ParallelBranch",
    "ParallelConfig",
    "ParallelExecutor",
    "ParallelResult",
    "ParallelErrorStrategy",
    "ParallelExecutionError",
    "default_aggregator",
    "list_aggregator",
    "keyed_aggregator",
    # Subgraph composition
    "Subgraph",
    "SubgraphConfig",
    "SubgraphBuilder",
    "SubgraphNode",
    # Streaming modes
    "StreamMode",
    "StreamModeAdapter",
    "StateValue",
    "StateUpdate",
    "StreamMessage",
    "DebugInfo",
    # Input collection
    "RequiredParam",
    "ToolInputRequirement",
    "inspect_tool_required_params",
    "inspect_data_handler_required_params",
    "analyze_graph_input_requirements",
    "inject_input_collection_interrupts",
    "setup_input_collection",
    "enable_input_collection_on_graph",
]
