# LangGraph Parity Features

This directory contains examples demonstrating Mesh's LangGraph parity features:

## Examples

### 1. Checkpointing (`01_checkpointing.py`)
Save and restore execution state for long-running workflows.
- Manual checkpoints with `executor.checkpoint(context)`
- Automatic checkpoint restore with `executor.restore(checkpoint_id)`
- List and manage checkpoints via backend

### 2. Interrupts (`02_interrupts.py`)
Human-in-the-loop patterns for approval workflows.
- Set interrupt points with `graph.set_interrupt_before("node_id")`
- Resume execution with `executor.resume_from_interrupt(context, InterruptResume())`
- Reject and abort with `InterruptReject(reason="...")`

### 3. Parallel Execution (`03_parallel_execution.py`)
Fan-out/fan-in patterns for concurrent node execution.
- Dynamic branching with `Send(node_id, input)`
- Parallel branches with `ParallelBranch`
- Custom aggregation with `ParallelConfig`

### 4. Subgraph Composition (`04_subgraph_composition.py`)
Nested graph execution for modular workflows.
- Embed compiled graphs as nodes
- State isolation with `SubgraphConfig(isolated=True)`
- Input/output mapping between graphs

### 5. Streaming Modes (`05_streaming_modes.py`)
Different views of execution for various use cases.
- `VALUES`: Full state after each node
- `UPDATES`: State deltas only
- `MESSAGES`: Chat messages only
- `EVENTS`: All execution events (default)
- `DEBUG`: Everything including internals

## Running Examples

```bash
# Run individual examples
python examples/08_upgrades/01_checkpointing.py
python examples/08_upgrades/02_interrupts.py
python examples/08_upgrades/03_parallel_execution.py
python examples/08_upgrades/04_subgraph_composition.py
python examples/08_upgrades/05_streaming_modes.py
```

## Feature Summary

| Feature | Key Classes | Use Case |
|---------|-------------|----------|
| Checkpointing | `Checkpoint`, `CheckpointConfig` | Resume long workflows, crash recovery |
| Interrupts | `InterruptResume`, `InterruptReject` | Human approval, confirmation steps |
| Parallel | `Send`, `ParallelBranch`, `ParallelConfig` | Concurrent research, multi-source queries |
| Subgraphs | `Subgraph`, `SubgraphConfig`, `SubgraphBuilder` | Modular agents, reusable workflows |
| Streaming | `StreamMode`, `StateValue`, `StateUpdate` | UI updates, debugging, logging |
