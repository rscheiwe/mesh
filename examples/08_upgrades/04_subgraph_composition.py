"""Subgraph Composition Example - Nested graph execution.

Architecture:
============

    ┌─────────────────────────────────────────────────────────────┐
    │                    PARENT GRAPH                             │
    │                                                             │
    │   [Plan] ──► ┌─────────────────────────────┐ ──► [Report]  │
    │              │       SUBGRAPH              │                │
    │              │  ┌───────────────────────┐  │                │
    │              │  │ [Search] ──► [Analyze]│  │                │
    │              │  └───────────────────────┘  │                │
    │              │                             │                │
    │              │  State: isolated or shared  │                │
    │              │  Events: prefixed           │                │
    │              └─────────────────────────────┘                │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    Nested Subgraphs (3 levels):
    ┌─────────────────────────────────────────────────────────────┐
    │  LEVEL 1 (Root)                                             │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  LEVEL 2                                            │    │
    │  │  ┌─────────────────────────────────────────────┐    │    │
    │  │  │  LEVEL 3 (Innermost)                        │    │    │
    │  │  │  [Deep Tool]                                │    │    │
    │  │  └─────────────────────────────────────────────┘    │    │
    │  └─────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────┘

This example demonstrates subgraph patterns for:
- Modular workflow composition
- Reusable agent components
- State isolation between graphs
- Input/output mapping

Key APIs:
- Subgraph(compiled_graph, name, config) - Wrap a graph for embedding
- SubgraphConfig(isolated, inherit_keys, input_mapping, output_mapping)
- SubgraphBuilder - Fluent API for building subgraphs
- graph.add_node("id", subgraph, node_type="subgraph")
"""

import asyncio
from mesh import (
    StateGraph,
    Executor,
    ExecutionContext,
    MemoryBackend,
    Subgraph,
    SubgraphConfig,
    SubgraphBuilder,
)
from mesh.core.events import EventType


# =============================================================================
# Tool Functions
# =============================================================================


async def planner(input, context):
    """Create a research plan."""
    query = input if isinstance(input, str) else input.get("query", "default")
    context.state["query"] = query
    return {
        "plan": f"Research plan for: {query}",
        "steps": ["search", "analyze", "summarize"],
    }


async def searcher(input, context):
    """Search for information."""
    query = context.state.get("query", "unknown")
    return {
        "findings": [f"Finding 1 for {query}", f"Finding 2 for {query}"],
        "source_count": 5,
    }


async def analyzer(input, context):
    """Analyze search results."""
    findings = []
    for key in context.state:
        if "findings" in str(context.state.get(key, "")):
            findings.append(context.state[key])
    return {
        "analysis": f"Analyzed {len(findings)} sources",
        "insights": ["Insight A", "Insight B"],
    }


async def summarizer(input, context):
    """Create final summary."""
    return {
        "summary": "Executive summary of research",
        "word_count": 150,
    }


async def formatter(input, context):
    """Format output for display."""
    return {
        "formatted": True,
        "output_type": "markdown",
    }


# =============================================================================
# Example: Basic Subgraph Embedding
# =============================================================================


async def basic_subgraph_example():
    """Demonstrate basic subgraph embedding."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Subgraph Embedding")
    print("=" * 60)

    # Build inner research subgraph
    research_graph = StateGraph()
    research_graph.add_node("search", searcher, node_type="tool")
    research_graph.add_node("analyze", analyzer, node_type="tool")
    research_graph.add_edge("search", "analyze")
    research_graph.set_entry_point("search")
    research_compiled = research_graph.compile()

    # Build parent graph with embedded subgraph
    parent_graph = StateGraph()
    parent_graph.add_node("plan", planner, node_type="tool")
    parent_graph.add_node(
        "research",
        Subgraph(research_compiled, name="research_module"),
        node_type="subgraph"
    )
    parent_graph.add_node("summarize", summarizer, node_type="tool")
    parent_graph.add_edge("plan", "research")
    parent_graph.add_edge("research", "summarize")
    parent_graph.set_entry_point("plan")
    parent_compiled = parent_graph.compile()

    backend = MemoryBackend()
    executor = Executor(parent_compiled, backend)

    context = ExecutionContext(
        graph_id="subgraph-demo",
        session_id="session-001",
        chat_history=[],
        variables={},
        state={},
    )

    print("\n[1] Executing parent graph with embedded research subgraph...")

    async for event in executor.execute("Research AI agents", context):
        if event.type == EventType.SUBGRAPH_START:
            print(f"    >> Entering subgraph: {event.metadata.get('subgraph_name')}")
        elif event.type == EventType.NODE_COMPLETE:
            print(f"    Completed: {event.node_id}")
        elif event.type == EventType.SUBGRAPH_COMPLETE:
            print(f"    << Exiting subgraph: {event.metadata.get('subgraph_name')}")
        elif event.type == EventType.EXECUTION_COMPLETE:
            print("\n[2] Execution complete!")

    print(f"\n[3] Final state keys: {list(context.state.keys())}")


# =============================================================================
# Example: State Isolation
# =============================================================================


async def state_isolation_example():
    """Demonstrate state isolation between graphs."""
    print("\n" + "=" * 60)
    print("Example 2: State Isolation")
    print("=" * 60)

    async def inner_tool(input, context):
        """Tool that modifies state."""
        context.state["inner_data"] = "created in subgraph"
        context.state["shared_key"] = "modified by subgraph"
        return {"inner_result": True}

    # Build isolated subgraph
    inner = StateGraph()
    inner.add_node("tool", inner_tool, node_type="tool")
    inner.set_entry_point("tool")
    inner_compiled = inner.compile()

    # Build parent with isolated subgraph
    parent = StateGraph()
    parent.add_node("init", planner, node_type="tool")
    parent.add_node(
        "isolated_sub",
        Subgraph(
            inner_compiled,
            name="isolated_module",
            config=SubgraphConfig(isolated=True)  # Enable isolation
        ),
        node_type="subgraph"
    )
    parent.add_edge("init", "isolated_sub")
    parent.set_entry_point("init")
    parent_compiled = parent.compile()

    backend = MemoryBackend()
    executor = Executor(parent_compiled, backend)

    context = ExecutionContext(
        graph_id="isolation-demo",
        session_id="session-002",
        chat_history=[],
        variables={},
        state={"shared_key": "original value", "parent_data": "parent only"},
    )

    print("\n[1] Initial parent state:")
    print(f"    shared_key: {context.state.get('shared_key')}")
    print(f"    parent_data: {context.state.get('parent_data')}")

    print("\n[2] Executing with isolated subgraph...")
    async for event in executor.execute("Test isolation", context):
        if event.type == EventType.SUBGRAPH_COMPLETE:
            print(f"    Subgraph completed with isolation={True}")

    print(f"\n[3] Parent state after execution:")
    print(f"    shared_key: {context.state.get('shared_key')} (should be 'original value')")
    print(f"    inner_data: {context.state.get('inner_data', 'NOT PRESENT')} (isolated)")
    print(f"    parent_data: {context.state.get('parent_data')}")


# =============================================================================
# Example: Input/Output Mapping
# =============================================================================


async def io_mapping_example():
    """Demonstrate input/output mapping between graphs."""
    print("\n" + "=" * 60)
    print("Example 3: Input/Output Mapping")
    print("=" * 60)

    async def process_query(input, context):
        """Process a query with expected input key."""
        user_query = context.state.get("user_query", "default")
        return {"processed_result": f"Processed: {user_query}"}

    # Build subgraph expecting specific state keys
    inner = StateGraph()
    inner.add_node("process", process_query, node_type="tool")
    inner.set_entry_point("process")
    inner_compiled = inner.compile()

    # Build parent with input/output mapping
    parent = StateGraph()
    parent.add_node("init", planner, node_type="tool")
    parent.add_node(
        "mapped_sub",
        Subgraph(
            inner_compiled,
            name="mapped_module",
            config=SubgraphConfig(
                # Map parent's 'query' to subgraph's 'user_query'
                input_mapping={"query": "user_query"},
                # Map subgraph's 'processed_result' to parent's 'final_result'
                output_mapping={"processed_result": "final_result"},
            )
        ),
        node_type="subgraph"
    )
    parent.add_edge("init", "mapped_sub")
    parent.set_entry_point("init")
    parent_compiled = parent.compile()

    backend = MemoryBackend()
    executor = Executor(parent_compiled, backend)

    context = ExecutionContext(
        graph_id="mapping-demo",
        session_id="session-003",
        chat_history=[],
        variables={},
        state={},
    )

    print("\n[1] Executing with input/output mapping...")
    async for event in executor.execute("Map this query", context):
        if event.type == EventType.NODE_COMPLETE:
            print(f"    Completed: {event.node_id}")

    print(f"\n[2] State after mapping:")
    print(f"    query (parent): {context.state.get('query')}")
    print(f"    final_result (mapped output): {context.state.get('final_result', 'NOT SET')}")


# =============================================================================
# Example: SubgraphBuilder Fluent API
# =============================================================================


async def builder_api_example():
    """Demonstrate SubgraphBuilder fluent API."""
    print("\n" + "=" * 60)
    print("Example 4: SubgraphBuilder Fluent API")
    print("=" * 60)

    # Build subgraph using fluent builder API
    research_subgraph = (
        SubgraphBuilder("research_pipeline")
        .add_node("search", searcher, node_type="tool")
        .add_node("analyze", analyzer, node_type="tool")
        .add_edge("search", "analyze")
        .set_entry_point("search")
        .with_isolation(False)
        .with_input_mapping({"query": "search_query"})
        .with_output_mapping({"analysis": "research_output"})
        .build()
    )

    # Use in parent graph
    parent = StateGraph()
    parent.add_node("plan", planner, node_type="tool")
    parent.add_node("research", research_subgraph, node_type="subgraph")
    parent.add_node("format", formatter, node_type="tool")
    parent.add_edge("plan", "research")
    parent.add_edge("research", "format")
    parent.set_entry_point("plan")
    parent_compiled = parent.compile()

    backend = MemoryBackend()
    executor = Executor(parent_compiled, backend)

    context = ExecutionContext(
        graph_id="builder-demo",
        session_id="session-004",
        chat_history=[],
        variables={},
        state={},
    )

    print("\n[1] Executing graph built with SubgraphBuilder...")
    async for event in executor.execute("Build with fluent API", context):
        if event.type == EventType.SUBGRAPH_START:
            print(f"    >> {event.metadata.get('subgraph_name')}")
        elif event.type == EventType.NODE_COMPLETE:
            print(f"    Completed: {event.node_id}")
        elif event.type == EventType.SUBGRAPH_COMPLETE:
            print(f"    << {event.metadata.get('subgraph_name')}")

    print("\n[2] Builder configuration was applied successfully!")


# =============================================================================
# Example: Nested Subgraphs
# =============================================================================


async def nested_subgraphs_example():
    """Demonstrate deeply nested subgraphs."""
    print("\n" + "=" * 60)
    print("Example 5: Nested Subgraphs (3 levels)")
    print("=" * 60)

    async def level_tool(input, context):
        """Tool that tracks nesting level."""
        level = context.state.get("nesting_level", 0) + 1
        context.state["nesting_level"] = level
        return {"level": level, "message": f"Executed at level {level}"}

    # Level 3 (deepest)
    level3 = StateGraph()
    level3.add_node("deep", level_tool, node_type="tool")
    level3.set_entry_point("deep")
    level3_compiled = level3.compile()

    # Level 2
    level2 = StateGraph()
    level2.add_node("mid", level_tool, node_type="tool")
    level2.add_node(
        "nested",
        Subgraph(level3_compiled, name="level3"),
        node_type="subgraph"
    )
    level2.add_edge("mid", "nested")
    level2.set_entry_point("mid")
    level2_compiled = level2.compile()

    # Level 1 (root)
    level1 = StateGraph()
    level1.add_node("top", level_tool, node_type="tool")
    level1.add_node(
        "nested",
        Subgraph(level2_compiled, name="level2"),
        node_type="subgraph"
    )
    level1.add_edge("top", "nested")
    level1.set_entry_point("top")
    level1_compiled = level1.compile()

    backend = MemoryBackend()
    executor = Executor(level1_compiled, backend)

    context = ExecutionContext(
        graph_id="nested-demo",
        session_id="session-005",
        chat_history=[],
        variables={},
        state={},
    )

    print("\n[1] Executing 3-level nested graph...")
    subgraph_depth = 0

    async for event in executor.execute("Test nesting", context):
        if event.type == EventType.SUBGRAPH_START:
            subgraph_depth += 1
            indent = "    " * subgraph_depth
            print(f"{indent}>> Entering: {event.metadata.get('subgraph_name')}")
        elif event.type == EventType.NODE_COMPLETE:
            indent = "    " * (subgraph_depth + 1)
            print(f"{indent}Completed: {event.node_id}")
        elif event.type == EventType.SUBGRAPH_COMPLETE:
            indent = "    " * subgraph_depth
            print(f"{indent}<< Exiting: {event.metadata.get('subgraph_name')}")
            subgraph_depth -= 1

    print(f"\n[2] Maximum nesting level reached: {context.state.get('nesting_level')}")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all subgraph examples."""
    print("\n" + "=" * 60)
    print("MESH SUBGRAPH COMPOSITION EXAMPLES")
    print("=" * 60)

    await basic_subgraph_example()
    await state_isolation_example()
    await io_mapping_example()
    await builder_api_example()
    await nested_subgraphs_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
