"""Parallel Execution Example - Fan-out/Fan-in patterns.

Architecture:
============

    ┌─────────────────────────────────────────────────────────────┐
    │  HARNESS LAYER (Claude Code / deepagents / your harness)    │
    │                                                             │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  SUPERVISOR / LEAD AGENT                            │    │
    │  │  - Receives aggregated results                      │    │
    │  │  - LLM synthesizes intelligently                    │    │
    │  │  - Generates coherent response                      │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                          ▲                                  │
    │                          │ Aggregated results               │
    └──────────────────────────┼──────────────────────────────────┘
                               │
    ┌──────────────────────────┼──────────────────────────────────┐
    │  MESH LAYER (orchestration)                                 │
    │                          │                                  │
    │                 ┌────────┴────────┐                         │
    │                 │   AGGREGATOR    │  (structural only)      │
    │                 │  list/merge/key │                         │
    │                 └────────┬────────┘                         │
    │           ┌──────────────┼──────────────┐                   │
    │           │              │              │                   │
    │           ▼              ▼              ▼                   │
    │      ┌────────┐    ┌────────┐    ┌────────┐                │
    │      │  Web   │    │  Docs  │    │  Code  │                │
    │      │ Search │    │ Search │    │ Search │                │
    │      └────────┘    └────────┘    └────────┘                │
    │           ▲              ▲              ▲                   │
    │           └──────────────┼──────────────┘                   │
    │                          │                                  │
    │                 ┌────────┴────────┐                         │
    │                 │     ROUTER      │  Fan-out                │
    │                 └─────────────────┘                         │
    └─────────────────────────────────────────────────────────────┘

DESIGN PRINCIPLE: Aggregation vs Synthesis
==========================================
Mesh handles STRUCTURAL AGGREGATION only:
- Collecting results into a data structure
- Merging dictionaries
- Flattening lists

Mesh does NOT perform intelligent operations (summarizing, deciding, interpreting).
That's the job of the downstream agent node in the HARNESS LAYER (e.g., Claude Code,
deepagents, or your custom harness).

This separation keeps Mesh focused on orchestration while the harness handles intelligence.

Key APIs:
- Send(node, input) - Dynamic dispatch to specific nodes
- ParallelExecutor.execute_parallel(branches, executor_fn, context)
- ParallelExecutor.execute_sends(sends, executor_fn, context)
- ParallelConfig(max_concurrency, error_strategy, timeout)
- Aggregators: default_aggregator, list_aggregator, keyed_aggregator
"""

import asyncio
from mesh import (
    StateGraph,
    Executor,
    ExecutionContext,
    MemoryBackend,
    Send,
    ParallelConfig,
    ParallelExecutor,
    ParallelResult,
    default_aggregator,
    list_aggregator,
    keyed_aggregator,
)
from mesh.core.events import EventType


# =============================================================================
# Research Tool Functions (Simulated parallel workers)
# =============================================================================


async def web_search(input, context):
    """Simulate web search."""
    query = input if isinstance(input, str) else input.get("query", "default")
    await asyncio.sleep(0.1)  # Simulate network delay
    return {
        "source": "web",
        "results": [f"Web result 1 for: {query}", f"Web result 2 for: {query}"],
        "confidence": 0.85,
    }


async def doc_search(input, context):
    """Simulate documentation search."""
    query = input if isinstance(input, str) else input.get("query", "default")
    await asyncio.sleep(0.15)  # Simulate processing
    return {
        "source": "docs",
        "results": [f"Doc finding for: {query}"],
        "confidence": 0.92,
    }


async def code_search(input, context):
    """Simulate code repository search."""
    query = input if isinstance(input, str) else input.get("query", "default")
    await asyncio.sleep(0.08)  # Simulate search
    return {
        "source": "code",
        "results": [f"Code example for: {query}", f"Test case for: {query}"],
        "confidence": 0.78,
    }


# =============================================================================
# Example: Basic Parallel Execution with ParallelExecutor
# =============================================================================


async def basic_parallel_example():
    """Demonstrate basic parallel execution using ParallelExecutor."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Parallel Execution")
    print("=" * 60)

    # Node functions mapping
    node_functions = {
        "web": web_search,
        "docs": doc_search,
        "code": code_search,
    }

    # Create branches as (node_id, input_data) tuples
    branches = [
        ("web", {"query": "AI agents"}),
        ("docs", {"query": "AI agents"}),
        ("code", {"query": "AI agents"}),
    ]

    # Create context
    context = ExecutionContext(
        graph_id="parallel-demo",
        session_id="session-001",
        chat_history=[],
        variables={},
        state={},
    )

    # Create executor function
    async def node_executor(node_id, input_data, ctx):
        fn = node_functions[node_id]
        return await fn(input_data, ctx)

    # Configure parallel execution
    config = ParallelConfig(max_concurrency=3)
    executor = ParallelExecutor(config)

    # Execute in parallel
    print("\n[1] Executing 3 searches in parallel...")
    result: ParallelResult = await executor.execute_parallel(branches, node_executor, context)

    print(f"\n[2] Results from {len(result.results)} branches:")
    for node_id, data in result.results.items():
        print(f"    {node_id}: {data.get('source', 'unknown')} - {len(data.get('results', []))} results")

    print(f"\n[3] Completed branches: {result.completed}")
    print(f"    All succeeded: {result.all_succeeded}")


# =============================================================================
# Example: Dynamic Branching with Send
# =============================================================================


async def dynamic_branching_example():
    """Demonstrate dynamic branching using Send objects."""
    print("\n" + "=" * 60)
    print("Example 2: Dynamic Branching with Send")
    print("=" * 60)

    # Node functions mapping
    node_functions = {
        "web": web_search,
        "docs": doc_search,
        "code": code_search,
    }

    # Create Send objects dynamically
    query = "How to write code examples"
    sends = []

    # Determine which sources to search based on query content
    if "code" in query.lower() or "example" in query.lower():
        sends.append(Send(node="code", input={"query": query}))
    if "doc" in query.lower() or "how" in query.lower():
        sends.append(Send(node="docs", input={"query": query}))
    # Always include web search
    sends.append(Send(node="web", input={"query": query}))

    print(f"\n[1] Query: '{query}'")
    print(f"    Dynamically created {len(sends)} Send objects")
    print(f"    Targets: {[s.node for s in sends]}")

    context = ExecutionContext(
        graph_id="send-demo",
        session_id="session-002",
        chat_history=[],
        variables={},
        state={},
    )

    async def node_executor(node_id, input_data, ctx):
        fn = node_functions[node_id]
        return await fn(input_data, ctx)

    config = ParallelConfig(max_concurrency=3, preserve_order=True)
    executor = ParallelExecutor(config)

    print("\n[2] Executing Send objects in parallel...")
    result = await executor.execute_sends(sends, node_executor, context)

    print(f"\n[3] Results:")
    if "_ordered" in result.results:
        for i, res in enumerate(result.results["_ordered"]):
            print(f"    Send {i}: {res.get('source')} - confidence {res.get('confidence')}")
    print(f"\n[4] Completed: {len(result.completed)} branches")


# =============================================================================
# Example: Structural Aggregation (Mesh's job)
# =============================================================================


async def structural_aggregation_example():
    """Demonstrate Mesh's structural aggregation - collecting, not synthesizing."""
    print("\n" + "=" * 60)
    print("Example 3: Structural Aggregation (Mesh Layer)")
    print("=" * 60)
    print("\nMesh aggregators handle STRUCTURE only:")
    print("- Collecting results into data structures")
    print("- Merging dictionaries")
    print("- Flattening lists")
    print("\nThey do NOT interpret, summarize, or decide.")

    node_functions = {
        "web": web_search,
        "docs": doc_search,
        "code": code_search,
    }

    branches = [
        ("web", {"query": "test"}),
        ("docs", {"query": "test"}),
        ("code", {"query": "test"}),
    ]

    context = ExecutionContext(
        graph_id="structural-agg-demo",
        session_id="session-003",
        chat_history=[],
        variables={},
        state={},
    )

    async def node_executor(node_id, input_data, ctx):
        fn = node_functions[node_id]
        return await fn(input_data, ctx)

    config = ParallelConfig(max_concurrency=3)
    executor = ParallelExecutor(config)

    print("\n[1] Executing parallel searches...")
    result = await executor.execute_parallel(branches, node_executor, context)

    # Built-in structural aggregators
    print("\n[2] Built-in aggregators (structural only):")

    # default_aggregator - merges dicts (last wins on conflicts)
    merged = default_aggregator(result.results)
    print(f"\n    default_aggregator:")
    print(f"      Merges all dicts into one: {list(merged.keys())}")

    # list_aggregator - collects to list
    listed = list_aggregator(result.results)
    print(f"\n    list_aggregator:")
    print(f"      Collects into list: {len(listed['results'])} items")

    # keyed_aggregator - stores under custom key
    findings_agg = keyed_aggregator("research_findings")
    keyed = findings_agg(result.results)
    print(f"\n    keyed_aggregator('research_findings'):")
    print(f"      Wraps in key: {list(keyed.keys())}")

    print("\n[3] These aggregators prepare data FOR the synthesis node.")
    print("    They don't interpret - that's the agent's job.")


# =============================================================================
# Example: Aggregation + Synthesis Pattern (Full Stack)
# =============================================================================


async def aggregation_synthesis_pattern():
    """Demonstrate the complete pattern: Mesh aggregates, Agent synthesizes."""
    print("\n" + "=" * 60)
    print("Example 4: Aggregation + Synthesis Pattern")
    print("=" * 60)
    print("\nThis is how Claude Code / deepagents works:")
    print("1. MESH: Runs parallel subagents, structurally aggregates results")
    print("2. HARNESS: Agent node synthesizes intelligently")

    # ==========================================================================
    # STEP 1: Mesh runs parallel workers and aggregates structurally
    # ==========================================================================

    node_functions = {
        "web": web_search,
        "docs": doc_search,
        "code": code_search,
    }

    branches = [
        ("web", {"query": "AI agent frameworks"}),
        ("docs", {"query": "AI agent frameworks"}),
        ("code", {"query": "AI agent frameworks"}),
    ]

    context = ExecutionContext(
        graph_id="full-pattern-demo",
        session_id="session-004",
        chat_history=[],
        variables={},
        state={},
    )

    async def node_executor(node_id, input_data, ctx):
        fn = node_functions[node_id]
        return await fn(input_data, ctx)

    config = ParallelConfig(max_concurrency=3)
    executor = ParallelExecutor(config)

    print("\n[1] MESH LAYER: Running parallel research...")
    result = await executor.execute_parallel(branches, node_executor, context)

    # Mesh's structural aggregation - just collects data
    aggregated = {
        "parallel_results": result.results,
        "instruction": "Synthesize these findings into a coherent report"
    }

    print(f"    Collected results from: {list(result.results.keys())}")
    print(f"    Aggregated structure: {list(aggregated.keys())}")

    # ==========================================================================
    # STEP 2: Harness layer does intelligent synthesis (simulated)
    # ==========================================================================

    print("\n[2] HARNESS LAYER: Agent synthesizes intelligently...")

    # In a real harness, this would be an LLM call like:
    # response = await agent.run(f"Synthesize: {aggregated}")
    #
    # The agent would:
    # - Understand the semantic meaning of each source
    # - Identify conflicts or agreements
    # - Prioritize by confidence/relevance
    # - Generate a coherent narrative

    # Simulated synthesis (what the agent would produce)
    synthesis = {
        "summary": "Based on web, documentation, and code analysis...",
        "key_findings": [
            "Finding 1: Documentation shows high confidence (0.92)",
            "Finding 2: Web provides broad coverage (2 results)",
            "Finding 3: Code examples validate implementation",
        ],
        "recommendation": "Use docs as primary source, validate with code",
        "sources_used": len(result.results),
    }

    print(f"    Agent produced synthesis with {len(synthesis['key_findings'])} findings")
    print(f"    Recommendation: {synthesis['recommendation']}")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n[3] PATTERN SUMMARY:")
    print("    ┌─────────────────────────────────────────────────┐")
    print("    │  HARNESS (Claude Code / deepagents / yours)     │")
    print("    │  - Spawns subagent tasks                        │")
    print("    │  - Receives aggregated results                  │")
    print("    │  - LLM synthesizes intelligently                │")
    print("    └─────────────────────────────────────────────────┘")
    print("                          │")
    print("                          ▼")
    print("    ┌─────────────────────────────────────────────────┐")
    print("    │  MESH (orchestration layer)                     │")
    print("    │  - Parallel execution                           │")
    print("    │  - Structural aggregation                       │")
    print("    │  - State management                             │")
    print("    └─────────────────────────────────────────────────┘")


# =============================================================================
# Example: Error Handling Strategies
# =============================================================================


async def error_handling_example():
    """Demonstrate different error handling strategies."""
    print("\n" + "=" * 60)
    print("Example 5: Error Handling Strategies")
    print("=" * 60)

    async def failing_search(input, context):
        """A search that always fails."""
        raise ValueError("Simulated search failure")

    async def slow_search(input, context):
        """A search that takes too long."""
        await asyncio.sleep(0.2)
        return {"source": "slow", "results": ["result"]}

    node_functions = {
        "web": web_search,
        "fail": failing_search,
        "slow": slow_search,
    }

    context = ExecutionContext(
        graph_id="error-demo",
        session_id="session-005",
        chat_history=[],
        variables={},
        state={},
    )

    async def node_executor(node_id, input_data, ctx):
        fn = node_functions[node_id]
        return await fn(input_data, ctx)

    # CONTINUE_ALL strategy (default) - continue despite errors
    from mesh.parallel import ParallelErrorStrategy

    print("\n[1] CONTINUE_ALL strategy (default):")
    config = ParallelConfig(error_strategy=ParallelErrorStrategy.CONTINUE_ALL)
    executor = ParallelExecutor(config)

    branches = [("web", {"query": "test"}), ("fail", {"query": "test"})]
    result = await executor.execute_parallel(branches, node_executor, context)

    print(f"    Completed: {result.completed}")
    print(f"    Failed: {result.failed}")
    print(f"    Has errors: {result.has_errors}")
    print(f"    Partial success: {result.partial_success}")

    # CONTINUE_PARTIAL - fail only if all fail
    print("\n[2] CONTINUE_PARTIAL strategy:")
    config = ParallelConfig(error_strategy=ParallelErrorStrategy.CONTINUE_PARTIAL)
    executor = ParallelExecutor(config)

    result = await executor.execute_parallel(branches, node_executor, context)
    print(f"    Partial success: {result.partial_success}")
    print(f"    Results available: {len(result.results)}")

    # Timeout handling
    print("\n[3] Timeout handling:")
    config = ParallelConfig(timeout=0.05)  # 50ms timeout
    executor = ParallelExecutor(config)

    try:
        branches = [("slow", {"query": "test"})]
        await executor.execute_parallel(branches, node_executor, context)
    except Exception as e:
        print(f"    Timeout caught: {type(e).__name__}")


# =============================================================================
# Example: Graph Integration with Fan-In
# =============================================================================


async def graph_fan_in_example():
    """Demonstrate fan-out/fan-in pattern in a Mesh graph."""
    print("\n" + "=" * 60)
    print("Example 6: Graph Fan-Out/Fan-In Pattern")
    print("=" * 60)

    async def router(input, context):
        """Route to multiple research nodes."""
        query = input if isinstance(input, str) else input.get("query", "")
        context.state["query"] = query
        return {"routed": True, "query": query}

    async def synthesize_agent(input, context):
        """
        HARNESS LAYER: This would be an agent node in production.

        In a real harness (Claude Code, deepagents), this node would:
        1. Receive the structurally aggregated parallel results
        2. Use an LLM to intelligently synthesize them
        3. Generate a coherent response

        Here we simulate what the agent would produce.
        """
        # Collect all parallel results from state
        findings = []
        for key in context.state:
            if key.endswith("_output"):
                output = context.state[key]
                if isinstance(output, dict) and "output" in output:
                    inner = output["output"]
                    if "results" in inner:
                        findings.extend(inner["results"])
                    if "source" in inner:
                        findings.append(f"Source: {inner['source']}")

        # Simulated LLM synthesis (in production: await agent.run(...))
        return {
            "synthesis": f"Intelligently synthesized {len(findings)} findings",
            "findings_count": len(findings),
            "synthesized_by": "agent_node",  # Would be LLM in production
        }

    # Build graph
    graph = StateGraph()
    graph.add_node("router", router, node_type="tool")
    graph.add_node("web", web_search, node_type="tool")
    graph.add_node("docs", doc_search, node_type="tool")
    graph.add_node("code", code_search, node_type="tool")

    # In production, this would be node_type="agent" with an actual LLM
    graph.add_node("synthesize", synthesize_agent, node_type="tool")

    # Fan-out: router -> multiple research nodes
    graph.add_edge("router", "web")
    graph.add_edge("router", "docs")
    graph.add_edge("router", "code")

    # Fan-in: all research nodes -> synthesize
    # Mesh waits for all to complete before proceeding
    graph.add_edge("web", "synthesize")
    graph.add_edge("docs", "synthesize")
    graph.add_edge("code", "synthesize")

    graph.set_entry_point("router")
    compiled = graph.compile()

    backend = MemoryBackend()
    executor = Executor(compiled, backend)

    context = ExecutionContext(
        graph_id="fan-in-demo",
        session_id="session-006",
        chat_history=[],
        variables={},
        state={},
    )

    print("\n[1] Executing fan-out/fan-in graph...")
    print("    router -> [web, docs, code] -> synthesize")

    async for event in executor.execute("Research AI frameworks", context):
        if event.type == EventType.NODE_COMPLETE:
            print(f"    Completed: {event.node_id}")

    print("\n[2] Graph execution complete!")
    print(f"    Query: {context.state.get('query')}")

    # Show the synthesis result
    synth_output = context.state.get("synthesize_output", {})
    if synth_output:
        inner = synth_output.get("output", {})
        print(f"    Synthesis: {inner.get('synthesis', 'N/A')}")
        print(f"    Synthesized by: {inner.get('synthesized_by', 'N/A')}")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all parallel execution examples."""
    print("\n" + "=" * 60)
    print("MESH PARALLEL EXECUTION EXAMPLES")
    print("=" * 60)
    print("\nDESIGN PRINCIPLE:")
    print("  Mesh = Structural Aggregation (orchestration)")
    print("  Harness = Intelligent Synthesis (agent layer)")

    await basic_parallel_example()
    await dynamic_branching_example()
    await structural_aggregation_example()
    await aggregation_synthesis_pattern()
    await error_handling_example()
    await graph_fan_in_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
