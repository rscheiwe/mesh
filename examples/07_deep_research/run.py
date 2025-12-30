#!/usr/bin/env python3
"""Run the Deep Research workflow.

This script demonstrates the deep research pipeline using Mesh.
It shows how to:
1. Create and execute the research graph
2. Handle the approval workflow
3. Process and display results

Usage:
    python -m examples.07_deep_research.run "Your research topic"

    # Or with approval auto-accept:
    python -m examples.07_deep_research.run "Your research topic" --auto-approve

    # With mock nodes (no API keys required):
    python -m examples.07_deep_research.run "Your research topic" --mock
"""

import asyncio
import argparse
import sys
from typing import List

from mesh import Executor, ExecutionContext, MemoryBackend
from mesh.core.events import ExecutionEvent, EventType
from mesh.core.executor import ExecutionStatus
from mesh.nodes import approve, reject

from .graph import create_deep_research_graph, create_deep_research_graph_with_vel


async def run_deep_research(
    topic: str,
    auto_approve: bool = False,
    use_mock: bool = False,
    model: str = "gpt-4o",
    verbose: bool = True,
) -> str:
    """Run the deep research workflow.

    Args:
        topic: Research topic to investigate
        auto_approve: If True, automatically approve the plan
        use_mock: If True, use mock nodes (no API keys required)
        model: Model to use for Vel agents
        verbose: If True, print progress messages

    Returns:
        Final research report as string
    """
    # Create the graph
    if use_mock:
        graph = create_deep_research_graph()
        if verbose:
            print("Using mock nodes (no LLM calls)")
    else:
        try:
            graph = create_deep_research_graph_with_vel(model=model)
            if verbose:
                print(f"Using Vel agents with model: {model}")
        except ImportError as e:
            print(f"Vel not available: {e}")
            print("Falling back to mock nodes...")
            graph = create_deep_research_graph()

    # Create executor and context
    backend = MemoryBackend()
    executor = Executor(graph, backend)

    context = ExecutionContext(
        graph_id="deep-research",
        session_id="research-session-001",
        chat_history=[],
        variables={},
        state={},
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Deep Research Pipeline")
        print(f"{'='*60}")
        print(f"Topic: {topic}")
        print(f"{'='*60}\n")

    # Execute first phase (until approval)
    events: List[ExecutionEvent] = []
    final_output = None

    if verbose:
        print("Phase 1: Coordinator & Planning")
        print("-" * 40)

    async for event in executor.execute(topic, context):
        events.append(event)

        if verbose:
            _print_event(event)

        # Check for approval pending
        if event.type == EventType.EXECUTION_COMPLETE:
            status = event.metadata.get("status")

            if status == ExecutionStatus.WAITING_FOR_APPROVAL:
                # Handle approval
                approval_data = event.metadata.get("approval_data", {})

                if verbose:
                    print("\n" + "=" * 60)
                    print("APPROVAL REQUIRED")
                    print("=" * 60)
                    print(f"Plan: {approval_data.get('plan_title', 'Research Plan')}")
                    print(f"Steps: {approval_data.get('step_count', 'N/A')}")

                if auto_approve:
                    if verbose:
                        print("\nAuto-approving plan...")
                    approval_result = approve()
                else:
                    # In a real application, this would wait for user input
                    # For this example, we'll approve by default
                    if verbose:
                        print("\nApproving plan (use --auto-approve to skip prompt)...")
                    approval_result = approve()

                # Resume execution
                if verbose:
                    print("\n" + "-" * 40)
                    print("Phase 2: Research Execution")
                    print("-" * 40)

                async for resume_event in executor.resume(context, approval_result):
                    events.append(resume_event)

                    if verbose:
                        _print_event(resume_event)

                    if resume_event.type == EventType.EXECUTION_COMPLETE:
                        final_output = resume_event.output

            elif status == ExecutionStatus.COMPLETED:
                final_output = event.output

    # Extract and return the report
    report = context.state.get("final_report", "No report generated")

    if verbose:
        print("\n" + "=" * 60)
        print("FINAL REPORT")
        print("=" * 60)
        print(report)

    return report


def _print_event(event: ExecutionEvent) -> None:
    """Print event in a readable format."""
    event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)

    # Skip verbose events
    if event_type in ['text-delta', 'data-custom']:
        return

    # Format based on event type
    if event_type == 'data-node-start':
        node_id = event.node_id or event.metadata.get('node_id', 'unknown')
        print(f"  > Starting: {node_id}")

    elif event_type == 'data-node-complete':
        node_id = event.node_id or event.metadata.get('node_id', 'unknown')
        print(f"  < Completed: {node_id}")

    elif event_type == 'data-approval-pending':
        print(f"  ! Approval pending: {event.metadata.get('approval_id', 'unknown')}")

    elif event_type == 'data-approval-received':
        print(f"  + Approval received")

    elif event_type == 'data-execution-complete':
        status = event.metadata.get('status', 'unknown')
        print(f"  * Execution complete: {status}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the Deep Research workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m examples.07_deep_research.run "AI safety research"
  python -m examples.07_deep_research.run "Climate change solutions" --auto-approve
  python -m examples.07_deep_research.run "Quantum computing" --model gpt-4o
  python -m examples.07_deep_research.run "Test topic" --mock --auto-approve
        """,
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default="artificial intelligence trends in 2024",
        help="Research topic to investigate",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve the research plan",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock nodes instead of real LLM calls (no API keys required)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to use for Vel agents (default: gpt-4o)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    try:
        report = await run_deep_research(
            topic=args.topic,
            auto_approve=args.auto_approve,
            use_mock=args.mock,
            model=args.model,
            verbose=not args.quiet,
        )

        if args.quiet:
            print(report)

    except KeyboardInterrupt:
        print("\nResearch interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during research: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
