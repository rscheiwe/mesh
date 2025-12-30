---
layout: default
title: Deep Research
parent: Guides
nav_order: 4
---

# Deep Research Guide

Build sophisticated multi-agent research pipelines with human-in-the-loop approval.

## Overview

Deep Research is a workflow pattern inspired by ByteDance's [DeerFlow](https://github.com/bytedance/deer-flow) that orchestrates multiple specialized agents to conduct comprehensive research on any topic. This guide shows you how to build a complete deep research pipeline using Mesh.

### What You'll Build

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Coordinator │ ──▶ │   Planner   │ ──▶ │  Approval   │
│  (clarify)  │     │ (create plan)│     │ (human OK)  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
            ┌──────────────┐
            │ Step Router  │◀─────────────────┐
            │ (condition)  │                  │
            └───────┬──────┘                  │
                    │                         │
         ┌──────────┴──────────┐              │
         ▼                     ▼              │
   ┌───────────┐        ┌───────────┐         │
   │ Researcher│────────│  (loop)   │─────────┘
   │ (execute) │        └───────────┘
   └───────────┘
         │
         │ (when all steps done)
         ▼
   ┌───────────┐     ┌─────────┐
   │  Reporter │ ──▶ │   END   │
   │(synthesize)│     └─────────┘
   └───────────┘
```

### Key Features Demonstrated

| Feature | Purpose |
|---------|---------|
| **ApprovalNode** | Pause execution for human review |
| **ConditionNode** | Context-aware routing decisions |
| **Controlled Loops** | Iterate through research steps |
| **State Accumulation** | Collect observations across iterations |
| **Custom Nodes** | Specialized agent implementations |

## Prerequisites

```bash
# Install Mesh
pip install mesh

# Optional: Install Vel for LLM-powered agents
pip install vel
```

## Architecture Deep Dive

### Phase 1: Planning

**Coordinator** clarifies the research topic:
```
User: "AI safety"
       ↓
Coordinator: "AI safety research trends in 2024, focusing on
             alignment techniques, governance frameworks, and
             industry applications"
```

**Planner** creates a structured research plan:
```json
{
  "title": "Research Plan: AI Safety",
  "thought": "Breaking down into foundational, current state, and expert analysis",
  "steps": [
    {
      "id": "step_1",
      "title": "Background Research",
      "search_queries": ["what is AI safety", "AI alignment overview"]
    },
    {
      "id": "step_2",
      "title": "Current State Analysis",
      "search_queries": ["AI safety 2024", "recent alignment papers"]
    },
    {
      "id": "step_3",
      "title": "Expert Perspectives",
      "search_queries": ["AI safety experts", "alignment research leaders"]
    }
  ]
}
```

### Phase 2: Approval

**ApprovalNode** pauses execution and waits for human review:

```python
# Execution pauses here
APPROVAL_PENDING event emitted
  ↓
Plan displayed to user
  ↓
User approves/rejects
  ↓
APPROVAL_RECEIVED event
  ↓
Execution resumes
```

### Phase 3: Research Loop

**Step Router** (ConditionNode) checks if steps remain:
```python
def has_incomplete_steps(state: Dict) -> bool:
    current_index = state.get('current_step_index', 0)
    steps = state.get('plan', {}).get('steps', [])
    return current_index < len(steps)
```

**Researcher** executes each step:
1. Get current step from plan
2. Execute web searches
3. Accumulate findings in state
4. Increment step index
5. Loop back to router

### Phase 4: Reporting

**Reporter** synthesizes all findings into a final report:
- Executive summary
- Detailed findings by step
- Conclusions and recommendations
- Source citations

## Step-by-Step Implementation

### Step 1: Create Project Structure

```
examples/07_deep_research/
├── __init__.py      # Package exports
├── prompts.py       # Agent prompts
├── tools.py         # Helper functions
├── graph.py         # Graph definition
└── run.py           # Entry point
```

### Step 2: Define Agent Prompts

```python
# prompts.py

COORDINATOR_PROMPT = """You are a Research Coordinator. Your role is to
understand the user's research request and ensure it's clear enough for
the research team.

## Your Tasks:
1. Analyze the research topic provided
2. If the topic is vague or too broad, ask clarifying questions
3. Once the topic is clear, summarize what will be researched

## Guidelines:
- Be concise and professional
- If the topic is already clear and specific, proceed directly
- Focus on understanding: What specifically does the user want to know?
- Consider: scope, timeframe, depth, specific aspects of interest

Research Topic: {{$question}}
"""

PLANNER_PROMPT = """You are a Research Planner. Your role is to create a
structured research plan based on the clarified research topic.

## Your Task:
Create a detailed research plan with specific steps that can be executed
by researchers.

## Plan Requirements:
1. Each step should be independently executable
2. Steps should be ordered logically (foundational research first)
3. Include 3-5 steps for comprehensive coverage
4. Each step should have a clear objective

## Output Format (JSON):
```json
{
  "title": "Research Plan: [Topic]",
  "thought": "Brief explanation of the approach",
  "steps": [
    {
      "id": "step_1",
      "type": "research",
      "title": "Step title",
      "description": "What to research and why",
      "search_queries": ["query 1", "query 2"]
    }
  ]
}
```

Research Topic: {{$question}}
Previous Context: {{coordinator}}
"""

RESEARCHER_PROMPT = """You are a Web Researcher. Your role is to gather
information for a specific research step using web search.

## Your Task:
1. Perform web searches using the provided queries
2. Analyze the search results
3. Extract key findings and insights
4. Note important sources

## Current Step:
{{$current_step}}

## Output Format:
1. Key findings (bullet points)
2. Important sources consulted
3. Any gaps or areas needing more research
4. Confidence level (high/medium/low)

Previous observations: {{$observations}}
"""

REPORTER_PROMPT = """You are a Research Reporter. Your role is to synthesize
all research findings into a comprehensive final report.

## Report Structure:
1. **Executive Summary** - Key findings in 2-3 sentences
2. **Background** - Context and why this matters
3. **Key Findings** - Organized by theme or step
4. **Analysis** - What the findings mean
5. **Conclusions** - Main takeaways
6. **Sources** - List of key sources consulted

## Guidelines:
- Be objective and balanced
- Highlight the most important findings
- Note any limitations or gaps in the research
- Use clear, professional language
- Include specific data points when available

Research Topic: {{$question}}
Research Plan: {{planner}}
All Observations: {{$observations}}
"""

PROMPTS = {
    "coordinator": COORDINATOR_PROMPT,
    "planner": PLANNER_PROMPT,
    "researcher": RESEARCHER_PROMPT,
    "reporter": REPORTER_PROMPT,
}
```

### Step 3: Implement Helper Functions

```python
# tools.py

from typing import Dict, Any, Optional, List
import os


async def web_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search the web for information.

    This implementation uses Perplexity Sonar API when available,
    falling back to mock results for testing.

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        Dict with search results
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY")

    if api_key:
        return await _perplexity_search(query, limit, api_key)
    else:
        return _mock_search_results(query, limit)


async def _perplexity_search(
    query: str,
    limit: int,
    api_key: str
) -> Dict[str, Any]:
    """Execute actual Perplexity search."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url='https://api.perplexity.ai'
    )

    response = await client.chat.completions.create(
        model='sonar',
        messages=[{'role': 'user', 'content': query}],
        temperature=0.2,
        max_tokens=2000
    )

    content = response.choices[0].message.content

    # Extract citations if available
    citations = []
    if hasattr(response, 'citations') and response.citations:
        citations = response.citations

    results = []
    if citations:
        for i, citation in enumerate(citations[:limit]):
            citation_url = citation if isinstance(citation, str) else citation.get('url', '')
            results.append({
                'title': f'Source {i+1}',
                'url': citation_url,
                'snippet': content[:500] if i == 0 else '',
            })
    else:
        results.append({
            'title': 'Web Search Result',
            'url': 'https://perplexity.ai',
            'snippet': content,
        })

    return {
        'query': query,
        'results': results[:limit],
        'answer': content,
    }


def _mock_search_results(query: str, limit: int) -> Dict[str, Any]:
    """Return mock search results for testing."""
    return {
        'query': query,
        'results': [
            {
                'title': f'Mock Result for: {query[:30]}...',
                'url': 'https://example.com/result',
                'snippet': f'Mock search result for "{query}".',
            }
        ][:limit],
        'mock': True,
    }


def get_current_step(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get the current step from the plan based on index.

    Args:
        state: Execution state containing the plan

    Returns:
        Current step dict or None if all complete
    """
    plan = state.get('plan', {})
    steps = plan.get('steps', [])
    current_index = state.get('current_step_index', 0)

    if current_index < len(steps):
        return steps[current_index]

    return None


def has_incomplete_steps(state: Dict[str, Any]) -> bool:
    """Check if there are any incomplete steps remaining.

    Uses index-based tracking for reliability.

    Args:
        state: Execution state containing the plan

    Returns:
        True if incomplete steps exist
    """
    plan = state.get('plan', {})
    steps = plan.get('steps', [])
    current_index = state.get('current_step_index', 0)

    return current_index < len(steps)


def mark_step_complete(
    state: Dict[str, Any],
    step_id: str,
    result: str
) -> Dict[str, Any]:
    """Mark a step as complete and store its result.

    Args:
        state: Execution state
        step_id: ID of the step to mark complete
        result: Result/findings from the step

    Returns:
        Updated state dict
    """
    plan = state.get('plan', {})
    steps = plan.get('steps', [])

    for step in steps:
        if step.get('id') == step_id:
            step['completed'] = True
            step['result'] = result
            break

    # Increment step index
    current_index = state.get('current_step_index', 0)
    state['current_step_index'] = current_index + 1
    state['plan'] = plan

    return state
```

### Step 4: Implement Custom Nodes

```python
# graph.py

from typing import Dict, Any, Optional
from mesh import StateGraph
from mesh.nodes import ApprovalNode, ConditionNode, Condition
from mesh.nodes.base import BaseNode, NodeResult
from mesh.nodes.end import EndNode
from mesh.core.state import ExecutionContext

from .prompts import PROMPTS
from .tools import (
    has_incomplete_steps,
    get_current_step,
    mark_step_complete
)


class CoordinatorNode(BaseNode):
    """Coordinator agent that clarifies the research topic."""

    def __init__(self, id: str = "coordinator"):
        super().__init__(id)
        self.prompt = PROMPTS["coordinator"]

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext
    ) -> NodeResult:
        """Process the research topic and prepare for planning."""
        # Extract the research topic
        if isinstance(input, dict):
            topic = input.get("input", input.get("question", str(input)))
        else:
            topic = str(input)

        # In production, this would call an LLM to clarify
        output = {
            "research_topic": topic,
            "clarified_topic": topic,
            "coordinator_notes": f"Research topic received: {topic}",
        }

        return NodeResult(
            output=output,
            state={"research_topic": topic},
        )


class PlannerNode(BaseNode):
    """Planner agent that creates a structured research plan."""

    def __init__(self, id: str = "planner"):
        super().__init__(id)
        self.prompt = PROMPTS["planner"]

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext
    ) -> NodeResult:
        """Generate a research plan based on the topic."""
        topic = context.state.get("research_topic", "")

        # In production, this would call an LLM to generate the plan
        plan = {
            "title": f"Research Plan: {topic[:50]}...",
            "thought": "Breaking down the research into systematic steps",
            "steps": [
                {
                    "id": "step_1",
                    "type": "research",
                    "title": "Background Research",
                    "description": f"Gather foundational information about {topic}",
                    "search_queries": [f"what is {topic}", f"{topic} overview"],
                    "completed": False,
                },
                {
                    "id": "step_2",
                    "type": "research",
                    "title": "Current State Analysis",
                    "description": f"Research the current state of {topic}",
                    "search_queries": [f"{topic} latest news", f"{topic} 2024"],
                    "completed": False,
                },
                {
                    "id": "step_3",
                    "type": "research",
                    "title": "Expert Perspectives",
                    "description": f"Find expert opinions on {topic}",
                    "search_queries": [f"{topic} expert analysis", f"{topic} research"],
                    "completed": False,
                },
            ],
        }

        return NodeResult(
            output=plan,
            state={
                "plan": plan,
                "current_step_index": 0,
                "observations": [],
            },
        )


class ResearcherNode(BaseNode):
    """Researcher agent that executes web searches for each step."""

    def __init__(self, id: str = "researcher"):
        super().__init__(id)
        self.prompt = PROMPTS["researcher"]

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext
    ) -> NodeResult:
        """Execute research for the current step."""
        # Get current step directly from plan
        current_step = get_current_step(context.state)
        if not current_step:
            return NodeResult(
                output={"error": "No current step found"},
                metadata={"error": True},
            )

        step_id = current_step.get("id", "unknown")
        step_title = current_step.get("title", "Research Step")
        queries = current_step.get("search_queries", [])

        # In production, this would call web_search for each query
        findings = f"""
## {step_title}

**Search Queries Used:** {', '.join(queries)}

**Key Findings:**
- Finding 1: Important information discovered
- Finding 2: Additional context and details
- Finding 3: Expert perspectives gathered

**Sources:**
- https://example.com/source1
- https://example.com/source2

**Confidence:** High
"""

        # Accumulate observations using state helper
        context.append_to_state("observations", {
            "step_id": step_id,
            "step_title": step_title,
            "findings": findings,
        })

        # Mark step complete and increment index
        mark_step_complete(context.state, step_id, findings)

        return NodeResult(
            output={
                "step_id": step_id,
                "findings": findings,
                "step_completed": True,
            },
            state={
                "plan": context.state.get("plan", {}),
                "current_step_index": context.state.get("current_step_index", 0),
                "observations": context.state.get("observations", []),
            },
        )


class ReporterNode(BaseNode):
    """Reporter agent that synthesizes findings into a final report."""

    def __init__(self, id: str = "reporter"):
        super().__init__(id)
        self.prompt = PROMPTS["reporter"]

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext
    ) -> NodeResult:
        """Generate the final research report."""
        topic = context.state.get("research_topic", "Unknown Topic")
        observations = context.state.get("observations", [])
        plan = context.state.get("plan", {})

        # Compile findings
        findings_text = ""
        for obs in observations:
            findings_text += f"\n### {obs.get('step_title', 'Step')}\n"
            findings_text += obs.get("findings", "No findings recorded.")

        # Generate report
        report = f"""
# Research Report: {topic}

## Executive Summary
Comprehensive research investigation into {topic}.

## Research Plan
**Title:** {plan.get('title', 'N/A')}
**Approach:** {plan.get('thought', 'N/A')}
**Steps Completed:** {len(observations)}

## Detailed Findings
{findings_text}

## Conclusions
1. The topic has been thoroughly investigated
2. Key sources have been consulted and documented
3. The findings provide a solid foundation for further analysis

## Sources
All sources are documented within each research step above.

---
*Report generated by Mesh Deep Research Pipeline*
"""

        return NodeResult(
            output={
                "report": report,
                "topic": topic,
                "steps_completed": len(observations),
            },
            state={"final_report": report},
        )
```

### Step 5: Assemble the Graph

```python
# graph.py (continued)

def create_deep_research_graph() -> "StateGraph":
    """Create the deep research workflow graph.

    Graph Structure:
        START
          |
          v
        coordinator --> planner --> approval --> step_router
                                                     |
                                      +--------------+
                                      |              |
                                      v              v
                                  researcher      reporter
                                      |              |
                                      v              |
                                 (loop back)        |
                                      |              |
                                      +--------------+
                                                     |
                                                     v
                                                    END
    """
    graph = StateGraph()

    # Add agent nodes
    graph.add_node("coordinator", CoordinatorNode(), node_type="tool")
    graph.add_node("planner", PlannerNode(), node_type="tool")

    # Approval node for human review of the plan
    graph.add_node("approval", ApprovalNode(
        id="approval",
        approval_id="research_plan_approval",
        approval_message="Please review the research plan before execution.",
        data_extractor=lambda input: {
            "plan_title": input.get("title", "Research Plan"),
            "steps": input.get("steps", []),
            "step_count": len(input.get("steps", [])),
        },
    ))

    # Step router with context-aware predicate
    def route_predicate(input: Any, context: ExecutionContext) -> bool:
        """Returns True if there are incomplete steps."""
        return has_incomplete_steps(context.state)

    graph.add_node("step_router", ConditionNode(
        id="step_router",
        condition_routing="deterministic",
        conditions=[
            Condition(
                name="has_incomplete_steps",
                predicate=route_predicate,
                target_node="researcher",
            ),
        ],
        default_target="reporter",
    ))

    graph.add_node("researcher", ResearcherNode(), node_type="tool")
    graph.add_node("reporter", ReporterNode(), node_type="tool")
    graph.add_node("END", EndNode(id="END"))

    # Define edges
    graph.add_edge("START", "coordinator")
    graph.add_edge("coordinator", "planner")
    graph.add_edge("planner", "approval")
    graph.add_edge("approval", "step_router")
    graph.add_edge("step_router", "researcher")
    graph.add_edge("step_router", "reporter")

    # Loop edge: researcher -> step_router for next step
    graph.add_edge(
        "researcher",
        "step_router",
        is_loop_edge=True,
        max_iterations=10,  # Safety limit
    )

    graph.add_edge("reporter", "END")

    # Set entry point
    graph.set_entry_point("coordinator")

    return graph.compile()
```

### Step 6: Create the Runner

```python
# run.py

import asyncio
import argparse
from typing import List

from mesh import Executor, ExecutionContext, MemoryBackend
from mesh.core.events import ExecutionEvent, EventType
from mesh.core.executor import ExecutionStatus
from mesh.nodes import approve, reject

from .graph import create_deep_research_graph


async def run_deep_research(
    topic: str,
    auto_approve: bool = False,
    verbose: bool = True,
) -> str:
    """Run the deep research workflow.

    Args:
        topic: Research topic to investigate
        auto_approve: If True, automatically approve the plan
        verbose: If True, print progress messages

    Returns:
        Final research report as string
    """
    # Create the graph
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
                    print(f"Plan: {approval_data.get('plan_title', 'Plan')}")
                    print(f"Steps: {approval_data.get('step_count', 'N/A')}")

                if auto_approve:
                    if verbose:
                        print("\nAuto-approving plan...")
                    approval_result = approve()
                else:
                    # In production, wait for user input
                    user_input = input("\nApprove plan? (y/n): ")
                    if user_input.lower() == 'y':
                        approval_result = approve()
                    else:
                        approval_result = reject(reason="User rejected")
                        if verbose:
                            print("Plan rejected. Exiting.")
                        return "Research cancelled by user."

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

    if event_type == 'data-node-start':
        node_id = event.node_id or event.metadata.get('node_id', 'unknown')
        print(f"  > Starting: {node_id}")

    elif event_type == 'data-node-complete':
        node_id = event.node_id or event.metadata.get('node_id', 'unknown')
        print(f"  < Completed: {node_id}")

    elif event_type == 'data-approval-pending':
        print(f"  ! Approval pending")

    elif event_type == 'data-approval-received':
        print(f"  + Approval received")

    elif event_type == 'data-execution-complete':
        status = event.metadata.get('status', 'unknown')
        print(f"  * Execution complete: {status}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deep Research Pipeline")
    parser.add_argument(
        "topic",
        nargs="?",
        default="artificial intelligence trends",
        help="Research topic",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve the research plan",
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
            verbose=not args.quiet,
        )

        if args.quiet:
            print(report)

    except KeyboardInterrupt:
        print("\nResearch interrupted by user")
    except Exception as e:
        print(f"\nError during research: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
```

## Running the Example

### Basic Usage

```bash
# Run with real Vel agents (requires API keys)
python -m examples.07_deep_research.run "AI safety research"

# Auto-approve the plan
python -m examples.07_deep_research.run "quantum computing" --auto-approve

# Specify a different model
python -m examples.07_deep_research.run "quantum computing" --model gpt-4o-mini

# Use mock nodes (no API keys required, for testing)
python -m examples.07_deep_research.run "test topic" --mock --auto-approve

# Quiet mode (just output the report)
python -m examples.07_deep_research.run "climate change" --auto-approve --quiet
```

### Expected Output

```
============================================================
Deep Research Pipeline
============================================================
Topic: AI safety research
============================================================

Phase 1: Coordinator & Planning
----------------------------------------
  > Starting: START
  < Completed: START
  > Starting: coordinator
  < Completed: coordinator
  > Starting: planner
  < Completed: planner
  > Starting: approval
  ! Approval pending
  < Completed: approval
  * Execution complete: waiting_for_approval

============================================================
APPROVAL REQUIRED
============================================================
Plan: Research Plan: AI safety research...
Steps: 3

Auto-approving plan...

----------------------------------------
Phase 2: Research Execution
----------------------------------------
  + Approval received
  > Starting: step_router
  < Completed: step_router
  > Starting: researcher
  < Completed: researcher
  [... repeats for each step ...]
  > Starting: reporter
  < Completed: reporter
  > Starting: END
  < Completed: END
  * Execution complete: completed

============================================================
FINAL REPORT
============================================================

# Research Report: AI safety research
...
```

## Advanced Patterns

### Pattern 1: Adding Real LLM Calls

Replace the placeholder nodes with actual Vel agents:

```python
from vel import Agent as VelAgent

def create_deep_research_graph_with_vel(model: str = "gpt-4o"):
    """Create graph with real LLM-powered agents."""

    graph = StateGraph()

    # Create Vel agents with prompts
    coordinator = VelAgent(
        id="coordinator",
        model={"provider": "openai", "model": model},
        system_prompt=PROMPTS["coordinator"],
    )

    planner = VelAgent(
        id="planner",
        model={"provider": "openai", "model": model},
        system_prompt=PROMPTS["planner"],
    )

    # Create web search tool for researcher
    from vel.tools import ToolSpec
    web_search_tool = ToolSpec.from_function(web_search)

    researcher = VelAgent(
        id="researcher",
        model={"provider": "openai", "model": model},
        system_prompt=PROMPTS["researcher"],
        tools=[web_search_tool],
    )

    reporter = VelAgent(
        id="reporter",
        model={"provider": "openai", "model": model},
        system_prompt=PROMPTS["reporter"],
    )

    # Add as agent nodes
    graph.add_node("coordinator", coordinator, node_type="agent")
    graph.add_node("planner", planner, node_type="agent")
    graph.add_node("researcher", researcher, node_type="agent")
    graph.add_node("reporter", reporter, node_type="agent")

    # ... rest of graph setup
```

### Pattern 2: Parallel Research Steps

Execute multiple research steps in parallel:

```python
# Instead of looping, create parallel researcher nodes
graph.add_node("researcher_1", ResearcherNode(step_index=0), node_type="tool")
graph.add_node("researcher_2", ResearcherNode(step_index=1), node_type="tool")
graph.add_node("researcher_3", ResearcherNode(step_index=2), node_type="tool")

# All start after approval
graph.add_edge("approval", "researcher_1")
graph.add_edge("approval", "researcher_2")
graph.add_edge("approval", "researcher_3")

# All feed into reporter
graph.add_edge("researcher_1", "reporter")
graph.add_edge("researcher_2", "reporter")
graph.add_edge("researcher_3", "reporter")
```

### Pattern 3: Rejection Handling

Handle plan rejection with refinement loop:

```python
async for event in executor.execute(topic, context):
    if event.type == EventType.EXECUTION_COMPLETE:
        status = event.metadata.get("status")

        if status == ExecutionStatus.WAITING_FOR_APPROVAL:
            user_input = input("Approve plan? (y/n/refine): ")

            if user_input.lower() == 'y':
                approval_result = approve()
            elif user_input.lower() == 'refine':
                feedback = input("Refinement feedback: ")
                approval_result = reject(
                    reason=f"Refine plan: {feedback}",
                    modified_data={"refinement_feedback": feedback}
                )
                # Could loop back to planner with feedback
            else:
                approval_result = reject(reason="User cancelled")
```

### Pattern 4: Progress Tracking

Track detailed progress through state:

```python
class ResearcherNode(BaseNode):
    async def _execute_impl(self, input, context):
        # Update progress in state
        total_steps = len(context.state.get("plan", {}).get("steps", []))
        current_step = context.state.get("current_step_index", 0) + 1

        context.set_in_state("progress", {
            "current_step": current_step,
            "total_steps": total_steps,
            "percentage": (current_step / total_steps) * 100,
            "status": f"Researching step {current_step} of {total_steps}",
        })

        # ... rest of implementation
```

## Troubleshooting

### Loop Never Exits

**Problem:** Researcher keeps looping forever.

**Solution:** Ensure `mark_step_complete` increments `current_step_index`:

```python
def mark_step_complete(state, step_id, result):
    # ... mark step ...

    # CRITICAL: Increment the index
    current_index = state.get('current_step_index', 0)
    state['current_step_index'] = current_index + 1
```

### State Not Persisting

**Problem:** State changes in one node aren't visible in the next.

**Solution:** Return state in `NodeResult`:

```python
return NodeResult(
    output={...},
    state={  # Explicit state return
        "plan": context.state.get("plan", {}),
        "current_step_index": context.state.get("current_step_index", 0),
    },
)
```

### Approval Doesn't Pause

**Problem:** Execution continues without waiting for approval.

**Solution:** Check that you're using `executor.resume()`:

```python
if status == ExecutionStatus.WAITING_FOR_APPROVAL:
    # Get approval...

    # MUST call resume() to continue
    async for event in executor.resume(context, approval_result):
        # Process resumed events
```

## See Also

- [ApprovalNode API](../api-reference#approvalnode) - Detailed API reference
- [Loops Guide](loops) - Understanding controlled cycles
- [Nodes Concept](../concepts/nodes) - Node types overview
- [Events Concept](../concepts/events) - Event system
- [Examples](../examples) - More example workflows
