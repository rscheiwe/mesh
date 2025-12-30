"""Deep Research Graph Definition.

This module defines the deep research workflow graph using Mesh's StateGraph builder.
The graph implements a DeerFlow-inspired research pipeline with:
- Multi-agent coordination (Coordinator, Planner, Researcher, Reporter)
- Human-in-the-loop approval (ApprovalNode)
- Iterative step execution (ConditionNode + loop edges)
"""

import json
from typing import Dict, Any, Optional

from mesh import StateGraph
from mesh.nodes import (
    ApprovalNode,
    ConditionNode,
    Condition,
)
from mesh.nodes.base import BaseNode, NodeResult
from mesh.nodes.end import EndNode
from mesh.core.state import ExecutionContext

from .prompts import PROMPTS
from .tools import has_incomplete_steps, get_current_step, mark_step_complete


# Check for Vel availability
try:
    from vel import Agent as VelAgent
    HAS_VEL = True
except ImportError:
    HAS_VEL = False
    VelAgent = None


class CoordinatorNode(BaseNode):
    """Coordinator agent that clarifies the research topic.

    In a production setup with Vel, this would be a VelAgent.
    This is a simplified version for demonstration.
    """

    def __init__(self, id: str = "coordinator"):
        super().__init__(id)
        self.prompt = PROMPTS["coordinator"]

    async def _execute_impl(self, input: Any, context: ExecutionContext) -> NodeResult:
        """Process the research topic and prepare for planning."""
        # Extract the research topic
        if isinstance(input, dict):
            topic = input.get("input", input.get("question", str(input)))
        else:
            topic = str(input)

        # Store the clarified topic
        output = {
            "research_topic": topic,
            "clarified_topic": topic,  # In real impl, agent would clarify
            "coordinator_notes": f"Research topic received: {topic}",
        }

        return NodeResult(
            output=output,
            state={"research_topic": topic},
        )


class PlannerNode(BaseNode):
    """Planner agent that creates a structured research plan.

    Generates a JSON plan with steps for the research team.
    """

    def __init__(self, id: str = "planner"):
        super().__init__(id)
        self.prompt = PROMPTS["planner"]

    async def _execute_impl(self, input: Any, context: ExecutionContext) -> NodeResult:
        """Generate a research plan based on the topic."""
        topic = context.state.get("research_topic", "")

        # In production, this would call an LLM to generate the plan
        # For demonstration, we create a sample plan
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
                    "description": f"Research the current state and recent developments in {topic}",
                    "search_queries": [f"{topic} latest news", f"{topic} 2024 updates"],
                    "completed": False,
                },
                {
                    "id": "step_3",
                    "type": "research",
                    "title": "Expert Perspectives",
                    "description": f"Find expert opinions and analysis on {topic}",
                    "search_queries": [f"{topic} expert analysis", f"{topic} research papers"],
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
    """Researcher agent that executes web searches for each step.

    In production, this would use actual web search tools via Vel.
    """

    def __init__(self, id: str = "researcher"):
        super().__init__(id)
        self.prompt = PROMPTS["researcher"]

    async def _execute_impl(self, input: Any, context: ExecutionContext) -> NodeResult:
        """Execute research for the current step."""
        # Get current step directly from plan based on index
        current_step = get_current_step(context.state)
        if not current_step:
            return NodeResult(
                output={"error": "No current step found"},
                metadata={"error": True},
            )

        step_id = current_step.get("id", "unknown")
        step_title = current_step.get("title", "Research Step")
        queries = current_step.get("search_queries", [])

        # Simulate research findings
        # In production, this would call web_search tool
        findings = f"""
## {step_title}

**Search Queries Used:** {', '.join(queries)}

**Key Findings:**
- Finding 1: Important information discovered about the topic
- Finding 2: Additional context and details found
- Finding 3: Expert perspectives gathered

**Sources:**
- https://example.com/source1
- https://example.com/source2

**Confidence:** High
"""

        # Accumulate observations
        context.append_to_state("observations", {
            "step_id": step_id,
            "step_title": step_title,
            "findings": findings,
        })

        # Mark step complete and return updated state
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

    async def _execute_impl(self, input: Any, context: ExecutionContext) -> NodeResult:
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
This report presents the findings from a comprehensive research investigation into {topic}.

## Research Plan
**Title:** {plan.get('title', 'N/A')}
**Approach:** {plan.get('thought', 'N/A')}
**Steps Completed:** {len(observations)}

## Detailed Findings
{findings_text}

## Conclusions
Based on the research conducted, here are the key takeaways:
1. The topic has been thoroughly investigated from multiple angles
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


def create_deep_research_graph() -> "StateGraph":
    """Create the deep research workflow graph with mock nodes.

    This version uses placeholder implementations that don't require
    API keys. Use `create_deep_research_graph_with_vel()` for real
    LLM-powered agents.

    Returns:
        Compiled ExecutionGraph ready for execution

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
                                 step_complete      |
                                      |              |
                                      +--> (loop) --+
                                                     |
                                                     v
                                                    END
    """
    graph = StateGraph()

    # Add nodes
    graph.add_node("coordinator", CoordinatorNode(), node_type="tool")
    graph.add_node("planner", PlannerNode(), node_type="tool")

    # Approval node for human review of the plan
    graph.add_node("approval", ApprovalNode(
        id="approval",
        approval_id="research_plan_approval",
        approval_message="Please review the research plan before execution begins.",
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
    graph.add_edge("step_router", "researcher")  # When has_incomplete_steps=True
    graph.add_edge("step_router", "reporter")    # When has_incomplete_steps=False (default)

    # Loop back from researcher to step_router for next step
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


def create_deep_research_graph_with_vel(model: str = "gpt-4o") -> "StateGraph":
    """Create the deep research graph using Vel agents.

    This version uses actual Vel agents with LLM capabilities.
    Requires Vel to be installed and configured.

    Args:
        model: Model to use (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")

    Returns:
        Compiled ExecutionGraph

    Raises:
        ImportError: If Vel is not available
    """
    if not HAS_VEL:
        raise ImportError(
            "Vel is required for this graph. Install with: pip install vel"
        )

    from vel import Agent as VelAgent
    from .tools import create_web_search_tool

    graph = StateGraph()

    # Create Vel agents
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
    web_search_tool = create_web_search_tool()
    researcher_tools = [web_search_tool] if web_search_tool else []

    researcher = VelAgent(
        id="researcher",
        model={"provider": "openai", "model": model},
        system_prompt=PROMPTS["researcher"],
        tools=researcher_tools,
    )

    reporter = VelAgent(
        id="reporter",
        model={"provider": "openai", "model": model},
        system_prompt=PROMPTS["reporter"],
    )

    # Add agent nodes
    graph.add_node("coordinator", coordinator, node_type="agent")
    graph.add_node("planner", planner, node_type="agent")

    # Approval node
    graph.add_node("approval", ApprovalNode(
        id="approval",
        approval_id="research_plan_approval",
        approval_message="Please review the research plan before execution begins.",
    ))

    # Step router
    def route_predicate(input: Any, context: ExecutionContext) -> bool:
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

    graph.add_node("researcher", researcher, node_type="agent")
    graph.add_node("reporter", reporter, node_type="agent")
    graph.add_node("END", EndNode(id="END"))

    # Edges
    graph.add_edge("START", "coordinator")
    graph.add_edge("coordinator", "planner")
    graph.add_edge("planner", "approval")
    graph.add_edge("approval", "step_router")
    graph.add_edge("step_router", "researcher")
    graph.add_edge("step_router", "reporter")
    graph.add_edge(
        "researcher",
        "step_router",
        is_loop_edge=True,
        max_iterations=10,
    )
    graph.add_edge("reporter", "END")

    graph.set_entry_point("coordinator")

    return graph.compile()
