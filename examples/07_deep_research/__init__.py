"""Deep Research Example - DeerFlow-style research workflow with Mesh.

This example demonstrates a multi-agent deep research workflow inspired by
ByteDance's DeerFlow, built using Mesh's graph orchestration capabilities.

Workflow:
    1. Coordinator - Clarifies the research topic with the user
    2. Planner - Creates a structured research plan with steps
    3. Approval - Human reviews and approves the plan (ApprovalNode)
    4. Step Router - Routes to researcher for each step
    5. Researcher - Executes web searches and gathers information
    6. Reporter - Synthesizes findings into a final report

Key Mesh Features Demonstrated:
    - ApprovalNode for human-in-the-loop workflows
    - ConditionNode with context-aware predicates
    - Controlled loops for iterative step execution
    - State accumulation for observations
    - Multi-agent coordination
"""

from .graph import create_deep_research_graph, create_deep_research_graph_with_vel
from .prompts import PROMPTS

__all__ = ["create_deep_research_graph", "create_deep_research_graph_with_vel", "PROMPTS"]
