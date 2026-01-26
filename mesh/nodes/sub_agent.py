"""Sub-agent node for orchestrator pattern.

This node represents an agent flow that is connected to an orchestrator.
Unlike regular AgentFlowNodes which are expanded inline during parsing,
SubAgentNodes are kept as references so the orchestrator can discover
and invoke them dynamically at runtime.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext


@dataclass
class SubAgentInfo:
    """Information about a sub-agent for orchestrator discovery."""

    node_id: str
    flow_uuid: str
    name: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "flowUuid": self.flow_uuid,
            "name": self.name,
            "description": self.description,
        }


class SubAgentNode(BaseNode):
    """Placeholder node for agent flows connected to an orchestrator.

    This node doesn't execute directly - instead, the parent OrchestratorNode
    discovers these nodes via graph edges and creates callable tools from them.

    When the orchestrator invokes a sub-agent, it loads the flow_uuid and
    executes it via Mesh Executor.

    Attributes:
        flow_uuid: UUID of the agent flow to execute
        name: Display name of the sub-agent
        description: Description shown to the orchestrator LLM for tool selection
    """

    def __init__(
        self,
        id: str,
        flow_uuid: str,
        name: str = "",
        description: str = "",
        config: Dict[str, Any] = None,
    ):
        """Initialize sub-agent node.

        Args:
            id: Node identifier
            flow_uuid: UUID of the saved agent flow
            name: Display name for the sub-agent (used as tool name)
            description: Description for the orchestrator LLM to understand
                        when to call this sub-agent
            config: Additional configuration
        """
        super().__init__(id, config or {})
        self.flow_uuid = flow_uuid
        self.name = name or f"agent_{flow_uuid[:8]}" if flow_uuid else f"agent_{id}"
        self.description = description or f"Delegate tasks to {self.name}"

    def get_info(self) -> SubAgentInfo:
        """Get sub-agent info for orchestrator discovery."""
        return SubAgentInfo(
            node_id=self.id,
            flow_uuid=self.flow_uuid,
            name=self.name,
            description=self.description,
        )

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Sub-agent nodes should not be executed directly.

        They are invoked by the parent OrchestratorNode via tools.
        If execution reaches here, it means the graph structure is incorrect.
        """
        raise RuntimeError(
            f"SubAgentNode '{self.id}' should not be executed directly. "
            "It should be connected to an OrchestratorNode which invokes it as a tool. "
            "Check your graph structure."
        )
