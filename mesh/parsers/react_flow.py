"""React Flow JSON parser for Flowise compatibility.

This module parses React Flow JSON format (used by Flowise) into
Mesh ExecutionGraph structures.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, ValidationError, Field

from mesh.core.graph import ExecutionGraph, Edge, NodeConfig
from mesh.nodes.start import StartNode
from mesh.nodes.end import EndNode
from mesh.nodes.agent import AgentNode
from mesh.nodes.llm import LLMNode
from mesh.nodes.tool import ToolNode
from mesh.nodes.condition import ConditionNode, Condition
from mesh.nodes.loop import LoopNode
from mesh.utils.registry import NodeRegistry
from mesh.utils.errors import GraphValidationError


class ReactFlowNode(BaseModel):
    """Schema for a node in React Flow JSON."""
    id: str
    type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    position: Optional[Dict[str, float]] = None
    parent_node: Optional[str] = Field(None, alias="parentNode")

    class Config:
        populate_by_name = True


class ReactFlowEdge(BaseModel):
    """Schema for an edge in React Flow JSON."""
    source: str
    target: str
    source_handle: Optional[str] = Field(None, alias="sourceHandle")
    target_handle: Optional[str] = Field(None, alias="targetHandle")

    class Config:
        populate_by_name = True


class ReactFlowJSON(BaseModel):
    """Schema for React Flow JSON from Flowise."""
    nodes: List[ReactFlowNode]
    edges: List[ReactFlowEdge]
    viewport: Optional[Dict[str, float]] = None


class ReactFlowParser:
    """Parse React Flow JSON into Mesh ExecutionGraph.

    This parser maps Flowise node types to Mesh node implementations:
    - startAgentflow -> StartNode
    - agentAgentflow -> AgentNode
    - llmAgentflow -> LLMNode
    - toolAgentflow -> ToolNode
    - conditionAgentflow -> ConditionNode
    - loopAgentflow -> LoopNode
    - endAgentflow -> EndNode

    Example:
        >>> registry = NodeRegistry()
        >>> registry.register_agent("my_agent", agent_instance)
        >>>
        >>> parser = ReactFlowParser(registry)
        >>> graph = parser.parse(flow_json)
    """

    # Map Flowise node types to Mesh node types
    NODE_TYPE_MAP = {
        "startAgentflow": "start",
        "agentAgentflow": "agent",
        "llmAgentflow": "llm",
        "toolAgentflow": "tool",
        "conditionAgentflow": "condition",
        "conditionflow": "condition",
        "foreachAgentflow": "loop",  # ForEach uses loop handler (distinguishes via config)
        "loopAgentflow": "loop",  # Loop backward jump uses same handler
        "endAgentflow": "end",
    }

    def __init__(self, registry: NodeRegistry):
        """Initialize parser with node registry.

        Args:
            registry: NodeRegistry for resolving agents and tools
        """
        self.registry = registry

    def parse(self, json_data: Dict[str, Any]) -> ExecutionGraph:
        """Parse React Flow JSON into ExecutionGraph.

        Args:
            json_data: React Flow JSON dictionary

        Returns:
            ExecutionGraph ready for execution

        Raises:
            ValidationError: If JSON structure is invalid
            GraphValidationError: If graph construction fails
        """
        # Validate JSON structure
        try:
            flow_data = ReactFlowJSON(**json_data)
        except ValidationError as e:
            raise GraphValidationError(f"Invalid React Flow JSON: {e}")

        # Build nodes
        nodes = {}
        for node_data in flow_data.nodes:
            node_id = node_data.id
            # Get node type from data.name or type
            flowise_type = node_data.data.get("name") or node_data.type
            mesh_type = self.NODE_TYPE_MAP.get(flowise_type, flowise_type)

            # Get node config from data.inputs
            node_config = node_data.data.get("inputs", {})

            # Create node instance
            node = self._create_node(
                node_id=node_id,
                node_type=mesh_type,
                config=node_config,
                raw_data=node_data.dict(),
            )
            nodes[node_id] = node

        # Build edges
        edges = [
            Edge(
                source=edge.source,
                target=edge.target,
                source_handle=edge.source_handle,
                target_handle=edge.target_handle,
            )
            for edge in flow_data.edges
        ]

        # Construct and validate graph
        graph = ExecutionGraph.from_nodes_and_edges(nodes, edges)
        graph.validate()

        return graph

    def _create_node(
        self,
        node_id: str,
        node_type: str,
        config: Dict[str, Any],
        raw_data: Dict[str, Any],
    ) -> Any:
        """Create node instance from configuration.

        Args:
            node_id: Node identifier
            node_type: Mesh node type
            config: Node configuration
            raw_data: Raw React Flow node data

        Returns:
            Node instance

        Raises:
            GraphValidationError: If node creation fails
        """
        try:
            if node_type == "start":
                return StartNode(
                    id=node_id,
                    event_mode=config.get("eventMode", "full"),
                    config=config
                )

            elif node_type == "end":
                return EndNode(id=node_id, config=config)

            elif node_type == "agent":
                return self._create_agent_node(node_id, config)

            elif node_type == "llm":
                return self._create_llm_node(node_id, config)

            elif node_type == "tool":
                return self._create_tool_node(node_id, config)

            elif node_type == "condition":
                return self._create_condition_node(node_id, config)

            elif node_type == "loop":
                return self._create_loop_node(node_id, config)

            else:
                raise GraphValidationError(f"Unknown node type: {node_type}")

        except Exception as e:
            raise GraphValidationError(
                f"Failed to create node '{node_id}' of type '{node_type}': {e}"
            ) from e

    def _create_agent_node(self, node_id: str, config: Dict[str, Any]) -> AgentNode:
        """Create AgentNode from config."""
        agent_ref = config.get("agent")
        if not agent_ref:
            raise GraphValidationError(
                f"Agent node '{node_id}' missing 'agent' reference"
            )

        # Get agent from registry
        agent = self.registry.get_agent(agent_ref)

        system_prompt = config.get("systemPrompt") or config.get("system_prompt")
        use_native_events = config.get("useNativeEvents", False)
        event_mode = config.get("eventMode", "full")

        return AgentNode(
            id=node_id,
            agent=agent,
            system_prompt=system_prompt,
            use_native_events=use_native_events,
            event_mode=event_mode,
            config=config,
        )

    def _create_llm_node(self, node_id: str, config: Dict[str, Any]) -> LLMNode:
        """Create LLMNode from config."""
        return LLMNode(
            id=node_id,
            model=config.get("model", "gpt-4"),
            system_prompt=config.get("systemPrompt") or config.get("system_prompt"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("maxTokens") or config.get("max_tokens"),
            provider=config.get("provider", "openai"),
            event_mode=config.get("eventMode", "full"),
            config=config,
        )

    def _create_tool_node(self, node_id: str, config: Dict[str, Any]) -> ToolNode:
        """Create ToolNode from config."""
        tool_ref = config.get("tool")
        if not tool_ref:
            raise GraphValidationError(
                f"Tool node '{node_id}' missing 'tool' reference"
            )

        # Get tool from registry
        tool_fn = self.registry.get_tool(tool_ref)

        return ToolNode(
            id=node_id,
            tool_fn=tool_fn,
            event_mode=config.get("eventMode", "full"),
            config=config,
        )

    def _create_condition_node(
        self, node_id: str, config: Dict[str, Any]
    ) -> ConditionNode:
        """Create ConditionNode from config."""
        conditions_config = config.get("conditions", [])
        if not conditions_config:
            raise GraphValidationError(
                f"Condition node '{node_id}' has no conditions"
            )

        conditions = []
        for cond_config in conditions_config:
            name = cond_config.get("name")
            target = cond_config.get("target")
            expression = cond_config.get("expression")

            if not all([name, target, expression]):
                raise GraphValidationError(
                    f"Condition in node '{node_id}' missing required fields"
                )

            # Create predicate from expression
            # Simple expression evaluation - can be extended
            predicate = self._create_predicate(expression)

            conditions.append(
                Condition(
                    name=name,
                    predicate=predicate,
                    target_node=target,
                )
            )

        return ConditionNode(
            id=node_id,
            conditions=conditions,
            default_target=config.get("defaultTarget") or config.get("default_target"),
            event_mode=config.get("eventMode", "full"),
            config=config,
        )

    def _create_loop_node(self, node_id: str, config: Dict[str, Any]):
        """Create LoopNode or ForEachNode from config based on parameters."""
        from mesh.nodes.loop import ForEachNode, LoopNode as ActualLoopNode

        # Check if this is a ForEach node (has arrayPath) or Loop node (has loopBackTo)
        if "arrayPath" in config or "array_path" in config:
            # ForEach node
            return ForEachNode(
                id=node_id,
                array_path=config.get("arrayPath") or config.get("array_path", "$.items"),
                max_iterations=config.get("maxIterations") or config.get("max_iterations", 100),
                event_mode=config.get("eventMode", "full"),
                config=config,
            )
        elif "loopBackTo" in config or "loop_back_to" in config:
            # Loop node (backward jump)
            loop_back_to = config.get("loopBackTo") or config.get("loop_back_to")
            if not loop_back_to:
                raise GraphValidationError(
                    f"Loop node '{node_id}' missing 'loopBackTo' parameter"
                )
            return ActualLoopNode(
                id=node_id,
                loop_back_to=loop_back_to,
                max_loop_count=config.get("maxLoopCount") or config.get("max_loop_count", 5),
                event_mode=config.get("eventMode", "full"),
                config=config,
            )
        else:
            # Default to ForEach for backward compatibility
            return ForEachNode(
                id=node_id,
                array_path=config.get("arrayPath", "$.items"),
                max_iterations=config.get("maxIterations", 100),
                event_mode=config.get("eventMode", "full"),
                config=config,
            )

    def _create_predicate(self, expression: str) -> Any:
        """Create a predicate function from an expression string.

        This is a simple implementation that handles basic patterns.
        Can be extended to support more complex expressions.

        Args:
            expression: Expression string (e.g., "{{node_id}}.contains('success')")

        Returns:
            Predicate function
        """

        def predicate(input: Any) -> bool:
            # Simple string matching
            input_str = str(input).lower()

            if "contains" in expression:
                # Extract substring to search for
                import re

                match = re.search(r"contains\(['\"](.+?)['\"]\)", expression)
                if match:
                    substring = match.group(1).lower()
                    return substring in input_str

            elif "equals" in expression or "==" in expression:
                # Extract value to compare
                import re

                match = re.search(r"['\"](.+?)['\"]", expression)
                if match:
                    value = match.group(1).lower()
                    return value == input_str

            # Default: evaluate expression directly (unsafe - for dev only)
            # In production, use a safe expression evaluator
            return False

        return predicate
