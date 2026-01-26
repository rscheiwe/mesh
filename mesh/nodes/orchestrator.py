"""Orchestrator node for LLM-driven delegation to Mesh graph sub-agents.

This node creates an orchestrator agent that can dynamically delegate
tasks to sub-agents at runtime using Vel's ToolSpec to expose each
sub-agent as a callable tool.

Key Design Decision: Sub-agents are discovered from graph edges.
Connect AgentFlowNodes as children of the OrchestratorNode in the canvas,
and they become available as tools for the orchestrator LLM to invoke.

Visual flow:
                    ┌─→ AgentFlowNode (Researcher)
StartNode → Orchestrator ─┼─→ AgentFlowNode (Analyst)
                    └─→ AgentFlowNode (Writer)
"""

from typing import Any, Dict, List, Optional
import re

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext
from mesh.core.events import (
    ExecutionEvent,
    EventType,
    create_mesh_node_start_event,
    create_mesh_node_complete_event,
    transform_event_for_transient_mode,
)
from mesh.utils.variables import VariableResolver


class OrchestratorNode(BaseNode):
    """LLM-driven orchestrator that delegates to sub-agent flows.

    At runtime:
    1. Discovers sub-agents from connected SubAgentNodes (via graph edges)
    2. Creates a ToolSpec for each sub-agent flow
    3. Creates a Vel orchestrator agent with sub-agents as callable tools
    4. Orchestrator LLM decides which sub-agents to call based on their descriptions
    5. Sub-agent flows are executed via Mesh Executor
    6. Results are synthesized or streamed through based on resultMode

    In React Flow canvas:
        StartNode → OrchestratorNode → AgentFlowNode (sub-agent 1)
                                     → AgentFlowNode (sub-agent 2)
                                     → AgentFlowNode (sub-agent 3)

    The orchestrator automatically discovers connected AgentFlowNodes as sub-agents.
    """

    def __init__(
        self,
        id: str,
        provider: str = "openai",
        model_name: str = "gpt-4o",
        instruction: str = "",
        temperature: float = 0.3,
        result_mode: str = "synthesize",
        max_iterations: int = 5,
        show_sub_agent_events: bool = True,
        event_mode: str = "full",
        config: Dict[str, Any] = None,
    ):
        """Initialize orchestrator node.

        Args:
            id: Node identifier
            provider: LLM provider (openai, anthropic, gemini)
            model_name: Model name for orchestration
            instruction: Instructions for the orchestrator LLM
            temperature: Sampling temperature (lower = more deterministic)
            result_mode: How to handle sub-agent outputs:
                - "synthesize": Orchestrator combines outputs into coherent response
                - "stream_through": Stream sub-agent outputs directly
                - "raw": Return structured output with all sub-agent responses
            max_iterations: Maximum number of sub-agent calls
            show_sub_agent_events: Whether to stream events from sub-agent execution
            event_mode: Event emission mode (full, status_only, transient_events, silent)
            config: Additional configuration
        """
        super().__init__(id, config or {})
        self.provider = provider
        self.model_name = model_name
        self.instruction = instruction
        self.temperature = temperature
        self.result_mode = result_mode
        self.max_iterations = max_iterations
        self.show_sub_agent_events = show_sub_agent_events
        self.event_mode = event_mode

        # Injected by backend/executor after parsing
        self._flow_loader: Optional[callable] = None
        self._registry = None
        self._graph = None  # ExecutionGraph reference for discovering children

    def set_flow_loader(self, loader: callable):
        """Set flow loader for loading sub-agent flow definitions.

        Args:
            loader: Function (flow_uuid, version) -> flow_json
        """
        self._flow_loader = loader

    def set_registry(self, registry):
        """Set node registry for creating sub-graphs.

        Args:
            registry: NodeRegistry instance
        """
        self._registry = registry

    def set_graph(self, graph):
        """Set graph reference for discovering child sub-agents.

        Args:
            graph: ExecutionGraph instance
        """
        self._graph = graph

    def _discover_sub_agents(self) -> List[Dict[str, Any]]:
        """Discover sub-agents from connected SubAgentNodes.

        Looks at this node's children in the graph and collects info
        from any SubAgentNode instances.

        Returns:
            List of sub-agent configurations (flowUuid, name, description)
        """
        from mesh.nodes.sub_agent import SubAgentNode

        if not self._graph:
            return []

        sub_agents = []
        child_ids = self._graph.get_children(self.id)

        for child_id in child_ids:
            child_node = self._graph.get_node(child_id)
            if isinstance(child_node, SubAgentNode):
                info = child_node.get_info()
                sub_agents.append(info.to_dict())

        return sub_agents

    def _create_sub_agent_tools(self, context: ExecutionContext, sub_agents: List[Dict[str, Any]]) -> List:
        """Create ToolSpec for each sub-agent flow.

        Each sub-agent flow is wrapped as a ToolSpec that, when called,
        executes the flow via Mesh Executor.

        Args:
            context: Execution context
            sub_agents: List of sub-agent configurations from discovery

        Returns:
            List of ToolSpec instances
        """
        from vel.tools import ToolSpec

        tools = []

        for sa in sub_agents:
            flow_uuid = sa.get("flowUuid")
            if not flow_uuid:
                continue

            name = sa.get("name", f"agent_{flow_uuid[:8]}")
            description = sa.get("description", f"Delegate tasks to {name}")

            # Sanitize name for tool compatibility
            tool_name = re.sub(r'[:\-.\s]', '_', name).lower()

            # Load flow JSON
            if not self._flow_loader:
                raise RuntimeError(
                    f"OrchestratorNode '{self.id}' requires flow_loader. "
                    "Call set_flow_loader() first."
                )

            flow_json = self._flow_loader(flow_uuid, None)
            if not flow_json:
                continue

            # Create handler using factory function to properly close over variables
            def make_handler(fj, fuuid, ctx, show_ev, ev_mode, nid, fl, reg):
                async def handler(message: str) -> Dict[str, Any]:
                    """Execute sub-agent flow and return result."""
                    from mesh.parsers.react_flow import ReactFlowParser
                    from mesh.core.executor import Executor
                    from mesh.backends.memory import MemoryBackend
                    from mesh.core.state import ExecutionContext as MeshContext

                    # Parse and execute the subflow
                    parser = ReactFlowParser(reg, flow_loader=fl)
                    graph = parser.parse(fj)

                    executor = Executor(graph, MemoryBackend())
                    sub_context = MeshContext(
                        graph_id=fuuid,
                        session_id=ctx.session_id,
                    )

                    result = None
                    full_output = ""

                    async for event in executor.execute({"message": message}, sub_context):
                        # Forward events if enabled
                        if show_ev and ev_mode not in ("silent", "status_only"):
                            prefixed_event = ExecutionEvent(
                                type=event.type,
                                node_id=f"{nid}.{event.node_id}",
                                content=getattr(event, 'content', None),
                                delta=getattr(event, 'delta', None),
                                output=getattr(event, 'output', None),
                                error=getattr(event, 'error', None),
                                metadata=getattr(event, 'metadata', {}),
                                raw_event=getattr(event, 'raw_event', None),
                            )
                            await ctx.emit_event(prefixed_event)

                        if event.type == EventType.TEXT_DELTA:
                            delta = getattr(event, 'delta', '') or ''
                            full_output += delta

                        if event.type == EventType.EXECUTION_COMPLETE:
                            result = getattr(event, 'output', None)

                    if result is not None:
                        if isinstance(result, str):
                            return {"response": result}
                        elif isinstance(result, dict):
                            return result
                        else:
                            return {"response": str(result)}
                    elif full_output:
                        return {"response": full_output}
                    else:
                        return {"response": "Sub-agent completed"}

                return handler

            handler = make_handler(
                flow_json, flow_uuid, context,
                self.show_sub_agent_events, self.event_mode,
                self.id, self._flow_loader, self._registry
            )

            tool = ToolSpec(
                name=tool_name,
                description=description,
                input_schema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": f"Task or query to delegate to {name}"
                        }
                    },
                    "required": ["message"],
                },
                handler=handler,
            )
            tools.append(tool)

        return tools

    async def _emit_event_if_enabled(self, context: ExecutionContext, event: ExecutionEvent) -> None:
        """Emit event based on event_mode configuration."""
        if self.event_mode == "silent":
            return

        if self.event_mode == "status_only":
            # Only mesh node start/complete events are allowed (handled separately)
            return

        if self.event_mode == "transient_events":
            transformed = transform_event_for_transient_mode(event, "orchestrator")
            await context.emit_event(transformed)
        else:
            # event_mode == "full"
            await context.emit_event(event)

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute orchestrator with sub-agent delegation.

        Args:
            input: Input message
            context: Execution context

        Returns:
            NodeResult with orchestrator response
        """
        # Emit node start event
        if context.graph_metadata and self.event_mode != "silent":
            is_final = self.id in context.graph_metadata.final_nodes
            is_intermediate = self.id in context.graph_metadata.intermediate_nodes

            start_event = create_mesh_node_start_event(
                node_id=self.id,
                node_type="orchestrator",
                is_final=is_final,
                is_intermediate=is_intermediate,
            )
            await context.emit_event(start_event)

        try:
            from vel import Agent as VelAgent
        except ImportError:
            raise RuntimeError(
                f"OrchestratorNode '{self.id}' requires Vel SDK. "
                "Install with: pip install vel"
            )

        # Discover sub-agents from graph edges
        sub_agents = self._discover_sub_agents()

        # Create sub-agent tools
        sub_agent_tools = self._create_sub_agent_tools(context, sub_agents)

        if not sub_agent_tools:
            raise RuntimeError(
                f"OrchestratorNode '{self.id}' has no valid sub-agents. "
                "Connect AgentFlowNodes as children of the orchestrator in the canvas."
            )

        # Resolve instruction variables
        resolved_instruction = self.instruction
        if self.instruction:
            resolver = VariableResolver(context)
            resolved_instruction = await resolver.resolve(self.instruction)

        # Create orchestrator agent with sub-agents as tools
        orchestrator = VelAgent(
            id=f"{self.id}_orchestrator",
            model={
                "provider": self.provider,
                "model": self.model_name,
            },
            tools=sub_agent_tools,
            instruction=resolved_instruction,
            generation_config={"temperature": self.temperature},
            policies={"max_steps": self.max_iterations * 2},
        )

        # Extract message from input
        message = self._extract_message(input)

        # Execute orchestrator with streaming
        full_response = ""
        sub_agent_outputs = []

        async for event in orchestrator.run_stream(
            {"message": message},
            session_id=context.session_id,
        ):
            if isinstance(event, dict):
                event_type = event.get("type", "")

                if event_type == "text-delta":
                    delta = event.get("delta", "")
                    if delta:
                        full_response += delta
                        await self._emit_event_if_enabled(
                            context,
                            ExecutionEvent(
                                type=EventType.TEXT_DELTA,
                                node_id=self.id,
                                delta=delta,
                                metadata={"node_type": "orchestrator"},
                                raw_event=event,
                            )
                        )

                elif event_type == "tool-input-available":
                    # Log tool calls for debugging
                    tool_name = event.get("toolName", "")
                    await self._emit_event_if_enabled(
                        context,
                        ExecutionEvent(
                            type=EventType.TOOL_INPUT_AVAILABLE,
                            node_id=self.id,
                            metadata={
                                "tool_name": tool_name,
                                "input": event.get("input"),
                                "node_type": "orchestrator",
                            },
                            raw_event=event,
                        )
                    )

                elif event_type == "tool-output-available":
                    output = event.get("output", {})
                    sub_agent_outputs.append(output)
                    await self._emit_event_if_enabled(
                        context,
                        ExecutionEvent(
                            type=EventType.TOOL_OUTPUT_AVAILABLE,
                            node_id=self.id,
                            output=output,
                            metadata={"node_type": "orchestrator"},
                            raw_event=event,
                        )
                    )

        # Build output based on result_mode
        if self.result_mode == "raw":
            output = {
                "orchestrator_response": full_response,
                "sub_agent_outputs": sub_agent_outputs,
            }
        else:
            # synthesize or stream_through - use full response
            output = {"content": full_response}

        # Emit node complete event
        if context.graph_metadata and self.event_mode != "silent":
            is_final = self.id in context.graph_metadata.final_nodes
            is_intermediate = self.id in context.graph_metadata.intermediate_nodes

            output_preview = None
            if full_response:
                output_preview = full_response[:100] + "..." if len(full_response) > 100 else full_response

            complete_event = create_mesh_node_complete_event(
                node_id=self.id,
                node_type="orchestrator",
                is_final=is_final,
                is_intermediate=is_intermediate,
                output_preview=output_preview,
            )
            await context.emit_event(complete_event)

        return NodeResult(
            output=output,
            metadata={
                "orchestrator_model": self.model_name,
                "sub_agents_called": len(sub_agent_outputs),
                "sub_agents_available": len(sub_agents),
                "result_mode": self.result_mode,
            },
        )

    def _extract_message(self, input: Any) -> str:
        """Extract message string from various input formats."""
        if isinstance(input, str):
            return input
        elif isinstance(input, dict):
            # Try common keys
            for key in ["message", "content", "text", "input", "question"]:
                if key in input:
                    return str(input[key])
            # If dict with 'output' key (from previous node)
            if "output" in input:
                return self._extract_message(input["output"])
        return str(input)
