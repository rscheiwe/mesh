"""React Flow JSON parser for Flowise compatibility.

This module parses React Flow JSON format (used by Flowise) into
Mesh ExecutionGraph structures.
"""

import json
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
from mesh.nodes.rag import RAGNode
from mesh.nodes.data_handler import DataHandlerNode
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
        "ragAgentflow": "rag",
        "dataHandlerAgentflow": "data_handler",
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

            elif node_type == "rag":
                return self._create_rag_node(node_id, config)

            elif node_type == "data_handler":
                return self._create_data_handler_node(node_id, config)

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
        """Create AgentNode from config.

        Supports two modes:
        1. Registry mode: config has 'agent' reference (legacy, for pre-configured agents)
        2. Inline mode: config has model/provider/etc. (instantiates Vel Agent on the fly)

        Tools can be provided inline via 'tools' array with:
        - code: Python function code
        - name: Optional tool name
        - description: Optional tool description

        Structured output can be enabled via 'outputSchema' (JSON Schema).
        Non-streaming mode can be enabled via 'streaming: false'.
        """
        agent_ref = config.get("agent")

        # Parse output schema if provided (JSON Schema -> Pydantic model)
        output_type = None
        output_schema = config.get("outputSchema")
        if output_schema:
            output_type = self._create_output_type_from_schema(output_schema, node_id)

        # Mode 1: Registry mode (legacy)
        if agent_ref:
            # Get pre-registered agent from registry
            agent = self.registry.get_agent(agent_ref)
            # If output_type specified, we need to set it on the agent
            if output_type and hasattr(agent, 'output_type'):
                agent.output_type = output_type

        # Mode 2: Inline mode (preferred) - instantiate Vel Agent from config
        else:
            try:
                from vel import Agent as VelAgent
                from vel.tools import ToolSpec
            except ImportError:
                raise GraphValidationError(
                    f"Agent node '{node_id}' requires Vel SDK. Install with: pip install vel"
                )

            # Get agent configuration
            model_config = config.get("model", {})
            if isinstance(model_config, str):
                # Simple string model name
                model_config = {"model": model_config}

            # Extract model parameters
            provider = config.get("provider") or model_config.get("provider", "openai")
            model = config.get("modelName") or model_config.get("model", "gpt-4o-mini")
            temperature = config.get("temperature")
            if temperature is None:
                temperature = model_config.get("temperature", 0.7)
            max_tokens = config.get("maxTokens") or config.get("max_tokens") or model_config.get("max_tokens")

            # Process tools array (inline tool definitions)
            tools = []
            tools_config = config.get("tools", [])
            if tools_config:
                tools = self._create_tools_from_config(tools_config, node_id)

            # Instantiate Vel Agent with tools and optional output_type
            agent = VelAgent(
                id=node_id,
                model={
                    "provider": provider,
                    "model": model,
                    "temperature": float(temperature),
                    **({"max_tokens": max_tokens} if max_tokens else {}),
                },
                tools=tools if tools else None,  # Pass tools to agent
                output_type=output_type,  # Pass structured output type
            )

        system_prompt = config.get("systemPrompt") or config.get("system_prompt")
        use_native_events = config.get("useNativeEvents", False)
        event_mode = config.get("eventMode", "full")
        streaming = config.get("streaming", True)  # Default to streaming

        return AgentNode(
            id=node_id,
            agent=agent,
            system_prompt=system_prompt,
            use_native_events=use_native_events,
            event_mode=event_mode,
            streaming=streaming,
            config=config,
        )

    def _create_output_type_from_schema(
        self,
        schema: Any,
        node_id: str
    ) -> type:
        """Create a Pydantic model from JSON Schema.

        Args:
            schema: JSON Schema (dict or string)
            node_id: Node ID for error messages

        Returns:
            Pydantic model class or List[model] for arrays
        """
        from typing import List, Optional, Any as TypingAny
        from pydantic import create_model, Field

        # Parse schema if it's a string
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except json.JSONDecodeError as e:
                raise GraphValidationError(
                    f"Invalid JSON in outputSchema for node '{node_id}': {e}"
                )

        if not isinstance(schema, dict):
            raise GraphValidationError(
                f"outputSchema for node '{node_id}' must be a JSON Schema object"
            )

        schema_type = schema.get("type")

        # Handle array type - returns List[ItemModel]
        if schema_type == "array":
            items_schema = schema.get("items", {})
            item_model = self._create_pydantic_model_from_schema(
                items_schema,
                f"{node_id}_Item"
            )
            return List[item_model]

        # Handle object type - returns single model
        elif schema_type == "object":
            return self._create_pydantic_model_from_schema(schema, f"{node_id}_Output")

        else:
            raise GraphValidationError(
                f"outputSchema for node '{node_id}' must have type 'object' or 'array'"
            )

    def _create_pydantic_model_from_schema(
        self,
        schema: Dict[str, Any],
        model_name: str
    ) -> type:
        """Create a Pydantic model from JSON Schema object definition.

        Args:
            schema: JSON Schema object definition
            model_name: Name for the generated model

        Returns:
            Pydantic model class
        """
        from typing import Optional, Any as TypingAny, List
        from pydantic import create_model, Field

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        # Map JSON Schema types to Python types
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        fields = {}
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "string")
            python_type = type_map.get(prop_type, TypingAny)

            # Handle nested arrays
            if prop_type == "array":
                items_type = prop_schema.get("items", {}).get("type", "string")
                item_python_type = type_map.get(items_type, TypingAny)
                python_type = List[item_python_type]

            # Make optional if not required
            if prop_name not in required:
                python_type = Optional[python_type]
                default = prop_schema.get("default", None)
            else:
                default = ...  # Required field

            description = prop_schema.get("description", "")
            fields[prop_name] = (python_type, Field(default=default, description=description))

        return create_model(model_name, **fields)

    def _create_tools_from_config(
        self,
        tools_config: List[Dict[str, Any]],
        node_id: str
    ) -> List[Any]:
        """Create ToolSpec instances from inline tool definitions.

        Args:
            tools_config: List of tool configurations, each with:
                - code: Python function code (required)
                - name: Optional tool name (auto-generated if not provided)
                - description: Optional tool description (extracted from docstring if not provided)
            node_id: Parent node ID (for error messages)

        Returns:
            List of ToolSpec instances

        Raises:
            GraphValidationError: If tool creation fails
        """
        try:
            from vel.tools import ToolSpec
        except ImportError:
            raise GraphValidationError(
                f"Tools require Vel SDK. Install with: pip install vel"
            )

        tools = []
        for idx, tool_config in enumerate(tools_config):
            code = tool_config.get("code")
            if not code:
                raise GraphValidationError(
                    f"Tool {idx} in node '{node_id}' missing 'code' field"
                )

            # Execute code to extract function
            namespace = {}
            try:
                exec(code, namespace)
            except Exception as e:
                raise GraphValidationError(
                    f"Failed to execute tool code in node '{node_id}' tool {idx}: {e}"
                )

            # Find the function
            func_name = tool_config.get("name")
            if func_name:
                if func_name not in namespace:
                    raise GraphValidationError(
                        f"Function '{func_name}' not found in tool code for node '{node_id}' tool {idx}"
                    )
                func = namespace[func_name]
            else:
                # Find first callable (skip builtins)
                func = None
                for name, obj in namespace.items():
                    if callable(obj) and not name.startswith('_'):
                        func = obj
                        func_name = name
                        break

                if not func:
                    raise GraphValidationError(
                        f"No callable function found in tool code for node '{node_id}' tool {idx}"
                    )

            # Create ToolSpec using new dynamic tools API
            tool_spec = ToolSpec.from_function(
                func,
                name=func_name,
                description=tool_config.get("description"),  # Optional override
            )

            tools.append(tool_spec)

        return tools

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
        """Create ToolNode from config.

        Supports three modes:
        1. Registry mode: config has 'tool' reference (for pre-registered tools)
        2. Placeholder mode: config has 'toolUuid' (tool function injected later by backend)
        3. Inline mode: config has 'code' (executes Python code directly)
        """
        # Parse bindings if it's a JSON string (from frontend)
        bindings = config.get("bindings")
        if isinstance(bindings, str):
            try:
                config["bindings"] = json.loads(bindings)
            except json.JSONDecodeError:
                # If invalid JSON, keep as empty dict
                config["bindings"] = {}

        tool_ref = config.get("tool")
        tool_uuid = config.get("toolUuid")
        code = config.get("code")

        # Mode 1: Registry mode (for pre-registered tools)
        if tool_ref:
            tool_fn = self.registry.get_tool(tool_ref)

        # Mode 2: Placeholder mode - tool UUID stored, function injected later
        elif tool_uuid:
            # Create a placeholder function that will be replaced by backend
            # The backend is responsible for loading from DB and calling set_tool_function()
            def placeholder_tool(*args, **kwargs):
                raise RuntimeError(
                    f"Tool '{tool_uuid}' not injected. "
                    f"Backend must load tool from DB and call node.set_tool_function() before execution."
                )
            tool_fn = placeholder_tool

        # Mode 3: Inline mode - execute code directly
        elif code:
            tool_fn = self._execute_tool_code(code, config)

        else:
            raise GraphValidationError(
                f"Tool node '{node_id}' missing 'tool' reference, 'toolUuid', or 'code'"
            )

        return ToolNode(
            id=node_id,
            tool_fn=tool_fn,
            event_mode=config.get("eventMode", "full"),
            config=config,
        )

    def _execute_tool_code(self, code: str, config: Dict[str, Any], imports=None, func_name=None):
        """Execute Python code to get tool function.

        Args:
            code: Python code defining the tool function
            config: Tool configuration
            imports: Optional list of import statements
            func_name: Optional function name to extract

        Returns:
            Callable tool function
        """
        # Execute imports
        if imports:
            for imp in imports:
                exec(imp)

        # Execute code
        namespace = {}
        exec(code, namespace)

        # Find the function
        if func_name:
            if func_name not in namespace:
                raise GraphValidationError(f"Function '{func_name}' not found in code")
            return namespace[func_name]
        else:
            # Find first callable
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    return obj

            raise GraphValidationError("No callable function found in code")

    def _create_rag_node(self, node_id: str, config: Dict[str, Any]) -> RAGNode:
        """Create RAGNode from config.

        Note: Retriever instance must be injected after parsing via set_retriever().
        """
        return RAGNode(
            id=node_id,
            query_template=config.get("queryTemplate") or config.get("query_template", "{{$question}}"),
            top_k=config.get("topK") or config.get("top_k", 5),
            similarity_threshold=config.get("similarityThreshold") or config.get("similarity_threshold", 0.7),
            file_id=config.get("fileId") or config.get("file_id"),
            folder_uuid=config.get("folderUuid") or config.get("folder_uuid"),
            retriever_type=config.get("retrieverType") or config.get("retriever_type", "postgres"),
            event_mode=config.get("eventMode", "full"),
            config=config,
        )

    def _create_data_handler_node(self, node_id: str, config: Dict[str, Any]) -> DataHandlerNode:
        """Create DataHandlerNode from config.

        Note: DB session getter must be injected after parsing via set_db_session_getter().
        """
        # Parse params - handle both string and dict
        params_value = config.get("params", {})
        if isinstance(params_value, str):
            import json
            try:
                params = json.loads(params_value) if params_value else {}
            except json.JSONDecodeError:
                params = {}
        else:
            params = params_value or {}

        return DataHandlerNode(
            id=node_id,
            db_source=config.get("dbSource") or config.get("db_source", "postgres"),
            query=config.get("query", ""),
            params=params,
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
