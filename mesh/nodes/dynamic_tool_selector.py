"""Dynamic Tool Selector node for runtime tool discovery.

This node creates a meta-tool that agents can call to discover and inject
additional tools at runtime. When called by an agent, it searches the
connected tools and returns an inject_tools array that Vel processes.

Unlike regular tool nodes that auto-execute, this node generates a ToolSpec
that gets injected into downstream agents. The agent decides if/when to call it.
"""

from typing import Any, Dict, List, Optional, Callable
import re
import inspect

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext


class DynamicToolSelectorNode(BaseNode):
    """Dynamic Tool Selector - generates a searchable tool from connected tools.

    This node collects tool metadata from connected upstream ToolNodes and
    generates a ToolSpec that downstream agents can call to discover and
    inject tools at runtime.

    When the agent calls the generated tool with a search query:
    1. The handler searches the tool catalog using keyword scoring
    2. Matching tools are returned as inject_tools array
    3. Vel processes inject_tools and adds them to the agent's session

    Example graph layout:
        Tool A --\\
                  --> DynamicToolSelector --> Agent
        Tool B --/

    The Agent will have a "find_tools" capability that it can call
    when it needs additional tools not in its initial toolkit.
    """

    def __init__(
        self,
        id: str,
        description: str = "Search for additional tools. Use when asked to perform a task you don't have a tool for.",
        max_results: int = 5,
        event_mode: str = "full",
        config: Dict[str, Any] = None,
    ):
        """Initialize Dynamic Tool Selector node.

        Args:
            id: Node identifier (also used as the tool name exposed to agents)
            description: Description shown to agents for this meta-tool
            max_results: Maximum number of tools to return per query
            event_mode: Event emission mode:
                - "full": Streams reasoning events during discovery
                - "silent": No events
            config: Additional configuration
        """
        super().__init__(id, config or {})
        self.description = description
        self.max_results = max_results
        self.event_mode = event_mode
        self._connected_tools: List = []
        self._tool_catalog: Dict[str, Dict[str, Any]] = {}

    def set_connected_tools(self, tools: List) -> None:
        """Set connected tool nodes (called by parser after edge resolution).

        Args:
            tools: List of ToolNode instances connected to this selector
        """
        self._connected_tools = tools
        self._build_tool_catalog()

    def _build_tool_catalog(self) -> None:
        """Build searchable catalog from connected tools.

        This may be called multiple times:
        1. Initially during parsing (tool functions may not be loaded yet)
        2. After tool functions are injected by the backend
        """
        self._tool_catalog = {}  # Clear existing catalog

        for tool_node in self._connected_tools:
            # Get tool name - prefer function_name if set, otherwise use config or node id
            tool_name = getattr(tool_node, 'function_name', None)
            if not tool_name:
                # Fallback to config or node ID
                tool_name = tool_node.config.get('name') or tool_node.id

            # Get documentation
            tool_doc = getattr(tool_node, 'function_doc', '') or ""

            # Also check config for tool metadata from DB
            if not tool_doc and tool_node.config.get('toolName'):
                tool_doc = f"Tool: {tool_node.config.get('toolName')}"

            keywords = self._extract_keywords(tool_name, tool_doc)

            self._tool_catalog[tool_name] = {
                "name": tool_name,
                "description": tool_doc,
                "keywords": keywords,
                "node": tool_node,
            }

    def _extract_keywords(self, name: str, doc: str) -> List[str]:
        """Extract searchable keywords from tool name and documentation.

        Args:
            name: Tool function name (e.g., "send_email")
            doc: Tool docstring

        Returns:
            List of lowercase keywords for search matching
        """
        # Extract words from camelCase/snake_case name
        words = re.findall(r'[a-zA-Z][a-z]*', name)
        words = [w.lower() for w in words]

        # Extract words from docstring (first 20)
        if doc:
            doc_words = re.findall(r'\b\w+\b', doc.lower())
            words.extend(doc_words[:20])

        return list(set(words))

    def _search_tools(self, query: str) -> List[Dict[str, Any]]:
        """Search tool catalog using keyword scoring.

        Scoring:
        - Name match: +10 points
        - Description term match: +5 points per term
        - Keyword match: +8 points per term

        Args:
            query: Search query from agent

        Returns:
            List of matching tool definitions, sorted by score
        """
        if not query:
            # Empty query returns all tools
            return list(self._tool_catalog.values())

        query_terms = query.lower().split()
        matches = []

        for tool_name, tool_def in self._tool_catalog.items():
            score = 0

            # Name match: +10
            if query.lower() in tool_name.lower():
                score += 10

            # Description match: +5 per term
            description = tool_def.get("description", "").lower()
            for term in query_terms:
                if term in description:
                    score += 5

            # Keyword match: +8 per term
            keywords = tool_def.get("keywords", [])
            for term in query_terms:
                if term in keywords:
                    score += 8

            if score > 0:
                matches.append((score, tool_def))

        # Sort by score descending
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches]

    def _build_input_schema(self, tool_node) -> Dict[str, Any]:
        """Build JSON Schema from ToolNode's function signature.

        Args:
            tool_node: ToolNode instance

        Returns:
            JSON Schema dict for the tool's parameters
        """
        schema = {"type": "object", "properties": {}, "required": []}

        try:
            tool_fn = getattr(tool_node, 'tool_fn', None)
            if tool_fn and callable(tool_fn):
                sig = inspect.signature(tool_fn)
                for param_name, param in sig.parameters.items():
                    # Skip special parameters that are auto-injected
                    if param_name in ("input", "context", "state", "variables", "chat_history"):
                        continue

                    # Default to string type
                    schema["properties"][param_name] = {"type": "string"}

                    # Mark as required if no default
                    if param.default == inspect.Parameter.empty:
                        schema["required"].append(param_name)
        except Exception:
            # If inspection fails, return empty schema
            pass

        return schema

    def get_tool_spec(self):
        """Generate ToolSpec that agents can call.

        Returns:
            ToolSpec instance for tool discovery

        Raises:
            RuntimeError: If Vel SDK is not installed
        """
        try:
            from vel.tools import ToolSpec
        except ImportError:
            raise RuntimeError(
                "DynamicToolSelectorNode requires Vel SDK. Install with: pip install vel"
            )

        selector = self

        async def find_tools_handler(inp: Dict[str, Any], ctx: Dict[str, Any] = None):
            """Handler for the find_tools tool.

            This is an async generator that:
            1. Emits reasoning events showing search progress
            2. Searches the tool catalog
            3. Returns inject_tools array for Vel to process
            """
            query = inp.get("query", "").strip()

            # Emit reasoning events if enabled
            if selector.event_mode == "full":
                yield {"type": "reasoning-start", "transient": True}
                yield {
                    "type": "reasoning-delta",
                    "delta": f"Searching for tools matching '{query}'...",
                    "transient": True,
                }

            # Search the catalog
            matches = selector._search_tools(query)

            if selector.event_mode == "full":
                yield {
                    "type": "reasoning-delta",
                    "delta": f" Found {len(matches)} matching tools.",
                    "transient": True,
                }
                yield {"type": "reasoning-end", "transient": True}

            # No matches case
            if not matches:
                available = list(selector._tool_catalog.keys())
                yield {
                    "type": "tool-output",
                    "output": {
                        "message": f"No tools found matching '{query}'. Available tools: {available}",
                        "inject_tools": [],
                    }
                }
                return

            # Build inject_tools array from matches
            inject_tools = []
            for tool_def in matches[:selector.max_results]:
                tool_node = tool_def["node"]
                tool_fn = getattr(tool_node, 'tool_fn', None)

                inject_tools.append({
                    "name": tool_def["name"],
                    "description": tool_def["description"],
                    "input_schema": selector._build_input_schema(tool_node),
                    "handler_fn": tool_fn,  # Direct callable for Vel
                })

            tool_names = ", ".join(t["name"] for t in inject_tools)
            yield {
                "type": "tool-output",
                "output": {
                    "message": f"Found tools: {tool_names}. These are now available.",
                    "inject_tools": inject_tools,
                }
            }

        return ToolSpec(
            name=self.id,
            description=self.description,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing the capability needed (e.g., 'send email', 'query database', 'create chart')"
                    }
                },
                "required": ["query"]
            },
            handler=find_tools_handler,
        )

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute node - outputs the generated ToolSpec.

        This node doesn't perform any action during graph execution.
        Instead, it outputs a ToolSpec that downstream agent nodes
        should inject into their agent's tool set.

        Args:
            input: Input data (unused)
            context: Execution context

        Returns:
            NodeResult containing the ToolSpec and connected tool info
        """
        tool_spec = self.get_tool_spec()

        return NodeResult(
            output={
                "tool_spec": tool_spec,
                "connected_tools": list(self._tool_catalog.keys()),
            },
            metadata={
                "node_type": "dynamic_tool_selector",
                "tool_count": len(self._tool_catalog),
                "selector_id": self.id,
            },
        )

    def __repr__(self) -> str:
        return f"DynamicToolSelectorNode(id='{self.id}', tools={len(self._tool_catalog)})"
