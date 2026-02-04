"""Auto-detect and inject input collection interrupts for upstream tools.

When Tool/DataHandler nodes are upstream of LLM/Agent nodes and have required
parameters without defaults, this module automatically injects interrupt_before
points to collect those parameters from the user via chat.

Example graph:
    Tool A ──┐
             ├──> LLM
    Tool B ──┘

If Tool A needs params (a, b) and Tool B needs params (query), the executor will:
1. Start execution
2. Hit Tool A → interrupt_before → emit INTERRUPT with param schema
3. User provides {a: 1, b: 2} → resume
4. Tool A executes
5. Hit Tool B → interrupt_before → emit INTERRUPT with param schema
6. User provides {query: "search term"} → resume
7. Tool B executes
8. LLM receives both tool outputs
"""

import inspect
from typing import Dict, Any, List, Optional, Set, Callable, get_type_hints
from dataclasses import dataclass, field


@dataclass
class RequiredParam:
    """A required parameter that needs user input."""
    name: str
    param_type: str  # String representation of the type
    description: Optional[str] = None
    default: Any = None
    has_default: bool = False


@dataclass
class ToolInputRequirement:
    """Input requirements for a tool node."""
    node_id: str
    tool_name: str
    tool_description: Optional[str]
    required_params: List[RequiredParam]


def inspect_tool_required_params(
    tool_fn: Callable,
    bindings: Optional[Dict[str, Any]] = None,
) -> List[RequiredParam]:
    """Inspect a tool function and return its required parameters.

    Parameters that are:
    - Auto-injected (input, context, state, variables, chat_history) are excluded
    - Already bound via config["bindings"] are excluded
    - Have defaults are marked but included (user can override)

    Args:
        tool_fn: The tool function to inspect
        bindings: Already-bound parameters from node config

    Returns:
        List of RequiredParam objects for params needing user input
    """
    # Auto-injected params that are handled by the executor
    AUTO_INJECTED = {"input", "context", "state", "variables", "chat_history"}

    bindings = bindings or {}
    required_params = []

    try:
        sig = inspect.signature(tool_fn)
        type_hints = get_type_hints(tool_fn) if hasattr(tool_fn, '__annotations__') else {}
    except (ValueError, TypeError):
        # Can't inspect this function
        return []

    # Get docstring for param descriptions
    docstring = inspect.getdoc(tool_fn) or ""
    param_docs = _parse_docstring_params(docstring)

    for param_name, param in sig.parameters.items():
        # Skip auto-injected params
        if param_name in AUTO_INJECTED:
            continue

        # Skip already-bound params
        if param_name in bindings:
            continue

        # Skip *args and **kwargs
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # Determine type string
        param_type = "any"
        if param_name in type_hints:
            hint = type_hints[param_name]
            param_type = _type_to_string(hint)
        elif param.annotation != inspect.Parameter.empty:
            param_type = _type_to_string(param.annotation)

        # Check for default
        has_default = param.default != inspect.Parameter.empty
        default_value = param.default if has_default else None

        required_params.append(RequiredParam(
            name=param_name,
            param_type=param_type,
            description=param_docs.get(param_name),
            default=default_value,
            has_default=has_default,
        ))

    return required_params


def _type_to_string(type_hint: Any) -> str:
    """Convert a type hint to a user-friendly string."""
    if type_hint is None:
        return "any"

    # Handle basic types
    if type_hint in (str, int, float, bool, list, dict):
        return type_hint.__name__

    # Handle typing module types
    type_str = str(type_hint)

    # Clean up typing module prefixes
    for prefix in ("typing.", "<class '", "'>"):
        type_str = type_str.replace(prefix, "")

    return type_str


def _parse_docstring_params(docstring: str) -> Dict[str, str]:
    """Parse parameter descriptions from docstring (Google/NumPy style)."""
    params = {}
    if not docstring:
        return params

    lines = docstring.split("\n")
    in_params = False
    current_param = None

    for line in lines:
        stripped = line.strip()

        # Detect Args/Parameters section
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            in_params = True
            continue

        # Detect end of params section
        if in_params and stripped.lower() in ("returns:", "raises:", "example:", "examples:", "note:", "notes:"):
            in_params = False
            continue

        if in_params:
            # Try to parse "param_name: description" or "param_name (type): description"
            if ":" in stripped and not stripped.startswith(" "):
                parts = stripped.split(":", 1)
                param_part = parts[0].strip()
                desc_part = parts[1].strip() if len(parts) > 1 else ""

                # Handle "param_name (type)" format
                if "(" in param_part:
                    param_name = param_part.split("(")[0].strip()
                else:
                    param_name = param_part

                if param_name and not param_name.startswith("-"):
                    params[param_name] = desc_part
                    current_param = param_name
            elif current_param and stripped:
                # Continuation of previous param description
                params[current_param] += " " + stripped

    return params


def inspect_data_handler_required_params(
    node: "DataHandlerNode",
) -> List[RequiredParam]:
    """Inspect a DataHandlerNode for required SQL query parameters.

    DataHandlerNodes use SQL queries with named parameters like :param_name.
    This function parses the query and checks which params don't have fixed values.

    Args:
        node: The DataHandlerNode to inspect

    Returns:
        List of RequiredParam objects for params needing user input
    """
    import re

    required_params = []

    # Parse SQL query for :param_name placeholders
    query = getattr(node, 'query', '') or ''
    param_pattern = r':(\w+)'
    query_params = set(re.findall(param_pattern, query))

    if not query_params:
        return required_params

    # Get fixed params (already provided)
    fixed_params = getattr(node, 'fixed_params', {}) or {}

    # Find params that are in query but not in fixed_params
    for param_name in query_params:
        if param_name not in fixed_params:
            required_params.append(RequiredParam(
                name=param_name,
                param_type="any",  # SQL params are typically strings/numbers
                description=f"SQL query parameter for: {param_name}",
                default=None,
                has_default=False,
            ))

    return required_params


def analyze_graph_input_requirements(
    graph: "ExecutionGraph",
    get_tool_fn: Optional[Callable[[str], Optional[Callable]]] = None,
) -> List[ToolInputRequirement]:
    """Analyze a graph to find tools that need user input.

    Identifies Tool/DataHandler nodes that are upstream of LLM/Agent nodes
    and have required parameters without defaults or bindings.

    Args:
        graph: The ExecutionGraph to analyze
        get_tool_fn: Optional callback to get the actual tool function for a node
                     (useful when functions are loaded from DB)

    Returns:
        List of ToolInputRequirement objects for nodes needing input
    """
    from mesh.nodes.tool import ToolNode
    from mesh.nodes.data_handler import DataHandlerNode
    from mesh.nodes.llm import LLMNode
    from mesh.nodes.agent import AgentNode

    requirements = []

    # Find all LLM and Agent nodes
    llm_agent_nodes = set()
    for node_id, node in graph.nodes.items():
        if isinstance(node, (LLMNode, AgentNode)):
            llm_agent_nodes.add(node_id)

    if not llm_agent_nodes:
        return requirements

    # Find all nodes that are upstream of LLM/Agent nodes
    upstream_of_llm = _find_upstream_nodes(graph, llm_agent_nodes)

    # Check each Tool/DataHandler node
    for node_id, node in graph.nodes.items():
        # Only process if upstream of LLM/Agent
        if node_id not in upstream_of_llm:
            continue

        # Handle DataHandlerNode specially (SQL query params)
        if isinstance(node, DataHandlerNode):
            required_params = inspect_data_handler_required_params(node)

            if required_params:
                requirements.append(ToolInputRequirement(
                    node_id=node_id,
                    tool_name=f"DataHandler ({node.db_source})",
                    tool_description=f"SQL Query: {node.query[:100]}..." if len(node.query) > 100 else f"SQL Query: {node.query}",
                    required_params=required_params,
                ))
            continue

        # Handle regular ToolNode
        if isinstance(node, ToolNode):
            tool_fn = node.tool_fn

            # Allow override via callback (for dynamically loaded tools)
            if get_tool_fn:
                override_fn = get_tool_fn(node_id)
                if override_fn:
                    tool_fn = override_fn

            if not tool_fn:
                continue

            # Get bindings from node config
            bindings = node.config.get("bindings", {})

            # Inspect for required params
            required_params = inspect_tool_required_params(tool_fn, bindings)

            # Only include if there are params WITHOUT defaults that need input
            params_needing_input = [p for p in required_params if not p.has_default]

            if params_needing_input:
                requirements.append(ToolInputRequirement(
                    node_id=node_id,
                    tool_name=getattr(node, 'function_name', node_id),
                    tool_description=getattr(node, 'function_doc', None),
                    required_params=params_needing_input,
                ))

    return requirements


def _find_upstream_nodes(graph: "ExecutionGraph", target_nodes: Set[str]) -> Set[str]:
    """Find all nodes that are upstream (ancestors) of the target nodes.

    Uses reverse BFS from target nodes to find all ancestors.
    """
    upstream = set()
    queue = list(target_nodes)

    while queue:
        current = queue.pop(0)
        parents = graph.get_parents(current)

        for parent in parents:
            if parent not in upstream and parent not in target_nodes:
                upstream.add(parent)
                queue.append(parent)

    return upstream


def inject_input_collection_interrupts(
    graph_builder: "StateGraph",
    requirements: List[ToolInputRequirement],
) -> "StateGraph":
    """Inject interrupt_before points for tools that need input collection.

    For each tool with required params, adds an interrupt_before that will
    pause execution and emit metadata for the frontend to render an input form.

    Args:
        graph_builder: The StateGraph builder (before compilation)
        requirements: List of ToolInputRequirement from analyze_graph_input_requirements

    Returns:
        The modified StateGraph builder
    """
    for req in requirements:
        # Create metadata extractor for this tool
        def make_metadata_extractor(requirement: ToolInputRequirement):
            def extractor(state: Dict[str, Any], input_data: Any) -> Dict[str, Any]:
                return {
                    "interrupt_type": "input_collection",
                    "tool_node_id": requirement.node_id,
                    "tool_name": requirement.tool_name,
                    "tool_description": requirement.tool_description,
                    "required_params": [
                        {
                            "name": p.name,
                            "type": p.param_type,
                            "description": p.description,
                            "default": p.default,
                            "has_default": p.has_default,
                        }
                        for p in requirement.required_params
                    ],
                }
            return extractor

        graph_builder.set_interrupt_before(
            node_id=req.node_id,
            metadata_extractor=make_metadata_extractor(req),
        )

    return graph_builder


def create_input_collection_condition(
    check_state_key: Optional[str] = None,
) -> Callable[[Dict[str, Any], Any], bool]:
    """Create a condition function for input collection interrupts.

    By default, always triggers. Can be configured to check state for
    a flag indicating params were already collected.

    Args:
        check_state_key: If provided, only interrupt if this key is not in state

    Returns:
        Condition function for use with set_interrupt_before
    """
    def condition(state: Dict[str, Any], input_data: Any) -> bool:
        if check_state_key:
            # Don't interrupt if params already collected
            return check_state_key not in state
        return True

    return condition


# Convenience function combining analysis and injection
def setup_input_collection(
    graph_builder: "StateGraph",
    compiled_graph: Optional["ExecutionGraph"] = None,
    get_tool_fn: Optional[Callable[[str], Optional[Callable]]] = None,
) -> List[ToolInputRequirement]:
    """Analyze graph and inject input collection interrupts.

    This is the main entry point for enabling input collection on a graph.
    Call this before final compilation.

    Args:
        graph_builder: The StateGraph builder
        compiled_graph: Optional pre-compiled graph for analysis
                       (if not provided, will compile temporarily)
        get_tool_fn: Optional callback to get tool functions by node_id

    Returns:
        List of requirements that were detected and will trigger interrupts

    Example:
        >>> graph = StateGraph()
        >>> graph.add_node("calculator", calc_fn, node_type="tool")
        >>> graph.add_node("llm", None, node_type="llm", model="gpt-4")
        >>> graph.add_edge("START", "calculator")
        >>> graph.add_edge("calculator", "llm")
        >>> graph.set_entry_point("calculator")
        >>>
        >>> # Enable input collection
        >>> requirements = setup_input_collection(graph)
        >>> print(f"Will collect input for: {[r.tool_name for r in requirements]}")
        >>>
        >>> # Now compile
        >>> compiled = graph.compile()
    """
    # Compile temporarily if needed for analysis
    if compiled_graph is None:
        # We need to compile to analyze, but this modifies the builder
        # So we do a shallow analysis using the builder's internal state
        compiled_graph = graph_builder.compile()

    # Analyze
    requirements = analyze_graph_input_requirements(compiled_graph, get_tool_fn)

    # Inject interrupts
    if requirements:
        inject_input_collection_interrupts(graph_builder, requirements)

    return requirements
