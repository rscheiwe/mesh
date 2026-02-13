"""Upstream context injection for LLM and Agent nodes.

When auto_inject_context is enabled on an LLMNode or AgentNode, this module
builds a structured XML preamble from upstream DataHandler, Tool, and RAG node
outputs and prepends it to the system prompt. This allows the LLM to use
upstream data without the user manually wiring template variables.
"""

import json
import re
from typing import Any, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from mesh.core.state import ExecutionContext

# Node types eligible for automatic context injection
_CONTEXT_ELIGIBLE_TYPES = {"DataHandlerNode", "ToolNode", "RAGNode", "APIHandlerNode"}

# Limits to prevent excessive token usage
_MAX_ROWS_PER_SOURCE = 50
_MAX_CHARS_PER_SOURCE = 8000


class UpstreamContextBuilder:
    """Builds XML-structured context preamble from upstream node outputs.

    Collects outputs from DataHandler, Tool, and RAG nodes that have already
    executed and formats them into a structured preamble for the LLM.

    Features:
        - Deduplication: Skips nodes already referenced via {{node_id...}} in prompt
        - Truncation: Limits rows and character count per source
        - Fallback heuristic: When node_type is unavailable, uses node_id patterns
    """

    @staticmethod
    def build(context: "ExecutionContext", system_prompt: str = "") -> str:
        """Build a structured context preamble from upstream node outputs.

        Args:
            context: Execution context with executed_data
            system_prompt: The raw (pre-resolved) system prompt, used for
                deduplication — nodes already referenced via {{node_id...}}
                are skipped.

        Returns:
            XML-structured context preamble, or empty string if no relevant data.
        """
        if not context.executed_data:
            return ""

        # Find node IDs already referenced in the system prompt
        referenced_ids = _extract_referenced_node_ids(system_prompt)

        sources = []
        for exec_entry in context.executed_data:
            node_id = exec_entry["node_id"]
            output = exec_entry.get("output")
            node_type = exec_entry.get("node_type")

            # Skip nodes already referenced in the system prompt
            if node_id in referenced_ids:
                continue

            # Determine eligibility and display type
            is_eligible, display_type = _classify_node(node_id, node_type)

            if not is_eligible or output is None:
                continue

            # Format the source output
            formatted = _format_source_output(node_id, display_type, output)
            if formatted:
                sources.append(formatted)

        if not sources:
            return ""

        preamble = "<upstream_context>\n"
        preamble += "The following data was retrieved by upstream nodes in this workflow. "
        preamble += "Use it to inform your response.\n\n"
        preamble += "\n\n".join(sources)
        preamble += "\n</upstream_context>\n\n"
        return preamble


def _extract_referenced_node_ids(system_prompt: str) -> Set[str]:
    """Extract node IDs referenced via {{node_id...}} in a system prompt.

    Args:
        system_prompt: Raw system prompt text

    Returns:
        Set of node IDs found in template variable references
    """
    if not system_prompt:
        return set()

    referenced = set()
    for match in re.finditer(r"\{\{\s*(\w+)", system_prompt):
        ref = match.group(1).strip()
        # Skip built-in variables like $input, $question, $vars, etc.
        if not ref.startswith("$"):
            referenced.add(ref)
    return referenced


def _classify_node(node_id: str, node_type: Optional[str]) -> tuple:
    """Determine if a node is eligible for context injection and its display type.

    Args:
        node_id: The node ID
        node_type: Python class name (may be None for older executed_data)

    Returns:
        Tuple of (is_eligible: bool, display_type: str)
    """
    if node_type:
        if node_type in _CONTEXT_ELIGIBLE_TYPES:
            type_map = {
                "DataHandlerNode": "database_query",
                "RAGNode": "document_retrieval",
                "ToolNode": "tool_output",
                "APIHandlerNode": "api_response",
            }
            return True, type_map.get(node_type, "data")
        return False, ""

    # Fallback heuristic when node_type is not available
    lower_id = node_id.lower()
    if "datahandler" in lower_id or "data_handler" in lower_id:
        return True, "database_query"
    elif "rag" in lower_id:
        return True, "document_retrieval"
    elif "api" in lower_id:
        return True, "api_response"
    elif "tool" in lower_id:
        return True, "tool_output"

    return False, ""


def _format_source_output(node_id: str, source_type: str, output: Any) -> str:
    """Format a single source's output for the context preamble.

    Args:
        node_id: The source node ID
        source_type: Type label (database_query, document_retrieval, tool_output)
        output: The node's output data

    Returns:
        Formatted XML string for this source
    """
    result = f'<source node_id="{node_id}" type="{source_type}">\n'

    if isinstance(output, dict):
        # DataHandler output: {"rows": [...], "count": N, "query": "...", "params": {...}}
        if "rows" in output:
            rows = output["rows"]
            count = output.get("count", len(rows) if isinstance(rows, list) else 0)
            query = output.get("query", "")

            result += f'  <metadata count="{count}"'
            if query:
                # Escape quotes in query for XML attribute
                safe_query = str(query).replace('"', "&quot;")
                result += f' query="{safe_query}"'
            result += " />\n"

            # Truncate rows if needed
            display_rows = rows[:_MAX_ROWS_PER_SOURCE] if isinstance(rows, list) else rows
            rows_json = json.dumps(display_rows, default=str, ensure_ascii=False)
            if len(rows_json) > _MAX_CHARS_PER_SOURCE:
                rows_json = rows_json[:_MAX_CHARS_PER_SOURCE] + "...(truncated)"
            result += f"  <rows>\n{rows_json}\n  </rows>\n"

            if isinstance(rows, list) and len(rows) > _MAX_ROWS_PER_SOURCE:
                result += f"  <note>Showing {_MAX_ROWS_PER_SOURCE} of {count} rows</note>\n"

        # RAG output with formatted context
        elif "formatted" in output:
            result += f"  {output['formatted']}\n"

        # Generic dict output
        else:
            output_json = json.dumps(output, default=str, ensure_ascii=False)
            if len(output_json) > _MAX_CHARS_PER_SOURCE:
                output_json = output_json[:_MAX_CHARS_PER_SOURCE] + "...(truncated)"
            result += f"  <data>\n{output_json}\n  </data>\n"

    elif isinstance(output, str):
        truncated = output[:_MAX_CHARS_PER_SOURCE]
        if len(output) > _MAX_CHARS_PER_SOURCE:
            truncated += "...(truncated)"
        result += f"  <data>\n{truncated}\n  </data>\n"

    else:
        result += f"  <data>\n{str(output)[:_MAX_CHARS_PER_SOURCE]}\n  </data>\n"

    result += "</source>"
    return result
