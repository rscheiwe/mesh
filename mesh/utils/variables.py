"""Variable resolution system for template strings.

This module implements variable resolution for templates using {{variable}} syntax.
Supports references to user input, node outputs, global variables, chat history,
and iteration context.

Reference: Flowise's resolveVariables() in buildAgentflow.ts:209-399
"""

import re
from typing import Any, Dict, Optional, List
from jsonpath_ng import parse as jsonpath_parse

from mesh.core.state import ExecutionContext
from mesh.utils.errors import VariableResolutionError


class VariableResolver:
    """Resolve {{variable}} references in template strings.

    Supported variable types:
    - {{$question}} - User input to the graph
    - {{node_id}} - Reference another node's output
    - {{node_id.output.path}} - JSONPath into node output
    - {{$vars.key}} - Global variables from context
    - {{$chat_history}} - Formatted chat history
    - {{$iteration}} or {{$iteration.field}} - Current iteration context

    Example:
        >>> resolver = VariableResolver(context)
        >>> template = "User asked: {{$question}}, agent said: {{agent_node.content}}"
        >>> resolved = await resolver.resolve(template)
    """

    # Pattern to match {{variable}} syntax
    PATTERN = re.compile(r"\{\{(.*?)\}\}")

    def __init__(self, context: ExecutionContext):
        """Initialize resolver with execution context.

        Args:
            context: ExecutionContext containing state and variables
        """
        self.context = context

    async def resolve(self, template: str) -> str:
        """Replace all {{variables}} with their values.

        Args:
            template: Template string with {{variable}} references

        Returns:
            Resolved string with variables replaced

        Raises:
            VariableResolutionError: If variable resolution fails
        """
        if not template:
            return template

        def replacer(match):
            variable = match.group(1).strip()
            try:
                value = self._resolve_single(variable)
                return str(value) if value is not None else ""
            except Exception as e:
                # Return placeholder for missing variables
                return f"{{{{MISSING: {variable}}}}}"

        return self.PATTERN.sub(replacer, template)

    def _resolve_single(self, variable: str) -> Any:
        """Resolve a single variable reference.

        Args:
            variable: Variable name (without {{ }})

        Returns:
            Resolved value

        Raises:
            VariableResolutionError: If resolution fails
        """
        # {{$question}} - User input
        if variable == "$question":
            # Find the input from the first node
            if self.context.executed_data:
                first_input = self.context.executed_data[0].get("output")
                if isinstance(first_input, dict) and "input" in first_input:
                    return first_input["input"]
            return ""

        # {{$vars.key}} - Global variables
        if variable.startswith("$vars."):
            key = variable[6:]  # Remove "$vars."
            return self._get_nested(self.context.variables, key)

        # {{$chat_history}} - Formatted chat history
        if variable == "$chat_history":
            return self._format_chat_history(self.context.chat_history)

        # {{$iteration}} or {{$iteration.field}} - Iteration context
        if variable.startswith("$iteration"):
            if not self.context.iteration_context:
                return ""

            if variable == "$iteration":
                return self.context.iteration_context.get("value", "")

            # {{$iteration.field}}
            if variable.startswith("$iteration."):
                key = variable[11:]  # Remove "$iteration."
                return self._get_nested(self.context.iteration_context.get("value", {}), key)

            # {{$iteration_index}}, {{$iteration_total}}, etc.
            if variable == "$iteration_index":
                return self.context.iteration_context.get("index", 0)
            if variable == "$iteration_total":
                return self.context.iteration_context.get("total", 0)

        # {{node_id}} or {{node_id.output.path}} - Node output reference
        return self._resolve_node_output(variable)

    def _resolve_node_output(self, variable: str) -> Any:
        """Resolve reference to a node's output.

        Args:
            variable: Node reference (e.g., "agent_node" or "agent_node.content")

        Returns:
            Node output or nested value

        Raises:
            VariableResolutionError: If node not found
        """
        # Split into node_id and path
        if "." in variable:
            parts = variable.split(".", 1)
            node_id = parts[0]
            path = parts[1]
        else:
            node_id = variable
            path = None

        # Find node output in executed data
        node_output = self.context.get_node_output(node_id)

        if node_output is None:
            raise VariableResolutionError(
                variable,
                f"Node '{node_id}' has not been executed or has no output",
            )

        # If no path specified, return entire output
        if not path:
            return node_output

        # Extract nested path
        return self._get_nested(node_output, path)

    def _get_nested(self, obj: Any, path: str) -> Any:
        """Get nested value using dot notation.

        Supports both dictionary key access and object attribute access.

        Args:
            obj: Object to extract from
            path: Dot-separated path (e.g., "user.name")

        Returns:
            Value at path or None if not found
        """
        if not path:
            return obj

        keys = path.split(".")
        current = obj

        for key in keys:
            if current is None:
                return None

            # Try dictionary access
            if isinstance(current, dict):
                current = current.get(key)
            # Try attribute access
            elif hasattr(current, key):
                current = getattr(current, key)
            # Try list/array index
            elif isinstance(current, (list, tuple)):
                try:
                    index = int(key)
                    current = current[index]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return current

    def _format_chat_history(self, history: List[Dict]) -> str:
        """Convert chat history to readable text format.

        Args:
            history: List of chat message dicts

        Returns:
            Formatted string with one message per line
        """
        if not history:
            return ""

        lines = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def resolve_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve variables in all string values of a dictionary.

        Args:
            data: Dictionary with potential template strings

        Returns:
            Dictionary with resolved values
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Resolve if it contains variables
                if "{{" in value and "}}" in value:
                    result[key] = self.resolve(value)
                else:
                    result[key] = value
            elif isinstance(value, dict):
                result[key] = self.resolve_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.resolve(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
