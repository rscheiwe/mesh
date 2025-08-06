"""Tool schematization utilities for converting Python functions to LLM tool schemas."""

import inspect
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints

from mesh.nodes.agent import FunctionTool


@dataclass
class ParameterSchema:
    """Schema for a function parameter."""

    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


def python_type_to_json_type(python_type: Type) -> str:
    """Convert Python type to JSON Schema type.

    Args:
        python_type: Python type annotation

    Returns:
        JSON Schema type string
    """
    # Handle Optional types
    if hasattr(python_type, "__origin__"):
        if python_type.__origin__ is Union:
            # Check if it's Optional (Union with None)
            args = python_type.__args__
            if type(None) in args:
                # It's Optional, get the actual type
                non_none_types = [t for t in args if t is not type(None)]
                if len(non_none_types) == 1:
                    return python_type_to_json_type(non_none_types[0])
        elif python_type.__origin__ is list:
            return "array"
        elif python_type.__origin__ is dict:
            return "object"

    # Simple type mappings
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    return type_map.get(python_type, "string")  # Default to string


def extract_function_metadata(func: Callable) -> Dict[str, Any]:
    """Extract metadata from a Python function.

    Args:
        func: Function to analyze

    Returns:
        Dict containing function metadata
    """
    # Get function signature
    sig = inspect.signature(func)

    # Get docstring
    docstring = inspect.getdoc(func) or ""

    # Parse docstring for parameter descriptions
    param_descriptions = _parse_docstring_params(docstring)

    # Get type hints
    type_hints = get_type_hints(func)

    # Extract parameters
    parameters = []
    required_params = []

    for param_name, param in sig.parameters.items():
        # Skip self parameter for methods
        if param_name == "self":
            continue

        # Get type
        param_type = type_hints.get(param_name, str)
        json_type = python_type_to_json_type(param_type)

        # Check if required (no default value)
        has_default = param.default != inspect.Parameter.empty
        if not has_default and param_name not in ["args", "kwargs"]:
            required_params.append(param_name)

        # Create parameter schema
        param_schema = ParameterSchema(
            name=param_name,
            type=json_type,
            description=param_descriptions.get(param_name, ""),
            required=not has_default,
            default=param.default if has_default else None,
        )

        parameters.append(param_schema)

    # Extract function description
    description = _extract_function_description(docstring)

    return {
        "name": func.__name__,
        "description": description,
        "parameters": parameters,
        "required_params": required_params,
        "return_type": type_hints.get("return", Any),
    }


def function_to_tool_schema(
    func: Callable, name: Optional[str] = None, description: Optional[str] = None
) -> Dict[str, Any]:
    """Convert a Python function to an OpenAI-compatible tool schema.

    Args:
        func: Function to convert
        name: Override function name
        description: Override function description

    Returns:
        Tool schema dict
    """
    metadata = extract_function_metadata(func)

    # Build parameters schema
    properties = {}
    for param in metadata["parameters"]:
        param_schema = {"type": param.type, "description": param.description}

        # Add enum if present
        if param.enum:
            param_schema["enum"] = param.enum

        # Handle array types
        if param.type == "array":
            param_schema["items"] = {"type": "string"}  # Default to string items

        properties[param.name] = param_schema

    # Build tool schema
    tool_schema = {
        "type": "function",
        "function": {
            "name": name or metadata["name"],
            "description": description or metadata["description"],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": metadata["required_params"],
            },
        },
    }

    return tool_schema


def create_tool_from_function(
    func: Callable, name: Optional[str] = None, description: Optional[str] = None
) -> FunctionTool:
    """Create a FunctionTool from a Python function.

    Args:
        func: Function to wrap
        name: Override function name
        description: Override function description

    Returns:
        FunctionTool instance
    """
    metadata = extract_function_metadata(func)

    # Build parameters schema
    properties = {}
    for param in metadata["parameters"]:
        properties[param.name] = {"type": param.type, "description": param.description}

    parameters_schema = {
        "type": "object",
        "properties": properties,
        "required": metadata["required_params"],
    }

    return FunctionTool(
        name=name or metadata["name"],
        description=description or metadata["description"],
        func=func,
        parameters_schema=parameters_schema,
    )


def _parse_docstring_params(docstring: str) -> Dict[str, str]:
    """Parse parameter descriptions from docstring.

    Args:
        docstring: Function docstring

    Returns:
        Dict mapping parameter names to descriptions
    """
    param_descriptions = {}

    # Simple parser for Google/NumPy style docstrings
    lines = docstring.split("\n")
    in_params_section = False
    current_param = None

    for line in lines:
        stripped = line.strip()

        # Check for parameter section
        if stripped.lower() in ["args:", "arguments:", "parameters:", "params:"]:
            in_params_section = True
            continue
        elif stripped.lower() in [
            "returns:",
            "return:",
            "raises:",
            "yields:",
            "examples:",
        ]:
            in_params_section = False
            continue

        if in_params_section and stripped:
            # Check if it's a parameter definition
            if ":" in stripped and not stripped.startswith(" "):
                parts = stripped.split(":", 1)
                param_name = parts[0].strip()
                param_desc = parts[1].strip() if len(parts) > 1 else ""
                param_descriptions[param_name] = param_desc
                current_param = param_name
            elif current_param and stripped.startswith(" "):
                # Continuation of previous parameter description
                param_descriptions[current_param] += " " + stripped

    return param_descriptions


def _extract_function_description(docstring: str) -> str:
    """Extract the main description from a docstring.

    Args:
        docstring: Function docstring

    Returns:
        Main description string
    """
    if not docstring:
        return ""

    lines = docstring.split("\n")
    description_lines = []

    for line in lines:
        stripped = line.strip()

        # Stop at first section header
        if stripped.lower() in [
            "args:",
            "arguments:",
            "parameters:",
            "params:",
            "returns:",
            "return:",
            "raises:",
            "yields:",
            "examples:",
            "note:",
            "notes:",
        ]:
            break

        if stripped:
            description_lines.append(stripped)

    return " ".join(description_lines)


# Example usage and testing
if __name__ == "__main__":
    # Example function
    def calculate_price(
        base_price: float, tax_rate: float = 0.08, discount: Optional[float] = None
    ) -> float:
        """Calculate the final price with tax and optional discount.

        Args:
            base_price: The base price of the item
            tax_rate: Tax rate as a decimal (default 0.08 for 8%)
            discount: Optional discount as a decimal

        Returns:
            Final price after tax and discount
        """
        price = base_price * (1 + tax_rate)
        if discount:
            price *= 1 - discount
        return price

    # Test extraction
    print("Function Metadata:")
    print(json.dumps(extract_function_metadata(calculate_price), indent=2))

    print("\nTool Schema:")
    print(json.dumps(function_to_tool_schema(calculate_price), indent=2))

    print("\nFunctionTool:")
    tool = create_tool_from_function(calculate_price)
    print(f"Name: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Schema: {json.dumps(tool.parameters_schema, indent=2)}")
