"""Input parsing utilities for structured data extraction from natural language.

This module provides automatic detection and parsing of multiple input variables
from natural language using LLM-based structured output extraction.
"""

import re
from typing import Dict, List, Optional, Any, Type
from pydantic import BaseModel, create_model


def detect_input_variables(text: str) -> List[str]:
    """Detect all {{$input.X}} variable references in a template string.

    Args:
        text: Template string that may contain {{$input.field}} references

    Returns:
        List of field names (e.g., ["topic1", "topic2"])
    """
    if not text:
        return []

    # Pattern to match {{$input.fieldname}} or {{$question.fieldname}}
    pattern = re.compile(r'\{\{\$(?:input|question)\.(\w+)\}\}')
    matches = pattern.findall(text)

    # Return unique field names
    return list(set(matches))


def create_input_schema(field_names: List[str]) -> Type[BaseModel]:
    """Dynamically create a Pydantic model for input fields.

    Args:
        field_names: List of field names to include in schema

    Returns:
        Pydantic model class with string fields

    Example:
        >>> schema = create_input_schema(["topic1", "topic2"])
        >>> # Creates: class InputSchema(BaseModel): topic1: str; topic2: str
    """
    if not field_names:
        raise ValueError("Must provide at least one field name")

    # Create fields dict with all fields as required strings
    fields = {
        field_name: (str, ...)
        for field_name in field_names
    }

    # Dynamically create Pydantic model
    InputSchema = create_model(
        'InputSchema',
        **fields
    )

    return InputSchema


async def parse_natural_language_input(
    raw_input: str,
    field_names: List[str],
    model_config: Dict[str, Any]
) -> Dict[str, str]:
    """Parse natural language input into structured data using LLM.

    Args:
        raw_input: Natural language input (e.g., "1. dogs, 2. cats")
        field_names: List of field names to extract (e.g., ["topic1", "topic2"])
        model_config: Model configuration for the parsing LLM

    Returns:
        Dictionary with extracted fields

    Example:
        >>> result = await parse_natural_language_input(
        ...     "1. dogs, 2. cats",
        ...     ["topic1", "topic2"],
        ...     {"provider": "openai", "model": "gpt-4o-mini"}
        ... )
        >>> # Returns: {"topic1": "dogs", "topic2": "cats"}
    """
    from vel import Agent

    # Create dynamic schema
    InputSchema = create_input_schema(field_names)

    # Create lightweight parsing agent with structured output
    parser_agent = Agent(
        id="input_parser",
        model=model_config,
        output_type=InputSchema,
        instruction=f"Extract the following fields from the user's message: {', '.join(field_names)}. Return only the extracted values without explanation."
    )

    # Run parser
    result = await parser_agent.run({"message": raw_input})

    # Convert Pydantic model to dict
    if hasattr(result, 'model_dump'):
        return result.model_dump()
    elif hasattr(result, 'dict'):
        return result.dict()
    else:
        raise ValueError(f"Unexpected result type from parser: {type(result)}")


def should_parse_input(system_prompt: Optional[str], input_data: Any) -> bool:
    """Determine if input parsing is needed.

    Parsing is needed when:
    1. System prompt has multiple {{$input.X}} variables
    2. Input is a string (not already structured)

    Args:
        system_prompt: Template string to check
        input_data: Input data to check

    Returns:
        True if parsing should be triggered, False otherwise
    """
    if not system_prompt:
        return False

    # Check if input is already structured
    if isinstance(input_data, dict):
        return False

    # Detect multiple input variables
    field_names = detect_input_variables(system_prompt)

    # Need at least 2 fields to warrant parsing
    return len(field_names) >= 2
