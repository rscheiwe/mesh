"""Message handling utilities for converting between different LLM message formats."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from mesh.nodes.llm import Message


class MessageFormat(Enum):
    """Supported message formats."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MESH = "mesh"


@dataclass
class MessageHandler:
    """Handles conversion between different message formats."""

    def __init__(self, default_format: MessageFormat = MessageFormat.MESH):
        self.default_format = default_format

    def convert(
        self,
        messages: Union[List[Dict[str, Any]], List[Message]],
        to_format: MessageFormat,
        from_format: Optional[MessageFormat] = None,
    ) -> List[Union[Dict[str, Any], Message]]:
        """Convert messages between formats.

        Args:
            messages: Messages to convert
            to_format: Target format
            from_format: Source format (auto-detected if not provided)

        Returns:
            Converted messages
        """
        # Auto-detect source format
        if from_format is None:
            from_format = self._detect_format(messages)

        # Convert to mesh format first (common format)
        mesh_messages = self._to_mesh_format(messages, from_format)

        # Convert from mesh to target format
        if to_format == MessageFormat.MESH:
            return mesh_messages
        elif to_format == MessageFormat.OPENAI:
            return convert_to_openai_format(mesh_messages)
        elif to_format == MessageFormat.ANTHROPIC:
            return convert_to_anthropic_format(mesh_messages)
        elif to_format == MessageFormat.GOOGLE:
            return convert_to_google_format(mesh_messages)
        else:
            raise ValueError(f"Unsupported format: {to_format}")

    def _detect_format(self, messages: List[Any]) -> MessageFormat:
        """Auto-detect message format."""
        if not messages:
            return self.default_format

        first_msg = messages[0]

        # Check if it's a Message object
        if isinstance(first_msg, Message):
            return MessageFormat.MESH

        # Check dict formats
        if isinstance(first_msg, dict):
            if "role" in first_msg and "content" in first_msg:
                # Could be OpenAI or generic format
                if "name" in first_msg or "function_call" in first_msg:
                    return MessageFormat.OPENAI
                return MessageFormat.OPENAI  # Default dict format
            elif "role" in first_msg and "text" in first_msg:
                return MessageFormat.ANTHROPIC
            elif "role" in first_msg and "parts" in first_msg:
                return MessageFormat.GOOGLE

        raise ValueError("Unable to detect message format")

    def _to_mesh_format(
        self, messages: List[Any], from_format: MessageFormat
    ) -> List[Message]:
        """Convert any format to mesh Message format."""
        if from_format == MessageFormat.MESH:
            return messages

        mesh_messages = []
        for msg in messages:
            if from_format == MessageFormat.OPENAI:
                mesh_messages.append(self._openai_to_mesh(msg))
            elif from_format == MessageFormat.ANTHROPIC:
                mesh_messages.append(self._anthropic_to_mesh(msg))
            elif from_format == MessageFormat.GOOGLE:
                mesh_messages.append(self._google_to_mesh(msg))

        return mesh_messages

    def _openai_to_mesh(self, msg: Dict[str, Any]) -> Message:
        """Convert OpenAI format to mesh Message."""
        metadata = {}

        # Handle function/tool messages
        if msg.get("name"):
            metadata["name"] = msg["name"]
        if msg.get("function_call"):
            metadata["function_call"] = msg["function_call"]
        if msg.get("tool_calls"):
            metadata["tool_calls"] = msg["tool_calls"]

        return Message(
            role=msg["role"], content=msg.get("content", ""), metadata=metadata
        )

    def _anthropic_to_mesh(self, msg: Dict[str, Any]) -> Message:
        """Convert Anthropic format to mesh Message."""
        # Anthropic uses 'text' instead of 'content'
        return Message(
            role=msg["role"],
            content=msg.get("text", msg.get("content", "")),
            metadata=msg.get("metadata", {}),
        )

    def _google_to_mesh(self, msg: Dict[str, Any]) -> Message:
        """Convert Google format to mesh Message."""
        # Google uses 'parts' which can contain multiple content types
        content = ""
        if "parts" in msg:
            # Extract text parts
            text_parts = [
                part.get("text", "")
                for part in msg["parts"]
                if isinstance(part, dict) and "text" in part
            ]
            content = " ".join(text_parts)

        return Message(
            role=msg["role"], content=content, metadata=msg.get("metadata", {})
        )


def convert_to_openai_format(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert mesh Messages to OpenAI format.

    Args:
        messages: List of Message objects

    Returns:
        List of dicts in OpenAI format
    """
    openai_messages = []

    for msg in messages:
        # Basic message structure
        openai_msg = {"role": msg.role, "content": msg.content}

        # Add metadata fields if present
        if msg.metadata:
            if "name" in msg.metadata:
                openai_msg["name"] = msg.metadata["name"]
            if "function_call" in msg.metadata:
                openai_msg["function_call"] = msg.metadata["function_call"]
            if "tool_calls" in msg.metadata:
                openai_msg["tool_calls"] = msg.metadata["tool_calls"]

        openai_messages.append(openai_msg)

    return openai_messages


def convert_to_anthropic_format(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert mesh Messages to Anthropic format.

    Args:
        messages: List of Message objects

    Returns:
        List of dicts in Anthropic format
    """
    anthropic_messages = []

    # Anthropic expects system message separately, not in messages array
    system_message = None

    for msg in messages:
        if msg.role == "system":
            # Combine multiple system messages if needed
            if system_message:
                system_message += "\n\n" + msg.content
            else:
                system_message = msg.content
        else:
            # Regular message
            anthropic_msg = {"role": msg.role, "content": msg.content}
            anthropic_messages.append(anthropic_msg)

    # Return format compatible with Anthropic API
    result = {"messages": anthropic_messages}
    if system_message:
        result["system"] = system_message

    return result


def convert_to_google_format(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert mesh Messages to Google (Gemini) format.

    Args:
        messages: List of Message objects

    Returns:
        List of dicts in Google format
    """
    google_messages = []

    for msg in messages:
        # Google uses 'parts' for content
        google_msg = {
            "role": msg.role if msg.role != "assistant" else "model",
            "parts": [{"text": msg.content}],
        }

        # Add metadata if present
        if msg.metadata:
            google_msg["metadata"] = msg.metadata

        google_messages.append(google_msg)

    return google_messages


def convert_to_mesh_format(
    messages: List[Dict[str, Any]], source_format: MessageFormat = MessageFormat.OPENAI
) -> List[Message]:
    """Convert messages from any format to mesh Message format.

    Args:
        messages: Messages in source format
        source_format: Format of input messages

    Returns:
        List of Message objects
    """
    handler = MessageHandler()
    return handler.convert(messages, MessageFormat.MESH, source_format)
