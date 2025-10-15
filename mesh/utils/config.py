"""Configuration utilities for loading environment variables."""

import os
from pathlib import Path
from typing import Optional


def load_env(env_file: Optional[str] = None) -> None:
    """Load environment variables from .env file.

    This function loads variables from a .env file into the environment.
    It will search for .env in the current directory and parent directories
    if no path is specified.

    Args:
        env_file: Optional path to .env file. If not specified, searches
                 for .env in current and parent directories.

    Example:
        >>> from mesh.utils.config import load_env
        >>> load_env()  # Loads from .env
        >>> import os
        >>> api_key = os.getenv("OPENAI_API_KEY")
    """
    try:
        from dotenv import load_dotenv

        if env_file:
            # Load from specified file
            load_dotenv(env_file)
        else:
            # Search for .env file
            load_dotenv(verbose=True)

    except ImportError:
        print(
            "Warning: python-dotenv not installed. "
            "Install with: pip install python-dotenv"
        )


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment.

    Returns:
        API key or None if not found
    """
    return os.getenv("OPENAI_API_KEY")


def get_config(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get configuration value from environment.

    Args:
        key: Configuration key
        default: Default value if not found

    Returns:
        Configuration value or default
    """
    return os.getenv(key, default)


def ensure_api_key() -> str:
    """Ensure OpenAI API key is available.

    Returns:
        API key

    Raises:
        ValueError: If API key not found
    """
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Please set it in .env file or environment variables."
        )
    return api_key
