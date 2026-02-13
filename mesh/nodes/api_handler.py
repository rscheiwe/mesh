"""API Handler Node for making HTTP API calls.

This is a specialized ToolNode that makes HTTP GET requests to external APIs
with support for variable resolution in URL, headers, and query parameters.
"""

import json
import re
from typing import Any, Dict, Optional

from mesh.nodes.tool import ToolNode
from mesh.nodes.base import NodeResult
from mesh.core.state import ExecutionContext


class APIHandlerNode(ToolNode):
    """Make HTTP API calls to external services.

    This is a specialized ToolNode that:
    1. Builds a full URL from base_url + endpoint + query_params
    2. Resolves {{variables}} in endpoint, headers, and query params
    3. Makes an async HTTP request via httpx
    4. Returns parsed response with metadata

    Output format:
        {
            "data": <parsed JSON or raw text>,
            "status_code": 200,
            "url": "https://api.example.com/v1/users?limit=10",
            "headers": {"content-type": "application/json", ...},
            "content_type": "application/json",
        }

    Access: {{api_0.data}}, {{api_0.status_code}}, {{api_0.url}}

    Example:
        >>> node = APIHandlerNode(
        ...     id="api_0",
        ...     base_url="https://api.example.com",
        ...     endpoint="/v1/users",
        ...     headers={"Authorization": "Bearer {{start.token}}"},
        ...     query_params={"limit": 10},
        ... )
    """

    def __init__(
        self,
        id: str,
        base_url: str,
        endpoint: str = "/",
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
        event_mode: str = "full",
        auto_inject_context: bool = False,
        config: Dict[str, Any] = None,
    ):
        """Initialize APIHandlerNode.

        Args:
            id: Node identifier
            base_url: Base URL of the API (e.g., "https://api.example.com")
            endpoint: Endpoint path (e.g., "/v1/users") — supports {{variables}}
            method: HTTP method (GET only for now)
            headers: HTTP headers dict — values support {{variables}}
            query_params: URL query parameters — values support {{variables}}
            timeout_seconds: Request timeout in seconds (default: 30)
            event_mode: Event emission mode
            auto_inject_context: Whether to auto-inject upstream context
            config: Additional configuration
        """
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.endpoint = endpoint or "/"
        self.method = method.upper()
        self.headers_config = headers or {}
        self.query_params_config = query_params or {}
        self.timeout_seconds = timeout_seconds
        self.auto_inject_context = auto_inject_context

        # Create the tool function that will execute the HTTP request
        tool_fn = self._create_http_executor()

        # Initialize as ToolNode
        super().__init__(
            id=id,
            tool_fn=tool_fn,
            event_mode=event_mode,
            config=config or {},
        )

    def _create_http_executor(self):
        """Create the async function that executes the HTTP request."""

        async def execute_http(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            """Execute the HTTP request with resolved parameters.

            Args:
                input: Input data (may contain interpolated params)
                context: Execution context

            Returns:
                Response data with metadata
            """
            import httpx

            # Resolve {{variables}} in endpoint, headers, and query params
            resolved_endpoint, resolved_headers, resolved_params = self._resolve_params(
                input, context
            )

            # Build full URL
            full_url = self.base_url + resolved_endpoint

            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.request(
                        method=self.method,
                        url=full_url,
                        headers=resolved_headers if resolved_headers else None,
                        params=resolved_params if resolved_params else None,
                    )

                    # Parse response body
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        try:
                            data = response.json()
                        except (json.JSONDecodeError, ValueError):
                            data = response.text
                    else:
                        data = response.text

                    # Convert response headers to a plain dict
                    resp_headers = dict(response.headers)

                    return {
                        "data": data,
                        "status_code": response.status_code,
                        "url": str(response.url),
                        "headers": resp_headers,
                        "content_type": content_type,
                    }

            except httpx.TimeoutException:
                return {
                    "data": {
                        "_error": "timeout",
                        "_message": (
                            f"API request timed out after {self.timeout_seconds} seconds. "
                            f"URL: {full_url}"
                        ),
                    },
                    "status_code": 0,
                    "url": full_url,
                    "headers": {},
                    "content_type": "",
                }
            except httpx.ConnectError as e:
                raise RuntimeError(
                    f"APIHandlerNode '{self.id}' connection failed: {str(e)}\n"
                    f"URL: {full_url}"
                ) from e

        return execute_http

    def _resolve_params(self, input: Any, context: ExecutionContext) -> tuple:
        """Resolve {{variables}} in endpoint, headers, and query params.

        Args:
            input: Input data
            context: Execution context

        Returns:
            Tuple of (resolved_endpoint, resolved_headers, resolved_query_params)
        """
        from mesh.utils.variables import VariableResolver

        resolver = VariableResolver(context)

        def resolve_value(value: Any) -> Any:
            """Resolve {{variables}} in a string value."""
            if not isinstance(value, str) or "{{" not in value:
                return value

            def replacer(match):
                variable = match.group(1).strip()
                try:
                    val = resolver._resolve_single(variable)
                    return str(val) if val is not None else ""
                except Exception:
                    return ""

            return re.sub(r"\{\{(.*?)\}\}", replacer, value)

        # Resolve endpoint path
        resolved_endpoint = resolve_value(self.endpoint)

        # Resolve headers
        resolved_headers = {}
        for key, value in self.headers_config.items():
            resolved_headers[key] = resolve_value(value)

        # Resolve query params
        resolved_params = {}
        for key, value in self.query_params_config.items():
            resolved_params[key] = resolve_value(value)

        return resolved_endpoint, resolved_headers, resolved_params

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute API request.

        Override ToolNode._execute_impl to return unwrapped output
        so {{api_0.data}} works directly (not {{api_0.output.data}}).
        """
        from mesh.core.events import ExecutionEvent, EventType

        # Emit start event
        await self._emit_event_if_enabled(
            context,
            ExecutionEvent(
                type=EventType.NODE_START,
                node_id=self.id,
                metadata={
                    "tool_name": self.function_name,
                    "node_type": "api_handler",
                },
            )
        )

        # Build kwargs from function signature
        kwargs = self._build_kwargs(input, context)

        # Execute the HTTP request
        try:
            if self.is_async:
                result = await self.tool_fn(**kwargs)
            else:
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: self.tool_fn(**kwargs))
        except Exception as e:
            await self._emit_event_if_enabled(
                context,
                ExecutionEvent(
                    type=EventType.NODE_ERROR,
                    node_id=self.id,
                    error=str(e),
                    metadata={
                        "tool_name": self.function_name,
                        "node_type": "api_handler",
                    },
                )
            )
            raise RuntimeError(
                f"APIHandlerNode '{self.id}' failed: {str(e)}"
            ) from e

        # Return result directly (not wrapped)
        # This allows {{api_0.data}} to work correctly
        return NodeResult(
            output=result,  # Direct: {"data": ..., "status_code": ..., ...}
            metadata={
                "tool_name": self.function_name,
                "node_type": "api_handler",
                "base_url": self.base_url,
                "method": self.method,
            },
        )

    def __repr__(self) -> str:
        return f"APIHandlerNode(id='{self.id}', base_url='{self.base_url}')"
