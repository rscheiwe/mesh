"""Utility nodes for common operations."""

import asyncio
import json
from typing import Any, Callable, Dict, Optional

import httpx
from pydantic import BaseModel, Field

from mesh.core.node import NodeConfig
from mesh.nodes.base import BaseNode
from mesh.state.state import GraphState


class HumanInputNode(BaseNode):
    """Node that waits for human input."""

    def __init__(
        self,
        prompt: str = "Please provide input:",
        input_handler: Optional[Callable[[], str]] = None,
        config: Optional[NodeConfig] = None,
    ):
        super().__init__(config)
        self.prompt = prompt
        self.input_handler = input_handler or input

    async def _execute_impl(
        self, input_data: Dict[str, Any], state: Optional[GraphState] = None
    ) -> Dict[str, Any]:
        """Get input from human.

        Args:
            input_data: Context data
            state: Optional shared state

        Returns:
            Dict with human input
        """
        # Display context if available
        context = input_data.get("context", "")
        if context:
            print(f"Context: {context}")

        print(self.prompt)

        # Get input
        if asyncio.iscoroutinefunction(self.input_handler):
            human_input = await self.input_handler()
        else:
            # Run blocking input in executor
            loop = asyncio.get_event_loop()
            human_input = await loop.run_in_executor(None, self.input_handler)

        return {
            "human_input": human_input,
            "context": context,
            "prompt": self.prompt,
        }


class CustomFunctionNode(BaseNode):
    """Node that executes a custom function."""

    def __init__(
        self,
        function: Callable[[Dict[str, Any], Optional[GraphState]], Any],
        config: Optional[NodeConfig] = None,
    ):
        super().__init__(config)
        self.function = function

    async def _execute_impl(
        self, input_data: Dict[str, Any], state: Optional[GraphState] = None
    ) -> Dict[str, Any]:
        """Execute custom function.

        Args:
            input_data: Input data for the function
            state: Optional shared state

        Returns:
            Dict with function result
        """
        try:
            if asyncio.iscoroutinefunction(self.function):
                result = await self.function(input_data, state)
            else:
                result = self.function(input_data, state)

            # Ensure result is JSON serializable
            if isinstance(result, dict):
                return result
            else:
                return {
                    "result": result,
                    "input": input_data,
                }

        except Exception as e:
            return {
                "error": str(e),
                "input": input_data,
            }


class HTTPRequest(BaseModel):
    """HTTP request configuration."""

    url: str
    method: str = "GET"
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    data: Optional[Any] = None
    json_data: Optional[Dict[str, Any]] = None
    timeout: float = 30.0


class HTTPNode(BaseNode):
    """Node that makes HTTP requests."""

    def __init__(
        self,
        default_request: Optional[HTTPRequest] = None,
        config: Optional[NodeConfig] = None,
    ):
        super().__init__(config)
        self.default_request = default_request

    async def _execute_impl(
        self, input_data: Dict[str, Any], state: Optional[GraphState] = None
    ) -> Dict[str, Any]:
        """Execute HTTP request.

        Args:
            input_data: Should contain request configuration
            state: Optional shared state

        Returns:
            Dict with response data
        """
        # Build request from input or defaults
        request = self._build_request(input_data)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method=request.method,
                    url=request.url,
                    headers=request.headers,
                    params=request.params,
                    data=request.data,
                    json=request.json_data,
                    timeout=request.timeout,
                )

                # Try to parse JSON response
                try:
                    response_data = response.json()
                except:
                    response_data = response.text

                return {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "data": response_data,
                    "url": str(response.url),
                    "method": request.method,
                }

            except httpx.TimeoutException:
                return {
                    "error": "Request timeout",
                    "url": request.url,
                    "method": request.method,
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "url": request.url,
                    "method": request.method,
                }

    def _build_request(self, input_data: Dict[str, Any]) -> HTTPRequest:
        """Build HTTP request from input data.

        Args:
            input_data: Input data containing request info

        Returns:
            HTTPRequest object
        """
        if "request" in input_data and isinstance(input_data["request"], dict):
            return HTTPRequest(**input_data["request"])
        elif self.default_request:
            # Override defaults with input data
            request_dict = self.default_request.dict()
            request_dict.update(input_data)
            return HTTPRequest(**request_dict)
        else:
            return HTTPRequest(**input_data)
