"""Tools for the deep research workflow.

This module provides tool implementations for the research agents,
including web search capabilities.
"""

import os
from typing import Dict, List, Any, Optional

# Check if Vel is available for ToolSpec
try:
    from vel.tools import ToolSpec
    HAS_VEL = True
except ImportError:
    HAS_VEL = False
    ToolSpec = None


async def web_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search the web for information.

    This is a placeholder implementation. In production, this would call
    an actual search API like Perplexity Sonar, Tavily, or similar.

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        Dict with search results containing:
        - query: The original query
        - results: List of result dicts with title, url, snippet
    """
    # Check for Perplexity API key
    api_key = os.environ.get("PERPLEXITY_API_KEY")

    if api_key:
        # Use real Perplexity API
        return await _perplexity_search(query, limit, api_key)
    else:
        # Return mock results for testing
        return _mock_search_results(query, limit)


async def _perplexity_search(query: str, limit: int, api_key: str) -> Dict[str, Any]:
    """Execute actual Perplexity search."""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=api_key,
            base_url='https://api.perplexity.ai'
        )

        response = await client.chat.completions.create(
            model='sonar',
            messages=[{'role': 'user', 'content': query}],
            temperature=0.2,
            max_tokens=2000
        )

        content = response.choices[0].message.content

        # Extract citations if available
        citations = []
        if hasattr(response, 'citations') and response.citations:
            citations = response.citations
        elif hasattr(response, 'model_extra') and isinstance(response.model_extra, dict):
            citations = response.model_extra.get('citations', [])

        results = []
        if citations:
            for i, citation in enumerate(citations[:limit]):
                citation_url = citation if isinstance(citation, str) else citation.get('url', '')
                results.append({
                    'title': f'Source {i+1}',
                    'url': citation_url,
                    'snippet': content[:500] if i == 0 else '',
                })
        else:
            results.append({
                'title': 'Web Search Result',
                'url': 'https://perplexity.ai',
                'snippet': content,
            })

        return {
            'query': query,
            'results': results[:limit],
            'answer': content,
        }

    except Exception as e:
        return {
            'query': query,
            'results': [{
                'title': 'Search Error',
                'url': '',
                'snippet': f'Error performing search: {str(e)}',
            }],
            'error': str(e),
        }


def _mock_search_results(query: str, limit: int) -> Dict[str, Any]:
    """Return mock search results for testing without API key."""
    mock_results = [
        {
            'title': f'Mock Result 1 for: {query[:30]}...',
            'url': 'https://example.com/result1',
            'snippet': f'This is a mock search result for the query "{query}". In production, this would contain actual web search results from Perplexity Sonar or similar API.',
        },
        {
            'title': f'Mock Result 2 for: {query[:30]}...',
            'url': 'https://example.com/result2',
            'snippet': 'Additional mock result with relevant information. Configure PERPLEXITY_API_KEY environment variable for real search results.',
        },
        {
            'title': f'Mock Result 3 for: {query[:30]}...',
            'url': 'https://example.com/result3',
            'snippet': 'Third mock result demonstrating the search functionality structure.',
        },
    ]

    return {
        'query': query,
        'results': mock_results[:limit],
        'mock': True,
    }


def create_web_search_tool() -> Optional["ToolSpec"]:
    """Create a ToolSpec for the web search tool.

    Returns:
        ToolSpec instance if Vel is available, None otherwise
    """
    if not HAS_VEL:
        return None

    return ToolSpec(
        name='web_search',
        description='Search the web for information on any topic. Returns relevant results with titles, URLs, and snippets.',
        input_schema={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The search query to execute',
                },
                'limit': {
                    'type': 'integer',
                    'description': 'Maximum number of results (default: 5)',
                    'default': 5,
                },
            },
            'required': ['query'],
        },
        handler=web_search,
    )


# Helper functions for step management

def get_current_step(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get the current step from the plan based on index.

    Args:
        state: Execution state containing the plan

    Returns:
        Current step dict or None if all complete
    """
    plan = state.get('plan', {})
    steps = plan.get('steps', [])
    current_index = state.get('current_step_index', 0)

    # Simply return the step at current index (index-based tracking)
    if current_index < len(steps):
        return steps[current_index]

    return None


def has_incomplete_steps(state: Dict[str, Any]) -> bool:
    """Check if there are any incomplete steps remaining.

    Args:
        state: Execution state containing the plan

    Returns:
        True if incomplete steps exist
    """
    plan = state.get('plan', {})
    steps = plan.get('steps', [])
    current_index = state.get('current_step_index', 0)

    # Use index-based check - more reliable than step.completed flag
    return current_index < len(steps)


def mark_step_complete(state: Dict[str, Any], step_id: str, result: str) -> Dict[str, Any]:
    """Mark a step as complete and store its result.

    Args:
        state: Execution state
        step_id: ID of the step to mark complete
        result: Result/findings from the step

    Returns:
        Updated state dict
    """
    plan = state.get('plan', {})
    steps = plan.get('steps', [])

    for step in steps:
        if step.get('id') == step_id:
            step['completed'] = True
            step['result'] = result
            break

    # Increment step index
    current_index = state.get('current_step_index', 0)
    state['current_step_index'] = current_index + 1
    state['plan'] = plan

    return state
