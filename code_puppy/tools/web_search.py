"""
Web search and page browsing tools for xAI Grok agents.

Uses xAI's /v1/responses endpoint (Agent Tools API) with built-in tools
(web_search, browse_page) to fetch information server-side, returning clean
summarized text. The main agent loop stays on /v1/chat/completions; only
these tool handlers call /v1/responses.
"""

import json
import os

import httpx


def _get_xai_api_key() -> str | None:
    """Resolve XAI_API_KEY from config then environment."""
    try:
        from code_puppy.model_factory import get_api_key

        return get_api_key("XAI_API_KEY")
    except Exception:
        return os.environ.get("XAI_API_KEY")


async def _call_xai_responses(query: str, tool_types: list[str]) -> str:
    """
    Call xAI's /v1/responses endpoint with the specified built-in tool types.
    Returns the text content of the response.
    """
    api_key = _get_xai_api_key()
    if not api_key:
        return "Error: XAI_API_KEY is not set. Cannot perform web search."

    tools = [{"type": t} for t in tool_types]

    payload = {
        "model": "grok-4-1-fast-reasoning",
        "input": [{"role": "user", "content": query}],
        "tools": tools,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.x.ai/v1/responses",
                headers=headers,
                content=json.dumps(payload),
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract text from the response output array
        output = data.get("output", [])
        text_parts = []
        for item in output:
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        text_parts.append(part.get("text", ""))
        if text_parts:
            return "\n".join(text_parts)

        # Fallback: try legacy choices format
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", str(data))

        return f"Search completed but no text content found in response: {json.dumps(data)[:500]}"

    except httpx.HTTPStatusError as e:
        return f"xAI search error {e.response.status_code}: {e.response.text[:300]}"
    except Exception as e:
        return f"Web search failed: {e}"


def register_web_search(agent):
    """Register the web_search tool with a pydantic-ai agent."""

    @agent.tool_plain
    async def web_search(query: str, include_x_posts: bool = False) -> str:
        """Search the web for current information using xAI's Agent Tools API.

        Use this whenever you need real-time information, recent news, current
        events, or data that may have changed since your training cutoff.
        Do NOT use shell commands (curl/wget) for this — use web_search instead.

        Args:
            query: The search query string.
            include_x_posts: If True, also search X (Twitter) posts in addition
                to the web. Default False.
        """
        tool_types = ["web_search"]
        if include_x_posts:
            tool_types.append("x_search")
        return await _call_xai_responses(query, tool_types)


def register_x_search(agent):
    """Register the x_search tool with a pydantic-ai agent."""

    @agent.tool_plain
    async def x_search(query: str) -> str:
        """Search X (Twitter) posts, users, and threads using xAI's Agent Tools API.

        Use this to find real-time X posts, trending topics, user opinions, or
        anything happening on X/Twitter. For general web information use web_search
        instead.

        Args:
            query: The search query string (keywords, hashtags, usernames, etc.).
        """
        return await _call_xai_responses(query, ["x_search"])


def register_browse_page(agent):
    """Register the browse_page tool with a pydantic-ai agent."""

    @agent.tool_plain
    async def browse_page(url: str, instructions: str = "Summarize the main content of this page.") -> str:
        """Fetch and summarize content from a specific URL using xAI's Agent Tools API.

        Use this to read the full content of a specific webpage — documentation,
        articles, GitHub files, etc. Prefer web_search for discovery; use
        browse_page when you already have the URL and need the page content.
        Do NOT use shell commands (curl/wget) for this.

        Args:
            url: The full URL of the page to fetch (must include https://).
            instructions: What to extract or summarize from the page.
                Defaults to a general summary.
        """
        # xAI's web_search type handles both web search and URL browsing.
        # ("browse_page" is not a valid tool type — web_search covers both.)
        query = f"{instructions}\n\nURL: {url}"
        result = await _call_xai_responses(query, ["web_search"])
        # If the primary call failed, fall back to a plain search query for the URL.
        if result.startswith("xAI search error") or result.startswith("Web search failed"):
            fallback_query = f"Summarize the content at this URL: {url}. {instructions}"
            result = await _call_xai_responses(fallback_query, ["web_search"])
        return result
