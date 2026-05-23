"""Thin async wrapper over the LinkedIn MCP server (stdio transport)."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

LINKEDIN_MCP_DIR = "/Users/allenzhang/Desktop/linkedin-mcp-server"


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="uv",
        args=["run", "--project", LINKEDIN_MCP_DIR, "-m", "linkedin_mcp_server"],
    )


@asynccontextmanager
async def linkedin_session():
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield LinkedInClient(session)


def _unwrap(result: Any) -> Any:
    """MCP tool results come back as a list of content blocks; unwrap to a dict/str."""
    if not getattr(result, "content", None):
        return None
    block = result.content[0]
    text = getattr(block, "text", None)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class LinkedInClient:
    def __init__(self, session: ClientSession):
        self.session = session

    async def search_jobs(self, keywords: str, location: str, limit: int) -> list[dict]:
        result = await self.session.call_tool(
            "search_jobs",
            {"search_term": keywords, "location": location, "limit": limit},
        )
        data = _unwrap(result)
        if isinstance(data, dict) and "jobs" in data:
            return data["jobs"]
        if isinstance(data, list):
            return data
        return []

    async def get_job_details(self, job_id: str) -> dict | None:
        result = await self.session.call_tool("get_job_details", {"job_id": job_id})
        data = _unwrap(result)
        return data if isinstance(data, dict) else None

    async def get_company_profile(self, company_url_or_name: str) -> dict | None:
        result = await self.session.call_tool(
            "get_company_profile", {"company_name": company_url_or_name}
        )
        data = _unwrap(result)
        return data if isinstance(data, dict) else None
