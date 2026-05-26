"""Thin async wrappers over MCP servers (stdio transport)."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

def _server_params() -> StdioServerParameters:
    # Installed via `uv tool install linkedin-scraper-mcp`,
    # which puts the `linkedin-scraper-mcp` console script on PATH.
    return StdioServerParameters(command="linkedin-scraper-mcp", args=[])


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

    async def search_jobs(self, **kwargs: Any) -> dict[str, Any]:
        """Pass-through to MCP search_jobs. Returns the full result dict:
        {url, sections, references, job_ids}. Caller extracts job_ids.

        kwargs are forwarded to the MCP tool (keywords, max_pages, date_posted,
        job_type, experience_level, work_type, easy_apply, sort_by, location).
        Only `keywords` is required by the MCP tool.
        """
        args = {k: v for k, v in kwargs.items() if v is not None}
        result = await self.session.call_tool("search_jobs", args)
        data = _unwrap(result)
        return data if isinstance(data, dict) else {}

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


# --- Notion MCP server (npm-installed `notion-mcp-server`) ---

def _notion_server_params() -> StdioServerParameters:
    """@notionhq/notion-mcp-server reads NOTION_TOKEN from env."""
    token = os.environ.get("NOTION_ACCESS_TOKEN", "")
    return StdioServerParameters(
        command="notion-mcp-server",
        args=[],
        env={**os.environ, "NOTION_TOKEN": token},
    )


@asynccontextmanager
async def notion_session():
    async with stdio_client(_notion_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield NotionClient(session)


class NotionClient:
    def __init__(self, session: ClientSession):
        self.session = session

    async def _call(self, tool: str, args: dict[str, Any]) -> dict | None:
        """Call a Notion MCP tool and raise on either MCP errors OR Notion API
        errors that the MCP passes through as structured data."""
        result = await self.session.call_tool(tool, args)
        data = _unwrap(result)
        if getattr(result, "isError", False):
            raise RuntimeError(f"Notion {tool} MCP error: {data!r}")
        if isinstance(data, dict) and data.get("object") == "error":
            raise RuntimeError(
                f"Notion {tool} API error: status={data.get('status')} "
                f"code={data.get('code')} message={data.get('message')!r}"
            )
        return data if isinstance(data, dict) else None

    async def create_database_page(
        self,
        database_id: str,
        job_title: str,
        date_iso: str,
        url: str,
        company: str = "",
    ) -> dict | None:
        """Create a page in the target database.

        Assumes the database has: 'Job Title' (title), 'Date' (date),
        'Url' (url), 'Company' (rich_text).
        """
        return await self._call("API-post-page", {
            "parent": {"database_id": database_id},
            "properties": {
                "Job Title": {"title": [{"text": {"content": job_title}}]},
                "Date": {"date": {"start": date_iso}},
                "Url": {"url": url},
                "Company": {"rich_text": [{"text": {"content": company or ""}}]},
            },
        })

    async def retrieve_database(self, database_id: str) -> dict | None:
        return await self._call("API-retrieve-a-database", {"database_id": database_id})
