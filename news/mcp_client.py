"""Thin async wrappers over MCP servers (stdio transport).

Two clients here:
  - XClient: talks to the local x-mcp-server (~/Projects/x-mcp-server).
  - NotionClient: same shape as jobhunting/, but writes to a "reading list"
    database (Title / Introduction / Url properties).
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# How long to back off when the x-mcp server reports an X rate limit (HTTP 429)
# before retrying the same scrape. The server's message is
# "Rate limited by X: HTTP 429 on <path>. Back off and retry later."
RATE_LIMIT_BACKOFF_SECONDS = 300  # 5 minutes


def _looks_rate_limited(text: Any) -> bool:
    """Detect an X rate-limit signal in an error message or tool payload."""
    s = str(text).lower()
    return "rate limit" in s or "429" in s


def _x_server_params(command: str, args: list[str]) -> StdioServerParameters:
    """x-mcp-server reads TWITTER_USERNAME / TWITTER_PASSWORD from env.

    The server resolves its cached login (playwright/.auth/twitter.json)
    relative to its own cwd unless AUTH_DIR is set. We launch it from a
    different working directory, so point AUTH_DIR at the server's own
    playwright/.auth — otherwise it starts an unauthenticated context and
    scrapes 0 posts.
    """
    env = {**os.environ}
    if "AUTH_DIR" not in env and args:
        # args[0] is <x-mcp-server>/dist/mcp.js; project root is two levels up.
        server_root = os.path.dirname(os.path.dirname(os.path.abspath(args[0])))
        auth_dir = os.path.join(server_root, "playwright", ".auth")
        if os.path.isdir(auth_dir):
            env["AUTH_DIR"] = auth_dir
    return StdioServerParameters(
        command=command,
        args=list(args),
        env=env,
    )


@asynccontextmanager
async def x_session(command: str = "node", args: list[str] | None = None):
    if args is None:
        # Default to the locally built bundle.
        args = [os.path.expanduser("~/Projects/x-mcp-server/dist/mcp.js")]
    async with stdio_client(_x_server_params(command, args)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield XClient(session)


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


class XClient:
    def __init__(self, session: ClientSession):
        self.session = session

    async def scrape_timeline(
        self,
        timeline_type: str = "following",
        max_posts: int = 20,
    ) -> dict[str, Any]:
        """Pass-through to the x-mcp `scrape_timeline` tool.

        Returns a dict shaped like:
          {timeline, count, avgEngagement, posts: [
              {postId, author, content, url, timestamp, likes, retweets, engagement},
              ...
          ]}
        """
        args = {"type": timeline_type, "maxPosts": max_posts}
        attempt = 0
        while True:
            attempt += 1
            try:
                result = await self.session.call_tool("scrape_timeline", args)
            except Exception as e:
                # A server-side rate limit is thrown as an McpError, which the
                # Python client re-raises here. Back off and retry; re-raise
                # anything else.
                if _looks_rate_limited(e):
                    print(
                        f"[x-mcp] rate limited (attempt {attempt}); backing off "
                        f"{RATE_LIMIT_BACKOFF_SECONDS}s then retrying: {e}"
                    )
                    await asyncio.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                    continue
                raise

            if getattr(result, "isError", False):
                payload = _unwrap(result)
                if _looks_rate_limited(payload):
                    print(
                        f"[x-mcp] rate limited (attempt {attempt}); backing off "
                        f"{RATE_LIMIT_BACKOFF_SECONDS}s then retrying: {payload!r}"
                    )
                    await asyncio.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                    continue
                raise RuntimeError(f"x-mcp scrape_timeline error: {payload!r}")

            data = _unwrap(result)
            return data if isinstance(data, dict) else {}


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

    async def create_reading_list_page(
        self,
        database_id: str,
        title: str,
        introduction: str,
        url: str,
    ) -> dict | None:
        """Create a page in the AI reading-list database.

        Assumes the database has: 'Title' (title), 'Summary' (rich_text),
        'URL' (url).
        """
        return await self._call("API-post-page", {
            "parent": {"database_id": database_id},
            "properties": {
                "Title": {"title": [{"text": {"content": title}}]},
                "Summary": {
                    "rich_text": [{"text": {"content": introduction or ""}}]
                },
                "URL": {"url": url or None},
            },
        })

    async def retrieve_database(self, database_id: str) -> dict | None:
        return await self._call("API-retrieve-a-database", {"database_id": database_id})
