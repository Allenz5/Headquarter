"""LangGraph workflow: fetch (x-mcp following timeline) → judge (AI filter +
intro generation) → write → notion."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict

import yaml
from dotenv import load_dotenv
from langchain_claude_code import ChatClaudeCode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from mcp_client import NotionClient, XClient, notion_session, x_session

HERE = Path(__file__).parent
DEFAULT_CONFIG = HERE / "config.yaml"
DEFAULT_ENV = HERE / ".env"
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "results.md"

# Bound concurrent LLM calls so we don't spawn too many Claude Code CLI
# subprocesses at once.
LLM_PARALLEL_LIMIT = 5

# Fallback time window (hours) when config doesn't set scrape.timeframe.
DEFAULT_TIMEFRAME_HOURS = 24


def _parse_post_timestamp(value: Any) -> datetime | None:
    """Parse an X post timestamp (ISO-8601, possibly 'Z') into aware UTC."""
    if not value:
        return None
    s = str(value)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

# JSON guardrail: how many extra attempts when LLM returns non-JSON
JSON_MAX_RETRIES = 2
_JSON_REPAIR_HINT = (
    "Your previous response could not be parsed as JSON. Reply with ONLY a "
    "single JSON object — no prose, no markdown, no code fences, nothing "
    "before { or after }."
)


class GraphState(TypedDict, total=False):
    config: dict[str, Any]
    posts: list[dict[str, Any]]   # raw x-mcp posts, plus phase verdict fields
    x: XClient
    llm: ChatClaudeCode
    notion: NotionClient | None
    notion_database_id: str | None


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


async def _llm_json(
    llm: ChatClaudeCode,
    system_prompt: str,
    user_content: str,
    on_error: dict[str, Any],
    tag: str = "",
) -> dict[str, Any]:
    """Call the LLM and force a JSON dict out, with retries and a fallback."""
    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    last_text = ""
    for attempt in range(JSON_MAX_RETRIES + 1):
        try:
            resp = await llm.ainvoke(messages)
        except Exception as e:
            print(f"[llm{(' ' + tag) if tag else ''}] invoke error (attempt {attempt + 1}): {e}")
            if attempt == JSON_MAX_RETRIES:
                return dict(on_error)
            continue

        last_text = getattr(resp, "content", "") or ""
        parsed = _extract_json(last_text)
        if parsed is not None:
            return parsed

        if attempt < JSON_MAX_RETRIES:
            print(f"[llm{(' ' + tag) if tag else ''}] non-JSON response; retry {attempt + 1}/{JSON_MAX_RETRIES}")
            messages.append(AIMessage(content=last_text))
            messages.append(HumanMessage(content=_JSON_REPAIR_HINT))

    print(
        f"[llm{(' ' + tag) if tag else ''}] gave up after {JSON_MAX_RETRIES + 1} "
        f"attempts; using fallback. last response head: {last_text[:200]!r}"
    )
    return dict(on_error)


async def fetch_node(state: GraphState) -> dict[str, Any]:
    cfg = state["config"]["scrape"]
    client = state["x"]
    timeline = cfg.get("timeline_type", "following")
    timeframe = float(cfg.get("timeframe", DEFAULT_TIMEFRAME_HOURS))
    print(f"[fetch] x-mcp scrape_timeline type={timeline} timeframe={timeframe}h")
    # The scraper bounds the scroll by time (stops past the window). No post cap.
    data = await client.scrape_timeline(timeline_type=timeline, max_age_hours=timeframe)
    posts = data.get("posts") or []
    print(f"[fetch] got {len(posts)} posts (avgEngagement={data.get('avgEngagement')})")

    # Keep originals from the last `timeframe` hours; ignore reposts entirely
    # (a repost's timestamp is the original tweet's, not its feed-appearance
    # time, so it can't be dated reliably and the user doesn't want them).
    cutoff = datetime.now(timezone.utc) - timedelta(hours=timeframe)
    kept: list[dict[str, Any]] = []
    reposts = 0
    too_old = 0
    for p in posts:
        if p.get("isRetweet"):
            reposts += 1
            continue
        ts = _parse_post_timestamp(p.get("timestamp"))
        if ts is not None and ts < cutoff:
            too_old += 1
            continue
        kept.append(p)
    print(
        f"[fetch] kept {len(kept)} original post(s) within {timeframe}h "
        f"(ignored {reposts} repost(s), dropped {too_old} older)"
    )
    return {"posts": kept}


async def judge_node(state: GraphState) -> dict[str, Any]:
    """Ask the LLM whether each post is about AI; in parallel, capped."""
    llm = state["llm"]
    prompt = state["config"]["judge"]["prompt"]
    posts = state["posts"]

    sem = asyncio.Semaphore(LLM_PARALLEL_LIMIT)

    async def judge(post: dict[str, Any]) -> dict[str, Any]:
        # Trim very long posts so we don't blow up the context for nothing.
        content = (post.get("content") or "")[:4000]
        user = "X/Twitter post:\n" + json.dumps(
            {
                "author": post.get("author"),
                "content": content,
                "url": post.get("url"),
            },
            indent=2,
            default=str,
        )
        async with sem:
            verdict = await _llm_json(
                llm, prompt, user,
                on_error={"is_ai": False, "title": "", "introduction": "",
                          "reason": "non-JSON; defaulting to reject"},
                tag=f"judge {post.get('postId') or post.get('author')}",
            )
        verdict.setdefault("is_ai", False)
        verdict.setdefault("title", "")
        verdict.setdefault("introduction", "")
        verdict.setdefault("reason", "")
        passed = bool(verdict["is_ai"])
        print(
            f"[judge] @{post.get('author')} :: is_ai={verdict['is_ai']} "
            f"-> {'PASS' if passed else 'reject'} ({verdict.get('reason')!r})"
        )
        return {**post, "judge": verdict, "is_ai": passed}

    t0 = time.monotonic()
    judged = list(await asyncio.gather(*(judge(p) for p in posts)))
    dt = time.monotonic() - t0
    survivors = sum(1 for p in judged if p["is_ai"])
    print(f"[judge] judged {len(judged)} in {dt:.1f}s ({LLM_PARALLEL_LIMIT}-way parallel); {survivors} AI posts")
    return {"posts": judged}


async def write_node(state: GraphState) -> dict[str, Any]:
    posts = state["posts"]
    RESULTS_JSON.write_text(
        json.dumps({"config": state["config"], "posts": posts}, indent=2, default=str)
    )

    ai_posts = [p for p in posts if p.get("is_ai")]
    lines = [f"# AI reading list ({len(ai_posts)} of {len(posts)})", ""]
    for p in ai_posts:
        j = p.get("judge") or {}
        title = j.get("title") or "(no title)"
        intro = j.get("introduction") or ""
        url = p.get("url") or ""
        author = p.get("author") or ""
        lines.append(f"- **{title}** — @{author}  ")
        if url:
            lines.append(f"  {url}  ")
        if intro:
            lines.append(f"  _{intro}_")
        lines.append("")
    RESULTS_MD.write_text("\n".join(lines))
    print(f"[write] wrote {RESULTS_JSON.name} and {RESULTS_MD.name}")
    return {}


async def notion_node(state: GraphState) -> dict[str, Any]:
    notion = state.get("notion")
    db_id = state.get("notion_database_id")
    if not notion or not db_id:
        print("[notion] skipping (NOTION_ACCESS_TOKEN or NOTION_DATABASE_ID not set)")
        return {}
    pushed = 0
    failed = 0
    for p in state["posts"]:
        if not p.get("is_ai"):
            continue
        j = p.get("judge") or {}
        title = j.get("title") or (p.get("content") or "")[:80] or "(untitled)"
        introduction = j.get("introduction") or ""
        url = p.get("url") or ""
        try:
            await notion.create_reading_list_page(db_id, title, introduction, url)
            pushed += 1
            print(f"[notion] pushed {title!r}")
        except Exception as e:
            failed += 1
            print(f"[notion] failed to push {title!r}: {e}")
    print(f"[notion] created {pushed} pages ({failed} failed)")
    return {}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("fetch", fetch_node)
    g.add_node("judge", judge_node)
    g.add_node("write", write_node)
    g.add_node("notion", notion_node)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "judge")
    g.add_edge("judge", "write")
    g.add_edge("write", "notion")
    g.add_edge("notion", END)
    return g.compile()


async def run(config_path: Path, fetch_only: bool = False) -> None:
    load_dotenv(DEFAULT_ENV)
    config = yaml.safe_load(config_path.read_text())

    scrape_cfg = config.get("scrape", {})
    x_cmd = scrape_cfg.get("x_mcp_command", "node")
    x_args = scrape_cfg.get("x_mcp_args") or None

    async with x_session(command=x_cmd, args=x_args) as x:
        if fetch_only:
            data = await x.scrape_timeline(
                timeline_type=scrape_cfg.get("timeline_type", "following"),
                max_age_hours=float(scrape_cfg.get("timeframe", DEFAULT_TIMEFRAME_HOURS)),
            )
            posts = data.get("posts") or []
            print(f"\n[fetch-only] {len(posts)} posts:")
            for p in posts:
                rt = " (RT)" if p.get("isRetweet") else ""
                print(f"  - @{p.get('author')}{rt}: {(p.get('content') or '')[:120]!r}")
                print(f"    {p.get('url')}")
            return

        judge_cfg = config.get("judge", {})
        llm_kwargs: dict[str, Any] = {}
        if judge_cfg.get("model"):
            llm_kwargs["model"] = judge_cfg["model"]
        if judge_cfg.get("effort"):
            llm_kwargs["effort"] = judge_cfg["effort"]
        llm = ChatClaudeCode(**llm_kwargs)

        notion_token = os.environ.get("NOTION_ACCESS_TOKEN")
        notion_db_id = os.environ.get("NOTION_DATABASE_ID")

        async def _invoke(notion_client: NotionClient | None) -> None:
            initial = {
                "config": config,
                "x": x,
                "llm": llm,
                "notion": notion_client,
                "notion_database_id": notion_db_id,
            }
            graph = build_graph()
            await graph.ainvoke(initial)

        if notion_token and notion_db_id:
            async with notion_session() as notion:
                await _invoke(notion)
        else:
            await _invoke(None)


def main() -> None:
    p = argparse.ArgumentParser(description="X following → AI reading-list workflow (LangGraph).")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml")
    p.add_argument("--fetch-only", action="store_true",
                   help="Run only the fetch node, print posts, and exit. Useful for debugging scraping.")
    args = p.parse_args()
    asyncio.run(run(args.config, fetch_only=args.fetch_only))


if __name__ == "__main__":
    main()
