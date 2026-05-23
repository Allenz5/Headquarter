"""LangGraph workflow: search LinkedIn jobs, enrich, judge, write results."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from judge import judge_job
from mcp_client import LinkedInClient, linkedin_session

HERE = Path(__file__).parent
DEFAULT_CONFIG = HERE / "config.yaml"
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "results.md"
ENRICH_CONCURRENCY = 3
JUDGE_CONCURRENCY = 3


class GraphState(TypedDict, total=False):
    config: dict[str, Any]
    job_stubs: list[dict[str, Any]]
    enriched: list[dict[str, Any]]
    verdicts: list[dict[str, Any]]
    company_cache: dict[str, dict[str, Any]]
    linkedin: LinkedInClient
    openai: AsyncOpenAI


async def search_node(state: GraphState) -> dict[str, Any]:
    cfg = state["config"]
    client = state["linkedin"]
    print(f"[search] keywords={cfg['keywords']!r} location={cfg['location']!r} limit={cfg['limit']}")
    jobs = await client.search_jobs(cfg["keywords"], cfg["location"], cfg["limit"])
    print(f"[search] got {len(jobs)} job stubs")
    return {"job_stubs": jobs}


def _job_id(stub: dict[str, Any]) -> str | None:
    return stub.get("job_id") or stub.get("id") or stub.get("jobId")


def _company_key(details: dict[str, Any]) -> str | None:
    return (
        details.get("company_url")
        or details.get("company_link")
        or details.get("company_linkedin_url")
        or details.get("company")
        or details.get("company_name")
    )


async def enrich_node(state: GraphState) -> dict[str, Any]:
    client = state["linkedin"]
    cache: dict[str, dict[str, Any]] = state.get("company_cache") or {}
    sem = asyncio.Semaphore(ENRICH_CONCURRENCY)
    company_locks: dict[str, asyncio.Lock] = {}

    async def fetch_company(key: str) -> dict[str, Any] | None:
        if key in cache:
            return cache[key]
        lock = company_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in cache:
                return cache[key]
            try:
                profile = await client.get_company_profile(key)
            except Exception as e:
                print(f"[enrich]   company fetch failed for {key!r}: {e}")
                profile = None
            if profile is not None:
                cache[key] = profile
            return profile

    async def enrich_one(stub: dict[str, Any]) -> dict[str, Any] | None:
        jid = _job_id(stub)
        if not jid:
            return None
        async with sem:
            try:
                details = await client.get_job_details(jid)
            except Exception as e:
                print(f"[enrich] job {jid} details failed: {e}")
                return None
            if not details:
                return None
            ckey = _company_key(details) or _company_key(stub)
            company = await fetch_company(ckey) if ckey else None
            print(f"[enrich] {jid} :: {details.get('title') or details.get('job_title')!r}")
            return {"stub": stub, "details": details, "company": company}

    results = await asyncio.gather(*(enrich_one(s) for s in state["job_stubs"]))
    enriched = [r for r in results if r is not None]
    print(f"[enrich] enriched {len(enriched)}/{len(state['job_stubs'])} jobs")
    return {"enriched": enriched, "company_cache": cache}


async def judge_node(state: GraphState) -> dict[str, Any]:
    cfg = state["config"]
    oai = state["openai"]
    model = cfg.get("llm", {}).get("model", "gpt-4o-mini")
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def judge_one(item: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            try:
                verdict = await judge_job(oai, model, item["details"], item.get("company"))
            except Exception as e:
                verdict = {"worth_notice": True, "reason": f"judge error, kept by lenient policy: {e}", "criteria": {}}
        verdict["job_id"] = _job_id(item["details"]) or _job_id(item["stub"])
        verdict["title"] = item["details"].get("title") or item["details"].get("job_title")
        verdict["company"] = (item.get("company") or {}).get("name") or item["details"].get("company")
        verdict["url"] = item["details"].get("job_url") or item["stub"].get("job_url")
        return verdict

    verdicts = await asyncio.gather(*(judge_one(it) for it in state["enriched"]))
    kept = sum(1 for v in verdicts if v.get("worth_notice"))
    print(f"[judge] {kept}/{len(verdicts)} jobs marked worth notice")
    return {"verdicts": verdicts}


async def write_node(state: GraphState) -> dict[str, Any]:
    payload = {
        "config": state["config"],
        "jobs": [
            {**v, "details": e["details"], "company": e.get("company")}
            for v, e in zip(state["verdicts"], state["enriched"])
        ],
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2, default=str))

    shortlist = [v for v in state["verdicts"] if v.get("worth_notice")]
    lines = [f"# Job shortlist ({len(shortlist)} of {len(state['verdicts'])})", ""]
    for v in shortlist:
        title = v.get("title") or "(no title)"
        company = v.get("company") or "(unknown company)"
        url = v.get("url") or ""
        reason = v.get("reason") or ""
        lines.append(f"- **{title}** — {company}  ")
        if url:
            lines.append(f"  {url}  ")
        lines.append(f"  _{reason}_")
        lines.append("")
    RESULTS_MD.write_text("\n".join(lines))
    print(f"[write] wrote {RESULTS_JSON.name} and {RESULTS_MD.name}")
    return {}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("search", search_node)
    g.add_node("enrich", enrich_node)
    g.add_node("judge", judge_node)
    g.add_node("write", write_node)
    g.add_edge(START, "search")
    g.add_edge("search", "enrich")
    g.add_edge("enrich", "judge")
    g.add_edge("judge", "write")
    g.add_edge("write", END)
    return g.compile()


async def run(config_path: Path) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set in the environment.")
    config = yaml.safe_load(config_path.read_text())
    graph = build_graph()
    async with linkedin_session() as linkedin:
        oai = AsyncOpenAI()
        initial: GraphState = {
            "config": config,
            "linkedin": linkedin,
            "openai": oai,
            "company_cache": {},
        }
        await graph.ainvoke(initial)


def main() -> None:
    p = argparse.ArgumentParser(description="LinkedIn job-hunting workflow (LangGraph).")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml")
    args = p.parse_args()
    asyncio.run(run(args.config))


if __name__ == "__main__":
    main()
