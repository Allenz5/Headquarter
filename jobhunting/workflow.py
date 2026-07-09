"""LangGraph workflow: search → phase1 (AI + experience filter) → phase2
(company size filter, only on survivors) → write."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

import yaml
from datetime import date as _date
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from mcp_client import LinkedInClient, NotionClient, linkedin_session, notion_session

HERE = Path(__file__).parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codex_headless import ChatCodexHeadless

DEFAULT_CONFIG = HERE / "config.yaml"
DEFAULT_ENV = HERE / ".env"
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "results.md"
# LinkedIn calls are serialized inside ThrottledLinkedIn (anti-scrape).
# LLM calls are independent → run in parallel up to this cap to bound CLI
# subprocess pressure on the host.
LLM_PARALLEL_LIMIT = 5

# Throttle settings (LinkedIn anti-scrape mitigation)
BASE_INTERVAL = 3.0       # seconds between any two LinkedIn MCP calls
SENTINEL_FLOOR = 60.0     # when empty-result detected, jump interval to at least this
MAX_INTERVAL = 300.0      # absolute cap
# When a detail/company fetch comes back empty (a rate-limit tell), retry it
# this many times — each retry waits the now-bumped interval. Without this, a
# throttled job is silently dropped for the whole run, which is how a fully
# rate-limited run produces an empty "0 of 0" shortlist with no error.
FETCH_MAX_RETRIES = 3

# JSON guardrail: how many extra attempts when LLM returns non-JSON
JSON_MAX_RETRIES = 2
_JSON_REPAIR_HINT = (
    "Your previous response could not be parsed as JSON. Reply with ONLY a "
    "single JSON object — no prose, no markdown, no code fences, nothing "
    "before { or after }."
)


class GraphState(TypedDict, total=False):
    config: dict[str, Any]
    job_ids: list[str]
    search_raw: dict[str, Any]
    jobs: list[dict[str, Any]]  # accumulates fields as it moves through phases
    company_cache: dict[str, dict[str, Any]]
    linkedin: "ThrottledLinkedIn"
    llm: ChatCodexHeadless
    notion: NotionClient | None
    notion_database_id: str | None


class ThrottledLinkedIn:
    """Serializes every LinkedIn MCP call, enforces a minimum gap between calls,
    and bumps the gap when a response looks rate-limited (empty sections)."""

    def __init__(self, client: LinkedInClient):
        self._client = client
        self._lock = asyncio.Lock()
        self._interval = BASE_INTERVAL
        self._last_call_at = 0.0

    async def _wait_turn(self) -> None:
        loop = asyncio.get_event_loop()
        wait = (self._last_call_at + self._interval) - loop.time()
        if wait > 0:
            print(f"[rate] waiting {wait:.1f}s (current interval={self._interval:.0f}s)")
            await asyncio.sleep(wait)
        self._last_call_at = loop.time()

    def _bump(self, kind: str, key: str) -> None:
        new = min(max(self._interval * 2, SENTINEL_FLOOR), MAX_INTERVAL)
        if new > self._interval:
            print(f"[rate] empty {kind} for {key!r}; bumping interval {self._interval:.0f}s -> {new:.0f}s")
        self._interval = new

    @staticmethod
    def _company_empty(r: dict[str, Any] | None) -> bool:
        if not r:
            return True
        about = (r.get("sections") or {}).get("about") or ""
        return not about.strip()

    @staticmethod
    def _job_empty(r: dict[str, Any] | None) -> bool:
        if not r:
            return True
        posting = (r.get("sections") or {}).get("job_posting") or ""
        return not posting.strip()

    async def search_jobs(self, **kwargs: Any) -> dict[str, Any]:
        async with self._lock:
            await self._wait_turn()
            return await self._client.search_jobs(**kwargs)

    async def get_job_details(self, jid: str) -> dict[str, Any] | None:
        async with self._lock:
            last: dict[str, Any] | None = None
            for attempt in range(FETCH_MAX_RETRIES + 1):
                await self._wait_turn()
                r = await self._client.get_job_details(jid)
                if not self._job_empty(r):
                    return r
                last = r
                if attempt < FETCH_MAX_RETRIES:
                    self._bump("job-details", jid)
                    print(f"[rate] empty job-details for {jid!r}; retry {attempt + 1}/{FETCH_MAX_RETRIES}")
            return last

    async def get_company_profile(self, slug: str) -> dict[str, Any] | None:
        async with self._lock:
            last: dict[str, Any] | None = None
            for attempt in range(FETCH_MAX_RETRIES + 1):
                await self._wait_turn()
                r = await self._client.get_company_profile(slug)
                if not self._company_empty(r):
                    return r
                last = r
                if attempt < FETCH_MAX_RETRIES:
                    self._bump("company-profile", slug)
                    print(f"[rate] empty company-profile for {slug!r}; retry {attempt + 1}/{FETCH_MAX_RETRIES}")
            return last


def _job_posting_text(details: dict[str, Any]) -> str:
    return (details.get("sections") or {}).get("job_posting") or ""


def _job_references(details: dict[str, Any]) -> list[dict[str, Any]]:
    return (details.get("references") or {}).get("job_posting") or []


def _parse_display(details: dict[str, Any]) -> tuple[str | None, str | None]:
    lines = [ln.strip() for ln in _job_posting_text(details).split("\n") if ln.strip()]
    company = lines[0] if lines else None
    title = lines[1] if len(lines) > 1 else None
    return title, company


def _company_slug(details: dict[str, Any]) -> str | None:
    for ref in _job_references(details):
        if ref.get("kind") != "company":
            continue
        url = (ref.get("url") or "").strip("/")
        if url.startswith("company/"):
            url = url[len("company/"):]
        slug = url.split("/", 1)[0]
        if slug:
            return slug
    return None


def _extract_json(text: str) -> dict[str, Any] | None:
    """Guardrail step 1: strip everything before the first '{' and after the
    last '}', then attempt json.loads. Returns None on failure."""
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
    llm: ChatCodexHeadless,
    system_prompt: str,
    user_content: str,
    on_error: dict[str, Any],
    tag: str = "",
) -> dict[str, Any]:
    """Call the LLM and force a JSON dict out.

    Guardrails:
      1. Extract the substring from first '{' to last '}' before parsing.
      2. If parsing still fails, retry up to JSON_MAX_RETRIES times with
         a corrective hint appended to the conversation. Falls back to
         on_error after the final attempt.
    """
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
        f"attempts; using lenient fallback. last response head: {last_text[:200]!r}"
    )
    return dict(on_error)


async def search_node(state: GraphState) -> dict[str, Any]:
    cfg = state["config"]["search"]
    client = state["linkedin"]
    print(f"[search] keywords={cfg['keywords']!r} max_pages={cfg.get('max_pages', 3)}")
    # The server stops paginating before its tool timeout and returns the pages
    # it already has, so a normal run yields a (possibly partial) dict. Still
    # guard against a hard transport/timeout error so the workflow degrades to
    # "no jobs this run" instead of crashing with a traceback.
    try:
        data = await client.search_jobs(**cfg)
    except Exception as e:
        print(f"[search] search_jobs failed ({type(e).__name__}: {e}); "
              f"continuing with no job ids this run")
        data = {}
    job_ids = [str(j) for j in (data.get("job_ids") or [])]
    print(f"[search] LinkedIn URL: {data.get('url')}")
    print(f"[search] got {len(job_ids)} job ids")
    return {"job_ids": job_ids, "search_raw": data}


async def phase1_node(state: GraphState) -> dict[str, Any]:
    """Fetch job details serially (anti-scrape), then judge all in parallel."""
    client = state["linkedin"]
    llm = state["llm"]
    cfg = state["config"]["judge"]
    prompt = cfg["phase1_prompt"]
    job_ids = state["job_ids"]

    # Step A: serial scrape (ThrottledLinkedIn enforces the 3s gap)
    t0 = time.monotonic()
    fetched: list[tuple[str, dict[str, Any]]] = []
    empty = 0
    for jid in job_ids:
        try:
            details = await client.get_job_details(jid)
        except Exception as e:
            print(f"[phase1] {jid} fetch failed: {e}")
            continue
        # An empty posting (after retries) is a rate-limit drop, not a real job —
        # skip it rather than judge the LLM on blank text.
        if details and _job_posting_text(details).strip():
            fetched.append((jid, details))
        else:
            empty += 1
            print(f"[phase1] {jid} returned empty posting (rate-limited?); skipped")
    t_scrape = time.monotonic() - t0
    print(f"[phase1] scraped {len(fetched)}/{len(job_ids)} job details in {t_scrape:.1f}s")
    if job_ids and len(fetched) < max(1, len(job_ids) // 2):
        print(
            f"[phase1] WARNING: only {len(fetched)}/{len(job_ids)} details scraped "
            f"({empty} empty) — likely LinkedIn rate-limiting. The shortlist will be "
            f"incomplete; consider rerunning later or lowering search.max_pages."
        )

    # Step B: judge in parallel (capped concurrency to avoid CLI subprocess flood)
    llm_sem = asyncio.Semaphore(LLM_PARALLEL_LIMIT)

    async def judge(jid: str, details: dict[str, Any]) -> dict[str, Any]:
        title, company_display = _parse_display(details)
        user = "Job posting:\n" + json.dumps(
            {"posting_text": _job_posting_text(details)[:6000]}, indent=2
        )
        async with llm_sem:
            verdict = await _llm_json(
                llm, prompt, user,
                on_error={"is_ai": True, "experience_ok": True, "reason": "non-JSON; kept by lenient policy"},
                tag=f"phase1 {jid}",
            )
        verdict.setdefault("is_ai", True)
        verdict.setdefault("experience_ok", True)
        verdict.setdefault("reason", "")
        passed = bool(verdict["is_ai"]) and bool(verdict["experience_ok"])
        print(
            f"[phase1] {jid} :: {title!r} :: is_ai={verdict['is_ai']} "
            f"exp_ok={verdict['experience_ok']} -> {'PASS' if passed else 'reject'}"
        )
        return {
            "job_id": jid,
            "title": title,
            "company_display": company_display,
            "url": details.get("url"),
            "details": details,
            "phase1": verdict,
            "phase1_passed": passed,
        }

    t1 = time.monotonic()
    jobs = list(await asyncio.gather(*(judge(jid, d) for jid, d in fetched)))
    t_judge = time.monotonic() - t1
    survivors = sum(1 for j in jobs if j["phase1_passed"])
    print(f"[phase1] judged in {t_judge:.1f}s ({LLM_PARALLEL_LIMIT}-way parallel); {survivors}/{len(jobs)} survive to phase 2")
    return {"jobs": jobs}


async def phase2_node(state: GraphState) -> dict[str, Any]:
    """Fetch company profiles serially (anti-scrape), then judge in parallel."""
    client = state["linkedin"]
    llm = state["llm"]
    cfg = state["config"]["judge"]
    prompt = cfg["phase2_prompt"]
    cache: dict[str, dict[str, Any]] = state.get("company_cache") or {}
    survivors = [j for j in state["jobs"] if j["phase1_passed"]]

    # Step A: serial scrape (cached so the same company isn't fetched twice)
    t0 = time.monotonic()
    enriched: list[tuple[dict[str, Any], str | None, dict[str, Any] | None]] = []
    for job in survivors:
        slug = _company_slug(job["details"])
        if slug and slug in cache:
            profile = cache[slug]
        elif slug:
            try:
                profile = await client.get_company_profile(slug)
            except Exception as e:
                print(f"[phase2]   company fetch failed for {slug!r}: {e}")
                profile = None
            if profile is not None:
                cache[slug] = profile
        else:
            profile = None
        enriched.append((job, slug, profile))
    t_scrape = time.monotonic() - t0
    print(f"[phase2] scraped {len(enriched)} company profiles in {t_scrape:.1f}s")

    # Step B: judge in parallel
    llm_sem = asyncio.Semaphore(LLM_PARALLEL_LIMIT)

    async def judge(job: dict[str, Any], slug: str | None, profile: dict[str, Any] | None) -> dict[str, Any]:
        if profile is None:
            verdict = {
                "company_size_ok": True,
                "reason": "no company profile available; default pass (lenient)",
            }
        else:
            user = "Company profile:\n" + json.dumps(profile, indent=2, default=str)[:8000]
            async with llm_sem:
                verdict = await _llm_json(
                    llm, prompt, user,
                    on_error={"company_size_ok": True, "reason": "non-JSON; kept by lenient policy"},
                    tag=f"phase2 {job['job_id']}",
                )
            verdict.setdefault("company_size_ok", True)
            verdict.setdefault("reason", "")
        print(f"[phase2] {job['job_id']} :: slug={slug} size_ok={verdict['company_size_ok']}")
        return {**job, "company_profile": profile, "company_slug": slug, "phase2": verdict}

    t1 = time.monotonic()
    results = list(await asyncio.gather(*(judge(j, s, p) for j, s, p in enriched)))
    t_judge = time.monotonic() - t1
    print(f"[phase2] judged in {t_judge:.1f}s ({LLM_PARALLEL_LIMIT}-way parallel)")

    by_id = {r["job_id"]: r for r in results}
    merged = [by_id.get(j["job_id"], j) for j in state["jobs"]]
    return {"jobs": merged, "company_cache": cache}


def _final_verdict(job: dict[str, Any]) -> tuple[bool, str]:
    if not job.get("phase1_passed"):
        return False, job["phase1"].get("reason") or "rejected at phase 1"
    p2 = job.get("phase2") or {}
    if p2.get("company_size_ok") is False:
        return False, p2.get("reason") or "company size rejected"
    return True, p2.get("reason") or job["phase1"].get("reason") or "passed all filters"


async def write_node(state: GraphState) -> dict[str, Any]:
    annotated = []
    for j in state["jobs"]:
        worth, reason = _final_verdict(j)
        annotated.append({**j, "worth_notice": worth, "final_reason": reason})

    RESULTS_JSON.write_text(
        json.dumps({"config": state["config"], "jobs": annotated}, indent=2, default=str)
    )

    shortlist = [j for j in annotated if j["worth_notice"]]
    lines = [f"# Job shortlist ({len(shortlist)} of {len(annotated)})", ""]
    for j in shortlist:
        title = j.get("title") or "(no title)"
        company = j.get("company_display") or "(unknown company)"
        url = j.get("url") or ""
        lines.append(f"- **{title}** — {company}  ")
        if url:
            lines.append(f"  {url}  ")
        lines.append(f"  _{j['final_reason']}_")
        lines.append("")
    RESULTS_MD.write_text("\n".join(lines))
    print(f"[write] wrote {RESULTS_JSON.name} and {RESULTS_MD.name}")
    print(f"[llm] subscription mode via Codex headless — no per-call cost tracked")
    return {}


async def notion_node(state: GraphState) -> dict[str, Any]:
    notion = state.get("notion")
    db_id = state.get("notion_database_id")
    if not notion or not db_id:
        print("[notion] skipping (NOTION_ACCESS_TOKEN or NOTION_DATABASE_ID not set)")
        return {}
    today = _date.today().isoformat()
    pushed = 0
    failed = 0
    for j in state["jobs"]:
        worth, _ = _final_verdict(j)
        if not worth:
            continue
        title = j.get("title") or "(no title)"
        url = j.get("url") or ""
        company = j.get("company_display") or ""
        try:
            await notion.create_database_page(db_id, title, today, url, company=company)
            pushed += 1
            print(f"[notion] pushed {title!r} @ {company!r}")
        except Exception as e:
            failed += 1
            print(f"[notion] failed to push {title!r}: {e}")
    print(f"[notion] created {pushed} pages ({failed} failed)")
    return {}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("search", search_node)
    g.add_node("phase1", phase1_node)
    g.add_node("phase2", phase2_node)
    g.add_node("write", write_node)
    g.add_node("notion", notion_node)
    g.add_edge(START, "search")
    g.add_edge("search", "phase1")
    g.add_edge("phase1", "phase2")
    g.add_edge("phase2", "write")
    g.add_edge("write", "notion")
    g.add_edge("notion", END)
    return g.compile()


async def run(config_path: Path, search_only: bool = False) -> None:
    load_dotenv(DEFAULT_ENV)

    # The judge runs via Codex headless (`codex exec`), which uses the local
    # Codex subscription login. Drop legacy vendor credentials inherited from
    # the shell so no old path can be selected accidentally.
    if os.environ.pop("ANTHROPIC_API_KEY", None):
        print("[setup] dropped ANTHROPIC_API_KEY so the judge uses Codex headless")

    config = yaml.safe_load(config_path.read_text())
    async with linkedin_session() as linkedin:
        throttled = ThrottledLinkedIn(linkedin)
        if search_only:
            initial: GraphState = {"config": config, "linkedin": throttled}
            state = dict(initial)
            result = await search_node(state)  # type: ignore[arg-type]
            ids = result.get("job_ids") or []
            print("\n[search-only] job IDs:")
            for jid in ids:
                print(f"  - https://www.linkedin.com/jobs/view/{jid}/")
            return

        judge_cfg = config.get("judge", {})
        llm_kwargs: dict[str, Any] = {}
        if judge_cfg.get("model"):
            llm_kwargs["model"] = judge_cfg["model"]
        if judge_cfg.get("effort"):
            llm_kwargs["effort"] = judge_cfg["effort"]
        llm = ChatCodexHeadless(**llm_kwargs)

        notion_token = os.environ.get("NOTION_ACCESS_TOKEN")
        notion_db_id = os.environ.get("NOTION_DATABASE_ID")

        async def _invoke(notion_client: NotionClient | None) -> None:
            initial = {
                "config": config,
                "linkedin": throttled,
                "llm": llm,
                "company_cache": {},
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
    p = argparse.ArgumentParser(description="LinkedIn job-hunting workflow (LangGraph).")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml")
    p.add_argument("--search-only", action="store_true",
                   help="Run only the search node, print results, and exit. Useful for debugging filters.")
    args = p.parse_args()
    asyncio.run(run(args.config, search_only=args.search_only))


if __name__ == "__main__":
    main()
