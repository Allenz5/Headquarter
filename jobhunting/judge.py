"""LLM judge: decide whether a job is worth noticing, given 3 criteria."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

SYSTEM_PROMPT = """You screen LinkedIn job postings against three rejection criteria.

Reject a job ONLY when evidence is clear. If a criterion can't be evaluated from the data, treat it as a pass for that criterion.

Rejection criteria:
1. Company size is "2-10 employees" OR "11-50 employees" (any other size = pass)
2. Company is NOT a technology company (software, internet, SaaS, AI/ML, fintech, devtools, cloud, semiconductors, consumer tech, etc. all count as tech)
3. Required experience is strictly MORE than 3 years (3 years or fewer = pass; "entry level" / "new grad" / unspecified = pass)

Return strict JSON: {"worth_notice": bool, "reason": "<one short sentence>", "criteria": {"company_size_ok": bool, "is_tech": bool, "experience_ok": bool}}"""


def _build_user_prompt(job: dict[str, Any], company: dict[str, Any] | None) -> str:
    payload = {
        "job_title": job.get("title") or job.get("job_title"),
        "job_description": (job.get("description") or job.get("job_description") or "")[:4000],
        "required_experience": job.get("seniority_level")
        or job.get("experience_level")
        or job.get("required_experience"),
        "employment_type": job.get("employment_type"),
        "company_name": (company or {}).get("name") or job.get("company"),
        "company_industry": (company or {}).get("industry"),
        "company_size": (company or {}).get("company_size") or (company or {}).get("size"),
        "company_specialties": (company or {}).get("specialties"),
        "company_about": ((company or {}).get("about") or "")[:1500],
    }
    return "Job posting to evaluate:\n" + json.dumps(payload, indent=2, default=str)


async def judge_job(
    client: AsyncOpenAI,
    model: str,
    job: dict[str, Any],
    company: dict[str, Any] | None,
) -> dict[str, Any]:
    response = await client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(job, company)},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        verdict = {"worth_notice": True, "reason": "judge returned non-JSON; kept by lenient policy", "criteria": {}}
    verdict.setdefault("worth_notice", True)
    verdict.setdefault("reason", "")
    verdict.setdefault("criteria", {})
    return verdict
