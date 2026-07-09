# jobhunting

LangGraph workflow that searches LinkedIn jobs, enriches each with job + company
details (via [linkedin-mcp-server](https://github.com/stickerdaniel/linkedin-mcp-server)),
then uses a Codex headless judge to filter against the configured criteria.

## Pipeline

```
START → search → enrich (parallel, company cache) → judge (parallel) → write → END
```

## Criteria (rejection)

A job is dropped only when evidence is clear; missing fields default to "pass" (lenient).

1. Company size is `2-10 employees` or `11-50 employees`
2. Company is not a tech company
3. Required experience strictly more than 3 years

## Setup

```sh
# from repo root
.venv/bin/pip install -r /dev/stdin <<'EOF'
langgraph
pyyaml
python-dotenv
mcp
EOF
```

Install the Codex CLI and log in once with your Codex subscription:

```sh
codex login
```

The first MCP call opens a browser for LinkedIn login (one-time per machine; the MCP server caches the session).

## Run

```sh
.venv/bin/python jobhunting/workflow.py --config jobhunting/config.yaml
```

Outputs `jobhunting/results.json` (full data + verdicts) and `jobhunting/results.md` (shortlist).

## Files

- `workflow.py` — LangGraph entry point, 4-node pipeline
- `mcp_client.py` — stdio MCP wrapper around `search_jobs` / `get_job_details` / `get_company_profile`
- `../codex_headless.py` — adapter around `codex exec`
- `config.yaml` — search params + judge prompt
