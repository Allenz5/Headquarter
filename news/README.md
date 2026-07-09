# news

LangGraph workflow that scrapes posts from your X/Twitter **Following** timeline
(via the local [x-mcp-server](../../Projects/x-mcp-server)), filters them down
to AI-related posts with an LLM judge, and pushes the survivors into a Notion
database as a reading list.

## Pipeline

```
START → fetch → judge (parallel) → write → notion → END
```

## Setup

```sh
# from repo root
.venv/bin/pip install -r /dev/stdin <<'EOF'
langgraph
pyyaml
python-dotenv
mcp
EOF

# One-time Codex login for the headless judge
codex login

# Build the local x-mcp-server once (Playwright will need browsers too)
cd ~/Projects/x-mcp-server && npm install && npm run build
npx playwright install chromium

# One-time X/Twitter login (caches session under playwright/.auth)
cd ~/Projects/x-mcp-server && npm run cli login
```

Then fill in `news/.env`:

```
TWITTER_USERNAME=...
TWITTER_PASSWORD=...
NOTION_ACCESS_TOKEN=ntn_...
NOTION_DATABASE_ID=...
```

The Notion database needs three properties: **Title** (title),
**Summary** (rich_text), **URL** (url). The database must also be shared with
the integration that owns `NOTION_ACCESS_TOKEN` (••• → Connections), or pushes
fail with a 404 `object_not_found`.

## Run

```sh
.venv/bin/python news/workflow.py --config news/config.yaml
```

Outputs `news/results.json` (full data + verdicts) and `news/results.md`
(AI reading list).

## Files

- `workflow.py` — LangGraph entry point, 4-node pipeline
- `mcp_client.py` — stdio MCP wrappers around `scrape_timeline` (x-mcp) and
  `API-post-page` (notion-mcp-server)
- `config.yaml` — scrape params + judge prompt
