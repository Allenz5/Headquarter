#!/usr/bin/env bash
# Run the jobhunting and news workflows concurrently.
#
# Both judge steps spawn Codex headless (`codex exec`), which uses the local
# Codex subscription login. We unset ANTHROPIC_API_KEY here too so legacy
# vendor credentials cannot affect the run.
#
# Each workflow streams to its own timestamped log under logs/, and its lines
# are also echoed here with a [job]/[news] prefix. Extra args are passed to BOTH
# workflows (e.g. ./run-all --search-only is meaningless for news, so prefer
# running them separately when you need workflow-specific flags).
set -u

cd "$(dirname "$0")"
unset ANTHROPIC_API_KEY

PY=.venv/bin/python
mkdir -p logs
STAMP="$(date +%Y%m%d-%H%M%S)"
JOB_LOG="logs/jobhunting-$STAMP.log"
NEWS_LOG="logs/news-$STAMP.log"

echo "Starting both workflows (logs: $JOB_LOG, $NEWS_LOG)"

# Run each workflow, tee to its log, and prefix the live output so the two
# interleaved streams stay readable. The pipefail subshell makes each wait
# return the workflow's exit status, not sed's.
( set -o pipefail; "$PY" jobhunting/workflow.py --config jobhunting/config.yaml "$@" 2>&1 \
  | tee "$JOB_LOG" | sed 's/^/[job]  /' ) &
job_pid=$!

( set -o pipefail; "$PY" news/workflow.py --config news/config.yaml "$@" 2>&1 \
  | tee "$NEWS_LOG" | sed 's/^/[news] /' ) &
news_pid=$!

# Wait for both; capture each exit status independently.
wait "$job_pid";  job_rc=$?
wait "$news_pid"; news_rc=$?

echo "----------------------------------------"
echo "jobhunting exited $job_rc  (log: $JOB_LOG)"
echo "news       exited $news_rc  (log: $NEWS_LOG)"

# Non-zero if either workflow failed.
[ "$job_rc" -eq 0 ] && [ "$news_rc" -eq 0 ]
