#!/usr/bin/env bash
# Record one dealer gamma snapshot and commit it.
#
# Every run appends a line to data/gex/routine.log and commits it, even when
# the snapshot itself is a no-op. That log is the only way to see what a
# scheduled firing actually did: the Routine runs in its own container, its
# reply is not readable from here, and "nothing new today" and "the push
# silently failed" look identical from outside. A heartbeat that must be
# committed collapses those two into one observable.
set -uo pipefail

BRANCH=claude/prop-firm-strategies-bvor8w
LOG=data/gex/routine.log
cd "$(dirname "$0")/.." || exit 1

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Record the outcome, then commit and push it along with whatever data landed.
# Called on every exit path, including the failures.
finish() {
  local outcome="$1"
  echo "$(now) $outcome" >> "$LOG"
  echo "record_gex: $outcome"

  git add "$LOG" data/gex/daily data/gex/metrics.jsonl 2>/dev/null
  git diff --cached --quiet && exit "${2:-0}"

  git -c user.email=mikhailhoh@gmail.com -c user.name="Mikhail Hoh" \
      commit -q -m "Record GEX run: $outcome" || {
        echo "record_gex: commit failed" >&2; exit 1; }

  for wait in 2 4 8 16 0; do
    git push -u origin "$BRANCH" 2>&1 | tail -1 && exit "${2:-0}"
    [ "$wait" = 0 ] && break
    sleep "$wait"
  done
  echo "record_gex: push failed after 5 attempts, the commit is local only" >&2
  exit 1
}

git rev-parse --git-dir >/dev/null 2>&1 || { echo "record_gex: not a git repo: $PWD" >&2; exit 1; }
git checkout "$BRANCH" 2>&1 | tail -1
git pull --ff-only origin "$BRANCH" 2>&1 | tail -1

python3 -c 'import numpy, pandas' 2>/dev/null \
  || finish "FAILED: container has no numpy/pandas" 1

out=$(python3 src/gexdb.py 2>&1) || finish "FAILED: src/gexdb.py errored -- $(echo "$out" | tail -1)" 1

date=$(python3 -c "
import json
rows=[json.loads(l) for l in open('data/gex/metrics.jsonl')]
print(max(r['trade_date'] for r in rows))
" 2>/dev/null) || date=unknown

if git diff --quiet -- data/gex/daily data/gex/metrics.jsonl; then
  finish "no-op: chain still reports $date, already stored"
fi
finish "recorded $date"
