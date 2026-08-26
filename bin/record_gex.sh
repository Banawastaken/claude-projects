#!/usr/bin/env bash
# Record one dealer gamma snapshot and commit it.
#
# This exists because the scheduled Routine that calls it used to be a list of
# prose steps, and three of the four asked the model to decide whether a commit
# was warranted. "Nothing new today" was always the cheaper branch, so three
# firings in a row reported success and left the repository untouched. Here the
# decision is the script's: it commits, or it exits non-zero and says why.
set -uo pipefail

BRANCH=claude/prop-firm-strategies-bvor8w
cd "$(dirname "$0")/.." || exit 1

die() { echo "record_gex: $*" >&2; exit 1; }

git rev-parse --git-dir >/dev/null 2>&1 || die "not a git repository: $PWD"

python3 -c 'import numpy, pandas' 2>/dev/null \
  || die "missing numpy/pandas -- the container image does not carry them"

git checkout "$BRANCH" 2>&1 | tail -1 || die "cannot checkout $BRANCH"
git pull --ff-only origin "$BRANCH" 2>&1 | tail -2

before=$(ls data/gex/daily/ 2>/dev/null | wc -l)
python3 src/gexdb.py || die "src/gexdb.py failed -- the CBOE fetch or the greeks did not complete"
after=$(ls data/gex/daily/ 2>/dev/null | wc -l)

git add data/gex/daily data/gex/metrics.jsonl
if git diff --cached --quiet; then
  # A duplicate day is the one legitimate no-op: the delayed chain still
  # carries the previous session's date, or the market was shut.
  [ "$before" = "$after" ] || die "wrote $((after - before)) file(s) that git will not stage"
  echo "record_gex: nothing new -- the chain still reports a date already stored"
  exit 0
fi

date=$(python3 -c "
import json
print(json.loads(open('data/gex/metrics.jsonl').read().strip().split(chr(10))[-1])['trade_date'])
")
git -c user.email=mikhailhoh@gmail.com -c user.name="Mikhail Hoh" \
    commit -q -m "Record GEX snapshot for $date" || die "commit failed"

for wait in 2 4 8 16 0; do
  git push -u origin "$BRANCH" && { echo "record_gex: pushed snapshot for $date"; exit 0; }
  [ "$wait" = 0 ] && break
  echo "record_gex: push failed, retrying in ${wait}s" >&2
  sleep "$wait"
done
die "push failed after 5 attempts -- the commit is local only"
