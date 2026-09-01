#!/usr/bin/env bash
# Fetch any still-missing consensus EPS histories for the PEAD basket, commit
# whatever landed, and push. Same reason as bin/record_gex.sh: the Routine that
# calls this used to be prose steps ending in "make no commit and stop", and it
# took that branch three firings running even though the fetch itself worked.
set -uo pipefail

BRANCH=claude/prop-firm-strategies-bvor8w
LOG=data/pead/routine.log
cd "$(dirname "$0")/.." || exit 1

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Same heartbeat as bin/record_gex.sh, and for the same reason: a firing that
# fetched nothing and a firing whose push failed are indistinguishable from
# outside this container unless every run is made to leave a committed mark.
finish() {
  local outcome="$1"
  echo "$(now) $outcome" >> "$LOG"
  echo "fill_pead: $outcome"

  git add "$LOG" data/pead/av_earnings.json data/pead/mega_earnings.json \
          data/pead/mega20.json data/av 2>/dev/null
  git diff --cached --quiet && exit "${2:-0}"

  git -c user.email=mikhailhoh@gmail.com -c user.name="Mikhail Hoh" \
      commit -q -m "PEAD basket run: $outcome" || {
        echo "fill_pead: commit failed" >&2; exit 1; }

  for wait in 2 4 8 16 0; do
    git push -u origin "$BRANCH" 2>&1 | tail -1 && exit "${2:-0}"
    [ "$wait" = 0 ] && break
    sleep "$wait"
  done
  echo "fill_pead: push failed after 5 attempts, the commit is local only" >&2
  exit 1
}

die() { finish "FAILED: $*" 1; }

git rev-parse --git-dir >/dev/null 2>&1 || die "not a git repository: $PWD"
git checkout "$BRANCH" 2>&1 | tail -1 || die "cannot checkout $BRANCH"
git pull --ff-only origin "$BRANCH" 2>&1 | tail -2

# Alpha Vantage's free tier answers a refusal with HTTP 200 and a prose body,
# so a name that fails today is a spent quota, not a broken symbol. Take what
# lands and leave the rest for tomorrow's firing.
python3 - <<'PY' || die "the fetch itself failed"
import json, sys
sys.path.insert(0, "src")
import av_earnings as A

# Two baskets share one 25-a-day quota, so they are filled in one pass with
# the original basket first: it needs two names and the mega-cap set needs
# seventeen, and a run that spent the quota on the second while leaving the
# first short would be the wrong way round.
BASKETS = [
    ("data/pead/basket20.json", "data/pead/av_earnings.json", "large"),
    ("data/pead/mega20.json", "data/pead/mega_earnings.json", "mega"),
]

total_added = 0
for basket_path, out_path, label in BASKETS:
    names = json.load(open(basket_path))
    try:
        have = json.load(open(out_path))
    except FileNotFoundError:
        have = {}
    missing = sorted(set(names) - set(have))
    if not missing:
        print(f"fill_pead: {label} complete ({len(have)}/{len(names)})")
        continue

    added = []
    for t in missing:
        qs = A.quarters(t)
        if qs:
            have[t] = qs
            added.append(t)
            print(f"  {label} {t}  {len(qs)} quarters  {qs[0]['date']} .. {qs[-1]['date']}")
        else:
            print(f"  {label} {t}: no data today (quota, most likely)")
    if added:
        json.dump(have, open(out_path, "w"), indent=1, sort_keys=True)
    total_added += len(added)
    print(f"fill_pead: {label} added {len(added)} -> {len(have)}/{len(names)}")
print(f"fill_pead: {total_added} name(s) added in total")
PY

count() { python3 -c "import json,os,sys; p=sys.argv[1]; print(len(json.load(open(p))) if os.path.exists(p) else 0)" "$1"; }
total=$(count data/pead/av_earnings.json)
mega=$(count data/pead/mega_earnings.json)

if git diff --quiet -- data/pead/av_earnings.json data/pead/mega_earnings.json data/av; then
  finish "no-op: nothing landed, ${total}/20 large and ${mega}/20 mega"
fi

python3 src/run_concordant.py > /dev/null || die "run_concordant.py errored on the new basket"
python3 src/pead_loo.py       > /dev/null || die "pead_loo.py errored on the new basket"
finish "${total}/20 large and ${mega}/20 mega"
