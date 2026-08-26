#!/usr/bin/env bash
# Fetch any still-missing consensus EPS histories for the PEAD basket, commit
# whatever landed, and push. Same reason as bin/record_gex.sh: the Routine that
# calls this used to be prose steps ending in "make no commit and stop", and it
# took that branch three firings running even though the fetch itself worked.
set -uo pipefail

BRANCH=claude/prop-firm-strategies-bvor8w
cd "$(dirname "$0")/.." || exit 1

die() { echo "fill_pead: $*" >&2; exit 1; }

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

basket = json.load(open("data/pead/basket20.json"))
names = basket if isinstance(basket, list) else list(basket)
have = json.load(open("data/pead/av_earnings.json"))
missing = sorted(set(names) - set(have))
if not missing:
    print(f"fill_pead: basket complete ({len(have)}/{len(names)})")
    sys.exit(0)

added = []
for t in missing:
    qs = A.quarters(t)
    if qs:
        have[t] = qs
        added.append(t)
        print(f"  {t}  {len(qs)} quarters  {qs[0]['date']} .. {qs[-1]['date']}")
    else:
        print(f"  {t}: no data today (quota, most likely)")
if added:
    json.dump(have, open("data/pead/av_earnings.json", "w"), indent=1, sort_keys=True)
print(f"fill_pead: added {len(added)} -> {len(have)}/{len(names)}")
PY

git add data/pead/av_earnings.json data/av 2>/dev/null
if git diff --cached --quiet; then
  echo "fill_pead: nothing new to commit"
  exit 0
fi

total=$(python3 -c "import json; print(len(json.load(open('data/pead/av_earnings.json'))))")
python3 src/run_concordant.py > /dev/null || die "run_concordant.py failed on the new basket"
python3 src/pead_loo.py       > /dev/null || die "pead_loo.py failed on the new basket"

git -c user.email=mikhailhoh@gmail.com -c user.name="Mikhail Hoh" \
    commit -q -m "Extend the PEAD basket to ${total}/20 names" || die "commit failed"

for wait in 2 4 8 16 0; do
  git push -u origin "$BRANCH" && { echo "fill_pead: pushed, basket now ${total}/20"; exit 0; }
  [ "$wait" = 0 ] && break
  echo "fill_pead: push failed, retrying in ${wait}s" >&2
  sleep "$wait"
done
die "push failed after 5 attempts -- the commit is local only"
