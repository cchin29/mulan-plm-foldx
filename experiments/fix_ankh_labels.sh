#!/usr/bin/env bash
# DRAFT — honest-naming fix for the mislabelled A1/C1 "ankh" scaffolding (Ankh3 masquerading as the
# Ankh-v1 reference). See experiments/ANKH_PROVENANCE_AUDIT.md.
#
# WHAT IT DOES (deferred edits — NOT applied automatically):
#   1. git mv the two ankh3 configs to honest names:
#        config_a1_ankh.sh -> config_a1_ankh3_large.sh
#        config_c1_ankh.sh -> config_c1_ankh3_large.sh
#   2. inside them, retarget the metrics + default result roots from *_ankh to *_ankh3_large.
#   3. point run_{a1,c1}_grid.sh at the renamed configs AND make CFG env-overridable
#      (CFG=${CFG:-...}) so the v1 configs (config_{a1,c1}_ankh_large.sh, already added) can drive
#      the reference arm without further edits.
#
# The v1 reference arms then run as:
#   TAG=ankh CFG=experiments/embedding_sweep/config_a1_ankh_large.sh MAXPAR=2 bash experiments/embedding_sweep/run_a1_grid.sh
#   TAG=ankh CFG=experiments/embedding_sweep/config_c1_ankh_large.sh MAXPAR=2 bash experiments/embedding_sweep/run_c1_grid.sh
#
# ⚠ DO NOT RUN while the A1/C1 grids are live (pueue #60 running, #54/#61 queued). run_sweep.sh and the
#   grid scripts source these files at launch; renaming mid-flight will break in-flight/queued launches.
#   Apply only during a code-freeze window after #60 finishes and #54/#61 are drained or re-pointed.
#
# Guard: refuses to run unless CONFIRM=1 AND no ankh3 grid task is Running/Queued in pueue.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.." || exit 1   # repo root

ES=experiments/embedding_sweep

if [ "${CONFIRM:-0}" != "1" ]; then
  echo "DRAFT guard: re-run with CONFIRM=1 once the A1/C1 grids are drained. Doing nothing."
  echo "Preview of planned git mv:"
  echo "  $ES/config_a1_ankh.sh -> $ES/config_a1_ankh3_large.sh"
  echo "  $ES/config_c1_ankh.sh -> $ES/config_c1_ankh3_large.sh"
  exit 0
fi

# Safety: bail if ANY task is still Running/Queued (inspect task statuses precisely via python —
# a group being in the "running" state is fine; only active TASKS block the rename).
if command -v pueue >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  active=$(pueue status --json 2>/dev/null | python3 -c '
import json,sys
t=json.load(sys.stdin).get("tasks",{})
def st(x):
    s=x["status"]; return s if isinstance(s,str) else list(s.keys())[0]
print(sum(1 for x in t.values() if st(x) in ("Running","Queued","Paused")))' 2>/dev/null)
  if [ "${active:-0}" != "0" ]; then
    echo "ABORT: $active task(s) still Running/Queued/Paused in pueue. Drain the queue first." >&2
    exit 1
  fi
fi

set -x
# 1 + 2: rename the ankh3 configs and make their internal roots honest.
git mv "$ES/config_a1_ankh.sh" "$ES/config_a1_ankh3_large.sh"
git mv "$ES/config_c1_ankh.sh" "$ES/config_c1_ankh3_large.sh"
sed -i '' -e 's#results_a1_ankh#results_a1_ankh3_large#g' \
          -e 's#a1_ankh/_scratch#a1_ankh3_large/_scratch#g' "$ES/config_a1_ankh3_large.sh"
sed -i '' -e 's#results_c1_ankh#results_c1_ankh3_large#g' \
          -e 's#c1_ankh/_scratch#c1_ankh3_large/_scratch#g' "$ES/config_c1_ankh3_large.sh"

# 3: point grid scripts at the renamed configs, CFG overridable.
sed -i '' -e 's#^CFG=experiments/embedding_sweep/config_a1_ankh.sh#CFG=${CFG:-experiments/embedding_sweep/config_a1_ankh3_large.sh}#' \
          "$ES/run_a1_grid.sh"
sed -i '' -e 's#^CFG=experiments/embedding_sweep/config_c1_ankh.sh#CFG=${CFG:-experiments/embedding_sweep/config_c1_ankh3_large.sh}#' \
          "$ES/run_c1_grid.sh"
set +x

echo "Done. Verify: grep -n CFG $ES/run_a1_grid.sh $ES/run_c1_grid.sh ; git status"
echo "Then run the v1 reference arms per experiments/ANKH_PROVENANCE_AUDIT.md."
