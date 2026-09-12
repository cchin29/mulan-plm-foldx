#!/usr/bin/env python3
"""Pueue watcher: auto-heal recoverable failures and write a compact status file.

A plain background daemon — pure Python plus the pueue CLI, no polling by any other process.
It reports by writing `scratch/queue_watch.status`, which is read on demand, and by raising an
ALERT file when a task exhausts its restart budget.

Every INTERVAL seconds it reads `pueue status --json` and, for each Failed / DependencyFailed task
in the watched groups, AUTO-RESTARTS it (`pueue restart --in-place`) up to CAP times. That is safe:
the training driver (run_bycomplex.sh) and score_multipoint.py are RESUMABLE — a restart skips any
fold whose all_results.json already exists, so it only re-does what actually failed (this is exactly
what would have auto-recovered the #86 exit-code poisoning). A DependencyFailed task is restarted
only once all its deps are Success (so a recovered dep re-triggers its dependents). After CAP
attempts a task is left alone and written to the ALERT file for a human — no crash-loop compute burn.

Files (all under scratch/, gitignored):
  queue_watch.status   compact snapshot: per-group counts + any failures/alerts + last poll time
  queue_watch.log      append-only action log (every restart / alert)
  queue_watch.state    taskid -> restart-count (survives watcher restarts)
  queue_watch.ALERT    exists ONLY when a task hit the cap and needs a human

Env knobs: WATCH_GROUPS (default "retrain mp_train mp_emb skempi_emb score"), WATCH_INTERVAL (600s),
WATCH_CAP (2).

  python3 scripts_plots/queue_watch.py --once     # single poll (test / cron)
  pueue group add watch && pueue add --group watch --label queue_watch -- \
      python3 scripts_plots/queue_watch.py        # daemon
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRATCH = os.path.join(ROOT, "scratch")
# `retrain` first, and it is load-bearing: without it the watcher writes a reassuring
# all-Success snapshot describing four drained groups while the only group doing work is unwatched.
GROUPS = os.environ.get("WATCH_GROUPS", "retrain mp_train mp_emb skempi_emb score").split()
INTERVAL = int(os.environ.get("WATCH_INTERVAL", "600"))
CAP = int(os.environ.get("WATCH_CAP", "2"))
STATUS = os.path.join(SCRATCH, "queue_watch.status")
LOG = os.path.join(SCRATCH, "queue_watch.log")
STATE = os.path.join(SCRATCH, "queue_watch.state")
ALERT = os.path.join(SCRATCH, "queue_watch.ALERT")

FAIL = {"Failed", "DependencyFailed"}


def now():
    return time.strftime("%m-%d %H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    with open(LOG, "a") as fh:
        fh.write(line + "\n")
    print(line, file=sys.stderr)


def load_state():
    d = {}
    if os.path.exists(STATE):
        for ln in open(STATE):
            p = ln.split()
            if len(p) == 2:
                d[p[0]] = int(p[1])
    return d


def save_state(d):
    with open(STATE + ".tmp", "w") as fh:
        for k, v in d.items():
            fh.write(f"{k} {v}\n")
    os.replace(STATE + ".tmp", STATE)


def status_of(task):
    """Robust across pueue versions: collapse the nested status to one token."""
    s = json.dumps(task.get("status"))
    # `Killed` is recognised but deliberately NOT in FAIL: a killed task was stopped on purpose,
    # and restarting it would fight the operator. Without the token it collapsed to Unknown.
    for tok in ("DependencyFailed", "Failed", "Killed", "Success", "Running", "Queued",
                "Stashed", "Paused"):
        if tok in s:
            return tok
    return "Unknown"


def pueue_json():
    out = subprocess.run(["pueue", "status", "--json"], capture_output=True, text=True, timeout=60)
    return json.loads(out.stdout)["tasks"]


def poll():
    try:
        tasks = pueue_json()
    except Exception as e:  # pueue busy / not running — skip this cycle, don't die
        log(f"poll error: {type(e).__name__}: {e}")
        return
    state = load_state()
    watched = {k: v for k, v in tasks.items() if v.get("group") in GROUPS}
    st = {k: status_of(v) for k, v in watched.items()}

    # per-group counts for the status snapshot
    counts = {g: {} for g in GROUPS}
    for k, v in watched.items():
        g = v["group"]
        counts[g][st[k]] = counts[g].get(st[k], 0) + 1

    alerts, actions = [], []
    for k, v in sorted(watched.items(), key=lambda kv: int(kv[0])):
        if st[k] not in FAIL:
            continue
        label = v.get("label") or ""
        n = state.get(k, 0)
        if n >= CAP:
            alerts.append(f"#{k} [{label}] {st[k]} — {n} restarts, CAP hit; NEEDS HUMAN")
            continue
        # DependencyFailed: wait until every dep is Success before re-triggering the dependent
        if st[k] == "DependencyFailed":
            deps = v.get("dependencies", [])
            if any(st.get(str(d), status_of(tasks.get(str(d), {}))) != "Success" for d in deps):
                continue  # a dep is still not green — let the dep recover first
        r = subprocess.run(["pueue", "restart", "--in-place", k], capture_output=True, text=True)
        ok = r.returncode == 0
        state[k] = n + 1
        msg = f"restart #{k} [{label}] ({st[k]}) attempt {n + 1}/{CAP} -> {'ok' if ok else r.stderr.strip()}"
        log(msg)
        actions.append(msg)

    save_state(state)

    # ALERT file: present iff something needs a human
    if alerts:
        with open(ALERT, "w") as fh:
            fh.write(f"[{now()}] queue_watch needs attention:\n" + "\n".join(alerts) + "\n")
    elif os.path.exists(ALERT):
        os.remove(ALERT)

    # compact status snapshot (overwrite each poll)
    lines = [f"queue_watch — last poll {now()}  (groups: {' '.join(GROUPS)}, cap {CAP})", ""]
    for g in GROUPS:
        c = counts.get(g, {})
        lines.append(f"  {g:12} " + ("  ".join(f"{s}={n}" for s, n in sorted(c.items())) or "(empty)"))
    running = [f"#{k} {v.get('label','')}" for k, v in sorted(watched.items(), key=lambda kv: int(kv[0]))
              if st[k] == "Running"]
    if running:
        lines += ["", "  running: " + ", ".join(running)]
    if actions:
        lines += ["", "  this poll:"] + [f"    {a}" for a in actions]
    if alerts:
        lines += ["", "  *** ALERT (see queue_watch.ALERT) ***"] + [f"    {a}" for a in alerts]
    with open(STATUS, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    once = "--once" in sys.argv[1:]
    log(f"queue_watch start (groups={GROUPS} interval={INTERVAL}s cap={CAP} once={once})")
    while True:
        poll()
        if once:
            print(open(STATUS).read())
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
