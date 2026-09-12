#!/usr/bin/env python3
"""Battery guard: pause the training queue before the machine loses power.

Why this exists: the trainer writes HF checkpoints (`training_run/checkpoint-NNNN`) but NOTHING
in this repo ever reads them back — there is no `resume_from_checkpoint` anywhere. So a pueue
pause is LOSSLESS (SIGSTOP; the process resumes at the exact step) but an uncontrolled power-off
is NOT: every in-flight fold restarts from epoch 0. With an undersized charger those two outcomes
are minutes apart, and the difference is worth hours of MPS compute.

Sibling of queue_watch.py and deliberately the same shape: pure Python plus the pueue CLI,
reporting by writing `scratch/battery_guard.status`, which is read on demand.

Every INTERVAL seconds it reads the SMC via ioreg and:
  * charge <= LOW   and the group is running -> `pueue pause  --group <g>`  (and remembers it did)
  * charge >= HIGH  and WE were the one who paused it -> `pueue start --group <g>`

That last condition is the important one. If YOU paused the group by hand, this will never resume
it behind your back — same rule that keeps queue_watch.py from fighting a deliberate suspend. The
"we paused it" flag lives in the state file, so it survives a guard restart.

Charge is read from AppleSmartBattery: AppleRawCurrentCapacity / AppleRawMaxCapacity (the true
pack numbers — pmset's rounded percentage lags, and its "charging" flag is unreliable on an
undersized adapter: it reads Yes while InstantAmperage is solidly negative). InstantAmperage is an
unsigned 64-bit field, so values above 2^63 are negative currents in disguise; negative = draining.

Threshold arithmetic (M4 Pro, ~74 Wh pack), so you can retune LOW/HIGH sensibly:
  training draw ~52-68 W total; on a 15 W brick that is a ~38-53 W net drain -> ~1.3 %/min
  idle draw     ~8-10 W (display on) / ~4-5 W (lid shut); on 15 W that is only +5 to +11 W net
So a LOW->HIGH round trip is minutes of training bought with hours of charging. Widen the band and
you get longer, rarer bursts; narrow it and you thrash. The default 30/75 is ~45 min of training
per ~4 h of charging on a 15 W adapter, and leaves real margin under the floor for the guard to
act. On a properly sized charger none of this triggers and the guard just idles.

Files (all under scratch/, gitignored):
  battery_guard.status   compact snapshot: charge, current, AC state, group states, last poll
  battery_guard.log      append-only action log (every pause / resume / alert)
  battery_guard.state    which groups WE paused (so we only resume those)
  battery_guard.ALERT    exists ONLY when charge is critical and still falling — needs a human

Env knobs: GUARD_GROUPS (default "retrain"), GUARD_LOW (30), GUARD_HIGH (75),
GUARD_CRIT (15), GUARD_INTERVAL (60).

  python3 scripts_plots/battery_guard.py --once     # single poll (test / cron)
  pueue parallel 2 --group watch && pueue add --group watch --label battery_guard -- \
      python3 scripts_plots/battery_guard.py        # daemon alongside queue_watch
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRATCH = os.path.join(ROOT, "scratch")
GROUPS = os.environ.get("GUARD_GROUPS", "retrain").split()
LOW = float(os.environ.get("GUARD_LOW", "30"))
HIGH = float(os.environ.get("GUARD_HIGH", "75"))
CRIT = float(os.environ.get("GUARD_CRIT", "15"))
INTERVAL = int(os.environ.get("GUARD_INTERVAL", "60"))
STATUS = os.path.join(SCRATCH, "battery_guard.status")
LOG = os.path.join(SCRATCH, "battery_guard.log")
STATE = os.path.join(SCRATCH, "battery_guard.state")
ALERT = os.path.join(SCRATCH, "battery_guard.ALERT")


def now():
    return time.strftime("%m-%d %H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    with open(LOG, "a") as fh:
        fh.write(line + "\n")
    print(line, file=sys.stderr)


def load_state():
    """Groups this guard paused itself (one per line). Survives a guard restart."""
    if not os.path.exists(STATE):
        return set()
    return {ln.strip() for ln in open(STATE) if ln.strip()}


def save_state(s):
    with open(STATE + ".tmp", "w") as fh:
        fh.write("".join(f"{g}\n" for g in sorted(s)))
    os.replace(STATE + ".tmp", STATE)


def battery():
    """(charge_pct, milliamps, on_ac) from the SMC. milliamps < 0 == draining."""
    out = subprocess.run(["ioreg", "-rn", "AppleSmartBattery"],
                         capture_output=True, text=True, timeout=30).stdout
    f = {}
    for key in ("AppleRawCurrentCapacity", "AppleRawMaxCapacity", "InstantAmperage",
                "ExternalConnected"):
        for ln in out.splitlines():
            if f'"{key}"' in ln:
                f[key] = ln.split("=", 1)[1].strip()
                break
    cur, mx = int(f["AppleRawCurrentCapacity"]), int(f["AppleRawMaxCapacity"])
    amp = int(f["InstantAmperage"])
    if amp >= 2 ** 63:          # unsigned 64-bit field holding a negative current
        amp -= 2 ** 64
    return 100.0 * cur / mx, amp, f.get("ExternalConnected") == "Yes"


def group_states():
    """{group: 'running'|'paused'} plus a count of tasks actually Running per group."""
    out = subprocess.run(["pueue", "status", "--json"], capture_output=True, text=True, timeout=60)
    d = json.loads(out.stdout)
    res = {}
    for g, info in d.get("groups", {}).items():
        st = info.get("status", info) if isinstance(info, dict) else info
        res[g] = str(st).lower()
    running = {g: 0 for g in res}
    for t in d.get("tasks", {}).values():
        s = t.get("status")
        tok = s if isinstance(s, str) else next(iter(s))
        if tok == "Running" and t.get("group") in running:
            running[t["group"]] += 1
    return res, running


def pueue(*args):
    r = subprocess.run(["pueue", *args], capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def poll():
    try:
        pct, amp, ac = battery()
        states, running = group_states()
    except Exception as e:   # ioreg/pueue hiccup — skip this cycle, never die
        log(f"poll error: {type(e).__name__}: {e}")
        return
    ours = load_state()
    draining = amp < 0
    actions, alerts = [], []

    for g in GROUPS:
        st = states.get(g)
        if st is None:
            continue                      # group went away; nothing to guard
        if pct <= LOW and st == "running":
            ok, msg = pueue("pause", "--group", g)
            if ok:
                ours.add(g)
            actions.append(f"pause {g} @ {pct:.1f}% -> {'ok' if ok else msg}")
            log(actions[-1])
        elif pct >= HIGH and st == "paused" and g in ours:
            ok, msg = pueue("start", "--group", g)
            if ok:
                ours.discard(g)
            actions.append(f"resume {g} @ {pct:.1f}% -> {'ok' if ok else msg}")
            log(actions[-1])
        elif pct >= HIGH and st == "paused" and g not in ours:
            # Paused by a human. Leave it alone — say so once per poll in the status file only.
            actions.append(f"{g} paused by hand @ {pct:.1f}% — not auto-resuming")

    save_state(ours)

    # Critical: below the floor AND still losing charge means the pause did not stop the bleeding
    # (something else is drawing, or the adapter is unplugged). That needs a human, not a retry.
    if pct <= CRIT and draining:
        with open(ALERT, "w") as fh:
            fh.write(f"[{now()}] battery {pct:.1f}% and still draining at {amp} mA "
                     f"(AC={'yes' if ac else 'NO'}). Groups {GROUPS} paused but charge is still "
                     f"falling — save work / find a charger.\n")
        alerts.append(f"CRITICAL {pct:.1f}% still draining ({amp} mA)")
    elif os.path.exists(ALERT):
        os.remove(ALERT)

    watts = abs(amp) * 11.86 / 1000.0     # nominal pack voltage; good enough for a status line
    lines = [
        f"battery_guard — last poll {now()}  (groups: {' '.join(GROUPS)}, "
        f"low {LOW:g}% / high {HIGH:g}% / crit {CRIT:g}%)",
        "",
        f"  charge      {pct:.1f}%",
        f"  current     {amp:+d} mA  (~{watts:.0f} W {'OUT of' if draining else 'INTO'} the pack)",
        f"  adapter     {'connected' if ac else 'NOT CONNECTED'}",
        "",
    ]
    for g in GROUPS:
        mark = "  (we paused it)" if g in ours else ""
        lines.append(f"  {g:12} {states.get(g, '?')}, {running.get(g, 0)} running{mark}")
    if actions:
        lines += ["", "  this poll:"] + [f"    {a}" for a in actions]
    if alerts:
        lines += ["", "  *** ALERT (see battery_guard.ALERT) ***"] + [f"    {a}" for a in alerts]
    with open(STATUS, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    once = "--once" in sys.argv[1:]
    log(f"battery_guard start (groups={GROUPS} low={LOW} high={HIGH} crit={CRIT} "
        f"interval={INTERVAL}s once={once})")
    while True:
        poll()
        if once:
            print(open(STATUS).read())
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
