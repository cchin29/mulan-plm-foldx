#!/usr/bin/env python3
"""Progress watcher for a FoldX per-complex run.

Sibling of queue_watch.py — a plain background daemon. It targets the
known FoldX failure mode: the run gets **stuck on a final complex and grinds for hours with no
progress**. The tell is a *filesystem activity clock*: FoldX BuildModel writes per-mutation output
files as it goes, so a healthy (even very slow) complex keeps touching files in its work dir, while
a hung FoldX touches nothing. So:

    idle = now - newest mtime across (results JSONs) ∪ (in-progress work dirs' files)
    STALL  ⇔  task still Running  AND  idle > --stall-hours

A legitimately slow deep-scan complex (3BT1 = 240 muts) does NOT trip it (its dir keeps growing);
a wedged FoldX does. Each cycle writes a human-readable status file; a stall (or task failure)
raises an ALERT file (self-clearing on recovery). Exits when the task is no longer Running.

    python3 scripts_plots/foldx_watch.py            # defaults: task 138, hourly, 2h stall
    FXWATCH_TASK=138 FXWATCH_INTERVAL=3600 FXWATCH_STALL_HRS=2 python3 scripts_plots/foldx_watch.py

Review:  cat scratch/foldx_watch.status   ·   test -f scratch/foldx_watch.ALERT && cat scratch/foldx_watch.ALERT
"""
import datetime
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = os.environ.get("FXWATCH_TASK", "138")
RESULTS = os.path.join(ROOT, os.environ.get("FXWATCH_RESULTS", "scratch/foldx_skempi_full/results_remainder"))
WORK = os.path.join(ROOT, os.environ.get("FXWATCH_WORK", "scratch/foldx_skempi_full/work_remainder"))
TOTAL = int(os.environ.get("FXWATCH_TOTAL", "159"))
INTERVAL = int(os.environ.get("FXWATCH_INTERVAL", "3600"))     # hourly
STALL_HRS = float(os.environ.get("FXWATCH_STALL_HRS", "2"))
STATUS = os.path.join(ROOT, "scratch/foldx_watch.status")
ALERT = os.path.join(ROOT, "scratch/foldx_watch.ALERT")


def task_status():
    try:
        d = json.loads(subprocess.check_output(["pueue", "status", "--json"], text=True))
        s = d["tasks"][TASK]["status"]
        if isinstance(s, str):
            return s, ""
        key = list(s.keys())[0]
        res = s[key].get("result", "") if isinstance(s[key], dict) else ""
        res = res if isinstance(res, str) else (list(res.keys())[0] if res else "")
        return key, res
    except Exception as e:
        return "UNKNOWN", str(e)[:60]


def done_set():
    return {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(RESULTS, "*.json"))}


def inprogress(done):
    """{complex: newest_mtime} for work dirs lacking a results JSON (active or wedged)."""
    out = {}
    for d in glob.glob(os.path.join(WORK, "*")):
        pdb = os.path.basename(d)
        if pdb in done or not os.path.isdir(d):
            continue
        mt = 0.0
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        mt = max(mt, e.stat().st_mtime)
                    except OSError:
                        pass
        except OSError:
            pass
        out[pdb] = mt
    return out


def newest_result_mtime(done):
    m = 0.0
    for f in glob.glob(os.path.join(RESULTS, "*.json")):
        try:
            m = max(m, os.path.getmtime(f))
        except OSError:
            pass
    return m


def write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


ACTIVE_WIN = float(os.environ.get("FXWATCH_ACTIVE_WIN_MIN", "20")) * 60  # a dir touched this recently = running


def cycle():
    now = time.time()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    state, res = task_status()
    done = done_set()
    ip = inprogress(done)               # {complex: newest_mtime} for all not-done (repairs pre-seeded)
    # Global activity clock = newest mtime anywhere in results ∪ work. Stays fresh while ANYTHING runs;
    # only goes stale if every worker is wedged -> the true stall signal (immune to the pre-seeded dirs).
    newest = max([newest_result_mtime(done)] + [mt for mt in ip.values() if mt > 0] or [0.0])
    idle_h = (now - newest) / 3600 if newest > 0 else -1
    # "active" = dirs touched within the window = what's actually being worked (~= --jobs); rest are pending.
    active = sorted(((p, mt) for p, mt in ip.items() if mt > 0 and (now - mt) <= ACTIVE_WIN),
                    key=lambda x: -x[1])
    pending = len(ip) - len(active)
    last_touched = max(ip.items(), key=lambda x: x[1])[0] if ip else None  # stuck candidate on stall

    lines = [f"[{ts}] task #{TASK}={state}{('/'+res) if res else ''}  done={len(done)}/{TOTAL}  "
             f"active={len(active)}  pending={pending}  idle={idle_h:.1f}h (stall>{STALL_HRS}h)"]
    for pdb, mt in active[:10]:
        lines.append(f"    running {pdb}: touched {(now-mt)/60:.0f} min ago")
    body = "\n".join(lines) + "\n"
    write(STATUS, body)

    running = state == "Running"
    stalled = running and idle_h >= 0 and idle_h > STALL_HRS
    failed = (state == "Done" and res and res != "Success") or state == "Failed"
    if stalled:
        write(ALERT, f"[{ts}] STALL: FoldX run #{TASK} idle {idle_h:.1f}h (>{STALL_HRS}h) — no file "
                     f"touched anywhere. Likely wedged on the last-touched complex: {last_touched}. "
                     f"done={len(done)}/{TOTAL}. Inspect scratch/foldx_skempi_full/work_remainder/"
                     f"{last_touched}/foldx.log ; unstick by `pkill -f {last_touched}` or kill the "
                     f"worker so process_complex records it build_failed and moves on.\n\n" + body)
    elif failed:
        write(ALERT, f"[{ts}] TASK #{TASK} ended {state}/{res}\n\n" + body)
    elif os.path.exists(ALERT):
        os.remove(ALERT)          # self-clear once progress resumes / task healthy
    return state


def main():
    print(f"[foldx_watch] task #{TASK}, interval {INTERVAL}s, stall {STALL_HRS}h -> {STATUS}", flush=True)
    while True:
        try:
            state = cycle()
        except Exception as e:
            state = "Running"
            try:
                write(STATUS, f"[watch-error] {e}\n")
            except Exception:
                pass
        if state != "Running":
            print(f"[foldx_watch] task #{TASK} no longer Running ({state}); exiting.", flush=True)
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
