"""Aggregate the embedding-sweep runs into tracked metric files.

Reads per-fold `all_results.json` from the (gitignored) results dir and writes small,
committed artifacts to ES_METRICS_DIR: one `<tag>.json` per model (per-fold + mean/std)
and a `summary.md` master table vs the paper's Table 1. Run from the repo root.

    python experiments/embedding_sweep/aggregate.py            # all models present
    python experiments/embedding_sweep/aggregate.py ankh esm2  # subset
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Paper Table 1, S1102 mutation-based (models not in the paper are shown with "-").
PAPER = {
    "ankh": {"pcc": 0.868, "rmse": 1.185},
    "esm2": {"pcc": 0.854, "rmse": 1.223},
}


def load_config(path=None, cfg=None):
    """Parse `export KEY=VALUE` lines, following `source <path>` first so a sibling
    config (e.g. config_balanced.sh) can source the base config then override keys.
    Honors ES_CONFIG (default: config.sh next to this file)."""
    if path is None:
        path = os.environ.get("ES_CONFIG", os.path.join(HERE, "config.sh"))
    cfg = {} if cfg is None else cfg
    with open(path) as f:
        for line in f:
            src = re.match(r"\s*(?:source|\.)\s+(\S+)", line)
            if src:
                load_config(src.group(1).strip().strip('"').strip("'"), cfg)
                continue
            m = re.match(r"\s*export\s+(\w+)=(.*)", line)
            if m:
                cfg[m.group(1)] = m.group(2).split("#")[0].strip().strip('"').strip("'")
    return cfg


def registry_tags(cfg):
    tags = []
    with open(cfg["ES_MODELS_TSV"]) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tags.append(line.split("\t")[0])
    return tags


def fold_metrics(fold_parent, num_folds):
    """Return (rows, missing): per-fold metrics for every fold whose all_results.json
    loaded, and the list of fold indices that were absent or unreadable. Callers use
    `missing` (and a null test_pcc within a row) to detect a truncated or no-op sweep
    instead of silently averaging over whatever folds happened to land. `fold_parent`
    is the directory that directly contains the `fold_*` subdirs."""
    rows, missing = [], []
    for k in range(num_folds):
        p = f"{fold_parent}/fold_{k}/training_run/all_results.json"
        try:
            r = json.load(open(p))
            rows.append({"fold": k, "pcc": r.get("test_pcc"), "rmse": r.get("test_rmse")})
        except Exception:
            missing.append(k)
    return rows, missing


def resolve_arg(cfg, arg):
    """Map a positional arg to (display_tag, fold_parent). A plain registry tag
    (`ankh`, `esm2`, ...) resolves under ES_RESULTS_DIR exactly as before. An explicit
    path to a directory holding `fold_*` subdirs — detected when the arg contains a `/`,
    is an existing dir, or has a `fold_0/training_run/all_results.json` — is treated as
    the fold-parent itself (e.g. the A1 grid's `scratch/results/a1_<tag>/<arm>/<tag>`).
    For an explicit path the display tag folds in the arm (`<leaf>@<arm>`) so the summary
    row is identifiable, two arms don't collide, and it can't clobber a standard `<tag>`."""
    fold0 = os.path.join(arg, "fold_0", "training_run", "all_results.json")
    if ("/" in arg) or os.path.isdir(arg) or os.path.exists(fold0):
        parent = arg.rstrip("/")
        leaf = os.path.basename(parent)
        arm = os.path.basename(os.path.dirname(parent))
        tag = f"{leaf}@{arm}" if arm else leaf
        return tag, parent
    return arg, os.path.join(cfg["ES_RESULTS_DIR"], arg)


def main():
    cfg = load_config()
    nfolds = int(cfg["ES_NUM_FOLDS"])
    os.makedirs(cfg["ES_METRICS_DIR"], exist_ok=True)
    tags = sys.argv[1:] or registry_tags(cfg)

    table = []
    warnings = []
    for arg in tags:
        tag, fold_parent = resolve_arg(cfg, arg)
        rows, missing = fold_metrics(fold_parent, nfolds)
        if not rows:
            msg = (f"{tag}: 0/{nfolds} folds found — SKIPPED. Likely a silent no-op "
                   f"(unknown-PLM tag) or wrong fold-parent dir ({fold_parent}).")
            print(f"!! {msg}", file=sys.stderr)
            warnings.append(msg)
            continue
        pcc = [r["pcc"] for r in rows if r["pcc"] is not None]
        rmse = [r["rmse"] for r in rows if r["rmse"] is not None]
        # A run is only trustworthy if EVERY expected fold produced a test_pcc. Flag
        # (don't silently average) any shortfall so a truncated sweep can't be blessed
        # as a complete result — the metrics are still written, but marked provisional.
        complete = (not missing) and (len(pcc) == nfolds)
        if not complete:
            null_pcc = [r["fold"] for r in rows if r["pcc"] is None]
            msg = (f"{tag}: PARTIAL — {len(pcc)}/{nfolds} folds have test_pcc "
                   f"(missing folds {missing or '[]'}, null-pcc folds {null_pcc or '[]'}); "
                   f"mean over a partial set — treat as provisional.")
            print(f"!! {msg}", file=sys.stderr)
            warnings.append(msg)
        summary = {
            "tag": tag,
            "n_folds": len(rows),
            "expected_folds": nfolds,
            "complete": complete,
            "pcc_mean": float(np.mean(pcc)) if pcc else None,
            "pcc_std": float(np.std(pcc)) if pcc else None,
            "rmse_mean": float(np.mean(rmse)) if rmse else None,
            "rmse_std": float(np.std(rmse)) if rmse else None,
            "folds": rows,
            "setup": {k: cfg[k] for k in ("ES_SPLIT_METHOD", "ES_SEED", "ES_NUM_EPOCHS",
                                          "ES_PATIENCE", "ES_MODEL_CONFIG")},
        }
        with open(os.path.join(cfg["ES_METRICS_DIR"], f"{tag}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        table.append(summary)

    # summary.md master table
    lines = [
        "# Embedding sweep — S1102 (paper-faithful CV)",
        "",
        f"Setup: split_method=`{cfg['ES_SPLIT_METHOD']}` seed={cfg['ES_SEED']} "
        f"epochs={cfg['ES_NUM_EPOCHS']} patience={cfg['ES_PATIENCE']} "
        f"config=`{os.path.basename(cfg['ES_MODEL_CONFIG'])}`. "
        f"Data: {cfg['ES_BASENAME']} (1100). Paper target: Ankh 0.868 / ESM2-3B 0.854.",
        "",
        "| model | n | PCC (mean±std) | RMSE (mean±std) | paper PCC | Δ PCC |",
        "|---|---|---|---|---|---|",
    ]
    def f4(x):
        return "—" if x is None else f"{x:.4f}"

    def f3(x):
        return "—" if x is None else f"{x:.3f}"

    for s in sorted(table, key=lambda x: -(x["pcc_mean"] or 0)):
        pap = PAPER.get(s["tag"])
        ppcc = f"{pap['pcc']:.3f}" if pap else "—"
        dpcc = (f"{s['pcc_mean']-pap['pcc']:+.3f}"
                if pap and s["pcc_mean"] is not None else "—")
        # Mark a partial run in the n cell so the table itself flags it (⚠ n/expected).
        ncell = (str(s["n_folds"]) if s["complete"]
                 else f"{s['n_folds']}/{s['expected_folds']} ⚠")
        lines.append(
            f"| {s['tag']} | {ncell} | {f4(s['pcc_mean'])}±{f3(s['pcc_std'])} | "
            f"{f3(s['rmse_mean'])}±{f3(s['rmse_std'])} | {ppcc} | {dpcc} |"
        )
    if warnings:
        lines += ["", "> ⚠ **Incomplete/skipped models** (metrics provisional):"]
        lines += [f"> - {w}" for w in warnings]
    out = os.path.join(cfg["ES_METRICS_DIR"], "summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(table)} model metric file(s) + {out}")
    print("\n".join(lines[4:]))
    if warnings:
        print(f"\n!! {len(warnings)} model(s) incomplete/skipped — see warnings above.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
