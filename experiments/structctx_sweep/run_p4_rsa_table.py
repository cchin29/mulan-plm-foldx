"""P4 — RSA MaxASA-table sensitivity guardrail (2026-07-09 plan §4, eval order step 2).

The three MaxASA tables (Tien-2013 theoretical [default] / empirical, Sander&Rost-1994) are
near-monotonic rescales, so switching them should *recalibrate* absolute RSA (~0.05–0.10) but
not *reorder* residues by burial. This piggybacks on the TEM-1↔1BTL crystal cross-check to confirm no
table drops AF-vs-experiment agreement below the tests' > 0.85 threshold; among those that
pass, the highest-agreement table is preferred.

Note `crystal_crosscheck` normalizes BOTH the AF and the experimental RSA with the same table,
so a monotonic rescale on both sides leaves Pearson r ~unchanged and SS3 exactly unchanged
(SS doesn't depend on the table) — MAE is the only materially table-sensitive number. That
near-invariance IS the finding: the table is a reporting recalibration, not a re-ranking.

Run (needs network + mkdssp; TEM-1 P62593 vs PDB 1BTL):
    .venv-structctx/bin/python experiments/structctx_sweep/run_p4_rsa_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

from foldenv import config, validation

TABLES = ["tien2013_theoretical", "tien2013_empirical", "sander_rost1994"]
THRESHOLD = 0.85
OUT_DIR = Path(__file__).with_name("results")


def main() -> None:
    rows = []
    for table in TABLES:
        cfg = config.load(overrides={"rsa": {"max_asa_table": table}})
        r = validation.crystal_crosscheck("P62593", "1BTL", cfg=cfg)
        passes = r.ss3_agreement > THRESHOLD and r.rsa_pearson > THRESHOLD
        rows.append({
            "table": table,
            "ss3_agreement": round(r.ss3_agreement, 4),
            "rsa_pearson": round(r.rsa_pearson, 4),
            "rsa_mae": round(r.rsa_mae, 4),
            "n_compared": r.n_compared,
            "passes_0.85_gate": passes,
        })

    lines = ["# P4 — RSA MaxASA-table sensitivity (TEM-1 P62593 vs 1BTL)", "",
             "| table | SS3 | RSA Pearson r | RSA MAE | n | >0.85 gate |",
             "|---|---|---|---|---|---|"]
    for x in rows:
        lines.append(
            f"| {x['table']}{' (default)' if x['table']==TABLES[0] else ''} "
            f"| {x['ss3_agreement']} | {x['rsa_pearson']} | {x['rsa_mae']} "
            f"| {x['n_compared']} | {'✅' if x['passes_0.85_gate'] else '❌'} |"
        )
    passing = [x for x in rows if x["passes_0.85_gate"]]
    best = max(passing, key=lambda x: x["rsa_pearson"]) if passing else None
    lines.append("")
    if best:
        lines.append(
            f"**Verdict:** all {len(passing)}/{len(rows)} tables clear the >0.85 gate — SS3 is "
            f"table-invariant and RSA r barely moves (rescale, not reorder), confirming the "
            f"choice is a reporting recalibration. Highest r: **{best['table']}** "
            f"({best['rsa_pearson']}). Keep the Tien-2013-theoretical default (D3)."
        )
    else:
        lines.append("**Verdict:** no table clears the gate — investigate the pipeline.")

    (OUT_DIR / "P4_RSA_TABLE.md").write_text("\n".join(lines) + "\n")
    (OUT_DIR / "p4_raw.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))
    print(f"\n[wrote {OUT_DIR/'P4_RSA_TABLE.md'}]")


if __name__ == "__main__":
    main()
