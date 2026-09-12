"""P1+P2 decisions sweep for `get_structural_context` (2026-07-09 plan §4, eval order step 1).

Sweeps the two highest-value contact knobs on the functional-site set:
  * P1 — D2 pLDDT contact mask: {50 (default), 70, none}
  * P2 — D1 contact primary:    {Cα ≤ 8 Å (default), Cβ ≤ 5 Å}
= a 3 × 2 grid, on TEM-1 (P62593) and TP53 (P04637).

RSA is DSSP-derived and depends only on the MaxASA table (fixed here), so it is IDENTICAL
across all six configs — the sweep is entirely about `contact_count` (the packing signal the
tool exists for). The embedding is disabled (`model: none`): the structural fields need no PLM.

Per 2026-07-09 plan §4 "Eval criteria", for each (protein, config) we report:
  1. the functional sites' RSA/contact + percentile ranks (the honest descriptive primary);
  2. a physical-sanity gate — median contact of the buried core (RSA<0.20) must stay ~10–20,
     surface (RSA>0.25) ~4–8 (a config that over-masks the core below ~10 fails);
  3. AUROC(contact_count → functional) and Cliff's δ (functional vs buried-core) — DIRECTIONAL
     only (n=3 sites/protein), reported with that caveat;
  4. Spearman ρ of per-residue contact_count vs the baseline (Cα/50) — rescale vs reorder.

Decision rule (from the plan): switch a default only if an alternative improves separation
consistently on BOTH proteins; a one-protein gain is noise.

Run (needs network + mkdssp; DSSP is disk-cached across configs by the new persistence layer):
    .venv-structctx/bin/python experiments/structctx_sweep/run_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from foldenv import analysis, config
from foldenv import context as ctx

PROTEINS = ["P62593", "P04637"]  # TEM-1, TP53
MASKS = [("50", 50.0), ("70", 70.0), ("none", 0.0)]  # 0.0 → nothing masked (pLDDT ≥ 0)
PRIMARIES = [("ca8", "ca"), ("cb5", "cb")]
BASELINE = "ca8/50"

OUT_DIR = Path(__file__).with_name("results")


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney rank formula; higher score ⇒ more likely positive.

    NaN if a class is empty. Ties get midranks (proper AUROC under ties).
    """
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # midranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's δ = P(a>b) − P(a<b) ∈ [−1, 1]; NaN if either group is empty."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    diff = np.sign(a[:, None] - b[None, :])
    return float(diff.mean())


def run_config(uniprot_id: str, mask_val: float, primary: str) -> dict:
    cfg = config.load(
        overrides={
            "embedding": {"model": "none"},
            "contacts": {"primary": primary},
            "plddt": {"mask_below": mask_val},
        }
    )
    profile = ctx.structural_profile(uniprot_id, cfg)
    positions = sorted(profile)
    cc = np.array([profile[p]["contact_count"] for p in positions], float)
    rsa = np.array(
        [profile[p]["rsa"] if profile[p]["rsa"] is not None else np.nan for p in positions],
        float,
    )

    sites = analysis.functional_site_stats(uniprot_id, cfg=cfg)
    func_positions = {s.position for s in sites}

    # physical-sanity gate: contact bands for the buried core vs the surface
    buried_thr = cfg["rsa"]["buried_threshold"]      # 0.20
    exposed_thr = cfg["rsa"]["exposed_threshold"]    # 0.25
    core_cc = cc[(rsa < buried_thr)]
    surf_cc = cc[(rsa > exposed_thr)]

    # discrimination (directional, n=3): functional vs bulk, and vs buried core
    labels = np.array([1 if p in func_positions else 0 for p in positions])
    func_cc = cc[labels == 1]

    return {
        "uniprot_id": uniprot_id,
        "mask": mask_val,
        "primary": primary,
        "positions": positions,          # for the cross-config Spearman join
        "contact_count": cc.tolist(),
        "sites": [
            {
                "position": s.position, "aa": s.aa, "rsa": s.rsa,
                "rsa_percentile": s.rsa_percentile,
                "contact_count": s.contact_count,
                "contact_percentile": s.contact_percentile,
                "buried": s.buried,
            }
            for s in sites
        ],
        "sanity": {
            "n_residues": len(positions),
            "core_median_cc": float(np.median(core_cc)) if len(core_cc) else float("nan"),
            "surf_median_cc": float(np.median(surf_cc)) if len(surf_cc) else float("nan"),
            "core_n": int(len(core_cc)), "surf_n": int(len(surf_cc)),
        },
        "discrimination": {
            "auroc_contact_vs_bulk": _auroc(cc, labels),
            "cliffs_delta_func_vs_core": _cliffs_delta(func_cc, core_cc),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, dict]] = {p: {} for p in PROTEINS}

    for uid in PROTEINS:
        for mlabel, mval in MASKS:
            for plabel, primary in PRIMARIES:
                key = f"{plabel}/{mlabel}"
                res = run_config(uid, mval, primary)
                results[uid][key] = res
                path = OUT_DIR / f"{uid}__{plabel}__plddt{mlabel}.json"
                path.write_text(json.dumps({k: v for k, v in res.items()
                                            if k not in ("contact_count", "positions")}, indent=2))
        # Spearman ρ of per-residue contact_count vs baseline (same positions, same order)
        base_cc = np.array(results[uid][BASELINE]["contact_count"], float)
        for key, res in results[uid].items():
            rho = spearmanr(base_cc, np.array(res["contact_count"], float)).statistic
            res["spearman_vs_baseline"] = float(rho)

    _report(results)
    (OUT_DIR / "sweep_raw.json").write_text(json.dumps(
        {uid: {k: {kk: vv for kk, vv in v.items() if kk != "contact_count"}
               for k, v in cfgs.items()} for uid, cfgs in results.items()}, indent=2))


def _report(results: dict) -> None:
    name = {"P62593": "TEM-1 (P62593)", "P04637": "TP53 (P04637)"}
    lines: list[str] = ["# P1+P2 structural-context sweep — results", ""]
    lines.append("RSA is table-fixed → identical across configs; only `contact_count` varies. "
                 "AUROC/Cliff's δ are **directional** (n=3 sites/protein).\n")

    for uid in PROTEINS:
        lines.append(f"## {name[uid]}\n")
        # functional-site contact_count across configs
        lines.append("### Functional-site contact_count (percentile) by config\n")
        site_ids = [s["position"] for s in results[uid][BASELINE]["sites"]]
        aas = {s["position"]: s["aa"] for s in results[uid][BASELINE]["sites"]}
        header = "| config | " + " | ".join(f"{aas[p]}{p}" for p in site_ids) + \
                 " | core_med | surf_med | AUROC | δ(func-core) | ρ_vs_base |"
        lines.append(header)
        lines.append("|" + "---|" * (len(site_ids) + 6))
        for key, res in results[uid].items():
            byp = {s["position"]: s for s in res["sites"]}
            cells = []
            for p in site_ids:
                s = byp[p]
                cells.append(f"{s['contact_count']} ({s['contact_percentile']:.2f})")
            san, disc = res["sanity"], res["discrimination"]
            lines.append(
                f"| {key} | " + " | ".join(cells) +
                f" | {san['core_median_cc']:.0f} | {san['surf_median_cc']:.0f}"
                f" | {disc['auroc_contact_vs_bulk']:.2f}"
                f" | {disc['cliffs_delta_func_vs_core']:+.2f}"
                f" | {res['spearman_vs_baseline']:.3f} |"
            )
        # RSA (config-invariant) shown once
        lines.append("\n**RSA / burial (config-invariant):** " + ", ".join(
            f"{aas[s['position']]}{s['position']} rsa={s['rsa']:.3f} "
            f"(pctile {s['rsa_percentile']:.2f}, {'buried' if s['buried'] else 'exposed'})"
            for s in results[uid][BASELINE]["sites"]) + "\n")

    out = OUT_DIR / "SWEEP_RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[wrote {out} and per-config JSON in {OUT_DIR}]")


if __name__ == "__main__":
    main()
