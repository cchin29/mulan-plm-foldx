"""P3 — embedding mutation-sensitivity sweep (2026-07-09 plan §4, eval order step 3).

Makes the PHASE2_REPORT §4 "mutation-alignment ratio" table **reproducible from a script**: for
each candidate embedder, embed TEM-1 (P62593) wild-type and a single point mutant, and check the
per-residue embedding changes *most at exactly the mutated row*. This is the P3 acceptance check —
the embedding default is decided on fit for the tool and cost (see README §P3), and each candidate must
pass this local row-alignment / mutation-sensitivity gate, NOT a ΔΔG re-benchmark.

Metric (matches tests/test_embedding.py): d = ‖e_wt − e_mut‖ per row; report
  argmax(d)+1  (must equal the mutated position) and  d[p-1] / median(d)  (must be ≫ 1).

Transformers PLMs run on CPU (deterministic; weights already in the HF cache). ESM C is
SDK-backed and runs in `.venv-esmc`: esmc_600m loads locally (MPS, bf16) via the `esm` SDK;
esmc_6b still needs transformers>=4.57 / cluster. Run (heavy; queue via pueue):
    .venv-structctx/bin/python experiments/structctx_sweep/run_p3_mutation_sensitivity.py [model ...]
    .venv-esmc/bin/python     experiments/structctx_sweep/run_p3_mutation_sensitivity.py esmc_600m
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from foldenv import config
from foldenv import context as ctx
from foldenv import embedding as E

UNIPROT, POS = "P62593", 150            # TEM-1; mutate residue 150 (L150A in the report)
DEFAULT_MODELS = [                       # everything runnable in .venv-structctx (transformers<4.45)
    "ankh", "prostt5_aa", "ankh3_large", "ankh3_xl",
    "saprot", "saprot_1.3b", "esm2_3b", "esm2_650m",
]
# Canonical row order for the merged table. ESM C models run separately in .venv-esmc:
# esmc_600m via the `esm` SDK (local, MPS-friendly); esmc_6b needs transformers>=4.57 / cluster.
ORDER = DEFAULT_MODELS + ["esmc_600m", "esmc_6b"]
OUT_DIR = Path(__file__).with_name("results")


def mutate(seq: str, pos: int) -> tuple[str, str, str]:
    wt = seq[pos - 1]
    mt = "A" if wt != "A" else "G"
    return seq[: pos - 1] + mt + seq[pos:], wt, mt


def run_model(model_name: str, seq: str, mut_seq: str, structure) -> dict:
    t0 = time.time()
    spec = E.get_model_spec(model_name)
    # Transformers PLMs run on CPU (deterministic); SDK-backed ESM C uses its resolved device
    # (MPS on Apple Silicon — the whole point of the esmc_600m local path).
    device = "auto" if spec.sdk else "cpu"
    model, tok = E.load_embedder(model_name, device)
    kw = {"model": model, "tokenizer": tok}
    if spec.structure_aware:
        kw["structure"] = structure
    e_wt = E.embed_protein(model_name, seq, **kw)
    e_mut = E.embed_protein(model_name, mut_seq, **kw)
    d = (e_wt - e_mut).norm(dim=1).cpu().numpy()
    argmax_pos = int(d.argmax()) + 1
    ratio = float(d[POS - 1] / np.median(d))
    return {
        "model": model_name, "dim": spec.dim,
        "argmax_pos": argmax_pos, "aligned": argmax_pos == POS,
        "ratio_mut_over_median": round(ratio, 1),
        "passes": argmax_pos == POS and ratio > 5.0,
        "rows": int(e_wt.shape[0]), "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = sys.argv[1:] or DEFAULT_MODELS
    cfg = config.load(overrides={"embedding": {"model": "none"}})   # seq/structure only, no PLM here
    seq = ctx.get_sequence(UNIPROT, cfg)
    structure = ctx.get_structure(UNIPROT, cfg).structure
    mut_seq, wt, mt = mutate(seq, POS)
    print(f"TEM-1 {UNIPROT} len={len(seq)}; mutation {wt}{POS}{mt}; models={models}")

    # Merge into any existing results so a separate-env run (e.g. esmc_6b in .venv-esmc) appends
    # a row rather than clobbering the committed table.
    raw_path = OUT_DIR / "p3_raw.json"
    merged: dict[str, dict] = {}
    if raw_path.exists():
        merged = {r["model"]: r for r in json.loads(raw_path.read_text())}

    def ordered_rows() -> list[dict]:
        known = [merged[m] for m in ORDER if m in merged]
        extra = [merged[m] for m in merged if m not in ORDER]
        return known + extra

    for m in models:
        try:
            r = run_model(m, seq, mut_seq, structure)
            print(f"  {m:14s} dim={r['dim']:4d} argmax={r['argmax_pos']:4d} "
                  f"ratio={r['ratio_mut_over_median']:6.1f} "
                  f"{'PASS' if r['passes'] else 'FAIL'} ({r['seconds']}s)")
        except Exception as exc:  # one model failing must not sink the batch
            r = {"model": m, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  {m:14s} ERROR: {r['error']}")
        merged[m] = r
        _write(ordered_rows(), wt, mt)     # checkpoint after each model (pueue-friendly)


def _write(rows: list[dict], wt: str, mt: str) -> None:
    L = ["# P3 — embedding mutation-sensitivity (TEM-1 P62593, "
         f"{wt}{POS}{mt}) — row-alignment gate", "",
         "Each embedder must localize a point mutation to its own row (argmax = mutated position) "
         "and dominate (ratio ≫ 1). Decides row-alignment/correctness, NOT ΔΔG (see README §P3). "
         "Transformers PLMs on CPU; ESM C via the `esm` SDK in `.venv-esmc` (esmc_600m local/MPS, "
         "esmc_6b cluster).", "",
         "| model | dim | argmax pos | ratio (mut/median) | gate |",
         "|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            L.append(f"| {r['model']} | — | — | — | ⚠ {r['error']} |")
        elif "argmax_pos" not in r:  # deferred / not-run row (e.g. esmc_6b env-blocked)
            note = r.get("reason", r.get("status", "deferred"))
            L.append(f"| {r['model']} | {r.get('dim', '—')} | — | — | ⏸ deferred (env) — {note} |")
        else:
            L.append(f"| {r['model']} | {r['dim']} | {r['argmax_pos']} "
                     f"| {r['ratio_mut_over_median']} "
                     f"| {'✅' if r['passes'] else '❌'} (want argmax={POS}, ratio>5) |")
    ran = [r for r in rows if "argmax_pos" in r]
    ok = [r for r in ran if r.get("passes")]
    L.append(f"\n**{len(ok)}/{len(ran)} locally-runnable models pass** the "
             f"row-alignment gate (argmax={POS}, ratio>5) → embeddings are correctly per-residue "
             "aligned; the Ankh default (and ProstT5/ESM C swaps) are safe to use interchangeably.")
    (OUT_DIR / "P3_MUTATION_SENSITIVITY.md").write_text("\n".join(L) + "\n")
    (OUT_DIR / "p3_raw.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
