#!/usr/bin/env python3
"""Tier-1-augmented FoldX splits (reverse-mutation + identity anchors) for the full-SKEMPI
FoldX arms — the FoldX×aug combo enabler for CATH and clustered(-all).

`experiments/augment.py` builds Tier-1 aug for the *base* (4-col) tables; the FoldX arms need each
augmented row to also carry its FoldX ΔΔG channel. FoldX binding ΔΔG is antisymmetric term-for-term
(ΔΔG(mut→wt) = −ΔΔG(wt→mut)), so a reverse row's 12-term vector is the *negated* forward vector, and
an identity (wt→wt) row's vector is zero. Single AND multi-point rows are reversed (unlike
augment.py, which skips multi) — required because the clustered split is 100% multi-point.

Rather than re-run the FoldX merge (which would have to reproduce the dual-key coverage fix), we read
the CANONICAL standardized foldxdec split (`--dec-src`) verbatim for the forward + val/test rows — so
coverage and values are *identical* to the reference `foldx`/`foldx_scalar` arms — and only synthesize
the reverse and identity train rows. The scalar arm is term-0 of the decomposed vector (Interaction
Energy), matching how the canonical `splits_*_foldx` were built (verified byte-identical).

Negation must happen in RAW units, not standardized ones
--------------------------------------------------------
The channel the split carries is z(x) = clip((x − mu)/sigma). A reverse row should carry the
standardized form of the negated raw value, z(−x), and that is NOT the negation of the standardized
forward value:

    z(-x) = (-x - mu)/sigma        what a reverse row must carry
    -z(x) = (mu - x)/sigma         what negating the standardized value gives
    -z(x) - z(-x) = 2*mu/sigma     a CONSTANT offset on every reverse row

`mu` is the raw train mean of each FoldX term and is not near zero — most mutations are destabilising
— so this is large: +0.88 to +1.12 sd on Interaction Energy, i.e. about one full channel width, on
every synthesized row (51% of the augmented training set). Until 2026-08-05 this script wrote `-z(x)`
and justified it with "each term is already train-centered at ~0 by the canonical standardization".
Train-centring makes the *standardized* values average zero, which is true and is not the question.
The label was always handled correctly (`-float(label)` on a raw kcal/mol value), so the defect put
the label and the channel in opposite directions on half the training data. See
`experiments/beyond_foldx/FOLDX_ANTISYMMETRY_RESULT.md` §1.

Identity (wt→wt) rows have the same defect: their true FoldX vector is raw zero, whose standardized
form is −mu/sigma, not 0. Writing 0 encoded them as "an average mutation" — and collided with the
0 the canonical merge writes for rows FoldX never scored.

Recovering (mu, sigma). The canonical merges do not persist their constants, so they are recovered by
regressing the committed standardized channel on the raw FoldX terms in `--results`, using unclipped
rows only. The relation is exactly linear, so this is reconstruction rather than estimation, and it is
verified as such: the recovered constants must reproduce every matched committed value to within
`--tol` (default 2e-5, i.e. the split's own 5-decimal rounding) or the build aborts. The constants are
written to `standardization.json` in `--out-dec`.

Antisymmetry itself is only approximate — reverting a mutation recovers ~0.78 of the forward effect,
not 1.0 (FOLDX_ANTISYMMETRY_RESULT.md §2). That is a property of FoldX, not of this script, and is not
repairable here; the only fix is to compute the reverse rows with FoldX instead of deriving them.

Two modes (build needs only the canonical dec split + WT fasta; materialize needs the embedding cache
→ run on the GPU box):

  build (default):  writes <out-scalar>/ <out-dec>/ (train augmented; val/test verbatim) + <out-fasta>
  --materialize:    reads <out-dec> train tsvs + --emb-dir, symlinks every missing embedding id to its
                    WT-chain base `{complex}_{chain}.pt` (reverse-mutant / identity / empty-side seqs
                    are all byte-identical to a WT chain → no PLM load), then reports any residual miss.

Build (CATH, on the Linux box):
  python experiments/full_skempi_seqonly/merge_foldx_aug.py \
    --dec-src scratch/foldx_skempi_full/splits_cath_foldxdec --base skempi_all \
    --fasta scratch/skempi_full/wt_sequences_cath_all.fasta \
    --out-scalar scratch/foldx_skempi_full/splits_cath_foldx_aug \
    --out-dec    scratch/foldx_skempi_full/splits_cath_foldxdec_aug \
    --out-fasta  scratch/skempi_full/wt_sequences_cath_all_aug.fasta
Materialize (GPU box, per model emb-dir):
  python .../merge_foldx_aug.py --materialize --base skempi_all \
    --out-dec scratch/foldx_skempi_full/splits_cath_foldxdec_aug \
    --emb-dir scratch/embeddings_skempi_full/esmc6b
"""
import argparse, glob, json, os

NTERMS = 12
STANDARD = set("ACDEFGHIKLMNPQRSTVWY")

# Must match merge_foldx_full_skempi.py exactly — order fixes which column is which term, and CLIP
# is applied on write, so a reverse row saturates the same bound a real row would.
TERMS = [
    "Interaction Energy", "Backbone Hbond", "Sidechain Hbond", "Van der Waals",
    "Electrostatics", "Solvation Polar", "Solvation Hydrophobic",
    "Van der Waals clashes", "entropy sidechain", "entropy mainchain",
    "torsional clash", "backbone clash",
]
CLIP = 4.0


def read_tsv(path):
    with open(path) as f:
        return [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]


def rev_mut(m):
    """DA11A -> AA11D  (swap wt/mut AA, keep canonical chain + pos)."""
    return f"{m[-1]}{m[1]}{m[2:-1]}{m[0]}"


def apply_muts(seq, muts):
    s = list(seq)
    for m in muts:
        s[int(m[2:-1]) - 1] = m[-1]
    return "".join(s)


def base_label(idv):
    """WT-chain base of any embedding id: `{complex}_{chain}` = first two '_'-tokens
    (complex names carry dots not underscores; chain is one token)."""
    return "_".join(idv.split("_")[:2])


# ------------------------------------------------------- standardization constant recovery
def load_raw_terms(results_glob):
    """(pdb, mut) -> the 12 raw FoldX terms, under both the JSON key and its `cleaned` alias.

    Union over every matching results dir. A key present in several dirs with different values would
    corrupt the fit, but cannot pass verify_constants() silently — that checks every matched row.
    """
    raw = {}
    for path in sorted(glob.glob(results_glob)):
        pdb = os.path.basename(path)[:-5]
        try:
            muts = json.load(open(path))["muts"]
        except Exception:
            continue
        for k, v in muts.items():
            if any(t not in v for t in TERMS):
                continue
            vec = [float(v[t]) for t in TERMS]
            raw.setdefault((pdb, k), vec)
            if v.get("cleaned"):
                raw.setdefault((pdb, v["cleaned"]), vec)
    return raw


def recover_constants(train_rows, raw, tol):
    """Reconstruct the (mu, sigma) the canonical merge standardized this fold's train split with.

    z = clip((x - mu)/sigma) is exactly linear off the clip bounds, so fitting z on x over unclipped
    rows RECONSTRUCTS the constants rather than estimating them. Only the rows whose (pdb, mut) key
    resolves against `raw` are usable — the canonical merge joins through a chain+resnum remap this
    script deliberately does not reproduce, so that is well under half of them, and it does not need
    to be more: two clean points determine a line.

    Returns (mu[12], sigma[12], diag). Raises if any matched row fails to reproduce within `tol`.
    """
    import numpy as np
    pairs = [(raw[k], z) for k, z in
             (((r[0].split(".")[0], r[2]), [float(x) for x in r[4:4 + NTERMS]]) for r in train_rows)
             if k in raw and any(x != 0.0 for x in z)]
    if len(pairs) < 50:
        raise SystemExit(f"only {len(pairs)} train rows resolve against --results; "
                         "cannot recover the standardization constants")
    X = np.array([p[0] for p in pairs])
    Z = np.array([p[1] for p in pairs])
    mu, sigma, worst, bad = [], [], 0.0, 0
    for j in range(NTERMS):
        free = np.abs(Z[:, j]) < CLIP - 1e-9
        if free.sum() < 10:
            raise SystemExit(f"term {j}: only {int(free.sum())} unclipped rows, cannot solve")
        slope, intercept = np.polyfit(X[free, j], Z[free, j], 1)
        # A mis-joined key shows up as a point off an otherwise exact line; drop and refit once.
        resid = np.abs(X[free, j] * slope + intercept - Z[free, j])
        keep = resid < 1e-3
        if keep.sum() < free.sum():
            bad += int(free.sum() - keep.sum())
            idx = np.where(free)[0][keep]
            slope, intercept = np.polyfit(X[idx, j], Z[idx, j], 1)
        s = 1.0 / slope
        mu.append(float(-intercept * s))
        sigma.append(float(s))
        err = np.abs(np.round(np.clip((X[:, j] - mu[j]) / sigma[j], -CLIP, CLIP), 5) - Z[:, j])
        worst = max(worst, float(err.max()))
        if err.max() > tol:
            raise SystemExit(
                f"term {j} ({TERMS[j]}): recovered constants reproduce the committed channel only to "
                f"{err.max():.2e} (tol {tol:.0e}) over {len(err)} rows — refusing to write reverse "
                f"rows from constants that do not describe the forward ones")
    return mu, sigma, dict(n_matched=len(pairs), max_err=worst, dropped=bad)


def row_emb_ids(s1, s2, muts):
    """The 4 embedding ids MuLAN loads for a row (matches data.py _fill_metadata)."""
    mA = "-".join(m for m in muts if m[1] == "A")
    mB = "-".join(m for m in muts if m[1] == "B")
    return [s1, s2, f"{s1}_{mA}", f"{s2}_{mB}"]


# ----------------------------------------------------------------------------- build
def build(a):
    from mulan import utils
    fasta = dict(utils.parse_fasta(a.fasta))
    aug_fasta = dict(fasta)
    folds = sorted(glob.glob(os.path.join(a.dec_src, "fold_*")))
    if not folds:
        raise SystemExit(f"no fold_* under {a.dec_src}")
    raw = load_raw_terms(a.results)
    print(f"[aug-build] raw FoldX terms for {len(raw)} (pdb, mut) keys from {a.results}")
    n_rev = n_ident = n_multi = n_skip_long = n_saturated = 0

    def augment_train(rows, off, ident_vec):
        """off[j] = 2*mu[j]/sigma[j]; ident_vec[j] = -mu[j]/sigma[j] (raw zero, standardized)."""
        nonlocal n_rev, n_ident, n_multi, n_skip_long, n_saturated

        def reverse_vec(vec):
            """z(-x) from z(x), the forward row's committed channel.

            -z(x) - off recovers z(-x) exactly wherever z(x) is off the clip bound. Where the forward
            row saturated, x is unrecoverable and this uses the bound itself — the reverse value is
            then correct in sign and at least as extreme as written, which is the honest floor.
            """
            nonlocal n_saturated
            if not any(x != 0.0 for x in vec):
                return [0.0] * NTERMS          # FoldX never scored it; reverse is missing too
            n_saturated += any(abs(x) >= CLIP - 1e-9 for x in vec)
            return [max(-CLIP, min(CLIP, -x - off[j])) for j, x in enumerate(vec)]

        out = []                                   # (s1,s2,muts,label_str, vec[12])
        seen = set()
        for r in rows:
            s1, s2, muts_str, label = r[0], r[1], r[2], r[3]
            vec = [float(x) for x in r[4:4 + NTERMS]]
            muts = muts_str.split(",")
            out.append((s1, s2, muts_str, label, vec))                       # forward verbatim
            mutsA = [m for m in muts if m[1] == "A"]
            mutsB = [m for m in muts if m[1] == "B"]
            labelA = f"{s1}_{'-'.join(mutsA)}" if mutsA else s1
            labelB = f"{s2}_{'-'.join(mutsB)}" if mutsB else s2
            revA = [rev_mut(m) for m in mutsA]
            revB = [rev_mut(m) for m in mutsB]
            # MuLAN will form the reverse mut_seq labels as `{labelX}_{joined-reverse-muts}`;
            # skip reversal when either would exceed the 255-byte filename limit (huge multipoints).
            rlab1 = f"{labelA}_{'-'.join(revA)}"
            rlab2 = f"{labelB}_{'-'.join(revB)}"
            if max(len(rlab1), len(rlab2)) + 3 > 255:
                n_skip_long += 1
                continue
            if mutsA:
                aug_fasta[labelA] = apply_muts(fasta[s1], mutsA)
            if mutsB:
                aug_fasta[labelB] = apply_muts(fasta[s2], mutsB)
            rev_str = ",".join(rev_mut(m) for m in muts)
            out.append((labelA, labelB, rev_str, f"{-float(label)}", reverse_vec(vec)))
            n_rev += 1; n_multi += (len(muts) > 1)
        if not a.no_identity:
            for r in rows:
                s1, s2 = r[0], r[1]
                if s1 not in fasta or s2 not in fasta or (s1, s2) in seen:
                    continue
                seen.add((s1, s2))
                p = next((i for i, aa in enumerate(fasta[s1]) if aa in STANDARD), None)
                if p is None:
                    continue
                aa = fasta[s1][p]
                out.append((s1, s2, f"{aa}A{p + 1}{aa}", "0.0", list(ident_vec)))
                aug_fasta[f"{s1}_{aa}A{p + 1}{aa}"] = fasta[s1]
                n_ident += 1
        return out

    os.makedirs(a.out_scalar, exist_ok=True); os.makedirs(a.out_dec, exist_ok=True)
    tot = cov = 0
    stats = {}
    for i in range(len(folds)):
        fd = os.path.join(a.dec_src, f"fold_{i}")
        os.makedirs(os.path.join(a.out_scalar, f"fold_{i}"), exist_ok=True)
        os.makedirs(os.path.join(a.out_dec, f"fold_{i}"), exist_ok=True)
        train_rows = read_tsv(os.path.join(fd, f"{a.base}_train.tsv"))
        mu, sigma, diag = recover_constants(train_rows, raw, a.tol)
        off = [2 * mu[j] / sigma[j] for j in range(NTERMS)]
        ident_vec = [max(-CLIP, min(CLIP, -mu[j] / sigma[j])) for j in range(NTERMS)]
        stats[f"fold_{i}"] = dict(terms=TERMS, mean=mu, std=sigma, clip=CLIP,
                                  reverse_offset_sd=off, identity_z=ident_vec, **diag)
        print(f"[aug-build] fold {i}: constants recovered from {diag['n_matched']} joinable train "
              f"rows (max reproduction error {diag['max_err']:.1e}"
              + (f", {diag['dropped']} outlier joins dropped" if diag["dropped"] else "") + ")")
        print(f"[aug-build]   Interaction Energy: mu {mu[0]:+.4f}  sigma {sigma[0]:.4f}  "
              f"-> reverse rows corrected by {-off[0]:+.3f} sd, identity rows set to "
              f"{ident_vec[0]:+.3f} sd")
        for split in ("train", "val", "test"):
            rows = train_rows if split == "train" else read_tsv(os.path.join(fd, f"{a.base}_{split}.tsv"))
            if split == "train":
                recs = augment_train(rows, off, ident_vec)
            else:
                recs = [(r[0], r[1], r[2], r[3], [float(x) for x in r[4:4 + NTERMS]]) for r in rows]
            fs = open(os.path.join(a.out_scalar, f"fold_{i}", f"{a.base}_{split}.tsv"), "w")
            fdz = open(os.path.join(a.out_dec, f"fold_{i}", f"{a.base}_{split}.tsv"), "w")
            for s1, s2, m, lab, vec in recs:
                tot += 1; cov += any(x != 0.0 for x in vec)
                head = [s1, s2, m, lab]
                fs.write("\t".join(head + [f"{vec[0]:.5f}"]) + "\n")
                fdz.write("\t".join(head + [f"{x:.5f}" for x in vec]) + "\n")
            fs.close(); fdz.close()
    utils.dict_to_fasta(aug_fasta, a.out_fasta)
    with open(os.path.join(a.out_dec, "standardization.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"[aug-build] folds={len(folds)}  +{n_rev} reverse ({n_multi} multi) +{n_ident} identity"
          f"  (skipped {n_skip_long} oversized-multipoint reversals > 255-char filename)")
    print(f"[aug-build] rows {tot}  covered {cov} ({100 * cov / max(1, tot):.1f}%)  "
          f"fasta {len(fasta)} -> {len(aug_fasta)}")
    print(f"[aug-build] {n_saturated}/{n_rev} reverse rows came from a forward row that hit the "
          f"+/-{CLIP:g} clip in at least one term; those terms use the bound (see docstring)")
    print(f"[aug-build] constants -> {os.path.join(a.out_dec, 'standardization.json')}")
    print(f"[aug-build] scalar -> {a.out_scalar}\n[aug-build] decomp -> {a.out_dec}\n[aug-build] fasta -> {a.out_fasta}")


# ------------------------------------------------------------------------ materialize
def materialize(a):
    folds = sorted(glob.glob(os.path.join(a.out_dec, "fold_*")))
    made = residual = 0
    seen = set()
    for fd in folds:
        for r in read_tsv(os.path.join(fd, f"{a.base}_train.tsv")):
            for idv in row_emb_ids(r[0], r[1], r[2].split(",")):
                if idv in seen:
                    continue
                seen.add(idv)
                dst = os.path.join(a.emb_dir, f"{idv}.pt")
                if os.path.exists(dst):
                    continue
                base = base_label(idv)
                if not os.path.exists(os.path.join(a.emb_dir, f"{base}.pt")):
                    residual += 1
                    if residual <= 20:
                        print(f"[materialize] !! no WT base for {idv}.pt (base {base}.pt absent)")
                    continue
                if not os.path.lexists(dst):
                    os.symlink(f"{base}.pt", dst); made += 1
    print(f"[materialize] emb-dir {a.emb_dir}: symlinks made {made}, residual-missing {residual}")
    if residual == 0:
        print("[materialize] OK — every augmented-train embedding id resolves (no PLM load needed)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--materialize", action="store_true", help="symlink-only pass (needs --emb-dir + --out-dec)")
    ap.add_argument("--dec-src", help="canonical foldxdec split dir (16-col) to read forward+val/test from")
    ap.add_argument("--base", required=True)
    ap.add_argument("--fasta")
    ap.add_argument("--out-scalar"); ap.add_argument("--out-dec", required=True)
    ap.add_argument("--out-fasta"); ap.add_argument("--emb-dir")
    ap.add_argument("--no-identity", action="store_true")
    ap.add_argument("--results", default="scratch/foldx_skempi_full/results*/*.json",
                    help="glob of raw FoldX JSONs, used ONLY to recover the standardization "
                         "constants the canonical merge used (they are not persisted)")
    ap.add_argument("--tol", type=float, default=2e-5,
                    help="max allowed disagreement between the recovered constants and the "
                         "committed channel; the split stores 5 decimals, so 2e-5 is rounding")
    a = ap.parse_args()
    if a.materialize:
        if not a.emb_dir:
            raise SystemExit("--materialize needs --emb-dir")
        materialize(a)
    else:
        for req in ("dec_src", "fasta", "out_scalar", "out_fasta"):
            if not getattr(a, req):
                raise SystemExit(f"build mode needs --{req.replace('_', '-')}")
        build(a)
        if a.emb_dir:
            materialize(a)


if __name__ == "__main__":
    main()
