"""Precompute per-residue PLM embeddings into a cache directory.

Two input modes, because both are needed in practice and each previously had its own script:

  * **a label FASTA** — embed every record, keyed by its label. This is how the CATH and
    full-SKEMPI caches were built.
  * **a mutation table + wild-type FASTA** — enumerate every wild-type and mutant chain the
    table references and embed those. That is exactly the set the trainer will look for, so a
    cache built this way is complete by construction.

Embeddings are written as ``<id>.pt`` holding ``[L, dim]``, which is what ``MulanDataset``
reads. Existing files are skipped, so a run is resumable — which matters when a single pass over
the full SKEMPI set with a large backbone takes hours.

Examples::

    plm-embed --list
    plm-embed --model ankh --fasta labels.fasta -o emb/ankh
    plm-embed --model prostt5 --table muts.tsv --wt-fasta wt.fasta -o emb/prostt5
"""

import os
import sys
import time
from argparse import ArgumentParser

# Route the handful of ops without a Metal kernel to the CPU rather than erroring. Must precede
# `import torch`.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402,F401  (import for the MPS fallback side effect above)

from mulan import plm  # noqa: E402
from mulan import utils  # noqa: E402


def _print_registry():
    """The answer to 'which backbones are there, and what does each need?'"""
    rows = []
    for tag in plm.list_plms():
        s = plm.get_spec(tag)
        rows.append((
            s.tag,
            str(s.dim),
            s.model_id or "(local weights)",
            s.extra or "-",
            "yes" if s.mps_ok else "no",
            ", ".join(s.aliases) or "-",
        ))
    head = ("tag", "dim", "model id", "needs", "mps", "aliases")
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*head))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))
    print(
        "\n'needs' names the environment a backbone requires. ESM-C/ESM3 have no pip extra by design --"
        "\nbuild a separate env from requirements-esmc.txt; see docs/EMBEDDING_SETUP.md.\n"
        "Backbones marked mps=no fall back to CPU on Apple Silicon.\n"
        "SaProt and MINT are not embeddable per-sequence — see their generators under "
        "experiments/."
    )


def get_args(argv=None):
    parser = ArgumentParser(prog="plm-embed", description=__doc__)
    parser.add_argument("--list", action="store_true",
                        help="List the registered backbones and exit.")
    parser.add_argument("-m", "--model", help="PLM tag (or alias). See --list.")
    parser.add_argument("--fasta", help="Label FASTA: embed every record, keyed by label.")
    parser.add_argument("--table", help="Mutation table: embed every id it references.")
    parser.add_argument("--wt-fasta", help="Wild-type FASTA, required with --table.")
    parser.add_argument("-o", "--output-dir", default="./embeddings",
                        help="Cache directory (default: ./embeddings).")
    parser.add_argument("--device", default=None,
                        help="Override device selection (cpu / cuda / mps).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-embed ids that are already cached.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Embed at most N sequences — for a smoke test.")

    # Positional form kept working: the original CLI was `plm-embed <fasta_file> <model_name>`.
    parser.add_argument("legacy", nargs="*", help="(deprecated) <fasta_file> <model_name>")
    args = parser.parse_args(argv)

    if args.list:
        return args
    if args.legacy and not (args.fasta or args.table):
        if len(args.legacy) != 2:
            parser.error("legacy positional form is: plm-embed <fasta_file> <model_name>")
        args.fasta, args.model = args.legacy
    if not args.model:
        parser.error("--model is required (see --list)")
    if not args.fasta and not args.table:
        parser.error("one of --fasta or --table is required")
    if args.table and not args.wt_fasta:
        parser.error("--table needs --wt-fasta")
    try:
        plm.resolve_tag(args.model)
    except KeyError as exc:
        parser.error(str(exc))
    return args


def collect_sequences(args):
    """``{id: sequence}`` for whichever input mode was given."""
    if args.table:
        return plm.enumerate_sequence_ids(args.table, args.wt_fasta)
    return utils.parse_fasta(args.fasta)


def run(args):
    spec = plm.get_spec(args.model)
    sequences = collect_sequences(args)

    os.makedirs(args.output_dir, exist_ok=True)
    todo = sorted(sequences if args.overwrite
                  else plm.missing_ids(sequences, args.output_dir))
    if args.limit:
        todo = todo[:args.limit]

    print(f"[plm-embed] {spec.tag} (dim {spec.dim}) -> {args.output_dir}")
    print(f"[plm-embed] {len(sequences)} ids | {len(sequences) - len(todo)} cached | "
          f"{len(todo)} to embed")
    if not todo:
        return 0

    backend = plm.load_backend(args.model, device=args.device)
    print(f"[plm-embed] loaded on {backend.device}")

    started = time.time()
    for n, id_ in enumerate(todo, 1):
        # embed_checked, not embed: a trimming mismatch must fail here rather than write a
        # misaligned tensor that surfaces later as an unexplained metric change.
        emb = backend.embed_checked(sequences[id_])
        utils.save_embedding(emb, args.output_dir, id_)
        if n % 100 == 0 or n == len(todo):
            per = (time.time() - started) / n
            print(f"[plm-embed]   {n}/{len(todo)} | {per:.2f}s/seq | "
                  f"ETA {(len(todo) - n) * per / 60:.0f} min", flush=True)

    print(f"[plm-embed] done: {len(todo)} embeddings in "
          f"{(time.time() - started) / 60:.1f} min")
    return 0


def main():
    args = get_args()
    if args.list:
        _print_registry()
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
