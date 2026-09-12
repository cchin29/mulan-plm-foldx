"""Generate (or verify) the paper-faithful CV folds for the embedding sweep.

Uses the real `mulan.data.split_data(split_method="random")` — the upstream/paper
`rng.integers` partition at the configured seed — and materializes the small fold TSVs
into the tracked `splits/` dir. Run from the repo root.

    python experiments/embedding_sweep/gen_splits.py            # (re)generate
    python experiments/embedding_sweep/gen_splits.py --verify   # check on-disk == regenerated
"""
import os
import re
import sys
import tempfile
import filecmp

from mulan.data import split_data

HERE = os.path.dirname(os.path.abspath(__file__))
# ES_CONFIG lets a sibling config (e.g. config_balanced.sh) drive a different
# split_method/dir while reusing this generator; defaults to the paper config.
CONFIG = os.environ.get("ES_CONFIG", os.path.join(HERE, "config.sh"))


def load_config(path=CONFIG, cfg=None):
    """Parse `export KEY=VALUE` lines from a config.sh (single source of truth).
    Follows `source <path>` lines first (so a sibling config, e.g.
    config_balanced.sh, can source the base config then override keys)."""
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


def generate(cfg, out_dir):
    split_data(
        cfg["ES_DATASET"],
        output_dir=out_dir,
        num_folds=int(cfg["ES_NUM_FOLDS"]),
        random_state=int(cfg["ES_SEED"]),
        split_method=cfg["ES_SPLIT_METHOD"],
    )


def main():
    cfg = load_config()
    verify = "--verify" in sys.argv[1:]
    dst = cfg["ES_SPLIT_DIR"]
    if not verify:
        generate(cfg, dst)
        n = int(cfg["ES_NUM_FOLDS"])
        print(f"Wrote {n}-fold '{cfg['ES_SPLIT_METHOD']}' splits (seed {cfg['ES_SEED']}) -> {dst}")
        for k in range(n):
            d = os.path.join(dst, f"fold_{k}")
            sizes = {
                s: sum(1 for _ in open(os.path.join(d, f"{cfg['ES_BASENAME']}_{s}.tsv")))
                for s in ("train", "val", "test")
            }
            print(f"  fold {k}: {sizes}")
        return

    # --verify: regenerate into a temp dir and diff against the committed splits
    with tempfile.TemporaryDirectory() as tmp:
        generate(cfg, tmp)
        ok = True
        for k in range(int(cfg["ES_NUM_FOLDS"])):
            for s in ("train", "val", "test"):
                rel = os.path.join(f"fold_{k}", f"{cfg['ES_BASENAME']}_{s}.tsv")
                a, b = os.path.join(dst, rel), os.path.join(tmp, rel)
                if not (os.path.exists(a) and filecmp.cmp(a, b, shallow=False)):
                    print(f"  MISMATCH: {rel}")
                    ok = False
        print("VERIFY:", "OK — committed splits match regeneration" if ok else "FAILED")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
