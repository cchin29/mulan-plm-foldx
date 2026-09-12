#!/usr/bin/env python3
"""Read the transcribed comparator numbers from ``data/benchmarks/frontier.tsv``.

They used to be hardcoded in five plotting scripts. Two of those scripts held the same five
values attached to different rungs, and because each owned its own copy nothing could detect the
disagreement -- see ``data/benchmarks/README.md``.

The loader is deliberately protocol-first: :func:`comparators` requires a protocol and returns
only the rows measured under it. There is no "give me everything for this method" accessor,
because that is the shape that produced the defect.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

TSV = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "frontier.tsv"


def load(path: Path = TSV):
    """-> list of row dicts, comments stripped."""
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def comparators(metric: str, protocol: str, only=None, path: Path = TSV) -> Dict[str, float]:
    """``{method: value}`` for one metric measured under one protocol.

    ``metric`` and ``protocol`` are required: a number is comparable to a MuLAN arm only when the
    protocol matches, so there is no default and no cross-protocol merge.

    ``only`` restricts to a named subset. It exists because a panel may legitimately plot fewer
    comparators than the protocol has -- and because widening a figure is a decision, not a
    side effect of centralising the data. A name in ``only`` that the file does not carry is an
    error rather than a silent omission.
    """
    rows = [r for r in load(path) if r["metric"] == metric and r["protocol"] == protocol]
    if only is not None:
        have = {r["method"] for r in rows}
        missing = set(only) - have
        if missing:
            raise KeyError(f"requested {sorted(missing)} for {metric}/{protocol}, "
                           f"which the file does not carry (has {sorted(have)})")
        rows = [r for r in rows if r["method"] in set(only)]
    if not rows:
        raise KeyError(
            f"no comparators for metric={metric!r} protocol={protocol!r}. "
            f"Available: {sorted({(r['metric'], r['protocol']) for r in load(path)})}"
        )
    return {r["method"]: float(r["value"]) for r in rows}


def source_of(method: str, metric: str, protocol: str, path: Path = TSV) -> str:
    for r in load(path):
        if r["method"] == method and r["metric"] == metric and r["protocol"] == protocol:
            return r["source"]
    raise KeyError(f"{method} / {metric} / {protocol}")


if __name__ == "__main__":
    for m, p in sorted({(r["metric"], r["protocol"]) for r in load()}):
        print(f"{m:<18} {p:<16} {len(comparators(m, p))} methods")
