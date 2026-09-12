#!/usr/bin/env python3
"""Locate, read and rewrite the pipe tables in experiments/BENCHMARK_MATRIX.md.

Why this module exists
----------------------
Two very different programs need to agree on what a table in the matrix *is*:

  * scripts_plots/comparators.py READS the published comparator values out of it,
    so the figures stop carrying typed copies.
  * experiments/gen_benchmark_matrix_data.py WRITES MuLAN's own rows into it, so
    nobody has to paste a recomputed number by hand.

If the reader and the writer disagree about where a table starts or how a cell is
spelled, the writer can emit something the reader silently misparses — which is
the same class of failure as the typed copies, just one level down. So the rule
lives here, once, and both import it.

The locating rule: a table is found by the *substrings its header row contains*,
never by line number. Insert a paragraph above a table and nothing shifts. Zero
matches or two matches is fatal — an ambiguous anchor is a bug in the caller, not
something to resolve by taking the first hit.

Usage
-----
    import mdtable
    lines = mdtable.read(path)
    t = mdtable.parse(path, lines, "papers", ["| Method — Date", "CATH"])
    t.value("RDE-Network", "byCplx")            # 0.401

    hdr, lo, hi = mdtable.locate(path, lines, "papers", ["| Method — Date"])
    lines[lo:hi] = new_rows                     # rewrite the body, header untouched
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

__all__ = ["read", "locate", "parse", "Table", "clean", "number", "cells", "fmt", "EMDASH"]

EMDASH = "—"


# --------------------------------------------------------------------------- cells


def clean(cell: str) -> str:
    """Drop the markdown emphasis a cell may be wearing."""
    return re.sub(r"\*\*|`|<[^>]+>", "", cell).strip()


def number(cell: str):
    """Cell -> float, or None if it does not carry one.

    Values in the matrix trail superscript provenance markers (0.401ˢ, 0.477ᵖ)
    and em-dashes stand in for 'not reported'. Anything that is not cleanly a
    number after stripping those becomes None — including the approximate forms
    (≈0.85), which are deliberately not usable as comparator values.
    """
    v = re.sub(r"[^\d.]+$", "", clean(cell))
    try:
        return float(v)
    except ValueError:
        return None


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def fmt(v, bold: bool = False) -> str:
    """Float -> the matrix's cell spelling. None is an em-dash, best-in-column is bold."""
    if v is None:
        return EMDASH
    s = f"{v:.3f}"
    return f"**{s}**" if bold else s


# --------------------------------------------------------------------------- locating


def locate(src, lines, name: str, required) -> tuple[int, int, int]:
    """(header index, body start, body end) for the one table whose header row
    contains every string in `required`. All indices 0-based, body end exclusive.

    The body runs from the line after the `|---|` separator to the first line that
    is not a table row, so a caller may replace lines[body_start:body_end] wholesale
    without touching the header or the separator.
    """
    hits = [i for i, l in enumerate(lines)
            if l.startswith("|") and all(r in l for r in required)]
    if len(hits) != 1:
        sys.exit(f"FATAL: {len(hits)} candidate header rows for the {name!r} table in {src} "
                 f"(looked for all of {list(required)!r}). Cannot address it unambiguously.")
    hdr = hits[0]
    end = hdr + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return hdr, hdr + 2, end


def read(path) -> list[str]:
    return Path(path).read_text().splitlines()


# --------------------------------------------------------------------------- parsing


class Table:
    """One pipe table, addressed by row-label prefix and column name."""

    def __init__(self, src, name, cols, rows, linenos, span):
        self.src = Path(src)
        self.name = name
        self.cols = cols
        self.rows = rows          # {row key: {column: float|None}}
        self.linenos = linenos    # {row key: 1-based line number}
        self.span = span          # (header, body_start, body_end), 0-based

    def row(self, prefix: str):
        """The single row whose label starts with `prefix`. Ambiguity is fatal."""
        hits = [k for k in self.rows if k.startswith(prefix)]
        if len(hits) != 1:
            sys.exit(f"FATAL: {len(hits)} rows in the {self.name!r} table of {self.src.name} "
                     f"start with {prefix!r}: {hits}")
        return hits[0]

    def value(self, prefix: str, col: str) -> float:
        label = self.row(prefix)
        if col not in self.cols:
            sys.exit(f"FATAL: the {self.name!r} table of {self.src.name} has no column {col!r}. "
                     f"Columns: {self.cols}")
        v = self.rows[label][col]
        if v is None:
            sys.exit(f"FATAL: {self.src.name}:{self.linenos[label]} — {label!r} has no numeric "
                     f"value in column {col!r}. A figure is asking for a number that the "
                     f"matrix does not report.")
        return v

    def column(self, col: str, methods) -> dict:
        """{short name: value} for the named methods, in the order given."""
        return {m: self.value(m, col) for m in methods}

    def lineno(self, prefix: str) -> int:
        return self.linenos[self.row(prefix)]


def parse(src, lines, name: str, required, keycols: int = 1) -> Table:
    """Locate a table and parse its body into a Table.

    `keycols` is how many leading columns form the row key (1 = method name,
    2 = the (embedding, arm) tables).
    """
    hdr, lo, hi = locate(src, lines, name, required)
    raw_cols = cells(lines[hdr])
    cols = [clean(c) for c in raw_cols[keycols:]]
    rows, linenos = {}, {}
    for i in range(lo, hi):
        cs = cells(lines[i])
        if len(cs) != len(raw_cols):
            continue
        key_parts = tuple(clean(c) for c in cs[:keycols])
        key = key_parts[0] if keycols == 1 else key_parts
        rows[key] = {c: number(cell) for c, cell in zip(cols, cs[keycols:])}
        linenos[key] = i + 1
    if not rows:
        sys.exit(f"FATAL: the {name!r} table in {src} parsed to zero rows")
    return Table(src, name, cols, rows, linenos, (hdr, lo, hi))
