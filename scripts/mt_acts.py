"""Step 0, activations: Marks and Tegmark's true/false statements, last-token residuals, all layers.

Usage: uv run python scripts/mt_acts.py [cities sp_en_trans] [--limit N]
Writes runs/step0/<dataset>.npy and .json. Statements are fed as plain text, no chat
template, as in the paper; position = last token of the statement (the full stop).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from knowing_probe import acts

RAW = Path("data/raw/geometry-of-truth")
SOURCE = {"repo": "https://github.com/saprmarks/geometry-of-truth", "commit": "5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb",
          "retrieved": "2026-10-29"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["cities", "sp_en_trans"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    adapter = acts.load()
    for name in args.datasets:
        path = RAW / f"{name}.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))[: args.limit]
        texts = [r["statement"] for r in rows]
        out, last = acts.read_last_token(adapter, texts, label=name)
        layers = list(range(out.shape[1]))
        side = acts.sidecar_base(adapter, layers, texts, last)
        side["dataset"] = {**SOURCE, "file": f"datasets/{name}.csv",
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n_rows": len(rows)}
        side["prompt"] = "statement text as-is, no chat template, no special tokens"
        acts.save(Path("runs/step0") / (name if args.limit is None else f"{name}-limit{args.limit}"), out, side)


if __name__ == "__main__":
    main()
