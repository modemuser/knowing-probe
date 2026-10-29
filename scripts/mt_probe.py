"""Step 0, probe: true vs false statements, split by item (city / Spanish word), arrow per layer.

Usage: uv run python scripts/mt_probe.py [cities sp_en_trans]
Reads runs/step0/<dataset>.npy/.json, writes figures/step0/ and runs/step0/<dataset>-summary.json.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from knowing_probe import analysis, arrow

SPLIT_SEED, HELD_OUT_FRAC = 20261029, 1 / 3
PANELS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]


def item_of(name: str, row: dict[str, str]) -> str:
    if name == "cities":
        return row["city"]
    m = re.match(r"The Spanish word '(.+?)' means", row["statement"])
    if not m:
        raise ValueError(f"cannot find the Spanish word in {row['statement']!r}")
    return m.group(1)


def main() -> None:
    for name in sys.argv[1:] or ["cities", "sp_en_trans"]:
        acts = np.load(f"runs/step0/{name}.npy")
        side = json.loads(Path(f"runs/step0/{name}.json").read_text())
        rows = list(csv.DictReader(open(f"data/raw/geometry-of-truth/{name}.csv", encoding="utf-8")))
        assert len(rows) == acts.shape[0]
        label = np.array([r["label"] == "1" for r in rows])
        held_out = arrow.split_groups([item_of(name, r) for r in rows], HELD_OUT_FRAC, SPLIT_SEED)
        cells = [("true", label), ("false", ~label)]
        summary = analysis.run(
            acts, side["layers"], cells, "true", "false", held_out, PANELS, "runs/step0", name,
            f"Step 0 · {name} (Marks & Tegmark) · Qwen3-8B · arrow = mean(true) − mean(false), training items",
            {"dataset": side["dataset"], "activations": f"runs/step0/{name}.npy", "activations_sha256": side["npy_sha256"],
             "position_rule": side["position_rule"], "layers": side["layers"],
             "split": {"by": "city" if name == "cities" else "Spanish word", "held_out_frac": HELD_OUT_FRAC,
                       "seed": SPLIT_SEED, "n_groups_held_out": len({item_of(name, r) for r, h in zip(rows, held_out) if h})}})
        Path("figures/step0").mkdir(parents=True, exist_ok=True)
        for f in Path("runs/step0").glob(f"{name}-*.png"):
            shutil.move(f, Path("figures/step0") / f.name)
        print(f"\n== {name}")
        analysis.print_table(summary)


if __name__ == "__main__":
    main()
