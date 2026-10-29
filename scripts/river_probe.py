"""River slots, probe: arrow per layer on training towns, every slot projected, two contrasts.

Usage: uv run python scripts/river_probe.py runs/river/acts
  contrast 1: known-by-retrieval vs unstable (the first probe)
  contrast 2: known-by-retrieval vs country-prior (does unstable land on the unknown side too?)
right-by-prior and parent-river never touch either arrow. Writes figures/river/ and
runs/river/<contrast>-summary.json.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from knowing_probe import analysis

ORDER = ["known-by-retrieval", "right-by-prior", "parent-river", "country-prior", "unstable"]
PANELS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]


def main() -> None:
    stem = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/river/acts")
    acts = np.load(stem.with_suffix(".npy"))
    side = json.loads(stem.with_suffix(".json").read_text())
    items = side["items"]
    held_out = np.array([it["held_out"] for it in items])
    cell = np.array([it["cell"] for it in items])
    cells = [(c, cell == c) for c in ORDER]
    meta = {"activations": str(stem.with_suffix(".npy")), "activations_sha256": side["npy_sha256"],
            "template": side["template"], "position_rule": side["position_rule"], "layers": side["layers"],
            "split": side["split"], "thresholds": side["thresholds"], "model": side["model"]["model"],
            "revision": side["model"]["revision"], "last_tokens": side["last_tokens"],
            "arrow_fit_on": "training towns of the two contrast cells only"}
    out = Path("runs/river")
    for neg in ("unstable", "country-prior"):
        name = f"known-vs-{neg}"
        s = analysis.run(acts, side["layers"], cells, "known-by-retrieval", neg, held_out, PANELS, out, name,
                         f"River slots · Qwen3-8B · template B · arrow = mean(known-by-retrieval) − mean({neg}), training towns",
                         meta)
        print(f"\n== {name}   cells: {s['cell_sizes']}")
        analysis.print_table(s, [c for c in ORDER if c not in ("known-by-retrieval", neg)])
    Path("figures/river").mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        shutil.move(f, Path("figures/river") / f.name)


if __name__ == "__main__":
    main()
