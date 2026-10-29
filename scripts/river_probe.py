"""River slots, probe: arrow per layer on training towns, every slot projected, two contrasts.

Usage: uv run python scripts/river_probe.py [runs/river/acts] [--centre country]
  contrast 1: known-by-retrieval vs unstable (the first probe)
  contrast 2: known-by-retrieval vs country-prior (does unstable land on the unknown side too?)
right-by-prior and parent-river never touch either arrow.
--centre country: subtract, from every slot, the mean of its country's TRAINING slots
(all five cells) before fitting, so the arrow cannot use country identity.
Writes figures/river/<variant>/ and runs/river/<variant>/<contrast>-summary.json, where
variant = prompt-last | entity-last, plus -country-centred.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from knowing_probe import analysis

ORDER = ["known-by-retrieval", "right-by-prior", "parent-river", "country-prior", "unstable"]
PANELS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", nargs="?", default="runs/river/acts")
    ap.add_argument("--centre", choices=["none", "country"], default="none")
    ap.add_argument("--cells", default="data/river-cells.jsonl")
    args = ap.parse_args()
    stem = Path(args.stem)
    acts = np.load(stem.with_suffix(".npy"))
    side = json.loads(stem.with_suffix(".json").read_text())
    items = side["items"]
    held_out = np.array([it["held_out"] for it in items])
    cell = np.array([it["cell"] for it in items])
    variant = "entity-last" if side.get("position") == "entity_last" else "prompt-last"
    if args.centre == "country":
        country_of = {json.loads(line)["qid"]: json.loads(line)["country_qid"] for line in open(args.cells, encoding="utf-8")}
        country = np.array([country_of[it["qid"]] for it in items])
        for c in set(country):
            m = country == c
            acts[m] -= acts[m & ~held_out].mean(axis=0)
        variant += "-country-centred"
    cells = [(c, cell == c) for c in ORDER]
    meta = {"activations": str(stem.with_suffix(".npy")), "activations_sha256": side["npy_sha256"],
            "template": side["template"], "position_rule": side["position_rule"], "layers": side["layers"],
            "split": side["split"], "thresholds": side["thresholds"], "model": side["model"]["model"],
            "revision": side["model"]["revision"], "last_tokens_sample": side["last_tokens"][:20],
            "centre": args.centre, "arrow_fit_on": "training towns of the two contrast cells only"}
    out = Path("runs/river") / variant
    where = "town-name token" if side.get("position") == "entity_last" else "last prompt token"
    for neg in ("unstable", "country-prior"):
        name = f"known-vs-{neg}"
        s = analysis.run(acts, side["layers"], cells, "known-by-retrieval", neg, held_out, PANELS, out, name,
                         f"River slots · {where}{' · country-centred' if args.centre == 'country' else ''} · "
                         f"arrow = mean(known-by-retrieval) − mean({neg}), training towns", meta)
        print(f"\n== {variant} · {name}")
        analysis.print_table(s, [c for c in ORDER if c not in ("known-by-retrieval", neg)])
    fig_dir = Path("figures/river") / variant
    fig_dir.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        shutil.move(f, fig_dir / f.name)


if __name__ == "__main__":
    main()
