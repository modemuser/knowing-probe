"""River slots, activations: one forward pass per slot under template B, no generation.

Usage: uv run python scripts/river_acts.py data/river-cells.jsonl GRADED.jsonl -o runs/river/acts
       [--position prompt_last|entity_last]
Position: prompt_last = last prompt token (right before the answer would begin);
entity_last = last token of the town name in the question. All layers.
The probe template is the grading template: the hash check refuses anything else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from contrastset.grade import Template, require_same_template

from knowing_probe import acts

TEMPLATE = Path(__file__).resolve().parents[2] / "contrastset/experiments/templates/qwen3-name-only.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells")
    ap.add_argument("graded")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--position", choices=["prompt_last", "entity_last"], default="prompt_last")
    args = ap.parse_args()
    cells = [json.loads(line) for line in open(args.cells, encoding="utf-8")]
    graded = [json.loads(line) for line in open(args.graded, encoding="utf-8")]
    template = Template.load(str(TEMPLATE))
    require_same_template(template, [SimpleNamespace(provenance=r["provenance"]) for r in graded])
    revision = {r["provenance"]["model"]["revision"] for r in graded}
    assert len(revision) == 1, revision
    adapter = acts.load(revision=revision.pop())
    prompts = [template.render(c["question"], c["entity"]) for c in cells]
    if args.position == "entity_last":
        located = [acts.entity_last_token(adapter, p, c["entity"]) for p, c in zip(prompts, cells)]
        positions = [i for i, _ in located]
    else:
        located, positions = [], [-1] * len(prompts)
    out, last = acts.read_at(adapter, prompts, positions, label=f"river {args.position}")
    side = acts.sidecar_base(adapter, list(range(out.shape[1])), prompts, last)
    if args.position == "entity_last":
        side["position_rule"] = acts.ENTITY_RULE
        side["token_runs_past_entity"] = sum(over for _, over in located)
    side.update({
        "template": {"path": str(TEMPLATE), "sha256": template.sha256},
        "cells_file": args.cells, "graded_file": args.graded,
        "position": args.position,
        "items": [{"entity": c["entity"], "qid": c["qid"], "cell": c["cell"], "held_out": c["held_out"],
                   "token_index": pos, "token": tok} for c, pos, tok in zip(cells, positions, last)],
        "split": cells[0]["provenance"]["split"], "thresholds": cells[0]["provenance"]["thresholds"],
    })
    acts.save(args.out, out, side)


if __name__ == "__main__":
    main()
