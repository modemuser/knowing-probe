"""Derived numbers about the river cells, from committed files only (no model).

Usage: uv run python scripts/river_cells_summary.py   (writes runs/river/cells-summary.json)
Inputs: data/wikidata-qwen3-8b-hf-logprob-B-priors.jsonl (the 294-town grade),
data/river-cells.jsonl (the cell assignment made from it by river_cells.py).
"""

from __future__ import annotations

import collections
import json

GRADED = "data/wikidata-qwen3-8b-hf-logprob-B-priors.jsonl"
CELLS = "data/river-cells.jsonl"
ORDER = ["known-by-retrieval", "right-by-prior", "parent-river", "country-prior", "unstable"]
COUNTRY = {"Q183": "DE", "Q40": "AT", "Q36": "PL", "Q213": "CZ", "Q214": "SK",
           "Q142": "FR", "Q38": "IT", "Q29": "ES", "Q45": "PT", "Q34": "SE"}
BANDS = [(0, 20), (20, 40), (40, 80), (80, None)]


def main() -> None:
    graded = [json.loads(line) for line in open(GRADED, encoding="utf-8")]
    cells = [json.loads(line) for line in open(CELLS, encoding="utf-8")]
    cell_of = {c["qid"]: c["cell"] for c in cells}
    sizes = {c: {"train": sum(1 for x in cells if x["cell"] == c and not x["held_out"]),
                 "held_out": sum(1 for x in cells if x["cell"] == c and x["held_out"])} for c in ORDER}
    by_country = {c: dict(collections.Counter(COUNTRY[x["country_qid"]] for x in cells if x["cell"] == c)) for c in ORDER}
    flagged_on_chain = [x for x in cells if x["prior_guess_flag"] and x["parent_qid"]]
    moved = [x for x in cells if x["cell"] in ("parent-river", "country-prior") and not x["prior_guess_flag"]]
    bands = []
    for lo, hi in BANDS:
        towns = [r for r in graded if r["meta"]["sitelinks"] >= lo and (hi is None or r["meta"]["sitelinks"] < hi)]
        k = sum(1 for r in towns if cell_of.get(r["meta"]["qid"]) == "known-by-retrieval")
        bands.append({"sitelinks": f">= {lo}" if hi is None else f"{lo}-{hi - 1}", "towns": len(towns),
                      "known_by_retrieval": k, "rate": round(k / len(towns), 3) if towns else None})
    out = {"inputs": {"graded": GRADED, "cells": CELLS}, "n_towns": len(graded),
           "n_gated": sum(r["gate"]["passed"] for r in graded), "n_slots_in_cells": len(cells),
           "cell_sizes": sizes, "cells_by_country": by_country,
           "country_prior_by_kind": dict(collections.Counter(x["prior_kind"] for x in cells if x["cell"] == "country-prior")),
           "flagged_prior_on_chain": len(flagged_on_chain),
           "flagged_prior_on_chain_failing_basin_check": sum(x["basin_check_failed"] for x in flagged_on_chain),
           "moved_out_of_unstable_by_new_rules": {"total": len(moved),
                                                  "to": dict(collections.Counter(x["cell"] for x in moved))},
           "known_by_retrieval_by_sitelinks": bands}
    json.dump(out, open("runs/river/cells-summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("cell_sizes", "flagged_prior_on_chain", "flagged_prior_on_chain_failing_basin_check",
                                          "moved_out_of_unstable_by_new_rules", "known_by_retrieval_by_sitelinks")}))


if __name__ == "__main__":
    main()
