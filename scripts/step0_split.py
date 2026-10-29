"""Write the Step 0 held-out split (cities / Spanish words) to data/step0-split.json.

Usage: uv run python scripts/step0_split.py
The split is a pure function of the CSVs and the seed (mt_probe.py); this file
records it so the held-out items can be read without rerunning anything.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mt_probe import HELD_OUT_FRAC, SPLIT_SEED, item_of  # noqa: E402

from knowing_probe import arrow  # noqa: E402


def main() -> None:
    out = {"seed": SPLIT_SEED, "held_out_frac": HELD_OUT_FRAC, "rule": "arrow.split_groups over items", "datasets": {}}
    for name in ("cities", "sp_en_trans"):
        rows = list(csv.DictReader(open(f"data/raw/geometry-of-truth/{name}.csv", encoding="utf-8")))
        items = [item_of(name, r) for r in rows]
        held = arrow.split_groups(items, HELD_OUT_FRAC, SPLIT_SEED)
        out["datasets"][name] = {"group": "city" if name == "cities" else "Spanish word",
                                 "n_statements": len(rows), "n_statements_held_out": int(held.sum()),
                                 "held_out_groups": sorted({i for i, h in zip(items, held) if h})}
    Path("data/step0-split.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, v in out["datasets"].items():
        print(k, len(v["held_out_groups"]), "groups,", v["n_statements_held_out"], "statements held out")


if __name__ == "__main__":
    main()
