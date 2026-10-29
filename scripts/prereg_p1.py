"""P1 (PREREGISTRATION.md): freeze the arrow on today's towns; later, score the new towns once.

  uv run python scripts/prereg_p1.py freeze [NAME ...]
      Builds runs/prereg/<name>-arrow.npz from today's 200 slots: p1 (entity token,
      layer 21; primary), s1 (last prompt token, layer 21; secondary), p1lex (entity
      token, layer 1; lexical control, P1a). Refuses to refreeze a name. Prints the
      sha256 of each, for NOTEBOOK.md.
  uv run python scripts/prereg_p1.py score NEW_CELLS.jsonl NEW_ACTS_ENTITY NEW_ACTS_PROMPT
      Scores the new towns once. Refuses if the arrow files' hashes do not match
      the ones recorded at freeze time, or if a cell has fewer than 30 towns.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from knowing_probe import arrow

MIN_PER_CELL = 30
N_PERM, N_BOOT = 10_000, 10_000
OUT = Path("runs/prereg")
# name: (today's activations, layer). p1lex is the lexical control of P1a (registered after P1).
TODAY = {"p1": ("runs/river/acts-entity", 21), "s1": ("runs/river/acts", 21), "p1lex": ("runs/river/acts-entity", 1)}
CELLS_TODAY = "data/river-cells.jsonl"


def country_of(cells_file: str) -> dict[str, str]:
    return {r["qid"]: r["country_qid"] for r in map(json.loads, open(cells_file, encoding="utf-8"))}


def freeze(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    countries = country_of(CELLS_TODAY)
    hash_file = OUT / "frozen-hashes.json"
    hashes = json.loads(hash_file.read_text()) if hash_file.exists() else {}
    for name in names:
        if name in hashes:
            raise SystemExit(f"{name} is already frozen ({hashes[name][:12]}…): never refreeze")
        stem, layer = TODAY[name]
        acts = np.load(f"{stem}.npy")[:, layer, :].astype(np.float64)
        side = json.loads(Path(f"{stem}.json").read_text())
        items = side["items"]
        country = np.array([countries[it["qid"]] for it in items])
        cell = np.array([it["cell"] for it in items])
        keys = sorted(set(country))
        means = np.stack([acts[country == c].mean(axis=0) for c in keys])  # all 200 slots, all cells
        centred = acts - means[[keys.index(c) for c in country]]
        mu_known = centred[cell == "known-by-retrieval"].mean(axis=0)  # all of today's towns, train and held-out
        mu_unst = centred[cell == "unstable"].mean(axis=0)
        d = mu_known - mu_unst
        path = OUT / f"{name}-arrow.npz"
        np.savez(path, arrow=d, lo=mu_unst @ d, hi=mu_known @ d, country_keys=np.array(keys), country_means=means,
                 layer=layer, position=side.get("position", "prompt_last"), template_sha256=side["template"]["sha256"],
                 n_known=int((cell == "known-by-retrieval").sum()), n_unstable=int((cell == "unstable").sum()))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name}: {path} sha256 {hashes[name]}  (known {int((cell == 'known-by-retrieval').sum())}, "
              f"unstable {int((cell == 'unstable').sum())}, countries {len(keys)})")
    hash_file.write_text(json.dumps(hashes, indent=1))


def score(new_cells: str, stems: dict[str, str]) -> None:
    out = OUT / "p1-result.json"
    if out.exists():
        raise SystemExit(f"{out} exists: P1 is scored once")
    recorded = json.loads((OUT / "frozen-hashes.json").read_text())
    rows = {r["qid"]: r for r in map(json.loads, open(new_cells, encoding="utf-8"))}
    today = set(country_of(CELLS_TODAY))
    if today & set(rows):
        raise SystemExit(f"{len(today & set(rows))} of the new towns were in today's set: P1 scores new towns only")
    result = {}
    for name, stem in stems.items():
        path = OUT / f"{name}-arrow.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != recorded[name]:
            raise SystemExit(f"{path} changed since freeze: refusing to score")
        z = np.load(path)
        keys = list(z["country_keys"])
        side = json.loads(Path(f"{stem}.json").read_text())
        acts = np.load(f"{stem}.npy")[:, int(z["layer"]), :].astype(np.float64)
        items = side["items"]
        cell = np.array([it["cell"] for it in items])
        country = [rows[it["qid"]]["country_qid"] for it in items]
        centred = acts - z["country_means"][[keys.index(c) for c in country]]
        s = (centred @ z["arrow"] - z["lo"]) / (z["hi"] - z["lo"])
        pos, neg = s[cell == "known-by-retrieval"], s[cell == "unstable"]
        if min(len(pos), len(neg)) < MIN_PER_CELL:
            raise SystemExit(f"{name}: {len(pos)} known / {len(neg)} unstable, below {MIN_PER_CELL}: pull more towns first")
        auc = arrow.auc(pos, neg)
        rng = np.random.default_rng(0)
        both = np.concatenate([pos, neg])
        perm = [arrow.auc(*np.split(rng.permutation(both), [len(pos)])) for _ in range(N_PERM)]
        p = (1 + sum(a >= auc for a in perm)) / (1 + N_PERM)
        boot = [arrow.auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(N_BOOT)]
        rbp = s[cell == "right-by-prior"]
        result[name] = {"auc": auc, "perm_p_one_sided": p, "pass": bool(auc > 0.5 and p < 0.05) if name == "p1" else None,
                        "auc_ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                        "balanced_acc_at_0.5": float(((pos > 0.5).mean() + (neg <= 0.5).mean()) / 2),
                        "n_known": len(pos), "n_unstable": len(neg),
                        "s2_right_by_prior_median": float(np.median(rbp)) if len(rbp) else None,
                        "s2_median_ci95": [float(x) for x in np.percentile(
                            [np.median(rng.choice(rbp, len(rbp))) for _ in range(N_BOOT)], [2.5, 97.5])] if len(rbp) else None,
                        "n_right_by_prior": len(rbp)}
        result[name]["_scores"] = (pos, neg)
        print(name, json.dumps({k: v for k, v in result[name].items() if k != "_scores"}))
    # P1a: does layer 21 add to the name's surface (layer 1)? Paired bootstrap over the same new towns.
    (p21, n21), (p01, n01) = result["p1"].pop("_scores"), result["p1lex"].pop("_scores")
    result["s1"].pop("_scores")
    rng = np.random.default_rng(1)
    diffs = []
    for _ in range(N_BOOT):
        i, j = rng.integers(0, len(p21), len(p21)), rng.integers(0, len(n21), len(n21))
        diffs.append(arrow.auc(p21[i], n21[j]) - arrow.auc(p01[i], n01[j]))
    ci = [float(x) for x in np.percentile(diffs, [2.5, 97.5])]
    result["p1a"] = {"auc_l21_minus_l1": result["p1"]["auc"] - result["p1lex"]["auc"], "ci95": ci,
                     "beyond_surface": bool(ci[0] > 0)}
    print("p1a", json.dumps(result["p1a"]))
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        freeze(sys.argv[2:] or list(TODAY))
    elif sys.argv[1] == "score":
        score(sys.argv[2], {"p1": sys.argv[3], "s1": sys.argv[4], "p1lex": sys.argv[3]})
