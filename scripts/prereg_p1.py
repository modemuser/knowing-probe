"""P1 (PREREGISTRATION.md): freeze the arrow on today's towns; later, score the new towns once.

  uv run python scripts/prereg_p1.py freeze [NAME ...]
      Builds runs/prereg/<name>-arrow.npz from today's 200 slots: p1 (entity token,
      layer 21; primary), s1 (last prompt token, layer 21; secondary), p1lex (entity
      token, layer 1; lexical control, P1a). Refuses to refreeze a name. Prints the
      sha256 of each, for NOTEBOOK.md.
  uv run python scripts/prereg_p1.py score NEW_CELLS.jsonl NEW_ACTS_ENTITY NEW_ACTS_PROMPT
      Scores the new towns once, as amended by P1b (new-pass country centring, >= 40
      per cell, surface controls P1a and P1c, descriptive D1/D2/S1/S2) and P1d (only
      countries with >= 10 new slots in the five cells are scored). Refuses if
      the arrow files' hashes differ from those recorded at freeze time, if a cell
      is short, or if any new town was in today's set.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from knowing_probe import arrow

MIN_PER_CELL = 40  # P1b (P1 said 30)
MIN_COUNTRY_SLOTS = 10  # P1d
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


def paired_diff(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], seed: int) -> dict:
    """AUC(a) - AUC(b) on the identical towns; 95 % interval from bootstraps that draw the same towns for both."""
    (pa, na), (pb, nb) = a, b
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(N_BOOT):
        i, j = rng.integers(0, len(pa), len(pa)), rng.integers(0, len(na), len(na))
        diffs.append(arrow.auc(pa[i], na[j]) - arrow.auc(pb[i], nb[j]))
    ci = [float(x) for x in np.percentile(diffs, [2.5, 97.5])]
    return {"diff": arrow.auc(pa, na) - arrow.auc(pb, nb), "ci95": ci, "above_zero": bool(ci[0] > 0)}


def score(new_cells: str, stems: dict[str, str]) -> None:
    """P1 as amended by P1b. stems: {"entity": new entity-token acts, "prompt": new last-prompt-token acts}."""
    out = OUT / "p1-result.json"
    if out.exists():
        raise SystemExit(f"{out} exists: P1 is scored once")
    recorded = json.loads((OUT / "frozen-hashes.json").read_text())
    rows = {r["qid"]: r for r in map(json.loads, open(new_cells, encoding="utf-8"))}
    today = set(country_of(CELLS_TODAY))
    if today & set(rows):
        raise SystemExit(f"{len(today & set(rows))} of the new towns were in today's set: P1 scores new towns only")
    frozen = {}
    for name in ("p1", "s1", "p1lex"):
        path = OUT / f"{name}-arrow.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != recorded[name]:
            raise SystemExit(f"{path} changed since freeze: refusing to score")
        frozen[name] = np.load(path)
    sides = {k: json.loads(Path(f"{v}.json").read_text()) for k, v in stems.items()}
    items = sides["entity"]["items"]
    if [it["qid"] for it in items] != [it["qid"] for it in sides["prompt"]["items"]]:
        raise SystemExit("entity and prompt activation files list different towns")
    cell = np.array([it["cell"] for it in items])
    country = np.array([rows[it["qid"]]["country_qid"] for it in items])
    # P1d: a country enters only with >= MIN_COUNTRY_SLOTS new slots in the five cells.
    counts = {c: int((country == c).sum()) for c in set(country)}
    keep = np.array([counts[c] >= MIN_COUNTRY_SLOTS for c in country])
    excluded = sorted(c for c, n in counts.items() if n < MIN_COUNTRY_SLOTS)
    P, N, R = (cell == "known-by-retrieval") & keep, (cell == "unstable") & keep, (cell == "right-by-prior") & keep
    if min(P.sum(), N.sum()) < MIN_PER_CELL:
        raise SystemExit(f"{P.sum()} known / {N.sum()} unstable, below {MIN_PER_CELL}: pull more towns first")

    def layer(stem_key: str, l: int) -> np.ndarray:
        """New towns at layer l, centred with NEW-pass country means (all new slots of the country; P1b)."""
        x = np.load(f"{stems[stem_key]}.npy")[:, l, :].astype(np.float64)
        for c in set(country):  # excluded countries are centred too, but never scored
            x[country == c] -= x[country == c].mean(axis=0)
        return x

    def scores(x: np.ndarray, z) -> np.ndarray:
        return (x @ z["arrow"] - z["lo"]) / (z["hi"] - z["lo"])

    e21, e1, p21 = layer("entity", 21), layer("entity", 1), layer("prompt", 21)
    u = frozen["p1lex"]["arrow"] / np.linalg.norm(frozen["p1lex"]["arrow"])
    variants = {
        "p1": scores(e21, frozen["p1"]),                      # primary
        "s1": scores(p21, frozen["s1"]),                      # secondary, last prompt token
        "p1lex": scores(e1, frozen["p1lex"]),                 # P1a: surface at layer 1
        "p1c_L1_arrow_on_L21": scores(e21, frozen["p1lex"]),  # P1c: surface direction carried to layer 21
        "d2_L1_removed": scores(e21 - np.outer(e21 @ u, u), frozen["p1"]),
        "d1_log_sitelinks": np.log(np.array([rows[it["qid"]]["sitelinks"] for it in items], dtype=float)),
    }
    rng = np.random.default_rng(0)
    result: dict = {"n_known": int(P.sum()), "n_unstable": int(N.sum()), "n_right_by_prior": int(R.sum()),
                    "centring": "new-pass country means (P1b)", "frozen_hashes": recorded,
                    "countries_excluded_p1d": {c: counts[c] for c in excluded},
                    "countries_scored": {c: counts[c] for c in sorted(counts) if c not in excluded}}
    for name, s in variants.items():
        pos, neg = s[P], s[N]
        auc = arrow.auc(pos, neg)
        both = np.concatenate([pos, neg])
        perm = [arrow.auc(*np.split(rng.permutation(both), [len(pos)])) for _ in range(N_PERM)]
        boot = [arrow.auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(N_BOOT)]
        result[name] = {"auc": auc, "perm_p_one_sided": (1 + sum(a >= auc for a in perm)) / (1 + N_PERM),
                        "auc_ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])]}
        if name in ("p1", "s1"):
            rbp = s[R]
            result[name]["balanced_acc_at_0.5"] = float(((pos > 0.5).mean() + (neg <= 0.5).mean()) / 2)
            result[name]["s2_right_by_prior_median"] = float(np.median(rbp)) if len(rbp) else None
            result[name]["s2_median_ci95"] = [float(x) for x in np.percentile(
                [np.median(rng.choice(rbp, len(rbp))) for _ in range(N_BOOT)], [2.5, 97.5])] if len(rbp) else None
        print(name, json.dumps(result[name]))
    pair = lambda k: (variants[k][P], variants[k][N])
    result["p1"]["pass"] = bool(result["p1"]["auc"] > 0.5 and result["p1"]["perm_p_one_sided"] < 0.05)
    result["p1a"] = paired_diff(pair("p1"), pair("p1lex"), seed=1)
    result["p1c"] = paired_diff(pair("p1"), pair("p1c_L1_arrow_on_L21"), seed=2)
    result["beyond_surface"] = bool(result["p1"]["pass"] and result["p1a"]["above_zero"] and result["p1c"]["above_zero"])
    print("p1a", json.dumps(result["p1a"]), "\np1c", json.dumps(result["p1c"]), "\nbeyond_surface", result["beyond_surface"])
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        freeze(sys.argv[2:] or list(TODAY))
    elif sys.argv[1] == "score":
        score(sys.argv[2], {"entity": sys.argv[3], "prompt": sys.argv[4]})
