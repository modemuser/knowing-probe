"""Controls for the river probe: country confound and power at the river's cell sizes.

Usage: uv run python scripts/river_controls.py   (writes runs/river/controls.json)

1. Country-only baseline: score each held-out town by the share of the positive cell
   among training towns of its country; no activations. If this matches the arrow's
   AUC, the arrow may be reading the country of the name.
2. Country-centred arrow: subtract each country's mean training activation from all
   its towns, refit the arrow, held-out AUC and permutation p (100 shuffles).
3. Power: Step 0 cities subsampled to the river's exact cell sizes (13 + 7 positive,
   55 + 28 negative, one statement per city, disjoint cities), 300 draws.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from knowing_probe import arrow

N_SHUFFLES, N_DRAWS = 100, 300


def country_controls() -> dict:
    acts = np.load("runs/river/acts.npy")
    side = json.load(open("runs/river/acts.json"))
    cells = {r["qid"]: r for r in map(json.loads, open("data/river-cells.jsonl", encoding="utf-8"))}
    items = side["items"]
    cell = np.array([i["cell"] for i in items])
    ho = np.array([i["held_out"] for i in items])
    country = np.array([cells[i["qid"]]["country_qid"] for i in items])
    centred = acts.copy()
    for c in set(country):
        m = country == c
        centred[m] -= acts[m & ~ho].mean(axis=0)
    out = {}
    for neg in ("unstable", "country-prior"):
        P, N = cell == "known-by-retrieval", cell == neg
        tp, tn, hp, hn = P & ~ho, N & ~ho, P & ho, N & ho
        share = {c: (tp & (country == c)).sum() / max(1, ((tp | tn) & (country == c)).sum()) for c in set(country)}
        base = np.array([share[c] for c in country])
        s = arrow.project(centred, arrow.fit(centred, tp, tn))
        sh = np.zeros((N_SHUFFLES, acts.shape[1]))
        for k in range(N_SHUFFLES):
            ss = arrow.project(centred, arrow.fit(centred, *arrow.shuffled(tp, tn, k)))
            sh[k] = [arrow.auc(ss[hp, l], ss[hn, l]) for l in range(acts.shape[1])]
        per_layer = []
        for l in range(acts.shape[1]):
            a = arrow.auc(s[hp, l], s[hn, l])
            per_layer.append({"layer": l, "centred_heldout_auc": a,
                              "centred_perm_p": None if np.isnan(a) else float(np.mean(sh[:, l] >= a))})
        out[f"known-vs-{neg}"] = {"country_only_baseline_heldout_auc": arrow.auc(base[hp], base[hn]),
                                  "country_centred": per_layer}
        print(f"known vs {neg}: country-only AUC {out[f'known-vs-{neg}']['country_only_baseline_heldout_auc']:.3f}; "
              "centred AUC by layer " + " ".join(f"{r['layer']}:{r['centred_heldout_auc']:.2f}" for r in per_layer[1:]))
    return out


def power() -> dict:
    acts = np.load("runs/step0/cities.npy", mmap_mode="r")
    rows = list(csv.DictReader(open("data/raw/geometry-of-truth/cities.csv", encoding="utf-8")))
    lab = np.array([r["label"] == "1" for r in rows])
    city = np.array([r["city"] for r in rows])
    layers = [8, 12, 16, 20, 24, 28, 32, 36]
    X = np.asarray(acts[:, layers, :])
    rng = np.random.default_rng(1)
    res = np.zeros((N_DRAWS, len(layers)))
    for k in range(N_DRAWS):
        cs = rng.permutation(np.unique(city))
        pick = lambda cl, want: np.array([rng.choice(np.flatnonzero((city == c) & (lab == want))) for c in cl])
        pi, ni = pick(cs[:20], True), pick(cs[20:103], False)
        m = lambda idx: np.isin(np.arange(len(rows)), idx)
        s = arrow.project(X, arrow.fit(X, m(pi[:13]), m(ni[:55])))
        res[k] = [arrow.auc(s[pi[13:], j], s[ni[55:], j]) for j in range(len(layers))]
    out = {str(l): {"median": float(np.median(res[:, j])), "p5": float(np.percentile(res[:, j], 5))}
           for j, l in enumerate(layers)}
    print("cities at river cell sizes, held-out AUC median (5th pct): " +
          " ".join(f"L{l} {v['median']:.3f} ({v['p5']:.3f})" for l, v in out.items()))
    return {"design": "13+7 true vs 55+28 false, one statement per city, disjoint cities", "draws": N_DRAWS, "by_layer": out}


def main() -> None:
    result = {"country": country_controls(), "power_cities_at_river_sizes": power()}
    with open("runs/river/controls.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
