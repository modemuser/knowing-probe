"""Is the entity-token separation the name's surface? Arrow overlap and projection-out, today's towns.

Usage: uv run python scripts/river_surface.py   (writes runs/river/surface.json)

Entity token, country-centred (training-slot country means, as in river_probe.py),
known-by-retrieval vs unstable, arrows fitted on training towns only.
1. |cos| between the layer-0 / layer-1 arrows and the arrow at each later layer. The
   residual stream is additive: unless something overwrote it, the token embedding
   is still present at layer 21.
2. Projection-out: remove the layer-1 arrow direction (and, separately, the span of
   the layer-0 and layer-1 arrows) from every slot's layer-L residual, refit the
   layer-L arrow on the same training towns, re-measure held-out AUC and
   permutation p. A single direction removed is a lower bound on removing the
   surface: surface information can live in other directions too.
"""

from __future__ import annotations

import json

import numpy as np

from knowing_probe import arrow

STEM = "runs/river/acts-entity"
N_SHUFFLES = 200
LAYERS = [4, 8, 12, 16, 19, 20, 21, 22, 23, 24, 28, 32, 36]


def held_out_auc(x: np.ndarray, tp, tn, hp, hn) -> tuple[float, float]:
    """x [n, d] at one layer. Held-out AUC of the training-fitted arrow and its permutation p."""
    def auc_for(p, n):
        d = x[p].mean(0) - x[n].mean(0)
        s = x @ d
        return arrow.auc(s[hp], s[hn])
    a = auc_for(tp, tn)
    sh = [auc_for(*arrow.shuffled(tp, tn, k)) for k in range(N_SHUFFLES)]
    return a, float(np.mean(np.array(sh) >= a))


def remove(x: np.ndarray, directions: list[np.ndarray]) -> np.ndarray:
    q, _ = np.linalg.qr(np.stack(directions, axis=1))  # orthonormal basis of the span
    return x - (x @ q) @ q.T


def main() -> None:
    acts = np.load(f"{STEM}.npy").astype(np.float64)
    side = json.load(open(f"{STEM}.json"))
    cells = {r["qid"]: r for r in map(json.loads, open("data/river-cells.jsonl", encoding="utf-8"))}
    items = side["items"]
    cell = np.array([it["cell"] for it in items])
    ho = np.array([it["held_out"] for it in items])
    country = np.array([cells[it["qid"]]["country_qid"] for it in items])
    for c in set(country):
        m = country == c
        acts[m] -= acts[m & ~ho].mean(axis=0)
    P, N = cell == "known-by-retrieval", cell == "unstable"
    tp, tn, hp, hn = P & ~ho, N & ~ho, P & ho, N & ho

    def arrow_at(l: int) -> np.ndarray:
        return acts[tp, l].mean(0) - acts[tn, l].mean(0)

    a0, a1 = arrow_at(0), arrow_at(1)
    cos = lambda u, v: float(abs(u @ v) / np.linalg.norm(u) / np.linalg.norm(v))
    rows = []
    print("layer | cos(L0,L) cos(L1,L) | AUC (p) | L1 removed: AUC (p) | L0+L1 removed: AUC (p)")
    for l in LAYERS:
        x = acts[:, l]
        base = held_out_auc(x, tp, tn, hp, hn)
        r1 = held_out_auc(remove(x, [a1]), tp, tn, hp, hn)
        r01 = held_out_auc(remove(x, [a0, a1]), tp, tn, hp, hn)
        row = {"layer": l, "abs_cos_L0": cos(a0, arrow_at(l)), "abs_cos_L1": cos(a1, arrow_at(l)),
               "auc": base[0], "perm_p": base[1], "auc_L1_removed": r1[0], "perm_p_L1_removed": r1[1],
               "auc_L0L1_removed": r01[0], "perm_p_L0L1_removed": r01[1],
               "share_of_L_arrow_norm_in_L1_direction": cos(a1, arrow_at(l))}
        rows.append(row)
        print(f"{l:5d} |   {row['abs_cos_L0']:.2f}      {row['abs_cos_L1']:.2f}   | {base[0]:.2f} ({base[1]:.3f}) |"
              f"      {r1[0]:.2f} ({r1[1]:.3f})     |       {r01[0]:.2f} ({r01[1]:.3f})")
    # Noise ceiling for the cosines: neighbouring layers, and split halves of the training towns.
    rng = np.random.default_rng(0)
    ip, ineg = np.flatnonzero(tp), np.flatnonzero(tn)
    halves = []
    for _ in range(50):
        sp, sn = rng.permutation(ip), rng.permutation(ineg)
        h1 = acts[sp[: len(sp) // 2], 21].mean(0) - acts[sn[: len(sn) // 2], 21].mean(0)
        h2 = acts[sp[len(sp) // 2:], 21].mean(0) - acts[sn[len(sn) // 2:], 21].mean(0)
        halves.append(cos(h1, h2))
    ceiling = {"abs_cos_L20_L21": cos(arrow_at(20), arrow_at(21)), "abs_cos_L1_L2": cos(a1, arrow_at(2)),
               "abs_cos_L21_split_halves_median": float(np.median(halves))}
    # Functional: the layer-1 arrow applied to layer-L activations.
    functional = {}
    for l in (1, 12, 21, 28):
        s = acts[:, l] @ a1
        functional[l] = arrow.auc(s[hp], s[hn])
    print("noise ceiling:", {k: round(v, 2) for k, v in ceiling.items()})
    print("layer-1 arrow scored on layer-L activations, held-out AUC:", {k: round(v, 2) for k, v in functional.items()})
    json.dump({"noise_ceiling": ceiling, "L1_arrow_on_layer_L_auc": functional, "activations": f"{STEM}.npy", "centre": "country (training-slot means)",
               "contrast": "known-by-retrieval vs unstable", "n_shuffles": N_SHUFFLES, "per_layer": rows},
              open("runs/river/surface.json", "w"), indent=1)


if __name__ == "__main__":
    main()
