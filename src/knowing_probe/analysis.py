"""One probe run: fit the arrow on training items, project everything, summarise, plot, shuffle control."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from knowing_probe import arrow, plot

N_SHUFFLES = 200


def run(acts: np.ndarray, layers: Sequence[int], cells: Sequence[tuple[str, np.ndarray]], pos: str, neg: str,
        held_out: np.ndarray, panel_layers: Sequence[int], out_dir: str | Path, name: str, title: str,
        meta: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks = dict(cells)
    train_pos, train_neg = masks[pos] & ~held_out, masks[neg] & ~held_out
    test_pos, test_neg = masks[pos] & held_out, masks[neg] & held_out
    fitted = arrow.fit(acts, train_pos, train_neg)
    scores = arrow.project(acts, fitted)

    per_layer = []
    for li, layer in enumerate(layers):
        s = scores[:, li]
        row: dict[str, Any] = {
            "layer": layer,
            "heldout_auc": arrow.auc(s[test_pos], s[test_neg]),
            "heldout_acc_at_midpoint": arrow.threshold_acc(s[test_pos], s[test_neg]),
            "train_auc": arrow.auc(s[train_pos], s[train_neg]),
        }
        for cname, m in cells:
            # cells the arrow was fitted on: held-out items only; other cells never touched it
            mm = m & held_out if cname in (pos, neg) else m
            ok = mm.any() and not np.isnan(s[mm]).any()
            row[f"{cname}: frac_above_threshold"] = float((s[mm] > 0.5).mean()) if ok else None
            row[f"{cname}: median_position"] = float(np.median(s[mm])) if ok else None
        per_layer.append(row)

    # Control: labels shuffled among the training items, arrow refitted, held-out AUC recomputed.
    shuffled_auc = np.zeros((N_SHUFFLES, len(layers)))
    for k in range(N_SHUFFLES):
        sp, sn = arrow.shuffled(train_pos, train_neg, seed=k)
        ss = arrow.project(acts, arrow.fit(acts, sp, sn))
        shuffled_auc[k] = [arrow.auc(ss[test_pos, li], ss[test_neg, li]) for li in range(len(layers))]
    # A shuffled arrow is the difference of two random halves' means, which points along the
    # highest-variance directions with a random sign. If the real contrast lies on such a
    # direction, shuffled held-out AUCs pile up near 0 AND 1 (FINDINGS, Step 0). So the
    # control is read by sign (fraction above 0.5, should be ~0.5) and by the permutation
    # p-value P(shuffled AUC >= real AUC), not by a band around 0.5.
    train = train_pos | train_neg
    for li, row in enumerate(per_layer):
        sh = shuffled_auc[:, li]
        row["shuffled_heldout_auc_p5_p50_p95"] = [float(x) for x in np.percentile(sh, [5, 50, 95])]
        row["shuffled_frac_auc_above_half"] = float(np.mean(sh > 0.5))
        row["perm_p_shuffled_ge_real"] = float(np.mean(sh >= row["heldout_auc"])) if not np.isnan(row["heldout_auc"]) else None
        x = acts[train, li, :].astype(np.float64)
        x -= x.mean(axis=0)
        _, sv, vt = np.linalg.svd(x, full_matrices=False)
        d = fitted[0][li]
        nd = np.linalg.norm(d)
        row["pc1_variance_share"] = float(sv[0] ** 2 / (sv ** 2).sum())
        row["abs_cos_arrow_pc1"] = float(abs(vt[0] @ d) / nd) if nd > 0 else None
    sp, sn = arrow.shuffled(train_pos, train_neg, seed=0)
    shuffled_scores = arrow.project(acts, arrow.fit(acts, sp, sn))

    titles = {r["layer"]: f"layer {r['layer']} · held-out AUC {r['heldout_auc']:.2f}" for r in per_layer}
    plot.strips(scores, cells, held_out, layers, panel_layers, titles, title, out_dir / f"{name}-strips.png")
    sh_titles = {l: f"layer {l} · held-out AUC {arrow.auc(shuffled_scores[test_pos, li], shuffled_scores[test_neg, li]):.2f}"
                 for li, l in enumerate(layers)}
    plot.strips(shuffled_scores, cells, held_out, layers, panel_layers, sh_titles,
                f"CONTROL, labels shuffled within training items · {title}", out_dir / f"{name}-shuffled-strips.png")
    plot.curves(layers, [("held-out AUC", [r["heldout_auc"] for r in per_layer]),
                         ("held-out accuracy at training midpoint", [r["heldout_acc_at_midpoint"] for r in per_layer]),
                         ("|cos(arrow, PC1 of training items)|", [r["abs_cos_arrow_pc1"] for r in per_layer]),
                         ("permutation p: P(shuffled AUC ≥ real)", [r["perm_p_shuffled_ge_real"] for r in per_layer])],
                (np.percentile(shuffled_auc, 5, axis=0), np.percentile(shuffled_auc, 95, axis=0)),
                f"{pos} vs {neg}", title, out_dir / f"{name}-by-layer.png")

    summary = {**meta, "contrast": {"positive": pos, "negative": neg},
               "cell_sizes": {c: {"train": int((m & ~held_out).sum()), "held_out": int((m & held_out).sum())} for c, m in cells},
               "n_shuffles": N_SHUFFLES, "projection": "0 = training negative mean, 1 = training positive mean; threshold 0.5",
               "per_layer": per_layer}
    (out_dir / f"{name}-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    np.save(out_dir / f"{name}-scores.npy", scores)
    return summary


def print_table(summary: dict[str, Any], extra_cells: Sequence[str] = ()) -> None:
    cols = "".join(f" | {c[:14]:>14s} >thr" for c in extra_cells)
    print(f"layer | AUC   | acc   | perm p | sh>0.5 | cosPC1{cols}")
    for r in summary["per_layer"]:
        ex = "".join(f" | {r[f'{c}: frac_above_threshold']:>18.2f}" if r[f"{c}: frac_above_threshold"] is not None
                     else f" | {'-':>18s}" for c in extra_cells)
        if r["perm_p_shuffled_ge_real"] is None:
            print(f"{r['layer']:5d} | zero arrow")
            continue
        print(f"{r['layer']:5d} | {r['heldout_auc']:.3f} | {r['heldout_acc_at_midpoint']:.3f} | "
              f"{r['perm_p_shuffled_ge_real']:.3f}  | {r['shuffled_frac_auc_above_half']:.2f}   | "
              f"{r['abs_cos_arrow_pc1'] or 0:.2f}  {ex}")
