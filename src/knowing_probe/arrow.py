"""The arrow: per layer, mean(positive) minus mean(negative) on training items only.

Difference of means (Marks and Tegmark's mass-mean probe without the covariance
correction). Every item is projected onto the arrow and rescaled so that, on the
TRAINING items of that layer, the negative class mean sits at 0 and the positive
class mean at 1. The threshold is the training midpoint, 0.5, fixed before any
held-out item is looked at.

AUC (area under the ROC curve): the probability that a random positive item projects
higher than a random negative one; 0.5 is chance, 1.0 is perfect separation.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence

import numpy as np


def split_groups(groups: Sequence[Hashable], held_out_frac: float, seed: int) -> np.ndarray:
    """Boolean mask, True = held out. Whole groups (towns, cities, words) go one way."""
    uniq = sorted(set(groups), key=str)
    rng = random.Random(seed)
    rng.shuffle(uniq)
    held = set(uniq[: round(len(uniq) * held_out_frac)])
    return np.array([g in held for g in groups])


def fit(acts: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """acts [n, layers, d]; pos/neg boolean masks over n (training items only).

    Returns arrows [layers, d], and per layer the projection of the negative and
    positive training means (for rescaling)."""
    mu_pos = acts[pos].mean(axis=0)
    mu_neg = acts[neg].mean(axis=0)
    d = mu_pos - mu_neg
    lo = np.einsum("ld,ld->l", mu_neg, d)
    hi = np.einsum("ld,ld->l", mu_pos, d)
    return d, lo, hi


def project(acts: np.ndarray, arrow: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """[n, layers] positions on the arrow: 0 = training negative mean, 1 = training positive mean."""
    d, lo, hi = arrow
    raw = np.einsum("nld,ld->nl", acts, d)
    span = hi - lo
    # a zero arrow (both class means equal, e.g. layer 0 when every item ends in the same token) projects to NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(span > 0, (raw - lo) / np.where(span > 0, span, 1), np.nan)


def auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    if len(pos_scores) == 0 or len(neg_scores) == 0 or np.isnan(pos_scores).any() or np.isnan(neg_scores).any():
        return float("nan")
    diff = pos_scores[:, None] - neg_scores[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def threshold_acc(pos_scores: np.ndarray, neg_scores: np.ndarray, t: float = 0.5) -> float:
    n = len(pos_scores) + len(neg_scores)
    if np.isnan(pos_scores).any() or np.isnan(neg_scores).any():
        return float("nan")
    return float(((pos_scores > t).sum() + (neg_scores <= t).sum()) / n) if n else float("nan")


def shuffled(pos: np.ndarray, neg: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The same training items with their pos/neg labels permuted among themselves."""
    idx = np.flatnonzero(pos | neg)
    labels = pos[idx].copy()
    np.random.default_rng(seed).shuffle(labels)
    p, n = np.zeros_like(pos), np.zeros_like(neg)
    p[idx[labels]] = True
    n[idx[~labels]] = True
    return p, n
