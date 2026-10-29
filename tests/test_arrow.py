import numpy as np

from knowing_probe import arrow


def test_split_keeps_groups_together() -> None:
    groups = ["a", "a", "b", "b", "c", "d", "d", "e", "f"]
    held = arrow.split_groups(groups, 1 / 3, seed=1)
    for g in set(groups):
        assert len({h for gg, h in zip(groups, held) if gg == g}) == 1
    assert sum(1 for g in set(groups) if held[groups.index(g)]) == 2


def test_projection_scale_and_auc() -> None:
    rng = np.random.default_rng(0)
    d = 16
    direction = rng.normal(size=d)
    pos = rng.normal(size=(40, 1, d)) + direction
    neg = rng.normal(size=(40, 1, d)) - direction
    acts = np.concatenate([pos, neg])
    p = np.array([True] * 40 + [False] * 40)
    fitted = arrow.fit(acts, p, ~p)
    s = arrow.project(acts, fitted)[:, 0]
    assert np.isclose(s[p].mean(), 1) and np.isclose(s[~p].mean(), 0)
    assert arrow.auc(s[p], s[~p]) > 0.95
    assert arrow.auc(np.array([1.0]), np.array([1.0])) == 0.5


def test_shuffle_permutes_within_training_items() -> None:
    pos = np.array([True, True, False, False, False, False])
    neg = np.array([False, False, True, True, False, False])
    sp, sn = arrow.shuffled(pos, neg, seed=3)
    assert sp.sum() == 2 and sn.sum() == 2
    assert not (sp | sn)[4:].any() and not (sp & sn).any()
