# Findings

Results of the probe experiments on Qwen/Qwen3-8B, kept with the data and code that
produced them. Datasets come from [contrastset](../contrastset) (graded river slots)
or from public labelled sets (Step 0). Each entry says what was measured and how far
it can be trusted.

## 2026-10-29 · Step 0: expected result, written before the run

Reproduction target: Marks and Tegmark 2023, *The Geometry of Truth*. On LLaMA-2-13B
(40 layers) true and false statements separate linearly from the early-middle layers
on (their visualisations use layer ~12 of 40), and the mass-mean probe reaches
held-out accuracy in the high 0.9s on `cities`; `sp_en_trans` is also high but
noisier.

Expected on Qwen3-8B (36 blocks, layers 0–36), same procedure:
- `cities`: held-out AUC near chance in layers 0–5, separation emerging around
  layers 8–12 (25–35 % depth), held-out threshold accuracy ≥ 0.90 from about layer
  12 to at least layer 28; may sag in the last few layers.
- `sp_en_trans`: the same shape, lower ceiling (≥ 0.80).
- Shuffled labels: held-out AUC ≈ 0.5 at every layer.

If neither set separates in the middle layers, the probe stage is broken (hook
position, layer indexing, split leakage, projection), and the river data waits.

## 2026-10-29 · Step 0: the probe stage reproduces Marks and Tegmark

**Setup.** `cities` (1496 statements) and `sp_en_trans` (354), from
saprmarks/geometry-of-truth at commit 5d1c630 (file hashes in the sidecars). Qwen3-8B
bf16, revision b968826, MPS. Plain statement text, no chat template. Residual at the
last token (the full stop in every statement), all 37 layers, read with forward
hooks before the final norm (`src/knowing_probe/acts.py`). Split by item: one third of
cities (249) or Spanish words (115) held out, seed 20261029. Arrow = mean(true) −
mean(false) on training items, per layer; threshold at the training midpoint. 200
label shuffles. `scripts/mt_acts.py`, `scripts/mt_probe.py`; figures in
`figures/step0/`, numbers in `runs/step0/*-summary.json`.

**Result** (held-out):

| layer | cities AUC | cities acc | sp_en_trans AUC | sp_en_trans acc |
|---|---|---|---|---|
| 1–3 | 0.49–0.52 | 0.51 | 0.49–0.58 | 0.52–0.57 |
| 4 | 0.87 | 0.78 | 0.71 | 0.65 |
| 8 | 0.94 | 0.87 | 0.99 | 0.96 |
| 12 | 0.96 | 0.88 | 1.00 | 0.98 |
| 16 | 1.00 | 0.97 | 1.00 | 1.00 |
| 24 | 1.00 | 0.99 | 1.00 | 1.00 |
| 32 | 0.99 | 0.97 | 1.00 | 1.00 |
| 36 | 0.99 | 0.95 | 1.00 | 0.99 |

Layer 0 gives a zero arrow: every statement ends in the same "." token, so its
embedding is identical across items. That is the expected outcome and confirms the position rule.

**Against the expectation.** Separation starts earlier than written down (layer 4
rather than 8–12, near-perfect from layer 7 for sp_en_trans and layer 15 for cities)
and does not sag at the end. The shape matches: chance in layers 1–3, a rise, then a
plateau across the middle and late layers. **The probe stage is validated**: correct
sign, held-out by item, accuracy ≥ 0.95 over most of the depth.

**The shuffle control does not collapse to chance the way the plan expected. The
reason is a property of the data, not a leak.** Shuffled-label arrows give held-out
AUCs spread over the whole 0–1 range in the middle layers (5th–95th percentile 0.00–1.00),
not a band around 0.5. From layer 9 (cities) or 8 (sp_en_trans) on, the true/false
arrow *is* the first principal component (PC1) of the training activations: |cos| 0.87–1.00,
and PC1 carries 28–48 % of the variance (cities, layers 16–24). The difference of two
random halves' means points along the highest-variance directions, so a shuffled
arrow lands on the truth axis with a random sign. The control collapses in sign
(47–56 % of shuffled arrows above 0.5 at every layer) but not in magnitude. Read as a
permutation test, P(shuffled AUC ≥ real) is 0.000 for cities in layers 4–13, then
0.005–0.075 in layers 14–36. For sp_en_trans it is 0.000–0.045 in layers 4–14 and 0.10–0.28
beyond, where the real AUC sits at 1.000 and a quarter of random-sign arrows also
reach it. In those layers the labels contribute only the sign: an unsupervised PC1
finds the same axis, as Marks and Tegmark's PCA plots show.

**Consequence for the river probe.** The analysis now records, per layer, the
permutation p, the sign fraction and |cos(arrow, PC1)|. If the known-vs-unstable
arrow is *not* PC1, a shuffled arrow has nothing to fall onto, and the control
is informative. If it is PC1, the labels contribute only the sign there too, and the
question becomes what PC1 is (fame, country, name length) rather than knowledge.
