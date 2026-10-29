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

## 2026-10-29 · River cells: most "parent-river" prior guesses are a country prior in a basin costume

**Setup.** Graded file `contrastset/runs/wikidata-qwen3-8b-hf-logprob-B-priors.jsonl`
(294 towns, HF Qwen3-8B revision b968826, template B sha256 be013947…, log-prob grade).
River question only. Binned offline by `scripts/river_cells.py` into
`data/river-cells.jsonl`; mouth chains from Wikidata P403 (mouth of the
watercourse), walked transitively (`data/river-mouths.json`, 281 rivers).
p = log-prob p_correct at T = 1. Every slot familiar: country p_correct ≥ 0.875.
Middle band (0.125 < p < 0.875) dropped.

**The basin check decides it.** Taking a wrong modal answer that lies downstream of
the gold river (any number of P403 hops) at face value gives 39 parent-river slots,
12 of them contrastset's country-shared prior guesses (Vltava ×5, Váh ×2, Tagus ×2,
Ebro ×2, Dordogne). These 12 are the "about 12 of 13" from the label run. **All 12
fail the sanity check**: the same answer is also the modal answer for same-country
towns outside that basin. Vltava is said for 16 Czech towns, 6 in its basin and 10
outside (six of them on the Elbe itself, which the Vltava flows *into*). Váh: 3 in,
6 out. Dordogne: 2 in, 4 out (Garonne, Rhône and Seine towns). Tagus in Portugal: 3
in, 4 out. Ebro: 4 in, 3 out. These are country defaults that happen to sit
downstream of some towns, not basin knowledge.

What survives the check comes entirely from slots *without* the country-shared flag:
19 towns whose wrong answer is downstream of the truth and never given for an
out-of-basin town of the same country (Leine for Dassel on the Ilme, Neckar for
Crailsheim on the Jagst, Seine for Noisiel on the Marne, Morava ×3, Rhine ×3, …).
Nine of the 19 answers occur for only one or two towns in the pool, so for those
the check has little to bite on.

**Cells** (train + held out; split by town, one third, stratified by cell, seed 20261029):

| cell | n | definition |
|---|---|---|
| known-by-retrieval | 20 (13 + 7) | p ≥ 0.875, gold ≠ applicable default |
| right-by-prior | 17 (11 + 6) | p ≥ 0.875, gold = applicable default |
| parent-river | 19 (13 + 6) | p ≤ 0.125, modal answer downstream of gold, passes basin check |
| country-prior | 61 (41 + 20) | p ≤ 0.125, modal answer a prior: basin-check failure 20, country-shared flag 34, regional/country default 7 |
| unstable | 83 (55 + 28) | p ≤ 0.125, none of the above (63 bin unstable, 20 confidently_wrong) |

Applicable default: modal gold river among the *other* towns of the same first-level
region (leave-one-out, ≥ 2 towns, unique mode) when the model knows the region
(region p_correct ≥ 0.875), else the same over the country. Right-by-prior: Inn ×3,
Arno ×3, Elbe ×4, Vistula ×3, Adda ×2, Mur, Tarn.

**Changes from the prompt's definitions, and why.** (1) Prior-guess is no longer
"contrastset flag true". It is the union of the flag, the default rule and
on-chain answers, split by the basin check. 34 slots that would have been
"unstable" (no flag) moved to parent-river (19) or country-prior (15), because a
downstream or default answer is not "nothing". (2) The arrow's positive cell is
known-by-retrieval only. Right-by-prior never touches the arrow, so its position is
a clean test. (3) The split is stratified by cell so that thin cells get a held-out
third.

**Reading for the probe.** The resolution ladder is now exact (known-by-retrieval),
basin-level (parent-river, 19, partly weakly checked), and nothing (unstable), with
country-prior as a separate rung: a confident name that carries no town-specific
information.

## 2026-10-29 · First river probe: no strong known/unknown direction before the answer; a weak late-layer trend

**Setup.** The 200 slots of the cells above. One forward pass per slot under template B
(the grading string; the hash check passed: be013947…). Residual at the last prompt
token (`ĊĊ` after the empty think block, the token before the answer would begin),
all 37 layers, same code path as Step 0 (`scripts/river_acts.py`). Arrow on training
towns only. Contrast 1: known-by-retrieval (13 train / 7 held out) vs unstable
(55 / 28). Contrast 2: known-by-retrieval vs country-prior (41 / 20). Right-by-prior
and parent-river never touch either arrow. 200 label shuffles per contrast.
`scripts/river_probe.py`, `scripts/river_controls.py`; figures in `figures/river/`,
numbers in `runs/river/*-summary.json` and `runs/river/controls.json`.

**Contrast 1, known vs unstable (the first probe).**

| layers | held-out AUC | permutation p | \|cos(arrow, PC1)\| |
|---|---|---|---|
| 1–18 | 0.49–0.64 | 0.13–0.65 | 0.13–0.37 |
| 19–24 | 0.70–0.75 | 0.05–0.075 | 0.22–0.84 |
| 25–36 | 0.60–0.69 | 0.15–0.32 | 0.12–0.40 |

Threshold accuracy is not reported: with 7 vs 28 held-out towns, the majority guess
already scores 0.80. Training AUC is 0.82–0.95 at every layer, held-out at most 0.75.
In layers 8–16 the 13 training known towns sit far right and the held-out known towns
do not (strip plot). That is an arrow fitted to the quirks of 13 items in 4096 dimensions.

**Contrast 2, known vs country-prior.** Held-out AUC 0.79–0.89 in layers 19–28,
permutation p 0.000–0.030. **Not evidence of knowing: it is matched by country
alone.** Scoring each held-out town by its country's share of known towns in training
(no activations) gives AUC 0.84. Known-by-retrieval has no Swedish, Portuguese or
Italian towns; country-prior is 14 CZ, 9 SE, 9 SK, 8 ES, 7 PT. Country-centring the
activations (subtracting each country's mean training activation) leaves 0.78–0.83
in layers 19–26, p 0.02–0.10. Contrast 1 survives country-centring unchanged
(0.70–0.75 in layers 19–29); there country alone gives only 0.58.

**Is it the cell size?** No, not for an effect of Step 0's strength. Cities subsampled
to the river's exact sizes (13 + 7 vs 55 + 28, one statement per city, 300 draws)
still give a median held-out AUC of 1.000 from layer 16 (5th percentile ≥ 0.94). A
known/unknown direction as strong as the truth direction would have shown up. A
weak one (AUC around 0.7) is neither established nor ruled out by 7 held-out towns.

**Where the other cells fall** (median position on the arrow, 0 = unstable/prior mean,
1 = known mean; held-out only for the two fitted cells):

| contrast, layer | known (held out) | right-by-prior | parent-river | country-prior | unstable |
|---|---|---|---|---|---|
| 1, L22 | 0.75 | 0.60 | 0.19 | 0.13 | 0.31 (held out) |
| 1, L28 | 0.66 | 0.40 | 0.69 | 0.47 | 0.26 (held out) |
| 2, L22 | 0.82 | 0.38 | 0.12 | 0.00 (held out) | 0.00 |
| 2, L28 | 0.94 | 0.48 | 0.63 | −0.01 (held out) | 0.32 |

- Unstable lands on the unknown side of the known-vs-country-prior arrow (median
  0.0–0.4, 25–42 % above threshold, layers 20–28), as the plan asked to check, but that
  arrow is partly a country arrow.
- Right-by-prior sits in the middle on both arrows (0.4–0.6), neither with known nor
  with the unknown cells. On the plan's reading this is neither "the arrow tracks
  retrieval" nor "the arrow reads confidence". With 17 towns and an arrow this weak it
  is not a result.
- Parent-river moves from the unknown side (L20–22) to the known side (L28–32) on
  both arrows. That would fit "having the fact at some resolution", read late. It
  rests on 19 towns, half of them weakly basin-checked: a hint, nothing more.

**Bracketing (Step 0 vs river).** Free labels separate perfectly through the same
code. Graded labels give at best a weak trend that a strong direction would not
hide behind at this n. On the plan's rule that points at the labels, not the probe.
Three readings remain open. (a) The pre-answer state carries no strong knowing
signal at this position. (b) Known-by-retrieval (20 towns, 6 FR, 5 ES, 0 SE/PT/IT) is too
thin and too country-skewed to define the positive pole. (c) The contrast is subtler
than true/false and needs more items, not a different probe. The targeted second
Wikidata pass should fatten the known-by-retrieval cell first, balanced by country,
as well as right-by-prior and the prior cells.

**Outcome.** Reported as a null for a strong direction (cell sizes above), with
a weak late-layer known-vs-unstable trend (AUC ≈ 0.7–0.75, layers 19–29, uncorrected
p ≈ 0.03–0.07 over 36 layers) that survives country-centring. Per the rules: no
logistic regression, 2D projection or ablation until this is followed up.
