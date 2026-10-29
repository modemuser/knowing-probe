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
`scripts/river_probe.py`, `scripts/river_controls.py`; figures in
`figures/river/prompt-last/`, numbers in `runs/river/prompt-last/*-summary.json` and
`runs/river/controls.json`.

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

**Is it the cell size?** Only for a dominant direction. Cities subsampled to the
river's exact sizes (13 + 7 vs 55 + 28, one statement per city, 300 draws) still
give a median held-out AUC of 1.000 from layer 16 (5th percentile ≥ 0.94). But
true/false is PC1 of those layers, the largest axis of variance in the stream: an
unusually easy target. A knowing direction a tenth that size would at 13 training
and 7 held-out towns look exactly like what we see: AUC 0.70–0.75, p ≈ 0.06, an
arrow that overfits its training towns. *(Corrected 11:40 after Thomas's review; the
first version said "not a lack of data", which the check does not support.)*

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
code. Graded labels give at best a weak trend; a dominant direction would not hide
behind it at this n, a modest one could. On the plan's rule that points at the
labels (or their number), not the probe.
Three readings remain open. (a) The pre-answer state carries no strong knowing
signal at this position. (b) Known-by-retrieval (20 towns, 6 FR, 5 ES, 0 SE/PT/IT) is too
thin and too country-skewed to define the positive pole. (c) The contrast is subtler
than true/false and needs more items, not a different probe. The targeted second
Wikidata pass is for known-by-retrieval, balanced by country, and right-by-prior
(the "correct because it is the default river" contamination of known that the
prior cells were meant to catch). The prior cells are not its target.

**Outcome.** A null for a dominant direction; **underpowered for a modest one**
(cell sizes above). A weak late-layer known-vs-unstable trend (AUC ≈ 0.7–0.75,
layers 19–29, uncorrected p ≈ 0.03–0.07 over 36 layers) survives country-centring.
It becomes a pre-registered test on the second pass (next entry). Per the rules: no
logistic regression, 2D projection or ablation until this is followed up.

## 2026-10-29 · Entity token and country-centring: higher AUC at the town name, mostly the name's surface (demoted, see next entry)

**Setup.** Same 200 slots, labels, split and code as the first river probe. Two
changes, each run alone and together. (1) Position: the last token of the town name
in the question (`river_acts.py --position entity_last`; the following "?" is always
its own token, so no pick runs past the name). (2) `--centre country`: before fitting,
subtract from every slot the mean of its country's *training* slots (all five cells).
P1 was pre-registered and committed (50500b5) before this pass ran. Figures in
`figures/river/{prompt-last,entity-last}[-country-centred]/`, numbers in
`runs/river/<variant>/`.

**Known vs unstable, held-out AUC (permutation p):**

| layer | last prompt token | + country-centred | entity token | + country-centred |
|---|---|---|---|---|
| 0 | zero arrow | zero arrow | 0.79 (0.00) | 0.66 (0.01) |
| 1 | 0.52 | 0.50 | 0.86 (0.00) | 0.76 (0.015) |
| 12 | 0.53 | 0.47 | 0.79 (0.05) | 0.85 (0.00) |
| 21 | 0.75 (0.055) | 0.74 (0.075) | 0.84 (0.01) | 0.83 (0.005) |
| 24 | 0.70 (0.055) | 0.72 (0.07) | 0.79 (0.02) | 0.80 (0.00) |
| 28 | 0.69 (0.17) | 0.72 (0.035) | 0.74 (0.085) | 0.82 (0.00) |

- **Country-centring does not explain the late-layer trend at the last prompt
  token**: 0.70–0.75 in layers 19–24 either way. It was explaining the known vs
  country-prior contrast (0.84 → 0.78–0.83 centred at the last prompt token).
- At the entity token, held-out AUC is 0.80–0.85 over layers 12–28 after centring
  (permutation p ≤ 0.005 at 21–28). *First written as "the separation is stronger
  [and] the entity-recognition position carries more than the end of the prompt".
  Demoted at 11:57: raw layer 0, the bare token embedding, is already 0.79, higher
  than the last prompt token ever reached. The next entry shows that a layer-1 arrow
  does as well at layer 21.*
- **But part of it is surface.** Layer 0 is the bare embedding of the name's last
  subword, with no context at all, and it already gives 0.66 centred (0.79 raw). Layer 1
  gives 0.76. Name length in tokens and word-initial endings do not differ between
  cells (3.5–4.3 tokens, no single-token names), so the cue is in which subwords
  obscure and familiar names end in. Mid layers add about 0.07–0.09 AUC over layer 1.
  Seven held-out known towns cannot tell that apart from zero. Pre-registered as
  control P1a (after seeing this, before any new town exists): P1 only counts as
  beyond-the-surface if AUC(L21) − AUC(L1) on the new towns has a paired bootstrap
  interval above 0.
- The layer-8 strip plot shows the overfitting directly. Training known towns sit
  at 0.8–1.3, held-out known towns at 0.0–0.4 (an arrow from 13 items in 4096
  dimensions).

**Right-by-prior moves with position.** Medians on the country-centred
known-vs-unstable arrow; held-out only for the two fitted cells:

| | known (held out) | right-by-prior | parent-river | country-prior | unstable (held out) |
|---|---|---|---|---|---|
| last prompt token, L22 | 0.75 | 0.60 | 0.19 | 0.13 | 0.31 |
| entity token, L21 | 0.69 | 0.23 | 0.19 | 0.07 | 0.03 |
| entity token, L28 | 0.54 | 0.27 | 0.10 | 0.12 | 0.12 |

At the town name, right-by-prior (correct, confident, and the default river of
its region) sits with the unknown cells, 24–29 % above threshold in layers 12–28.
At the end of the prompt it sat between. On the plan's reading, the entity-token
arrow tracks retrieval rather than output confidence. Hint only: 17 towns, and the
layer-1 arrow already puts them low (0.15), so a lexical template fitted on 13 known
names may do part of this too. Registered as descriptive S2 for the second pass.

**Status.** Underpowered for a modest direction, as before. The entity token is
the better candidate, with the lexical floor as its main threat. Both are now
fixed in PREREGISTRATION.md (P1, P1a, S1, S2), to be scored once on the second pass.

## 2026-10-29 · The layer-1 (surface) arrow does as well at layer 21 as layer 21's own arrow

**Question** (Thomas). The residual stream is additive, so unless something
overwrites it, the token embedding is still present at layer 21. Is the entity-token
separation one surface direction carried through the stream?

**Setup.** Entity token, country-centred (training-slot means), known-by-retrieval
vs unstable, arrows on the 13 + 55 training towns, held-out 7 + 28, 200 shuffles.
`scripts/river_surface.py` → `runs/river/surface.json`.

| layer | \|cos(L1 arrow, L arrow)\| | AUC (p) | L1 arrow removed: AUC (p) | L0 + L1 removed: AUC (p) |
|---|---|---|---|---|
| 4 | 0.67 | 0.76 (0.045) | 0.69 (0.11) | 0.69 (0.12) |
| 12 | 0.21 | 0.85 (0.000) | 0.83 (0.015) | 0.83 (0.015) |
| 21 | 0.16 | 0.83 (0.005) | 0.81 (0.010) | 0.81 (0.015) |
| 28 | 0.21 | 0.82 (0.000) | 0.80 (0.000) | 0.80 (0.000) |

Taken alone, this table says "different directions, removing the surface costs
nothing". **Both readings fail the noise check:**
- Two layer-21 arrows from random halves of the training towns agree at a median
  |cos| of only **0.12**. At 13 known towns an arrow in 4096 dimensions is mostly
  noise, so a cosine of 0.16 between the L1 and L21 arrows is at the noise floor and
  says nothing about shared direction. Adjacent layers (L20/L21: 0.95) share the
  same noise, which is why they look alike.
- Projecting out one noisy direction removes almost none of the surface
  information, so the unchanged AUC after removal is no evidence either.
- **Functional test, which does not depend on the arrows being clean:** the layer-1
  arrow applied to layer-L activations gives held-out AUC 0.76 at L1, **0.85 at
  L12, 0.81 at L21, 0.80 at L28**. That is the same as each layer's own arrow. A
  direction fitted on the name's surface ranks the held-out towns at layer 21 as
  well as anything fitted at layer 21.

**Reading.** Today's towns cannot tell knowledge from surface at the entity token.
The surface being the whole story is consistent with every number so far. The extra
0.07 over layer 1 is the only part not yet explained, and it is within the noise of
7 held-out towns. **P1 as originally written could pass for the wrong reason.**

**Why the floor exists, and what it means for the second pass** (Thomas). Known
towns are famous. Famous names occur more often in the tokenizer's training corpus
and end in different subwords, so the last subword is a fame proxy, and fame is
what the label tracks. Pulling famous towns for the known cell and leaving the
unstable towns obscure would strengthen the confound. Fix: draw *both* cells from
the same fame band. At ≥ 40 sitelinks, 82–92 % of towns are still not
known-by-retrieval, so known and unstable come out fame-matched and roughly
tokenization-matched. Registered as P1b, with a functional surface control and a
fame-only baseline.
