# knowing-probe

**Status: the pre-registered confirmatory test P1 has not been run. Everything below
is exploratory.** The second Wikidata pass that P1 needs (famous towns, ≥ 40
sitelinks, country-balanced, both cells drawn from the same fame band) has not been
graded.

Built on [contrastset](https://github.com/modemuser/contrastset) 0.1.0.post1 (PyPI),
which grades what the model knows; this repository reads activations and plots.
Records: [PREREGISTRATION.md](PREREGISTRATION.md) (tests fixed before their data
exist, with the outcome paragraphs written in advance), [FINDINGS.md](FINDINGS.md)
(dated results, with errata), [NOTEBOOK.md](NOTEBOOK.md) (every run and decision in
order, mistakes included).

## Question

Does Qwen3-8B's residual stream carry a direction that separates facts the model
retrieves from facts it doesn't, independently of the entity's surface form and its
country? The facts are the rivers of European towns ("Which river flows through
Tábor?"), graded by what the model answers. Terms used throughout:
- **Residual stream**: the vector each transformer block reads from and adds to, one
  per token. Layer 0 is the token embedding, layer *i* the output of block *i* (36
  blocks), read before the final norm.
- **Direction**: a fixed vector in that 4096-dimensional space; a slot's score is
  its activation projected onto it.
- **Difference-of-means arrow**: per layer, the mean activation of one cell minus
  the mean of another, computed on training towns only (Marks and Tegmark's
  mass-mean probe without the covariance correction). Scores are rescaled so that
  the negative cell's training mean is 0 and the positive cell's is 1; the threshold
  is 0.5.
- **AUC** (area under the ROC curve): the probability that a random positive item
  scores higher than a random negative one. 0.5 is chance, 1.0 is perfect separation.
  Reported on held-out items, with a permutation p (the share of 200 label-shuffled
  arrows that reach the real held-out AUC).

## Step 0: the probe stage reproduces Marks and Tegmark

The same code (hooks, position, split, arrow, plots) was first run on Marks and
Tegmark's true/false statements, where the direction is known to exist. Statements
were fed as plain text; the position was the last token (the full stop); the split was
by item, a third held out.
- `cities`, 1496 statements, held out 498 statements (249 cities): held-out AUC 0.94
  at layer 8, 0.999 at layer 15, 1.00 from layer 16 on.
- `sp_en_trans`, 354 statements, held out 118 (115 Spanish words): 0.99 at layer 8,
  1.00 from layer 13 on.

The library repeats this as
[example 00](https://github.com/modemuser/contrastset/tree/v0.1.0.post1/examples/00-marks-tegmark),
with identical AUCs (NOTEBOOK.md, 15:21).

**The shuffled-label control does not fall to a band around 0.5, and why.** With labels
shuffled among the training items, held-out AUCs at layer 16 spread from 0.00 to
1.00 (5th–95th percentile, both sets). In layers 16–36 the true/false arrow *is* the
first principal component (PC1) of the training activations: |cos| 0.97–1.00, with
PC1 carrying 28–48 % of the variance (cities, layers 16–24). A difference of two random
halves' means points along the highest-variance directions, so a shuffled arrow lands
on the truth axis with a random sign. The control is sign-balanced (47–57 % of shuffled
arrows above 0.5 at every layer) but not centred. Read as a permutation test, p is
0.005–0.105 for cities and 0.10–0.31 for sp_en_trans in layers 16–36, where the real AUC
is at or near 1.00 and random-sign arrows also reach it. In those layers the labels
contribute only the sign. Figures: `figures/step0/`.

## Cells

Input: contrastset's grade of 294 Wikidata towns from ten countries (DE, AT, PL, CZ,
SK, FR, IT, ES, PT, SE) on Qwen3-8B bf16 (revision b968826), template B ("Answer with
the name only", no "I don't know" line), log-prob grading at T = 1, committed as
`data/wikidata-qwen3-8b-hf-logprob-B-priors.jsonl`. 252 towns pass the country gate;
200 river slots fall into a cell once every slot is required to be familiar (country
p_correct ≥ 0.875) and the middle band (0.125 < p_correct < 0.875) is dropped.

| cell | towns (train + held out) | rule (p = p_correct for the river) |
|---|---|---|
| known-by-retrieval | 20 (13 + 7) | p ≥ 0.875, gold river is not the region's default |
| right-by-prior | 17 (11 + 6) | p ≥ 0.875, gold river is the region's default |
| parent-river | 19 (13 + 6) | p ≤ 0.125, wrong answer is a river downstream of the gold one, and passes the basin rule |
| country-prior | 61 (41 + 20) | p ≤ 0.125, wrong answer is a prior: a basin-rule failure (20), a country-shared answer (34) or the regional default (7) |
| unstable | 83 (55 + 28) | p ≤ 0.125, none of the above |

**The basin rule.** A wrong answer downstream of the truth (Wikidata P403 chain) counts
as basin-level only if no same-country town outside that basin gets the same answer.
All 12 country-flagged prior guesses that name a downstream river fail it. Vltava, for
example, is the modal answer for 16 Czech towns, 10 of them outside its basin (six on
the Elbe, which the Vltava flows into). Applying the rule to every slot with
p ≤ 0.125, not only to flagged ones, moved 34 slots out of unstable (19 to parent-river,
15 to country-prior).

**Why known-by-retrieval has only 20 towns: fame.** Its rate by Wikipedia sitelinks:
0 of 8 towns below 20, 1 of 93 at 20–39, 13 of 159 at 40–79 (8 %), 6 of 34 at 80 or
more (18 %). No Italian, Portuguese or Swedish town is in the cell. Numbers:
`runs/river/cells-summary.json`; cells: `data/river-cells.jsonl`; split by town, a
third held out, stratified by cell, seed 20261029.

## Exploratory results

Arrows are fitted on training towns; AUCs are held-out (7 known-by-retrieval vs 28
unstable, or vs 20 country-prior). Country-centring subtracts each country's mean
training activation before fitting. Figures: `figures/river/<variant>/`; numbers:
`runs/river/<variant>/*-summary.json`, `runs/river/controls.json`,
`runs/river/surface.json`.

1. **Last prompt token, known vs unstable.** Held-out AUC 0.70–0.75 at layers 19–24,
   permutation p 0.050–0.075 (uncorrected for 36 layers), 7 held-out known towns;
   0.71–0.75 after country-centring. *Caveat:* a null for a dominant direction,
   underpowered for a modest one. Cities subsampled to these exact cell sizes still
   give a median AUC of 1.000 from layer 16 (300 draws), but true/false is PC1 there, an
   unusually easy target.
2. **Last prompt token, known vs country-prior.** 0.79–0.89 at layers 19–28. *Caveat:*
   country alone, with no activations, scores 0.84: the known cell has no Swedish,
   Portuguese or Italian town, while country-prior is mostly CZ, SE, SK, ES and PT. After
   country-centring it is 0.78–0.83 at layers 19–24 (p 0.045–0.080).
3. **Town-name token** (last token of the name in the question), known vs unstable.
   Raw: 0.79 at layer 0 already. Country-centred: 0.66 at layer 0, 0.76 at layer 1,
   0.77–0.85 at layers 12–28 (0.83 at layer 21, p 0.005).
   *The lexical floor:* layer 0 is the name's last subword with no context, and it
   already separates the cells, because famous names are tokenized differently from
   obscure ones. Name length in tokens does not differ between cells (3.5–4.3 on
   average), so it is the identity of the subwords. The middle layers add 0.07–0.09 over
   layer 1, which 7 held-out known towns cannot distinguish from zero. **The
   entity-token result cannot yet be told apart from the name's surface form.** Two
   further checks were run (`scripts/river_surface.py`):
   - The layer-1 and layer-21 arrows overlap at |cos| 0.16, and projecting the layer-1
     direction out of layer 21 leaves AUC 0.81 (p 0.010). But two layer-21 arrows from
     random halves of the training towns overlap at a median |cos| of only 0.12, so
     both numbers sit at the noise floor of a 13-town arrow and decide nothing.
   - The layer-1 arrow applied to layer-21 activations gives 0.81, the same as layer
     21's own arrow (0.83). A direction fitted on the surface ranks held-out towns at
     layer 21 as well as anything fitted there.
4. **Right-by-prior** (17 towns, none used to fit any arrow). Median position on the
   country-centred known-vs-unstable arrow: 0.49 at the last prompt token (layer 22),
   0.23 at the town name (layer 21); raw arrows give 0.60 and 0.27. *Caveat:* a hint
   only, and the layer-1 arrow already puts these towns low. If it replicates on 30 or
   more new towns, it would support the sentence "the arrow tracks retrieval, not answer
   confidence". It has not replicated, because it has not been tested.

## What would settle it

- **P1** ([PREREGISTRATION.md](PREREGISTRATION.md)): the frozen known-vs-unstable
  arrow, town-name token, layer 21, country-centred, fitted on all 103 of today's
  towns in the two cells, scored once on new towns, one-sided permutation p < 0.05.
- **P1a and P1c**, the surface controls: P1 counts as more than the name's surface only
  if layer 21 beats the layer-1 arrow at layer 1 (P1a) and the layer-1 arrow read out at
  layer 21 (P1c). Both comparisons are paired on identical towns, with bootstrap
  intervals above zero.
- **The second pass** (P1b): every new town from the ≥ 40-sitelink band, so known and
  unstable are fame-matched; no population cap; ≥ 40 known-by-retrieval and ≥ 40
  unstable; new towns centred on their own country means; direct HF log-prob grading
  of about 350 towns (about 4.5 h at 46 s per town).
- **The outcome paragraphs** for pass and for failure are written in advance:
  [PREREGISTRATION.md, last section](PREREGISTRATION.md#2026-10-29--before-publication-hashes-confirmed-ratified-choices-outcome-paragraphs-p1-still-unrun).
- The frozen arrows and their sha256 are committed in `runs/prereg/`;
  `scripts/prereg_p1.py` refuses to refreeze an arrow or to score twice.

## Open

- **Country balance of the second pass.** The candidate pull (contrastset
  `experiments/wikidata-towns-pass2.jsonl`, 168 towns, ≥ 40 sitelinks) has 1 Polish,
  1 Portuguese and 2 Swedish towns. Few towns in those countries have both ≥ 40
  sitelinks and a Wikidata river statement. "Country-balanced" as registered is not
  reachable from this source; how to fill the pass is undecided.
- P1, P1a, P1c and the descriptive S1/S2/D1/D2 are unrun.
- No layer search, logistic regression, 2D projection, ablation or steering has been
  run on purpose: they wait for P1.

## Corrections on record

Claims that were withdrawn or corrected, all kept visible where they were made:
- The first river write-up called the result "not a lack of data". The power check
  supports that only for a PC1-strength direction, and the claim is now "null for a
  dominant direction, underpowered for a modest one" (FINDINGS.md).
- The town-name result was first headlined as stronger than the last prompt token. It
  is demoted: the bare token embedding already gives 0.79 (FINDINGS.md).
- Four ranges in FINDINGS.md were misquoted from the result files. Among them, a
  right-by-prior comparison mixed a raw and a country-centred arrow (0.60 → 0.23; like
  for like 0.49 → 0.23). They are corrected as errata, with the original sentences
  kept.
- In the library's history: a second-clause advantage for half-known towns (Tábor
  8/15 second vs 0/15 first) failed its pre-specified replication on 14 middle-band
  towns
  ([contrastset FINDINGS](https://github.com/modemuser/contrastset/blob/main/FINDINGS.md#2026-10-28--experiment-1-probe-no-second-clause-advantage-for-half-known-towns)).

The order of runs, reviews and decisions is in [NOTEBOOK.md](NOTEBOOK.md).

## Reproduce

Python ≥ 3.11, [uv](https://docs.astral.sh/uv/). Times are on an M1 Pro (32 GB, MPS,
bf16).

```bash
uv sync --extra hf                       # torch 2.14.1, transformers 5.21.0, contrastset 0.1.0.post1
uv run pytest -q                         # fast tests, no model
```

Step 0 (needs the model; about 10 min of activations at 0.28 s per statement):

```bash
./scripts/fetch_mt.sh                    # CSVs at geometry-of-truth commit 5d1c630, sha256-checked
uv run python scripts/mt_acts.py         # runs/step0/*.npy (1.1 GB, git-ignored) + .json sidecars
uv run python scripts/mt_probe.py        # figures/step0/, runs/step0/*-summary.json
uv run python scripts/step0_split.py     # data/step0-split.json
```

Cells, from the committed grade (no model, seconds). The grade itself is an input
(3 h 46 min with contrastset); it is not regenerated here:

```bash
G=data/wikidata-qwen3-8b-hf-logprob-B-priors.jsonl
uv run python scripts/river_wikidata.py $G -o data/river-mouths.json   # P403 chains (network)
uv run python scripts/river_cells.py $G data/river-mouths.json -o data/river-cells.jsonl
uv run python scripts/river_cells_summary.py                          # runs/river/cells-summary.json
```

Per-layer sweeps (activations about 1.5 min per position at 0.40–0.44 s per slot; each
probe variant a few minutes for 200 shuffles):

```bash
uv run python scripts/river_acts.py data/river-cells.jsonl $G -o runs/river/acts
uv run python scripts/river_acts.py data/river-cells.jsonl $G -o runs/river/acts-entity --position entity_last
for a in runs/river/acts runs/river/acts-entity; do
  uv run python scripts/river_probe.py $a
  uv run python scripts/river_probe.py $a --centre country
done
uv run python scripts/river_controls.py   # country-only baseline, centred arrows, power at river cell sizes
uv run python scripts/river_surface.py    # arrow overlap, projection-out, layer-1 arrow at layer 21
```

Do not run `scripts/prereg_p1.py freeze`: the arrows are frozen and committed, and the
script refuses. `score` is for the second pass only, once.

## Related work

- Marks & Tegmark, *The Geometry of Truth: Emergent Linear Structure in Large Language
  Model Representations of True/False Datasets*, arXiv:2310.06824 (difference-of-means
  probes, per-layer sweep; Step 0 reproduces it).
- Ferrando, Obeso, Rajamanoharan & Nanda, *Do I Know This Entity? Knowledge Awareness and
  Hallucinations in Language Models*, arXiv:2411.14257 (known/unknown entity latents
  at the entity's final token, middle layers, Gemma 2; the reason for the town-name
  position).
- Azaria & Mitchell, *The Internal State of an LLM Knows When It's Lying*,
  arXiv:2304.13734.
- Bürger, Hamprecht & Nadler, *Truth is Universal: Robust Detection of Lies in LLMs*,
  arXiv:2407.12831 (a general and a polarity-sensitive truth direction).
- Activation-based detectors are fragile when the model is trained against them
  (Gupta & Jenner, *RL-Obfuscation: Can Language Models Learn to Evade Latent-Space
  Monitors?*, arXiv:2506.14261). The single-direction picture of deception was rejected
  for the Gemma 3 family (Kumar, *Pressure-Testing Deception Probes in LLMs: Scaling,
  Robustness, and the Geometry of Deceptive Representations*, arXiv:2605.27958).

## Licence

MIT. The Marks and Tegmark CSVs are not redistributed; `scripts/fetch_mt.sh` fetches
them.
