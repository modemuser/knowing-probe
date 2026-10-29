# Pre-registered tests

Each test is written and committed before the data that decides it exist. Once
committed, an entry is not edited; corrections go in a new dated entry below it.

## P1 · 2026-10-29 · Known vs unstable at the entity token, layer 21, country-centred

**Registered** before any entity-token activation was computed, and before any town
of the second Wikidata pass was pulled or graded. Origin: the weak late-layer
known-vs-unstable trend of the first river probe (last prompt token, held-out AUC
0.70–0.75 in layers 19–24, perm p ≈ 0.05–0.07 uncorrected; FINDINGS, 2026-10-29).
Position and layer as chosen by Thomas.

**Test data.** Only towns new in the second Wikidata pass (none of today's 294).
Graded with HF Qwen/Qwen3-8B bf16, revision b968826, template B
(`contrastset/experiments/templates/qwen3-name-only.txt`, sha256 be013947…), `-n 0
--logprob`, gate country, same settings as `runs/hf-logprob-294-B.sh`. Binned by
`scripts/river_cells.py` with today's thresholds and rules unchanged (familiar =
country p_correct ≥ 0.875; known-by-retrieval = p ≥ 0.875 and gold not the
applicable default; unstable = p ≤ 0.125 and no prior cell). Only these two cells
enter the test.

**Position.** The last token of the town name inside the template-B river prompt:
the last token whose character span starts before the end of the entity string
(its occurrence in the question). One forward pass per slot, residual before the
final norm, forward hooks, as in `src/knowing_probe/acts.py`.

**Layer.** 21 (output of block 21). No other layer is part of this test.

**Arrow, frozen from today's towns.** Everything is fitted on today's 200 slots
(`data/river-cells.jsonl`), at the entity token, layer 21:
1. Country means: per country, the mean residual over all of today's slots of that
   country (all five cells).
2. Centre every slot (today's and new) by subtracting its country's mean from step 1.
3. Arrow d = mean(centred known-by-retrieval) − mean(centred unstable), over all of
   today's towns in the two cells (20 vs 83, train and held-out alike).
4. Scale: 0 = today's centred unstable mean, 1 = today's centred known mean;
   threshold 0.5.
The arrow, country means and scale are saved once
(`runs/prereg/p1-arrow.npz`, sha256 recorded in NOTEBOOK.md) and never refitted.

**Primary statistic.** Held-out AUC of known-by-retrieval vs unstable over the new
towns, projected on the frozen arrow. Test: one-sided, 10 000 permutations of the
cell labels among the new towns; **pass if AUC > 0.5 with p < 0.05.** Reported with
a 95 % bootstrap interval and balanced accuracy at the 0.5 threshold.

**Minimum size.** Scored only if the new towns give ≥ 30 known-by-retrieval and
≥ 30 unstable. (At a true AUC of 0.72 that gives power ≈ 0.9.) If either is short,
pull more towns first; the test is not run on fewer.

**Scored once.** No other layer, position or centring is evaluated on the new towns
for this claim. Exploratory analyses of the new towns are allowed afterwards and are
labelled as such.

**Secondary, descriptive (registered so they are not chosen later; they do not
decide P1).**
- S1: the same procedure at the last prompt token, layer 21 (where the trend was
  actually seen).
- S2: median position of new right-by-prior towns on the frozen P1 arrow, reported
  with its bootstrap interval. No pass/fail.

## P1a · 2026-10-29 · Lexical control for P1 (addendum; P1 itself unchanged)

**Registered after** the entity-token probe on today's 200 towns was seen (11:45),
and before any town of the second pass exists. The reason: at the entity token,
known-by-retrieval vs unstable already separates at layer 0 (bare token embedding,
country-centred held-out AUC 0.66) and layer 1 (0.76), so P1 could pass on the
surface of the names alone (subword identity of obscure vs famous names), with no
knowledge read out by the model's processing.

**Control.** The P1 procedure exactly (frozen on today's towns, country-centred,
entity token), at layer 1 instead of 21: `runs/prereg/p1lex-arrow.npz`.

**Rule.** P1 counts as evidence of something beyond the name's surface only if, on
the new towns, AUC(layer 21) − AUC(layer 1) > 0 with a paired bootstrap 95 %
interval (10 000 resamples of the same towns) excluding 0. If P1 passes and P1a
does not, report: "separates, but no more than the name's surface does".

## P1b · 2026-10-29 · Amendment: sampling, centring, size, surface controls (P1 and P1a unchanged in text; this entry overrides where they differ)

**Registered** at ~12:00, after the surface analysis on today's towns
(`runs/river/surface.json`, FINDINGS "The layer-1 (surface) arrow does as well at
layer 21…") and Thomas's review, before any town of the second pass was pulled.

**Sampling.** Every new town is drawn from the same fame band, sitelinks ≥ 40,
with no population cap, about 35 per country across all ten countries (Italy,
Portugal and Sweden included). None of today's 294 towns. Known-by-retrieval and
unstable both come from this one pull, so they are fame-matched by construction.
No separate pull per cell.

**Binning.** `river_cells.py` rules and thresholds unchanged, run on the merged pool
(today's 294 + the new towns) so that leave-one-out regional defaults and the basin
check see all towns. Only new towns enter the test.

**Centring (changes P1 step 2 for the new towns).** New towns are centred with
new-pass country means: per country, the mean residual over all new slots of that
country (all five cells), at the layer being scored. Today's means are not used for
new towns: today's Italian mean is an all-unstable mean, and it would push a new
known Italian town towards the unstable side by construction. The frozen arrows
(direction and 0/1 scale, from today's towns) are unchanged.

**Minimum size (changes P1).** ≥ 40 new known-by-retrieval and ≥ 40 new unstable.
Below that, pull more from the same band before scoring.

**Paired comparisons.** Every comparison of two AUCs uses the identical set of new
towns. The 95 % interval comes from 10 000 bootstrap resamples that draw the same
towns for both AUCs.

**Surface controls (decide whether a P1 pass counts as more than the surface).**
P1 counts as evidence of something beyond the name's surface only if BOTH:
- P1a: AUC(frozen L21 arrow on L21) − AUC(frozen L1 arrow on L1) > 0, paired CI above 0;
- P1c (new): AUC(frozen L21 arrow on L21) − AUC(frozen L1 arrow applied to the L21
  activations) > 0, paired CI above 0. This is the functional surface test that
  today's data could not decide: it asks whether layer 21 knows more than the
  surface direction carried along to layer 21.

**Descriptive, registered so they are not chosen later.**
- D1, fame baseline: AUC of log(sitelinks) alone, known vs unstable, new towns.
- D2, projection-out: the frozen L1 direction projected out of the new L21
  activations, then scored with the frozen L21 arrow.
- S1 and S2 as before (S2, right-by-prior position, is the line the write-up would
  want if it replicates on ≥ 30 new right-by-prior towns).
