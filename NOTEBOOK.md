# Experiment notebook

Chronological record of what was run, in what order, with what came out, including
dead ends, mistakes and changes of plan. Conclusions go to FINDINGS.md; this file is
the path that led there. Times are CET. Newest entries at the bottom.

## 2026-10-29

*11:00–11:30: reconstructed after the fact from file times, git log and the
session transcript (the notebook was started at 11:40, on Thomas's instruction).*

**~11:00 · Repo set up.** `~/dev/knowing-probe`, uv project, contrastset as an
editable path dependency (HFAdapter, Template, hash check), torch 2.14.1 /
transformers 5.21.0 pinned to contrastset's versions. Qwen/Qwen3-8B snapshot
b968826 already in the HF cache.

**~11:04 · Step 0 data.** Downloaded `cities.csv`, `sp_en_trans.csv` from
saprmarks/geometry-of-truth (main = 5d1c630) into `data/raw/` (git-ignored).
sha256 in the sidecars.

**~11:05 · Step 0 expectation written into FINDINGS before any activation existed.**
Expected separation from layers 8–12, accuracy ≥ 0.90 from ~12 to ≥ 28.

**11:05 · Timing probe.** `mt_acts.py cities --limit 10`: 1.53 s/item incl. MPS
warm-up, last token "." for every statement. Estimated 15–45 min for 1850 items.
Actual: 0.28 s/item steady, ~10 min.

**11:06 · Wikidata mouth chains.** `river_wikidata.py`: P403 walked breadth-first from
191 gold rivers, 281 rivers total after 5 rounds → `data/river-mouths.json`.

**11:06–11:15 · Step 0 activations** (background), all 37 layers, forward hooks.

**11:07 · First river binning.** `require_same_template` failed on plain dicts (it
expects objects with `.provenance`); wrapped records in SimpleNamespace. First cell
sizes: known-by-retrieval 20, right-by-prior 17, unstable 83, parent-river 39,
prior-unrelated 41.

**11:08 · Basin check changes the cells.** The check showed that the country-shared
prior answers (Vltava, Váh, Dordogne, Tagus, Ebro) are also given for towns outside
their basins (Vltava: 10 of 16 Czech towns outside, six on the Elbe). Folded the check
into the binning: on-chain answers that fail it → country-prior. Final: known-by-
retrieval 20, right-by-prior 17, parent-river 19, country-prior 61, unstable 83. All
12 flagged on-chain slots fail the check. Decisions (in FINDINGS): prior cells
defined over all p ≤ 1/8 slots, not only flagged ones (34 slots left unstable);
arrow's positive cell = known-by-retrieval only; split stratified by cell.
Could not reproduce the label run's "13 prior-guess slots" under any familiar rule
tried (country / country+region, p or bin); the 12 on-chain flagged slots do match
its "about 12".

**11:15 · Step 0 probe, first try: crash.** Layer 0 arrow is exactly zero (every
statement ends in "."; identical embeddings) → divide by zero in the projection.
Made a zero arrow project to NaN and plot as "zero arrow". Confirms the position rule.

**11:17 · Step 0 result.** Held-out AUC 0.94 at layer 8, 1.00 from 15 (cities) / 13
(sp_en_trans). Separation starts earlier than expected (layer 4).

**11:17 · Surprise: shuffle control does not collapse to a band around 0.5.**
Shuffled-label held-out AUCs span 0.00–1.00 in mid layers. Diagnosis
(scratch `shuffle_diag.py`): truth arrow = PC1 from layer ~9 (|cos| up to 1.00, PC1
28–48 % of variance); random-half mean differences fall onto PC1 with a random sign.
Sign fraction ≈ 0.5 at every layer → no leak. Added permutation p, sign fraction
and |cos(arrow, PC1)| to every summary.

**11:25 · Commit 911d9e0: Step 0.** FINDINGS staged without the river-cells section
(blob via `git hash-object`) so the first commit is Step 0 only.

**11:22 · River activations** (background, 200 slots, 0.40 s/item). Last prompt token
`ĊĊ` for all slots, template hash be013947… checked.

**11:26 · River probe.** Known vs unstable: held-out AUC 0.70–0.75 at layers 19–24,
perm p 0.05–0.075. Known vs country-prior: 0.79–0.89, p ≤ 0.03.

**11:27 · Country confound found.** Known-by-retrieval has no SE/PT/IT towns;
country-prior is 14 CZ, 9 SE, 9 SK, 8 ES, 7 PT. Country-only baseline (no
activations): 0.84 on known vs country-prior, 0.58 on known vs unstable.
Country-centred arrows: known vs unstable unchanged (0.70–0.75, layers 19–29,
p 0.03–0.07); known vs country-prior drops to 0.78–0.83 (p 0.02–0.10).

**11:28 · Power check.** Cities subsampled to the river cell sizes: median held-out
AUC 1.000 from layer 16. Wrote this up as "not a lack of data". Thomas disagreed
(11:40, below): it only shows a PC1-strength direction would survive.

**11:30 · Commit 9a5517b: river cells and probe.** A FINDINGS edit (training-AUC
sentence, which wrongly said 1.0; it is 0.82–0.95) failed to apply before the commit
because the file had changed on disk; fixed and amended.

**11:40 · Thomas's review, decisions for next steps.**
- Reframe: null for a *dominant* direction, *underpowered* for a modest one (a
  direction a tenth of PC1's size would look exactly like 0.70–0.75, p ≈ 0.06, with
  an overfitting arrow at 13 + 7 known towns).
- Position: the 2024 entity-recognition work found known/unknown latents at the
  entity's own token, in middle layers. Run one extra pass at the last token of the
  town name, same labels.
- Country-centre before fitting, rerun (already done at the last prompt token; repeat
  at the entity token, with plots).
- Pre-register the confirmatory test for the second Wikidata pass, before seeing
  the entity-token data.
- Second pass targets: known-by-retrieval balanced by country, and right-by-prior.
  Not the prior cells.

**11:50 · Reframed the river null in FINDINGS** ("null for a dominant direction,
underpowered for a modest one"), second-pass targets narrowed to known-by-retrieval
(country-balanced) and right-by-prior.

**11:52 · Pre-registered P1** (`PREREGISTRATION.md`), committed before the entity-token
pass: entity token, layer 21, country-centred, arrow frozen on all of today's 200
slots, scored once on the second pass, ≥ 30 per cell, one-sided permutation p < 0.05.
I added two descriptive secondaries (S1: last prompt token, layer 21, since that is where
the trend was seen; S2: right-by-prior position). They do not decide P1. Choices I
made inside Thomas's spec, flagged for review: arrow from all 20 + 83 of today's
towns (not the 13 + 55 training ones), country means from all 200 slots, threshold
0.5 on today's scale.
