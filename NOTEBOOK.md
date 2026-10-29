# Experiment notebook

Chronological record of what was run, in what order, with what came out, including
dead ends, mistakes and changes of plan. Conclusions go to FINDINGS.md; this file is
the path that led there. Times are CET. Newest entries at the bottom.

## 2026-10-29

*11:00–11:30: reconstructed after the fact from file times, git log and the
session transcript (the notebook was started at ~11:37, on Thomas's instruction).*

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

**11:25 · Commit e61627a: Step 0.** FINDINGS staged without the river-cells section
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
(~11:35, below): it only shows a PC1-strength direction would survive.

**11:30 · Commit 221646c: river cells and probe.** A FINDINGS edit (training-AUC
sentence, which wrongly said 1.0; it is 0.82–0.95) failed to apply before the commit
because the file had changed on disk; fixed and amended.

**~11:35 · Thomas's review, decisions for next steps.**
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

**~11:38 · Reframed the river null in FINDINGS** ("null for a dominant direction,
underpowered for a modest one"), second-pass targets narrowed to known-by-retrieval
(country-balanced) and right-by-prior.

**11:40 · Pre-registered P1** (`PREREGISTRATION.md`), committed (50500b5, 11:40:57) before
the entity-token pass started (11:41): entity token, layer 21, country-centred, arrow frozen on all of today's 200
slots, scored once on the second pass, ≥ 30 per cell, one-sided permutation p < 0.05.
I added two descriptive secondaries (S1: last prompt token, layer 21, since that is where
the trend was seen; S2: right-by-prior position). They do not decide P1. Choices I
made inside Thomas's spec, flagged for review: arrow from all 20 + 83 of today's
towns (not the 13 + 55 training ones), country means from all 200 slots, threshold
0.5 on today's scale.

**11:41 · Entity-token pass.** `river_acts.py --position entity_last` →
`runs/river/acts-entity`, 200 slots, 0.44 s/item. Rule: last token whose span starts
before the end of the town name; the following "?" is always its own token, 0 picks
run past the name, 164 distinct tokens (e.g. Tábor → "bor", Wiener Neustadt → "adt").

**11:43 · P1/S1 arrows frozen** (`scripts/prereg_p1.py freeze`), before any
entity-token probe result was looked at:
- p1: runs/prereg/p1-arrow.npz sha256 eca30c291be69827deb4baab329070565622f491becb9984803cc301a26cc463
- s1: runs/prereg/s1-arrow.npz sha256 33a7252ba3b6be5debf7ce524265189e73e9dd26e54917e99590a66dce0c0665

*Times of the ~11:35–11:40 entries corrected at 11:44: first written as guesses (11:40/11:50/11:52), now from git commit times.*

**11:44 · Four probe variants** (`river_probe.py` now writes per variant:
prompt-last / entity-last, each raw and `--centre country`). First attempt at the
centred runs silently did nothing: zsh does not word-split `$v`, so
`--centre country` arrived as one argument and argparse's error was hidden by my
grep. Rerun with explicit arguments.
- Last prompt token: centring leaves known vs unstable at 0.70–0.75 (L19–24).
- Entity token, centred: 0.80–0.85 over L12–28, p ≤ 0.005 at L21–28.
- **Surprise: entity token separates at layer 0/1** (centred 0.66 / 0.76): the bare
  subword embedding. Checked tokenisation per cell: names are 3.5–4.3 tokens on
  average in every cell, none single-token, endings almost never word-initial. So
  it is not length or whole-word endings; it is which subwords the names end in.
- Right-by-prior sits with the unknown cells at the entity token (median 0.23 at
  L21 centred), between them at the last prompt token (0.60 at L22).
- Superseded top-level outputs (`runs/river/known-vs-*`, `figures/river/*.png`)
  deleted. The same numbers are in `prompt-last/`.

**11:47 · Registered P1a** (lexical control: same frozen procedure at layer 1; P1
counts as beyond the surface only if AUC(L21) − AUC(L1) on new towns has a paired
bootstrap CI above 0). Written after seeing today's entity results, labelled as
such; P1's text untouched. `prereg_p1.py freeze p1lex`: sha256 93fef88e…bc000.
The freeze script now refuses to refreeze a frozen name (P1/S1 hashes unchanged).

**11:49 · Dry run of `prereg_p1.py score`** against a scratch copy of the frozen
arrows, with fake "new" towns (today's slots with renamed QIDs; known doubled to pass
the ≥ 30 gate). All paths run: hash check, size gate, permutation p, bootstraps, P1a
paired difference, S2 interval, refusal to score twice. Real `runs/prereg/` holds
only the three arrows and their hashes; no `p1-result.json` exists.

**11:52 · Second-pass sizing (no run yet).** Targets, per Thomas: known-by-retrieval
balanced by country, and right-by-prior. P1 needs ≥ 30 new known-by-retrieval AND
≥ 30 new unstable. Today's yield of known-by-retrieval is low and fame-dependent:
by sitelinks 0/8 (<20), 1/93 (20–40), 13/159 (40–80), 6/34 (≥ 80). By country it is
DE 3, AT 1, PL 1, CZ 2, SK 2, FR 6, IT 0, ES 5, PT 0, SE 0. The seed script caps
population at 200 000 and samples evenly over sitelinks, which starves the known
cell. Proposed pull: exclude today's 294 QIDs; per country, oversample towns with
sitelinks ≥ 60 and no population cap, aiming at ~30 graded towns per country for
known-by-retrieval. Add a right-by-prior arm: small towns whose river is the
leave-one-out regional modal river. Grading cost on HF log-prob: 46 s/town (294
towns took 3 h 46), so ~300 towns ≈ 3.8 h. Alternative: an Ollama pre-screen
(~14 s/town sampled) on ~600 candidates (≈ 2.3 h), then HF only on likely-known
plus a random unstable sample (~160 towns ≈ 2 h). Waiting for Thomas's choice before
pulling.

**~11:53 · Thomas's second review.** (1) Demote the entity-token headline: raw L0 is
already 0.79. (2) Test the surface directly: cos(L1 arrow, L21 arrow), and project the
L1 arrow out of L21. (3) The floor is a fame proxy; draw known and unstable from the
same fame band (≥ 40 sitelinks), no population cap. (4) Centre new towns with new-pass
country means; ≥ 40 known; paired comparisons on identical towns. (5) Direct HF, ~350
towns, balanced over all ten countries; split across both Macs if the second is free.

**11:55 · Surface test** (`scripts/river_surface.py` → `runs/river/surface.json`).
|cos(L1, L21 arrow)| 0.16; L1 direction removed → L21 AUC 0.81 (from 0.83). Looked like
"different direction", but the noise check kills that reading: split-half |cos| of
the L21 arrow itself is 0.12 (13 known towns). Functional test: the L1 arrow scored on
L21 activations gives 0.81, the same as L21's own arrow. Undecidable on today's towns;
the surface may be the whole story. FINDINGS headline demoted (11:57), new entry added.

**~12:00 · Registered P1b** (sampling ≥ 40 sitelinks for all new towns; merged-pool
binning; new-pass country centring; ≥ 40 per cell; paired CIs; new control P1c = the
frozen L1 arrow applied to L21; descriptive D1 log-sitelinks, D2 projection-out).
`prereg_p1.py score` rewritten for P1b; dry run on a scratch copy passes. Side
observation from the dry run (today's towns, in-sample): log(sitelinks) alone gives
AUC 0.67 for known vs unstable, so fame tracks the label even inside today's pool.
`river_cells.py` now also writes `sitelinks` (no rule change).

**15:21 · contrastset pinned to the release.** `pyproject.toml`: `contrastset[hf]==0.1.0.post1`
from PyPI instead of the editable `../contrastset` path; `uv.lock` changes only that
entry (0.1.0.dev0 → 0.1.0.post1), torch 2.14.1 / transformers 5.21.0 unchanged.
0.1.0.post1 has the library code these runs used: the contrastset release pass changed
no behaviour (its baseline grade and behaviour check reproduce bit for bit), and its
`examples/00-marks-tegmark/` reproduces Step 0 above exactly (max |AUC − runs/step0
summaries| = 0.0, both sets). Fast tests: 3 passed; `knowing_probe.acts/arrow/
analysis/plot` import against the pinned package. The graded file the river steps read
(`../contrastset/runs/wikidata-qwen3-8b-hf-logprob-B-priors.jsonl`) is now committed in
the contrastset repository.

**15:25 · Commit history rewritten: author name, no attribution lines.** All commits
now carry the author Thomas Misera; `Co-Authored-By` lines removed from the messages.
Trees, author dates and committer dates are unchanged (P1's commit is still 11:40:57,
before the pass it governs). Hashes cited in this notebook and in FINDINGS.md updated:
911d9e0 → e61627a (Step 0), 9a5517b → 221646c (river cells), a1fb877 → 50500b5 (P1).

**15:41 · Publication pass, section 0: freeze.** Before any change: frozen arrows
`runs/prereg/p1-arrow.npz` eca30c29…c463, `s1-arrow.npz` 33a7252b…0665,
`p1lex-arrow.npz` 93fef88e…c000, each equal to `runs/prereg/frozen-hashes.json`.
No `p1-result.json` exists (P1 unrun). Fast tests: 3 passed. The venv runs contrastset
0.1.0.post1 from the PyPI registry (`uv.lock` source = pypi.org/simple, not editable);
the pin itself was made at 15:21 (61a1907). Commits of this pass carry no
attribution lines, matching the 15:25 rewrite.

**15:43 · Section 1: repo hygiene.**
- Grep of tracked files: no secrets, tokens or machine names. One leak: the template's
  absolute path (`/Users/tom/dev/contrastset/experiments/templates/…`), 210 times, in
  the cells file, the river sidecars and the river summaries. Fixed by committing a
  byte-identical copy of the template as `data/templates/qwen3-name-only.txt`
  (file sha256 01122ebc…, body sha256 be013947…) and replacing the path strings by
  plain text substitution (no number touched). The same for the graded-file path:
  the 294-town grade is now committed here as
  `data/wikidata-qwen3-8b-hf-logprob-B-priors.jsonl` (2.8 MB, sha256 768070ea…,
  identical to contrastset 514ed22 and to `graded_sha256` in the cells provenance).
- History: two older commits still contain that absolute path in blobs. Not rewritten.
  Nothing tracked is ignored; no blob was committed and later ignored; largest
  blobs are `uv.lock` (413 kB) and strip-plot PNGs (~375 kB).
- Reproducibility checks with no model: `river_cells.py` on the committed grade and
  template regenerates the committed cell assignment exactly (200/200 slots: cell,
  split, p_correct, modal answer, parent, prior kind, default; provenance equal).
  `scripts/fetch_mt.sh` fetches the Marks & Tegmark CSVs at commit 5d1c630 and
  verifies the recorded sha256 (both OK). `data/step0-split.json` records the Step 0
  split (249 cities / 498 statements; 115 words / 118 statements held out), matching the
  summaries. `runs/river/cells-summary.json` (`river_cells_summary.py`) puts the cell
  numbers the README cites (sizes, 34 moved out of unstable, 12/12 basin failures,
  known-by-retrieval yield by sitelinks band) into a committed result file.
- `.gitignore`: `*.npy`, `*.npz` except `runs/prereg/*.npz`, `data/raw/`, model caches.
  Ignored on disk: 1.3 GB of activations; committed tree: 10 MB.
- `pyproject.toml`: 0.1.0, MIT (`LICENSE` copied from contrastset), Python ≥ 3.11,
  `contrastset==0.1.0.post1`; torch 2.14.1 / transformers 5.21.0 /
  `contrastset[hf]` moved to an `hf` extra. Resolved versions unchanged. With torch and
  transformers blocked, all four `knowing_probe` modules import. Tests: 3 passed.
