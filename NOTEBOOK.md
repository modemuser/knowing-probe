# Experiment notebook

Chronological record of the experiments: what was run on what, what came out,
surprises in the data, reviews and decisions, in the order they happened. Conclusions
go to FINDINGS.md (with errata), tests fixed in advance to PREREGISTRATION.md. Times
are CET.

## 2026-10-29

**~11:05 · Step 0 expectation, written into FINDINGS before any activation existed.**
Separation from layers 8–12, held-out accuracy ≥ 0.90 from ~12 to ≥ 28 (cities),
lower ceiling for sp_en_trans, shuffled labels at chance.

**11:05 · Step 0 inputs and timing.** Marks & Tegmark `cities.csv`, `sp_en_trans.csv`
at geometry-of-truth commit 5d1c630 (sha256 in the sidecars). Qwen/Qwen3-8B bf16,
revision b968826, MPS. 10-statement probe: last token "." for every statement; steady
0.28 s per statement, ~10 min for 1850.

**11:06 · Wikidata mouth chains.** P403 walked breadth-first from the 191 gold rivers of
the 294-town grade, 281 rivers after 5 rounds → `data/river-mouths.json`.

**11:07 · First river binning.** Cells before the basin check: known-by-retrieval 20,
right-by-prior 17, unstable 83, parent-river 39, prior-unrelated 41.

**11:08 · The basin check changes the cells.** The country-shared prior answers (Vltava,
Váh, Dordogne, Tagus, Ebro) are also given for towns outside their basins (Vltava: 10 of
16 Czech towns outside, six on the Elbe). On-chain answers that fail the check go to
country-prior. Final: known-by-retrieval 20, right-by-prior 17, parent-river 19,
country-prior 61, unstable 83. All 12 flagged on-chain slots fail. Decisions (in
FINDINGS): prior cells defined over all p ≤ 1/8 slots, not only flagged ones (34
slots leave unstable); the arrow's positive cell is known-by-retrieval only; the split is
stratified by cell. The label run's "13 prior-guess slots" could not be reproduced
under any familiar rule tried (country / country + region, p or bin); the 12
on-chain flagged slots match its "about 12".

**11:15 · Step 0, layer 0.** The layer-0 arrow is exactly zero: every statement ends in
".", so the embeddings are identical. This confirms the position rule; layer 0 is
reported as a zero arrow.

**11:17 · Step 0 result.** Held-out AUC 0.94 at layer 8, 1.00 from layer 15 (cities) /
13 (sp_en_trans). Separation starts earlier than expected (layer 4).

**11:17 · Surprise: the shuffle control does not collapse to a band around 0.5.**
Shuffled-label held-out AUCs span 0.00–1.00 in mid layers. The truth arrow is PC1 from
layer ~9 (|cos| up to 1.00, PC1 28–48 % of variance), and random-half mean differences
fall onto PC1 with a random sign. Sign fraction ≈ 0.5 at every layer, so no leak.
Permutation p, sign fraction and |cos(arrow, PC1)| added to every summary.

**11:22 · River activations.** 200 slots, last prompt token (`ĊĊ` for every slot),
0.40 s per slot, template hash be013947… checked against the grade.

**11:26 · River probe.** Known vs unstable: held-out AUC 0.70–0.75 at layers 19–24,
permutation p 0.05–0.075. Known vs country-prior: 0.79–0.89, p ≤ 0.03.

**11:27 · Country confound.** Known-by-retrieval has no SE/PT/IT towns; country-prior
is 14 CZ, 9 SE, 9 SK, 8 ES, 7 PT. Country-only baseline (no activations): 0.84 on known
vs country-prior, 0.58 on known vs unstable. Country-centred arrows: known vs unstable
unchanged (0.70–0.75, layers 19–29, p 0.03–0.07); known vs country-prior 0.78–0.83
(p 0.02–0.10).

**11:28 · Power check.** Cities subsampled to the river cell sizes: median held-out AUC
1.000 from layer 16 (300 draws).

**~11:35 · Thomas's review.**
- The power check only shows that a PC1-strength direction would survive. Read the
  river result as a null for a *dominant* direction, *underpowered* for a modest one:
  a direction a tenth of PC1's size would look like 0.70–0.75, p ≈ 0.06, with an
  overfitting arrow at 13 + 7 known towns.
- Position: the 2024 entity-recognition work found known/unknown latents at the
  entity's own token, middle layers. Run one pass at the last token of the town name,
  same labels.
- Country-centre before fitting (already done at the last prompt token; repeat at the
  entity token, with plots).
- Pre-register the confirmatory test for the second Wikidata pass before seeing the
  entity-token data.
- Second-pass targets: known-by-retrieval balanced by country, and right-by-prior; not
  the prior cells.

**11:40 · P1 pre-registered** and committed at 11:40:57 (now 50500b5), before the
entity-token pass started at 11:41. Town-name token, layer 21, country-centred, arrow
frozen on today's towns, scored once on the second pass, one-sided permutation
p < 0.05. Descriptive S1 (last prompt token, layer 21, where the trend was seen) and S2
(right-by-prior position) added; they decide nothing. Choices inside Thomas's spec,
flagged for review and later ratified: arrow from all 20 + 83 of today's towns,
country means from all 200 slots, threshold 0.5 on today's scale.

**11:41 · Entity-token pass.** 200 slots, 0.44 s per slot. Rule: the last token whose
span starts before the end of the town name; the following "?" is always its own
token, no pick runs past the name, 164 distinct tokens (Tábor → "bor", Wiener
Neustadt → "adt").

**11:43 · P1 and S1 arrows frozen** before any entity-token probe result was looked at:
p1 eca30c291be69827deb4baab329070565622f491becb9984803cc301a26cc463,
s1 33a7252ba3b6be5debf7ce524265189e73e9dd26e54917e99590a66dce0c0665.

**11:44 · Four probe variants** (last prompt token / town-name token, each raw and
country-centred).
- Last prompt token: centring leaves known vs unstable at 0.70–0.75 (layers 19–24).
- Town-name token, centred: 0.77–0.85 over layers 12–28, p ≤ 0.005 at 21–28.
- **Surprise: the town-name token separates at layer 0/1** (centred 0.66 / 0.76): the
  bare subword embedding. Names average 3.5–4.3 tokens in every cell, none is a single
  token, and endings are almost never word-initial. So it is not length or whole-word
  endings; it is which subwords the names end in.
- Right-by-prior sits with the unknown cells at the town-name token (median 0.23 at
  layer 21, centred) and between them at the last prompt token (0.49 at layer 22,
  centred; 0.60 raw).

**11:47 · P1a registered** (lexical control: the same frozen procedure at layer 1; P1
counts as beyond the surface only if AUC(L21) − AUC(L1) on new towns has a paired
bootstrap interval above 0). Written after seeing today's entity results and
labelled so; P1's text untouched. Arrow p1lex frozen:
93fef88e077f04709d4059bbc961da799a230f17809b31f44ea056269aebc000.

**11:49 · Scorer checked** on a scratch copy of the frozen arrows with fake "new" towns
(today's slots, renamed): hash check, size gate, permutation p, bootstraps, the P1a
paired difference and the refusal to score twice all work. The real `runs/prereg/` holds
only the three arrows and their hashes.

**11:52 · Second-pass sizing.** Today's known-by-retrieval yield is low and depends on
fame: by sitelinks 0/8 (< 20), 1/93 (20–39), 13/159 (40–79), 6/34 (≥ 80). By country:
DE 3, AT 1, PL 1, CZ 2, SK 2, FR 6, IT 0, ES 5, PT 0, SE 0. HF log-prob grading costs
46 s per town (294 towns took 3 h 46).

**~11:53 · Thomas's second review.**
- Demote the town-name headline: raw layer 0 is already 0.79, higher than anything the
  last prompt token reached.
- Test the surface directly: cos(L1 arrow, L21 arrow), and project the L1 arrow out of
  layer 21.
- The floor is a fame proxy: famous names tokenize differently, and fame tracks the
  label. Draw known and unstable from the same fame band (≥ 40 sitelinks), with no
  population cap.
- Centre new towns on new-pass country means; ≥ 40 known; paired comparisons on
  identical towns.
- Direct HF, ~350 towns, balanced over all ten countries.

**11:55 · Surface test** (`runs/river/surface.json`). |cos(L1, L21 arrow)| 0.16; with the
L1 direction removed, layer 21 gives 0.81 (from 0.83). The noise check undoes the "different
direction" reading: split-half |cos| of the L21 arrow itself is 0.12 at 13 known towns.
Functional test: the L1 arrow applied to layer-21 activations gives 0.81, the same as
layer 21's own arrow. Undecidable on today's towns; the surface may be the whole story.
The town-name headline in FINDINGS is demoted.

**~12:00 · P1b registered** before any new town was pulled: all new towns from the
≥ 40-sitelink band; binning on the merged pool; new-pass country centring; ≥ 40 per
cell; paired intervals; new control P1c (the frozen L1 arrow applied at layer 21);
descriptive D1 (log sitelinks) and D2 (projection-out). Seen in the scorer check
(today's towns, in-sample): log(sitelinks) alone gives AUC 0.67 for known vs unstable,
so fame tracks the label even inside today's pool.

**~12:00 · Pass-2 candidates pulled** (contrastset `experiments/wikidata-towns-pass2.jsonl`,
168 towns, ≥ 40 sitelinks, no population cap, today's 294 excluded). The band is thin
where Wikidata has few towns with a river statement: eligible before echo filtering DE
361, FR 720, SK 52, CZ 38, IT 38, ES 34, AT 21, PT 8, PL 6, SE 6. After filtering: FR 32,
DE 28, SK 27, ES 23, CZ 20, IT 20, AT 14, SE 2, PL 1, PT 1. Widening to generic
settlement classes gave PT 17 and timed out for PL, SE and AT. "~35 per country across
all ten" as registered is not reachable from this source. How to fill the pass is
undecided; nothing is graded.

**15:21 · Library release check.** contrastset 0.1.0.post1 (PyPI) has the code these runs
used, and its `examples/00-marks-tegmark/` reproduces Step 0 exactly (max |AUC − runs/step0
summaries| = 0.0, both sets). This repository now depends on that release.

**15:43 · Reproducibility without a second machine.** The 294-town grade (sha256
768070ea…) and the template-B file (body sha256 be013947…) are committed under `data/`.
From them, with no model, `river_cells.py` regenerates the committed cell assignment
exactly (200/200 slots: cell, split, p_correct, modal answer, parent, prior kind,
default). The Marks & Tegmark CSVs re-fetched at 5d1c630 match the recorded sha256. The
Step 0 split is recorded in `data/step0-split.json` (249 cities / 498 statements; 115
words / 118 statements held out). Derived cell numbers are in
`runs/river/cells-summary.json`.

**15:47 · Before publication.** The frozen arrow files hash to the registered values
(p1 eca30c29…, s1 33a7252b…, p1lex 93fef88e…; all unchanged since d0bc2c7). Outcome
paragraphs for pass and failure appended to PREREGISTRATION.md (bd456cb), before any new
town is graded. Re-reading every number from the committed summaries for the README
found four misquoted ranges in FINDINGS. The most consequential: the right-by-prior
"0.60 → 0.23" compared a raw arrow with a centred one; like for like it is
0.49 → 0.23 (centred) or 0.60 → 0.27 (raw). Errata appended to FINDINGS (61f81d9).
Related work checked against arXiv: all seven ids resolve to the cited titles and
authors.

**15:58 · Thomas's review of the publication draft.** Country balance as registered
cannot be executed; decide before P1 how P1 runs without it. Decision, registered as
P1d before any town is graded: a country enters P1 only with ≥ 10 new slots in the five
cells. Centring a country with one or two towns on its own mean would put them at the
zero vector. On the candidate list (FR 32, DE 28, SK 27, ES 23, CZ 20, IT 20, AT 14, SE 2,
PL 1, PT 1), Poland, Portugal and Sweden are therefore out; Austria is decided by the
rule. 168 candidates at the band's 8–18 % yield give about 22 known-by-retrieval
towns, so the pass is topped up from the same band with no per-country quota (mainly
DE, FR), in batches decided on cell counts alone, until ≥ 40 known and ≥ 40 unstable in
the scored countries. The scorer applies the rule and reports the excluded countries.
Checked on a scratch copy with a three-slot fake country (excluded and named; the rest
scored). Frozen arrows unchanged.

**16:01 · State at push.** P1 unrun; outcome paragraphs written (bd456cb); amendments
P1b and P1d registered; no town of the second pass graded. Arrows frozen at p1
eca30c291be69827deb4baab329070565622f491becb9984803cc301a26cc463, s1
33a7252ba3b6be5debf7ce524265189e73e9dd26e54917e99590a66dce0c0665, p1lex
93fef88e077f04709d4059bbc961da799a230f17809b31f44ea056269aebc000.
