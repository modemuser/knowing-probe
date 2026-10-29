# knowing-probe

Difference-of-means probes on the residual stream of Qwen/Qwen3-8B (bf16, MPS): does
the model's internal state separate facts it knows from facts it does not, before it
answers? Datasets come from [contrastset](../contrastset), which grades what the model
knows; this repo reads activations and plots. Results go in [FINDINGS.md](FINDINGS.md).

Terms: **residual stream**, the running vector each transformer block reads from and
adds to (layer 0 = embeddings, layer i = output of block i, before the final norm);
**arrow**, mean activation of one cell minus mean of another, per layer, fitted on
training items only; **AUC** (area under the ROC curve), the chance that a random
positive item projects higher than a random negative one (0.5 = chance).

## Steps

```bash
uv sync
# Step 0: reproduce Marks & Tegmark 2023 (true/false statements) to validate the stage
uv run python scripts/mt_acts.py            # ~10 min, runs/step0/*.npy + .json sidecars
uv run python scripts/mt_probe.py           # figures/step0/, runs/step0/*-summary.json
# River slots (graded file: contrastset, 294 towns, template B, HF log-prob)
G=../contrastset/runs/wikidata-qwen3-8b-hf-logprob-B-priors.jsonl
uv run python scripts/river_wikidata.py $G -o data/river-mouths.json     # P403 mouth chains
uv run python scripts/river_cells.py $G data/river-mouths.json -o data/river-cells.jsonl
uv run python scripts/river_acts.py data/river-cells.jsonl $G -o runs/river/acts
uv run python scripts/river_acts.py data/river-cells.jsonl $G -o runs/river/acts-entity --position entity_last
for a in runs/river/acts runs/river/acts-entity; do
  uv run python scripts/river_probe.py $a                                 # figures/river/<position>/
  uv run python scripts/river_probe.py $a --centre country                # .../<position>-country-centred/
done
uv run python scripts/river_controls.py                                   # country baseline, power check
uv run python scripts/prereg_p1.py freeze                                 # once: PREREGISTRATION.md P1
```

[NOTEBOOK.md](NOTEBOOK.md) is the chronological lab notebook: every run and decision,
including mistakes. [PREREGISTRATION.md](PREREGISTRATION.md) holds tests fixed before
their data exist.

Activations (`*.npy`) are git-ignored; sidecars (`*.json`: model revision, template
hash, position rule, layers, split seed, item list) and summaries are kept.
