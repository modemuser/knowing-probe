"""Residual-stream reads at the last prompt token, one forward pass per prompt, no generation.

Residual stream: the running vector each transformer block reads from and adds to.
Layer 0 is the embedding output, layer i the output of block i (contrastset's
convention). Read with forward hooks on the decoder blocks, so every layer, the last
one included, is the raw residual BEFORE the model's final norm (transformers'
hidden_states may hand back the last entry post-norm; hooks make it unambiguous).

The model is loaded through contrastset's HFAdapter so the weights, revision, dtype and
device are the ones the grade used, and its describe() goes into the sidecar.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from contrastset.adapters.hf import HFAdapter

POSITION_RULE = "last token of the input text (for a chat prompt: the token right before the answer would begin)"
ENTITY_RULE = ("last token of the entity name: the last token whose character span starts before the end of the "
               "entity string (its last occurrence in the prompt)")


def load(model: str = "Qwen/Qwen3-8B", revision: str = "b968826d9c46dd6066d109eabc6255188de91218") -> HFAdapter:
    return HFAdapter(model, revision=revision)


def _blocks(adapter: HFAdapter) -> Any:
    return adapter.model.model.layers


def entity_last_token(adapter: HFAdapter, text: str, entity: str) -> tuple[int, bool]:
    """Token index of the entity's last token in text, and whether that token runs past the entity's end."""
    end = text.rindex(entity) + len(entity)
    offsets = adapter.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    index = max(i for i, (a, _b) in enumerate(offsets) if a < end)
    return index, offsets[index][1] > end


def read_last_token(adapter: HFAdapter, texts: Sequence[str], layers: Sequence[int] | None = None,
                    label: str = "") -> tuple[np.ndarray, list[str]]:
    """[n_texts, n_layers, d_model] float32 residuals at each text's last token, and those tokens."""
    return read_at(adapter, texts, [-1] * len(texts), layers, label)


def read_at(adapter: HFAdapter, texts: Sequence[str], positions: Sequence[int], layers: Sequence[int] | None = None,
            label: str = "") -> tuple[np.ndarray, list[str]]:
    """[n_texts, n_layers, d_model] float32 residuals at token positions[k] of texts[k], and those tokens.

    Streams one progress line per 25 texts to stderr."""
    torch = adapter._torch
    blocks = _blocks(adapter)
    n_layers = len(blocks) + 1
    layers = list(range(n_layers)) if layers is None else [l if l >= 0 else n_layers + l for l in layers]
    captured: dict[int, Any] = {}
    target = [-1]

    def hook(i: int) -> Any:
        def fn(_module: Any, _inp: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            captured[i] = h[0, target[0]].float().cpu()
        return fn

    def embed_hook(_module: Any, _inp: Any, out: Any) -> None:
        captured[0] = out[0, target[0]].float().cpu()

    handles = [adapter.model.model.embed_tokens.register_forward_hook(embed_hook)]
    handles += [b.register_forward_hook(hook(i + 1)) for i, b in enumerate(blocks)]
    out = np.zeros((len(texts), len(layers), adapter.model.config.hidden_size), dtype=np.float32)
    last_tokens: list[str] = []
    t0 = time.time()
    try:
        with torch.no_grad():
            for k, text in enumerate(texts):
                ids = adapter._ids(text)
                target[0] = positions[k]
                captured.clear()
                adapter.model.model(ids)  # the backbone only: no LM head, nothing generated
                out[k] = np.stack([captured[l].numpy() for l in layers])
                last_tokens.append(adapter.tokenizer.convert_ids_to_tokens([int(ids[0, positions[k]])])[0])
                if (k + 1) % 25 == 0 or k + 1 == len(texts):
                    el = time.time() - t0
                    print(f"{label} {k + 1}/{len(texts)}  {el / (k + 1):.2f} s/item  "
                          f"eta {el / (k + 1) * (len(texts) - k - 1):.0f} s", file=sys.stderr, flush=True)
    finally:
        for h in handles:
            h.remove()
    return out, last_tokens


def save(stem: str | Path, acts: np.ndarray, sidecar: dict[str, Any]) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.save(stem.with_suffix(".npy"), acts)
    sidecar = {**sidecar, "shape": list(acts.shape), "dtype": "float32",
               "npy_sha256": hashlib.sha256(stem.with_suffix(".npy").read_bytes()).hexdigest()}
    stem.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=1))


def sidecar_base(adapter: HFAdapter, layers: Sequence[int], texts: Sequence[str], last_tokens: Sequence[str]) -> dict[str, Any]:
    return {
        "model": adapter.describe(),
        "position_rule": POSITION_RULE,
        "residual": "forward hooks on embed_tokens (layer 0) and each decoder block (layer i); pre final norm",
        "layers": list(layers),
        "n_items": len(texts),
        "texts_sha256": hashlib.sha256("\n".join(texts).encode()).hexdigest(),
        "last_tokens": sorted(set(last_tokens)),
    }
