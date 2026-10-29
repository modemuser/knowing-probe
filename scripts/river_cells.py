"""Bin the graded river slots into the probe's cells, offline, from the existing graded file.

Usage: uv run python scripts/river_cells.py GRADED.jsonl data/river-mouths.json -o data/river-cells.jsonl

Cells (river question; p = logprob p_correct at T = 1, the model's own distribution):
  familiar (every slot):  country p_correct >= 0.875, so "unknown" means fact-unknown,
                          not entity-unknown
  known-by-retrieval:     p >= 0.875 and the gold river is NOT the applicable default
  right-by-prior:         p >= 0.875 and the gold river IS the applicable default
  parent-river:           p <= 0.125, modal answer wrong and a river the gold river flows
                          into, any number of hops (Wikidata P403 chain), AND that answer
                          passes the basin check: no gated town of the same country whose
                          gold river lies outside its basin gets it as modal answer
  country-prior:          p <= 0.125, modal answer wrong and a prior guess: an on-chain
                          answer that fails the basin check (a country prior in a basin
                          costume), contrastset's country-shared flag, or the applicable
                          default; prior_kind says which
  unstable:               p <= 0.125 and none of the above
  (0.125 < p < 0.875, the middle band, is dropped.)

Applicable default: the regional default (modal gold river among the OTHER towns of
the same first-level region, leave-one-out, needing >= 2 such towns) if the model
knows the region (region p_correct >= 0.875) and the region has one; otherwise the
country default, built the same way over the country.

Held-out split: by town, one third, stratified by cell, fixed seed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

from contrastset.grade import Template, normalise, require_same_template

KNOWN_MIN, UNKNOWN_MAX = 0.875, 0.125
SPLIT_SEED, HELD_OUT_FRAC = 20261029, 1 / 3
TEMPLATE = Path(__file__).resolve().parents[2] / "contrastset/experiments/templates/qwen3-name-only.txt"
LANGS = ("en", "de")
GENERIC = {"river", "rio", "rzeka", "rieka", "reka", "fluss", "fleuve", "fiume", "alv", "alven"}
CELLS = ["known-by-retrieval", "right-by-prior", "unstable", "parent-river", "country-prior"]


def key(name: str) -> str:
    return " ".join(w for w in normalise(name, LANGS).split() if w not in GENERIC)


def q(record: dict, qid: str) -> dict | None:
    return next((x for x in record["questions"] if x["id"] == qid), None)


def leave_one_out_default(records: list[dict], group: str) -> dict[str, str | None]:
    """entity -> modal gold river QID among the other towns of its group (>= 2 of them, unique mode)."""
    by_group: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        if r["meta"].get(group):
            by_group[r["meta"][group]].append(r)
    out: dict[str, str | None] = {}
    for members in by_group.values():
        for r in members:
            counts = collections.Counter(q_ for o in members if o is not r for q_ in o["meta"].get("river_qids", []))
            top = counts.most_common(2)
            unique = top and top[0][1] >= 2 and (len(top) == 1 or top[1][1] < top[0][1])
            out[r["entity"]] = top[0][0] if unique else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("graded")
    ap.add_argument("mouths")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    records = [json.loads(line) for line in open(args.graded, encoding="utf-8")]
    template = Template.load(str(TEMPLATE))
    require_same_template(template, [SimpleNamespace(provenance=r["provenance"]) for r in records])
    wd = json.loads(open(args.mouths, encoding="utf-8").read())
    forms = {qid: {key(n) for n in [*e["labels"].values(), *(a for v in e["aliases"].values() for a in v)]} - {""}
             for qid, e in wd["names"].items()}
    label = {qid: e["labels"].get("en") or next(iter(e["labels"].values()), qid) for qid, e in wd["names"].items()}

    def downstream(qid: str) -> list[str]:
        seen, todo = [], list(wd["mouth"].get(qid, []))
        while todo:
            m = todo.pop(0)
            if m not in seen:
                seen.append(m)
                todo += wd["mouth"].get(m, [])
        return seen

    def matches(answer: str | None, qid: str | None) -> bool:
        return bool(answer and qid and key(answer) in forms.get(qid, set()))

    region_default = leave_one_out_default(records, "region_qid")
    country_default = leave_one_out_default(records, "country_qid")

    gated_by_country: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        if r["gate"]["passed"] and q(r, "river") is not None:
            gated_by_country[r["meta"]["country_qid"]].append(r)

    def in_basin(r: dict, pq: str) -> bool:
        gold = r["meta"].get("river_qids", [])
        return pq in gold or any(pq in downstream(g) for g in gold)

    def basin_check(pq: str, country_qid: str) -> tuple[list[dict], list[dict]]:
        """Same-country gated towns whose modal river answer is pq: (inside its basin, outside)."""
        same = [o for o in gated_by_country[country_qid] if matches(q(o, "river")["logprob"]["modal_answer"], pq)]
        return [o for o in same if in_basin(o, pq)], [o for o in same if not in_basin(o, pq)]

    rows = []
    for r in records:
        river, country, region = q(r, "river"), q(r, "country"), q(r, "region")
        if not r["gate"]["passed"] or river is None or country["logprob"]["p_correct"] < KNOWN_MIN:
            continue
        lp = river["logprob"]
        p, modal, modal_correct = lp["p_correct"], lp["modal_answer"], lp["modal_correct"]
        gold = r["meta"].get("river_qids", [])
        region_known = region is not None and region["logprob"]["p_correct"] >= KNOWN_MIN
        default, default_kind = (region_default[r["entity"]], "region") if region_known and region_default.get(r["entity"]) \
            else (country_default.get(r["entity"]), "country")
        chain = [m for g in gold for m in downstream(g) if m not in gold]
        parent = next((m for m in chain if matches(modal, m)), None)
        is_default_guess = not modal_correct and matches(modal, default) and default not in gold
        costume = bool(parent) and bool(basin_check(parent, r["meta"]["country_qid"])[1])
        prior_kind = None
        if p >= KNOWN_MIN:
            cell = "right-by-prior" if default in gold else "known-by-retrieval"
        elif p <= UNKNOWN_MAX and not modal_correct and parent and not costume:
            cell = "parent-river"
        elif p <= UNKNOWN_MAX and not modal_correct and (costume or river.get("prior_guess") or is_default_guess):
            cell = "country-prior"
            prior_kind = "costume-parent" if costume else "shared" if river.get("prior_guess") else "default"
        elif p <= UNKNOWN_MAX:
            cell = "unstable"
        else:
            continue
        rows.append({
            "entity": r["entity"], "qid": r["meta"]["qid"], "country_qid": r["meta"]["country_qid"],
            "region_qid": r["meta"].get("region_qid"), "question": river["q"], "cell": cell,
            "p_correct": p, "p_correct_sampler": river["logprob_sampler"]["p_correct"], "bin": river["bin"],
            "modal_answer": modal, "modal_correct": modal_correct, "p_modal": lp["p_modal"],
            "gold": [label.get(g, g) for g in gold], "gold_qids": gold,
            "parent_qid": parent, "parent": label.get(parent) if parent else None,
            "hops": (chain.index(parent) + 1) if parent else None, "basin_check_failed": costume,
            "prior_kind": prior_kind,
            "prior_guess_flag": bool(river.get("prior_guess")), "default_guess": is_default_guess,
            "default": label.get(default) if default else None, "default_qid": default, "default_kind": default_kind,
            "region_known": region_known,
        })

    # Split by town, stratified by cell.
    rng = random.Random(SPLIT_SEED)
    for cell in CELLS:
        members = sorted((x for x in rows if x["cell"] == cell), key=lambda x: x["qid"])
        rng.shuffle(members)
        n_held = round(len(members) * HELD_OUT_FRAC)
        for i, x in enumerate(members):
            x["held_out"] = i < n_held

    prov = {"graded_file": args.graded, "graded_sha256": hashlib.sha256(Path(args.graded).read_bytes()).hexdigest(),
            "template": str(TEMPLATE), "template_sha256": template.sha256, "mouths_file": args.mouths,
            "thresholds": {"known_min": KNOWN_MIN, "unknown_max": UNKNOWN_MAX, "familiar": "country p_correct >= 0.875",
                           "p": "logprob p_correct at T = 1"},
            "default_rule": "leave-one-out modal gold river, >= 2 other towns, unique mode; region if region known else country",
            "split": {"by": "town", "held_out_frac": HELD_OUT_FRAC, "seed": SPLIT_SEED, "stratified_by": "cell"}}
    with open(args.out, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps({**x, "provenance": prov}, ensure_ascii=False) + "\n")

    sizes = collections.Counter(x["cell"] for x in rows)
    held = collections.Counter(x["cell"] for x in rows if x["held_out"])
    print("cell sizes (train + held out):")
    for c in CELLS:
        print(f"  {c:20s} {sizes[c]:4d}  ({sizes[c] - held[c]} + {held[c]})")
    print("\nparent-river slots:")
    for x in sorted((x for x in rows if x["cell"] == "parent-river"), key=lambda x: x["parent"]):
        print(f"  {x['entity']:24s} gold {'/'.join(x['gold']):22s} said {x['modal_answer']:16s} -> {x['parent']} "
              f"({x['hops']} hop{'s' if x['hops'] > 1 else ''}) p_modal {x['p_modal']:.2f}  flag {x['prior_guess_flag']}")
    print("\ncountry-prior slots:")
    for x in sorted((x for x in rows if x["cell"] == "country-prior"), key=lambda x: (x["prior_kind"], x["modal_answer"])):
        print(f"  {x['prior_kind']:15s} {x['entity']:24s} gold {'/'.join(x['gold']):22s} said {x['modal_answer']:20s} "
              f"default {x['default']} ({x['default_kind']})")
    print("\nright-by-prior slots:")
    for x in rows:
        if x["cell"] == "right-by-prior":
            print(f"  {x['entity']:24s} gold {'/'.join(x['gold']):22s} default {x['default']} ({x['default_kind']})")

    # Sanity check: is a parent river also given for same-country towns in a DIFFERENT basin?
    print("\nbasin check (same answer, same country, gold river not draining into it):")
    for pq in sorted({x["parent_qid"] for x in rows if x["parent_qid"]}, key=lambda s: label.get(s, s)):
        for c in sorted({x["country_qid"] for x in rows if x["parent_qid"] == pq}):
            inside, outside = basin_check(pq, c)
            names = ", ".join(f"{o['entity']} ({'/'.join(label.get(g, g) for g in o['meta']['river_qids'])})" for o in outside)
            print(f"  {label.get(pq, pq):14s} {c:6s} said for {len(inside) + len(outside):2d} towns: {len(inside):2d} in basin, "
                  f"{len(outside):2d} outside{': ' + names if outside else ''}")
    kinds = collections.Counter(x["prior_kind"] for x in rows if x["cell"] == "country-prior")
    print(f"\ncountry-prior by kind: {dict(kinds)}")
    moved = sum(1 for x in rows if x["cell"] in ("parent-river", "country-prior") and not x["prior_guess_flag"])
    print(f"slots without contrastset's prior flag that left 'unstable' for a prior cell: {moved}")

if __name__ == "__main__":
    main()
