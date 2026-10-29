"""Mouth chains (Wikidata P403, "mouth of the watercourse") for every gold river in the graded file.

Usage: uv run python scripts/river_wikidata.py GRADED.jsonl -o data/river-mouths.json

Walks P403 breadth-first from each gold river until nothing new turns up, and fetches
labels plus aliases (English and the pool's local languages) for every river on the
way, so a model answer string can be matched against "a river the gold river flows
into, any number of hops".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

SPARQL = "https://query.wikidata.org/sparql"
API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "knowing-probe/0.0 (river mouth chains; https://github.com/modemuser/contrastset)"}
LANGUAGES = ["en", "de", "pl", "cs", "sk", "fr", "it", "es", "pt", "sv"]


def get(client: httpx.Client, url: str, params: dict) -> dict:
    for attempt in range(5):
        try:
            r = client.get(url, params=params)
            if r.status_code == 200:
                return r.json()
            print(f"  retry {attempt + 1}: HTTP {r.status_code}", file=sys.stderr)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            print(f"  retry {attempt + 1}: {exc}", file=sys.stderr)
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


def mouths(client: httpx.Client, qids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {q: [] for q in qids}
    for i in range(0, len(qids), 100):
        values = " ".join(f"wd:{q}" for q in qids[i:i + 100])
        data = get(client, SPARQL, {"format": "json", "query":
                   f"SELECT ?r ?m WHERE {{ VALUES ?r {{ {values} }} ?r wdt:P403 ?m . }}"})
        for b in data["results"]["bindings"]:
            r, m = (b[k]["value"].rsplit("/", 1)[1] for k in ("r", "m"))
            out[r].append(m)
    return out


def names(client: httpx.Client, qids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(qids), 50):
        data = get(client, API, {"action": "wbgetentities", "ids": "|".join(qids[i:i + 50]),
                                 "props": "labels|aliases|claims", "languages": "|".join(LANGUAGES), "format": "json"})
        for qid, ent in data["entities"].items():
            p31 = [c["mainsnak"].get("datavalue", {}).get("value", {}).get("id") for c in ent.get("claims", {}).get("P31", [])]
            out[qid] = {"labels": {k: v["value"] for k, v in ent.get("labels", {}).items()},
                        "aliases": {k: [a["value"] for a in v] for k, v in ent.get("aliases", {}).items()},
                        "instance_of": [p for p in p31 if p]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("graded")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    gold = sorted({q for line in open(args.graded, encoding="utf-8")
                   for q in json.loads(line)["meta"].get("river_qids", [])})
    client = httpx.Client(headers=HEADERS, timeout=90)
    edges: dict[str, list[str]] = {}
    frontier = gold
    while frontier:
        found = mouths(client, frontier)
        edges.update(found)
        frontier = sorted({m for ms in found.values() for m in ms} - set(edges))
        print(f"{len(edges)} rivers walked, {len(frontier)} new mouths", file=sys.stderr)
    info = names(client, sorted(edges))
    Path(args.out).write_text(json.dumps({
        "source": "Wikidata P403 (mouth of the watercourse), walked transitively",
        "retrieved": time.strftime("%Y-%m-%d"), "graded_file": args.graded, "languages": LANGUAGES,
        "gold": gold, "mouth": edges, "names": info}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
