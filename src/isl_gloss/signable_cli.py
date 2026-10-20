"""python run_signable.py --gloss data/translations/*.gloss.json --out data/signable [--level exact|close|off]

Reads Phase 1 gloss files (never modifies them) and writes `<id>.signable.json` next to a before/after sign-coverage summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from islrtc_index.vocab import Vocabulary, load_synonym_table, validate_synonyms

from .signable import LEVELS, consistently_capitalised, coverage, normalize_tokens

DISCLAIMER = ("Candidate only. Synonyms record that two ENGLISH words mean (nearly) the same; they are not reviewed by an ISL signer and do not claim ISL signs them alike. "
              "Every change is listed in `changes` and can be undone by using `final_tokens` instead.")


def process_file(path: Path, vocab: Vocabulary, synonyms: Dict[str, Dict[str, object]], level: str, spell_names: bool) -> Dict[str, object]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    results: List[Dict[str, object]] = doc.get("results", [])
    capitalised = consistently_capitalised(r.get("source_text", "") for r in results) if spell_names else None
    out_results = []
    before_all: List[str] = []
    after_all: List[str] = []
    for r in results:
        tokens = list(r.get("final_tokens") or [])
        norm = normalize_tokens(tokens, vocab, synonyms, level, capitalised)
        before_all += tokens
        after_all += norm.tokens
        out_results.append({"chunk_id": r.get("chunk_id"), "source_text": r.get("source_text"), "final_tokens": tokens, "signable_tokens": norm.tokens,
                            "changes": [c.__dict__ for c in norm.changes]})
    return {
        "report_type": "isl_gloss_signable", "source_id": doc.get("source_id"), "source_file": str(path), "level": level, "spell_consistently_capitalised_names": spell_names,
        "review_status": "unreviewed", "review_note": DISCLAIMER,
        "coverage_before": coverage(before_all, vocab), "coverage_after": coverage(after_all, vocab),
        "results": out_results,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="isl-signable", description="Make Phase 1 gloss more signable (synonyms and name spelling); never edits the gloss files.")
    ap.add_argument("--gloss", nargs="+", required=True, help="Phase 1 *.gloss.json files")
    ap.add_argument("--out", default="data/signable")
    ap.add_argument("--level", choices=LEVELS, default="exact", help="synonym level: exact (default), close (near-synonyms too) or off")
    ap.add_argument("--no-spell-names", action="store_true", help="do not fingerspell consistently capitalised words that have no sign")
    ap.add_argument("--manifest", default="data/islrtc/manifest.json")
    ap.add_argument("--mappings", default="data/islrtc/approved_mappings.json")
    ap.add_argument("--synonyms", default="data/vocab/synonyms.json")
    ap.add_argument("--yap-vocab", default="../yap-render/lib/islVocabulary.ts")
    args = ap.parse_args(argv)
    for label, p in (("manifest", args.manifest), ("yap vocabulary", args.yap_vocab), ("synonym table", args.synonyms)):
        if not Path(p).is_file():
            print(f"error: {label} not found: {p}", file=sys.stderr)
            return 2
    vocab = Vocabulary.from_files(Path(args.yap_vocab), Path(args.manifest), Path(args.mappings))
    synonyms, problems = validate_synonyms(load_synonym_table(Path(args.synonyms)), vocab)
    for p in problems:
        print("synonym table:", p, file=sys.stderr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'file':<48} {'tokens':>6}  {'no sign before':>14} -> {'after':<6} {'synonyms':>8} {'spelled':>7}")
    for g in args.gloss:
        report = process_file(Path(g), vocab, synonyms, args.level, not args.no_spell_names)
        name = Path(g).name.replace(".gloss.json", "")
        (out / f"{name}.signable.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        ch = [c for r in report["results"] for c in r["changes"]]
        print(f"{name[:48]:<48} {report['coverage_before']['total']:>6}  {report['coverage_before']['no_sign']:>14} -> {report['coverage_after']['no_sign']:<6} "
              f"{sum(1 for c in ch if c['kind'] == 'synonym'):>8} {sum(1 for c in ch if c['kind'] == 'spell'):>7}")
    return 0
