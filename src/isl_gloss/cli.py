"""Command line: translate cleaned transcripts, inspect sentence/group stages, compare modes, evaluate, list rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .config import TranslatorConfig
from .export import write_groups, write_results
from .grouping import group_sentences, group_stats
from .llm import LLMError, MissingApiKey, load_env_file, make_client
from .models import SourceChunk
from .prepare import chunks_from_doc, chunks_from_sentences
from .rules import RULES, RULE_SET_VERSION
from .scoring import load_annotations
from .sentences import build_sentences
from .store import ResultStore
from .translator import Translator
from .validate import load_vocabulary


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, help="JSON file overriding TranslatorConfig fields")
    p.add_argument("--provider", choices=("groq", "gemini"), help="LLM provider (default groq; gemini is still available)")
    p.add_argument("--model", help="model id (default openai/gpt-oss-120b on groq, gemini-3.6-flash on gemini)")
    p.add_argument("--rpm", type=float, help="max requests per minute (default 20 on groq, 5 on gemini)")
    p.add_argument("--tpm", type=int, help="max tokens per minute to send (default 7000 on groq; the free limit is reported as 8K)")
    p.add_argument("--prompt-version", choices=("5.0", "6.0", "7.0", "8.0"), help="prompt version (default 7.0; 5.0, 6.0 and 8.0 are kept for comparison)")
    p.add_argument("--references", help="reviewed ISL reference file (prompt 8.0 only; see docs/reference_set_format.md)")
    p.add_argument("--no-empty-gloss-retry", action="store_true", help="prompt 8.0 only: do not re-ask once when a gloss comes back empty")
    p.add_argument("--reasoning-effort", choices=("low", "medium", "high"), help="groq gpt-oss reasoning effort (default low; reasoning tokens count toward limits)")
    p.add_argument("--batch-size", type=int, help="sentences per request in the old direct mode (default 10)")
    p.add_argument("--enable-rule", action="append", default=[], metavar="RULE_ID", help="switch on a rule that is off by default")
    p.add_argument("--disable-rule", action="append", default=[], metavar="RULE_ID")
    p.add_argument("--vocab", help="file with one allowed gloss token per line; unknown tokens are flagged")
    p.add_argument("--env-file", default=".env", help="optional file with GROQ_API_KEY=... (or GEMINI_API_KEY=...); default ./.env")
    p.add_argument("--dry-run", action="store_true", help="show the requests that would be sent; call nothing")


def _add_grouping(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("sentence chunking and grouping (stages 2 and 3)")
    g.add_argument("--max-group-sentences", type=int, help="max sentences per group (default 8)")
    g.add_argument("--max-group-chars", type=int, help="max characters per group (default 1200)")
    g.add_argument("--max-group-tokens", type=int, help="max estimated tokens (chars/4) per group (default 300)")
    g.add_argument("--max-gap-seconds", type=float, help="start a new group after a pause longer than this (default 8)")
    g.add_argument("--no-time-gap", action="store_true", help="ignore timestamps when grouping")
    g.add_argument("--context-before", type=int, help="read-only sentences before a dependent group (default 1)")
    g.add_argument("--context-after", type=int, help="read-only sentences after a group that ends mid-thought (default 1)")
    g.add_argument("--runon-min-words", type=int, help="split unpunctuated chunks of at least this many words at capitalised openers; 0 = never (default 14)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="isl_gloss", description="Cleaned English transcript -> sentences -> context groups -> candidate ISL gloss (unreviewed).")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("translate", help="translate cleaned transcript JSON file(s)")
    t.add_argument("input", type=Path, help="a *.cleaned.json file, or a folder of them (e.g. data/structured)")
    t.add_argument("-o", "--out", type=Path, help="output folder (default: data/translations next to structured/)")
    t.add_argument("--mode", choices=("groups", "direct"), default="groups",
                   help="groups = sentence chunking + context grouping (default); direct = the old fixed windows of cleaned chunks")
    t.add_argument("--limit", type=int, help="only the first N sentences of each file (handy for a first free-tier test)")
    t.add_argument("--layer2", action="store_true", help="after translating, also run Layer 2 (spaCy candidate + comparison) on the saved "
                                                          "results into <out>/<name>.layer2/; makes no extra LLM calls and does not change Layer 1 output")
    t.add_argument("--layer2-config", type=Path, help="JSON for Layer 2 ({enabled, llm_field, policy{...}}); only used with --layer2, "
                                                      '{"enabled": false} switches Layer 2 off even if --layer2 is given')
    _add_common(t)
    _add_grouping(t)

    gp = sub.add_parser("group", help="run stages 2 and 3 only (no LLM, no key) and write <id>.groups.json")
    gp.add_argument("input", type=Path)
    gp.add_argument("-o", "--out", type=Path)
    gp.add_argument("--limit", type=int)
    gp.add_argument("--config", type=Path)
    _add_grouping(gp)

    c = sub.add_parser("compare", help="old direct mode vs sentence+group mode on a few representative passages")
    c.add_argument("--passages", type=Path, default=Path("evaluation/context_passages.json"))
    c.add_argument("-o", "--out", type=Path, default=Path("data/translations/comparison"))
    _add_common(c)
    _add_grouping(c)

    cp = sub.add_parser("compare-prompts", help="run prompt versions on the same inputs; keep outputs; record failure types")
    cp.add_argument("--inputs", type=Path, default=Path("evaluation/layer1_inputs.json"))
    cp.add_argument("-o", "--out", type=Path, default=Path("data/translations/prompt_comparison"))
    cp.add_argument("--versions", nargs="+", choices=("5.0", "6.0", "7.0", "8.0"), default=["6.0", "7.0"],
                    help="prompt versions to run on the same inputs (default: 6.0 and 7.0; the first is the base of the sheet)")
    cp.add_argument("--annotations", type=Path, default=Path("evaluation/layer1_annotations.json"),
                    help="provisional annotations used for the scored categories (engineering judgements, not ISL-reviewed)")
    cp.add_argument("--seed-cache-from", type=Path,
                    help="copy cache_prompt<version>.llm_cache.jsonl files from this folder into -o first (the folder itself is never written to)")
    _add_common(cp)
    _add_grouping(cp)

    sr = sub.add_parser("score-results", help="score saved results_prompt_<version>.json files side by side (no model call)")
    sr.add_argument("results", nargs="+", type=Path, help="results_prompt_<version>.json files from compare-prompts")
    sr.add_argument("--annotations", type=Path, default=Path("evaluation/layer1_annotations.json"))
    sr.add_argument("-o", "--out", type=Path, required=True)

    l2 = sub.add_parser("layer2", help="Layer 2: run the spaCy translator on SAVED Layer 1 results and compare the two candidates (no LLM calls)")
    l2.add_argument("results", type=Path, help="a Layer 1 results file (results_prompt_<version>.json or <id>.gloss.json); only read")
    l2.add_argument("-o", "--out", type=Path, required=True, help="output folder (must not be the folder holding the Layer 1 file)")
    l2.add_argument("--layer2-config", type=Path, help="JSON: {enabled, llm_field, policy{...}}")
    l2.add_argument("--llm-field", choices=("candidate_tokens", "final_tokens"), help="Candidate A: the model's own gloss (default) or Layer 1's post-rule gloss")
    sub.add_parser("layer2-check", help="show whether spaCy, the model and NLTK are installed, with versions and setup steps")

    rc = sub.add_parser("check-references", help="validate a reference file and show what would be accepted or rejected")
    rc.add_argument("file", type=Path, nargs="?", default=Path("references/reference_set.json"))

    e = sub.add_parser("evaluate", help="run the evaluation sentence set and write a reviewer sheet")
    e.add_argument("--sentences", type=Path, default=Path("evaluation/eval_sentences.json"))
    e.add_argument("-o", "--out", type=Path, default=Path("data/translations/evaluation"))
    _add_common(e)
    sub.add_parser("rules", help="list the rule set")
    return p


def _config(args) -> TranslatorConfig:
    provider = getattr(args, "provider", None)
    if getattr(args, "config", None):
        data = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if provider:
            data["provider"] = provider
        cfg = TranslatorConfig.from_dict(data)
    else:
        cfg = TranslatorConfig.for_provider(provider or "groq")
    simple = {"model": "model", "rpm": "requests_per_minute", "tpm": "tokens_per_minute", "batch_size": "batch_size",
              "reasoning_effort": "reasoning_effort", "prompt_version": "prompt_version", "vocab": "vocabulary_file",
              "max_group_sentences": "group_max_sentences", "max_group_chars": "group_max_chars",
              "max_group_tokens": "group_max_tokens", "max_gap_seconds": "group_max_gap_seconds",
              "context_before": "context_sentences_before", "context_after": "context_sentences_after",
              "runon_min_words": "runon_min_words"}
    for arg, field in simple.items():
        value = getattr(args, arg, None)
        if value is not None:
            setattr(cfg, field, value)
    if getattr(args, "references", None):
        cfg.references_file = args.references
    if getattr(args, "no_empty_gloss_retry", False):
        cfg.empty_gloss_retry = False
    if getattr(args, "no_time_gap", False):
        cfg.group_max_gap_seconds = None
    if getattr(args, "enable_rule", None):
        cfg.enable_rules = tuple(args.enable_rule)
    if getattr(args, "disable_rule", None):
        cfg.disable_rules = tuple(args.disable_rule)
    cfg.validate()
    return cfg


def _make_translator(cfg: TranslatorConfig, args, cache: Path) -> Translator:
    client = None
    if not args.dry_run:
        load_env_file(args.env_file)
        client = make_client(cfg)
    return Translator(cfg, client, ResultStore(cache), load_vocabulary(cfg.vocabulary_file), log=print)


def _default_out(first: Path) -> Path:
    return first.parent.parent / "translations" if first.parent.name == "structured" else first.parent / "translations"


def _inputs(path: Path) -> list[Path]:
    return sorted(path.glob("*.cleaned.json")) if path.is_dir() else [path]


def _stages(doc: dict, cfg: TranslatorConfig, limit: Optional[int]):
    sentences = build_sentences(doc, cfg)
    if limit:
        sentences = sentences[:limit]
    return sentences, group_sentences(sentences, cfg, source_id=doc["source_id"])


def _cmd_group(args) -> int:
    cfg = _config(args)
    files = _inputs(args.input)
    if not files:
        print(f"No *.cleaned.json files in {args.input}", file=sys.stderr)
        return 2
    out = args.out or _default_out(files[0])
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        sentences, groups = _stages(doc, cfg, args.limit)
        stats = {**group_stats(groups), "cleaned_chunks": len(doc["chunks"]),
                 "sentences_from_runon_splits": sum(bool(s.source_metadata.get("split_method")) for s in sentences),
                 "config": {k: v for k, v in cfg.to_dict().items() if k.startswith(("group_", "context_sentences", "runon"))}}
        path = write_groups(out, doc["source_id"], sentences, groups, stats)
        print(f"{f.name}: {stats['cleaned_chunks']} chunks -> {stats['sentences']} sentences -> {stats['groups']} groups "
              f"(avg {stats['sentences_per_group_avg']}, max {stats['sentences_per_group_max']}); boundaries {stats['boundary_reasons']}; "
              f"forced {stats['forced_boundaries']}, oversize {stats['oversize_groups']}\n  wrote {path}")
    return 0


def _cmd_translate(args) -> int:
    cfg = _config(args)
    files = _inputs(args.input)
    if not files:
        print(f"No *.cleaned.json files in {args.input}", file=sys.stderr)
        return 2
    out = args.out or _default_out(files[0])
    exit_code = 0
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        if "chunks" not in doc or "source_id" not in doc:
            raise ValueError(f"{f} is not a cleaned-transcript JSON (missing 'chunks'/'source_id').")
        if args.mode == "groups":
            sentences, groups = _stages(doc, cfg, args.limit)
            chunks, name = chunks_from_sentences(sentences), doc["source_id"]
            stats = group_stats(groups)
            note = f"{len(doc['chunks'])} chunks -> {stats['sentences']} sentences -> {stats['groups']} groups (one request each)"
        else:
            sentences, groups = [], None
            chunks = chunks_from_doc(doc, cfg)[:args.limit] if args.limit else chunks_from_doc(doc, cfg)
            name, stats = f"{doc['source_id']}.direct", None
            note = f"{len(chunks)} chunks in fixed windows of {cfg.batch_size} (old direct mode)"
        tr = _make_translator(cfg, args, out / f"{doc['source_id']}.llm_cache.jsonl")
        print(f"\n{f.name}: {note}; model {cfg.model}; rules {tr.engine.active_ids}")
        outcome = tr.run(chunks, dry_run=args.dry_run, groups=groups)
        if args.dry_run:
            print(f"dry run: {len(outcome.dry_run_requests)} request(s) would be sent. First one:\n")
            if outcome.dry_run_requests:
                print(outcome.dry_run_requests[0]["user_payload"])
            continue
        extra = {"source_type": doc.get("source_type"), "mode": args.mode, "limit": args.limit}
        if groups is not None:
            extra["grouping"] = stats
            extra["groups"] = [g.to_dict(include_sentences=False) for g in groups]
            write_groups(out, name, sentences, groups, stats)
        paths = write_results(out, name, outcome, tr.metadata(), str(f), extra)
        print(f"counts: {outcome.counts()}  requests made: {outcome.requests_made}")
        if not outcome.assembly.get("passed", True):
            print(f"ASSEMBLY CHECK FAILED: {outcome.assembly['errors']}")
            exit_code = 1
        if not outcome.complete:
            print(f"INCOMPLETE ({outcome.stop_reason}). Re-run the same command to resume; finished sentences are cached.")
            exit_code = 1
        print("wrote", paths["json"])
        if args.layer2:
            from .layer2 import Layer2Config, run_layer2

            l2cfg = Layer2Config.from_json(args.layer2_config) if args.layer2_config else Layer2Config()
            l2 = run_layer2(paths["json"], out / f"{name}.layer2", l2cfg, log=print)
            if not l2["enabled"]:
                print(f"layer 2: {l2['message']}")
            else:
                s = l2["summary"]
                print(f"layer 2: LLM kept as primary for {s['selection']['llm_primary_by_policy_default']}, spaCy selected for "
                      f"{s['selection']['spacy_selected']}, {s['selection']['needs_human_review_total']} flagged for review; wrote {l2['out_dir']}")
    return exit_code


def _cmd_compare(args) -> int:
    from .compare import run_compare

    cfg = _config(args)
    spec = json.loads(args.passages.read_text(encoding="utf-8"))
    make = None if args.dry_run else (lambda mode: _make_translator(cfg, args, args.out / f"compare.{mode}.llm_cache.jsonl"))
    summary = run_compare(spec, cfg, make, args.out, dry_run=args.dry_run)
    if args.dry_run:
        for p in summary["passages"]:
            print(f"{p['id']}: {p['cleaned_chunks']} chunks, {p['sentences']} sentences, {p['grouping']['groups']} groups; "
                  f"requests direct {p['direct_requests']} vs grouped {p['grouped_requests']}")
        print(f"total requests: {summary['requests']}")
        return 0
    t = summary["totals"]
    for mode in ("direct", "grouped"):
        m = t[mode]
        print(f"{mode:8} translated {m['translated']}/{m['sentences']}, failed {m['failed']}, uncertain {m['llm_uncertain']}, "
              f"review {m['validation_review_required']}, meaning warnings {m['meaning_warnings']}, requests {m['requests']}")
    print(f"{summary['chunks_with_different_gloss']} of {summary['chunks_compared']} chunks differ.\n{summary['automatic_verdict']}")
    for r in summary["automatic_regressions"]:
        print("  -", r)
    for mode, why in summary.get("incomplete_reasons", {}).items():
        print(f"  {mode} stopped: {why}")
    print(f"wrote {args.out / 'comparison_sheet.csv'} (give this to the ISL-fluent reviewer)")
    return 0 if t["grouped"]["complete"] and t["direct"]["complete"] else 1


def _cmd_compare_prompts(args) -> int:
    from .prompt_eval import run_prompt_comparison

    cfg = _config(args)
    spec = json.loads(args.inputs.read_text(encoding="utf-8"))

    def make(cfg_v):
        cache = args.out / f"cache_prompt{cfg_v.prompt_version}.llm_cache.jsonl"
        return _make_translator(cfg_v, args, cache)

    if args.seed_cache_from and not args.dry_run:
        import shutil
        args.out.mkdir(parents=True, exist_ok=True)
        for v in args.versions:
            src, dst = args.seed_cache_from / f"cache_prompt{v}.llm_cache.jsonl", args.out / f"cache_prompt{v}.llm_cache.jsonl"
            if src.is_file() and not dst.exists():
                shutil.copyfile(src, dst)
                print(f"seeded {dst.name} from {args.seed_cache_from}")
    ann = load_annotations(args.annotations)
    summary = run_prompt_comparison(spec, cfg, None if args.dry_run else make, args.out, args.versions, dry_run=args.dry_run,
                                    annotations=ann)
    if args.dry_run:
        for p in summary["passages"]:
            print(f"{p['id']}: {p['sentences']} sentences, {p['groups']} group(s)")
        print(f"requests per prompt version: {summary['requests_per_version']} (x {len(args.versions)} versions)")
        return 0
    for v, m in summary["per_version"].items():
        c = m["config"]
        print(f"prompt {v} [{c['provider']}/{c['model']}, reasoning {c['reasoning_effort']}]: translated {m['translated']}/{m['sentences']}, "
              f"failed {m['failed']}, model-flagged {m['model_flagged_review']}, sentences with an observed failure "
              f"{m['sentences_with_an_observed_failure']}, requests {m['requests']}, complete {m['complete']}")
        print(f"   observed failure types: {m['observed_failure_types'] or 'none'}")
        u = m["usage"]
        print(f"   usage: {u['llm_calls']} fresh call(s), {u['total_tokens']} tokens, {u['latency_seconds']}s, "
              f"empty-gloss retries {u['empty_gloss_retries']} (recovered {u['empty_gloss_recovered']})")
        if m["stop_reason"]:
            print(f"   stopped: {m['stop_reason'][:160]}")
    print(summary["interpretation"])
    print(f"wrote {args.out / 'comparison_sheet.csv'} and results_prompt_<version>.json")
    return 0 if summary["all_versions_complete"] else 1


def _cmd_score_results(args) -> int:
    from .prompt_eval import score_saved_results

    summary = score_saved_results(args.results, load_annotations(args.annotations), args.out)
    print(f"same sentences in every file: {summary['same_sentences_in_every_file']}")
    for v, m in summary["per_version"].items():
        print(f"\nprompt {v}:")
        for k, n in m["score"].get("categories", {}).items():
            print(f"  {k}: {n}")
    print(f"\n{summary['interpretation']}\nwrote {args.out / 'scored_comparison.json'} and scored_comparison_sheet.csv")
    return 0


def _cmd_layer2(args) -> int:
    from .layer2 import Layer2Config, run_layer2

    cfg = Layer2Config.from_json(args.layer2_config) if args.layer2_config else Layer2Config()
    if args.llm_field:
        cfg.llm_field = args.llm_field
    res = run_layer2(args.results, args.out, cfg, log=print)
    if not res["enabled"]:
        print(res["message"])
        return 0
    s = res["summary"]
    print(f"{s['sentences']} sentences. spaCy: {s['spacy']}. LLM: {s['llm']}.")
    print(f"both usable {s['both_candidates_usable']}, differing {s['candidates_differ_after_normalisation']}; "
          f"selection {s['selection']['status_counts']}; needs review {s['selection']['needs_human_review_total']}")
    print("Linguistic accuracy has NOT been established (no reviewer labels).")
    if s["spacy"]["dependency_unavailable"]:
        print("spaCy was unavailable for some sentences. Run: isl_gloss layer2-check")
    print("wrote", res["out_dir"])
    return 0


def _cmd_layer2_check(args) -> int:
    from .layer2.spacy_translator import SETUP_HELP, SpacyTranslator, environment_report

    print(json.dumps(environment_report(), indent=2))
    ok, why = SpacyTranslator().available()
    print("spaCy translator available:", ok)
    if not ok:
        print(why)
    return 0 if ok else 1


def _cmd_check_references(args) -> int:
    from .references import describe, load_references

    info = describe(load_references(args.file))
    print(json.dumps(info, indent=2))
    return 1 if info["rejected"] else 0


def _cmd_evaluate(args) -> int:
    cfg = _config(args)
    spec = json.loads(args.sentences.read_text(encoding="utf-8"))
    chunks = [SourceChunk(chunk_id=s["id"], text=s["text"]) for s in spec["sentences"]]
    tr = _make_translator(cfg, args, args.out / "eval.llm_cache.jsonl")
    outcome = tr.run(chunks, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry run: {len(outcome.dry_run_requests)} request(s) for {len(chunks)} sentences")
        return 0
    cats = {s["id"]: s.get("category") for s in spec["sentences"]}
    paths = write_results(args.out, "eval", outcome, tr.metadata(), str(args.sentences), {"categories": cats})
    print(f"counts: {outcome.counts()}\nwrote {paths['csv']}  (give this sheet to the ISL-fluent reviewer)")
    return 0 if outcome.complete else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "rules":
        print(f"rule set {RULE_SET_VERSION}")
        for r in RULES:
            print(f"\n{r.rule_id} v{r.version} [{r.kind}] {'ENABLED' if r.enabled_by_default else 'disabled by default'}")
            print(f"  does:        {r.description}\n  assumes:     {r.assumptions}\n  limitation:  {r.limitation}")
        return 0
    handlers = {"translate": _cmd_translate, "group": _cmd_group, "compare": _cmd_compare, "evaluate": _cmd_evaluate,
                "compare-prompts": _cmd_compare_prompts, "score-results": _cmd_score_results,
                "check-references": _cmd_check_references, "layer2": _cmd_layer2, "layer2-check": _cmd_layer2_check}
    try:
        return handlers[args.cmd](args)
    except MissingApiKey as exc:
        print(f"Configuration error: {exc}\nTip: add --dry-run to preview requests without a key.", file=sys.stderr)
        return 2
    except (ValueError, OSError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
