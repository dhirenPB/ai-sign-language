"""Build a searchable index (SQLite) and a resolver manifest (JSON) from the ISLRTC dataset directory, without copying or loading any video.

Status meanings (what the resolver may auto-select is decided in the player, not here):
  source_label  clean label taken verbatim from the dataset's file name; NOT reviewed by a person
  verified      a reviewer approved it (review CSV decision)
  needs_review  something in the name needs a human (sense qualifier, long phrase, trailing number, weak label)
  concept       an explanation / example entry, not a direct sign for the term
  duplicate     same label+variant+sense as an earlier file
  unreadable    ffprobe could not read a video stream
  rejected      a reviewer rejected it

Default sign: a headword (or source-provided alias) with several live signs gets ONE default so the same word always shows the same sign
(`compute_auto_defaults`, deterministic, recorded in approved_mappings.json). A reviewer's `default` decision always overrides it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .parse import Alias, ParsedLabel, normalize_label, parse_stem
from .vocab import Vocabulary, load_synonym_table, parse_yap_tokens, validate_synonyms

SCHEMA_VERSION = "1.0"
VIDEO_EXT = ".mp4"
REVIEW_FLAGS = {"sense_qualifier", "long_label", "trailing_number", "weak_label"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
  asset_id TEXT PRIMARY KEY, folder TEXT, rel_path TEXT UNIQUE, filename TEXT, label TEXT, label_norm TEXT, variant TEXT,
  qualifier TEXT, entry_type TEXT, language_tag TEXT, status TEXT, flags TEXT, duplicate_of TEXT, size_bytes INTEGER,
  container TEXT, codec TEXT, width INTEGER, height INTEGER, fps REAL, duration_s REAL, probe_error TEXT, reviewed INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS aliases (asset_id TEXT, text TEXT, norm TEXT, verified INTEGER, source TEXT);
CREATE TABLE IF NOT EXISTS probe_cache (rel_path TEXT PRIMARY KEY, size_bytes INTEGER, mtime_ns INTEGER, codec TEXT, width INTEGER,
  height INTEGER, fps REAL, duration_s REAL, error TEXT);
CREATE INDEX IF NOT EXISTS idx_assets_norm ON assets(label_norm);
CREATE INDEX IF NOT EXISTS idx_aliases_norm ON aliases(norm);
"""


def asset_id_for(rel_path: str) -> str:
    """Stable id: derived from the relative path only, so re-indexing the same dataset gives the same ids."""
    return "i" + hashlib.sha1(rel_path.lower().encode("utf-8")).hexdigest()[:12]


@dataclass
class Entry:
    asset_id: str
    folder: str
    rel_path: str
    filename: str
    size_bytes: int
    mtime_ns: int
    parsed: ParsedLabel
    status: str = "source_label"
    duplicate_of: Optional[str] = None
    reviewed: bool = False
    container: Optional[str] = "mp4"
    codec: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    duration_s: Optional[float] = None
    probe_error: Optional[str] = None


def scan(data_dir: Path) -> List[Entry]:
    """List videos (one directory level below the dataset root). No file is opened."""
    entries: List[Entry] = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for item in sorted(os.scandir(folder), key=lambda e: e.name):
            if not item.is_file() or not item.name.lower().endswith(VIDEO_EXT):
                continue
            rel = f"{folder.name}/{item.name}"
            st = item.stat()
            entries.append(Entry(asset_id_for(rel), folder.name, rel, item.name, st.st_size, st.st_mtime_ns,
                                 parse_stem(item.name[: -len(VIDEO_EXT)], folder.name)))
    return entries


def assign_status(entries: List[Entry]) -> None:
    """Mechanical status assignment from parse flags, then duplicate detection. Never invents a label."""
    first_seen: Dict[Tuple, str] = {}
    for e in entries:
        p = e.parsed
        if p.entry_type == "explanation":
            e.status = "concept"
        elif REVIEW_FLAGS & set(p.flags) or p.entry_type == "phrase":
            e.status = "needs_review"
        else:
            e.status = "source_label"
        if e.status == "concept":
            continue
        key = (p.label_norm, p.variant, (p.qualifier or "").lower(), p.entry_type)
        if key in first_seen:
            e.status, e.duplicate_of = "duplicate", first_seen[key]
        else:
            first_seen[key] = e.asset_id


def _ffprobe(path: Path, ffprobe: str) -> Dict[str, object]:
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=codec_name,width,height,r_frame_rate,duration:format=duration", "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"ffprobe failed: {exc}"}
    if out.returncode != 0:
        return {"error": (out.stderr or "ffprobe error").strip()[:200]}
    try:
        data = json.loads(out.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
    except (ValueError, IndexError):
        return {"error": "unparseable ffprobe output"}
    if not stream.get("codec_name"):
        return {"error": "no video stream"}
    fps = None
    num, _, den = str(stream.get("r_frame_rate", "")).partition("/")
    if num.replace(".", "").isdigit() and den.isdigit() and float(den) > 0:
        fps = round(float(num) / float(den), 3)
    duration = stream.get("duration") or (data.get("format") or {}).get("duration")
    return {"codec": stream["codec_name"], "width": stream.get("width"), "height": stream.get("height"), "fps": fps,
            "duration_s": round(float(duration), 3) if duration not in (None, "N/A") else None, "error": None}


def probe_all(entries: List[Entry], data_dir: Path, db: sqlite3.Connection, workers: int = 6, ffprobe: str = "ffprobe",
              progress=None) -> Tuple[int, int]:
    """Fill codec/size/duration with ffprobe, reusing cached results for files whose size and mtime are unchanged."""
    cache = {r[0]: r for r in db.execute("SELECT rel_path,size_bytes,mtime_ns,codec,width,height,fps,duration_s,error FROM probe_cache")}
    todo: List[Entry] = []
    reused = 0
    for e in entries:
        c = cache.get(e.rel_path)
        if c and c[1] == e.size_bytes and c[2] == e.mtime_ns:
            e.codec, e.width, e.height, e.fps, e.duration_s, e.probe_error = c[3], c[4], c[5], c[6], c[7], c[8]
            reused += 1
        else:
            todo.append(e)

    def work(e: Entry):
        return e, _ffprobe(data_dir / e.rel_path, ffprobe)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for e, r in pool.map(work, todo):
            e.codec, e.width, e.height = r.get("codec"), r.get("width"), r.get("height")
            e.fps, e.duration_s, e.probe_error = r.get("fps"), r.get("duration_s"), r.get("error")
            db.execute("INSERT OR REPLACE INTO probe_cache VALUES (?,?,?,?,?,?,?,?,?)",
                       (e.rel_path, e.size_bytes, e.mtime_ns, e.codec, e.width, e.height, e.fps, e.duration_s, e.probe_error))
            done += 1
            if progress and done % 500 == 0:
                progress(done, len(todo))
    db.commit()
    return len(todo), reused


def apply_probe_status(entries: Iterable[Entry]) -> None:
    for e in entries:
        if e.probe_error and e.status not in ("concept", "duplicate"):
            e.status = "unreadable"


@dataclass
class ReviewResult:
    applied: int
    select: Dict[str, str]
    unknown_ids: List[str]


def apply_reviews(entries: List[Entry], review_csv: Path) -> ReviewResult:
    """Apply human decisions. Columns used: asset_id, decision (verified|reject|concept|default), label_override, aliases_add."""
    by_id = {e.asset_id: e for e in entries}
    select: Dict[str, str] = {}
    unknown: List[str] = []
    applied = 0
    with open(review_csv, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            decision = (row.get("decision") or "").strip().lower()
            if not decision:
                continue
            e = by_id.get((row.get("asset_id") or "").strip())
            if e is None:
                unknown.append(row.get("asset_id", ""))
                continue
            if decision in ("verified", "default"):
                e.status, e.reviewed = "verified", True
            elif decision == "reject":
                e.status, e.reviewed = "rejected", True
            elif decision == "concept":
                e.status, e.reviewed = "concept", True
            else:
                unknown.append(f"{e.asset_id}:{decision}")
                continue
            override = (row.get("label_override") or "").strip()
            if override:
                e.parsed.label, e.parsed.label_norm = override, normalize_label(override)
            for extra in (row.get("aliases_add") or "").split(";"):
                if extra.strip():
                    e.parsed.aliases.append(Alias(extra.strip(), "review", True))
            if decision == "default":
                select[e.parsed.label_norm] = e.asset_id
            applied += 1
    return ReviewResult(applied, select, unknown)


STATUS_RANK = {"verified": 0, "source_label": 1, "needs_review": 2}
LIVE_STATUSES = ("verified", "source_label", "needs_review")


def _variant_rank(variant: Optional[str]) -> int:
    """No marker first, then `(Sign)`, then `(Sign_2)`, `(Sign_3)`... and numbered `(1)`, `(2)` by their number."""
    if not variant:
        return 0
    if variant == "sign":
        return 1
    digits = "".join(ch for ch in variant if ch.isdigit())
    return int(digits) if digits else 999


def default_sort_key(e: "Entry", key: str):
    """Lower sorts first = preferred as the default sign for `key`. Fully deterministic (ends with the relative path)."""
    p = e.parsed
    return (0 if p.label_norm == key else 1,              # the entry that IS the word beats one that merely lists it as an alias
            STATUS_RANK.get(e.status, 9),                  # verified, then clean source label, then needs_review
            1 if p.qualifier else 0,                       # no sense qualifier beats "(Financial Term)"
            _variant_rank(p.variant),                      # base file, then (Sign_2), (Sign_3)...
            (p.qualifier or "").lower(), e.rel_path.lower())


def compute_auto_defaults(entries: List["Entry"], reviewer_select: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """One default sign per ambiguous key (headword or source-provided alias), so every occurrence of a word gets the same sign.

    Returns (select, meta): select maps key -> asset_id; meta maps key -> [chosen, *alternates]. Keys a reviewer already decided are left alone.
    This is a consistency rule, not a claim that the chosen sign is the best or only one for the word.
    """
    reviewer_select = reviewer_select or {}
    by_key: Dict[str, List[Entry]] = {}
    for e in entries:
        if e.status not in LIVE_STATUSES:
            continue
        keys = {e.parsed.label_norm} | {normalize_label(a.text) for a in e.parsed.aliases if a.verified}
        for k in keys:
            if k:
                by_key.setdefault(k, []).append(e)
    select: Dict[str, str] = {}
    meta: Dict[str, List[str]] = {}
    for key in sorted(by_key):
        group = by_key[key]
        if len(group) < 2 or key in reviewer_select:
            continue
        ordered = sorted(group, key=lambda e: default_sort_key(e, key))
        select[key] = ordered[0].asset_id
        meta[key] = [e.asset_id for e in ordered]
    return select, meta


def build_synonyms(entries: List["Entry"], select: Dict[str, str], table_path: Optional[Path], yap_vocab: Optional[Path]) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    """Validate the curated synonym table against the real vocabulary and return the entries the resolver may use.

    Needs the Yap avatar vocabulary (yap-render lib/islVocabulary.ts) to know which targets have a sign; without it nothing is written.
    """
    info: Dict[str, object] = {"table": str(table_path) if table_path else None, "valid": 0, "problems": []}
    if not table_path or not table_path.is_file():
        info["problems"] = ["no synonym table found; synonyms disabled"]
        return {}, info
    if not yap_vocab or not yap_vocab.is_file():
        info["problems"] = [f"Yap vocabulary file not found ({yap_vocab}); synonyms disabled because their targets cannot be validated"]
        return {}, info
    vocab = Vocabulary(parse_yap_tokens(yap_vocab), to_manifest(entries)["assets"], select)
    valid, problems = validate_synonyms(load_synonym_table(table_path), vocab)
    info.update(valid=len(valid), exact=sum(1 for e in valid.values() if e["confidence"] == "exact"),
                close=sum(1 for e in valid.values() if e["confidence"] == "close"), problems=problems)
    return valid, info


def write_sqlite(entries: List[Entry], db: sqlite3.Connection) -> None:
    db.execute("DELETE FROM assets")
    db.execute("DELETE FROM aliases")
    for e in entries:
        p = e.parsed
        db.execute("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (e.asset_id, e.folder, e.rel_path, e.filename, p.label, p.label_norm, p.variant, p.qualifier, p.entry_type,
                    p.language_tag, e.status, json.dumps(p.flags), e.duplicate_of, e.size_bytes, e.container, e.codec, e.width, e.height,
                    e.fps, e.duration_s, e.probe_error, int(e.reviewed)))
        for a in p.aliases:
            db.execute("INSERT INTO aliases VALUES (?,?,?,?,?)", (e.asset_id, a.text, normalize_label(a.text), int(a.verified), a.source))
    db.commit()


def to_manifest(entries: List[Entry]) -> Dict[str, object]:
    assets = []
    for e in entries:
        p = e.parsed
        variant = " / ".join(x for x in (p.variant, p.qualifier) if x) or None
        assets.append({
            "asset_id": e.asset_id, "label": p.label, "label_norm": p.label_norm,
            "aliases": [{"text": a.text, "verified": a.verified, "source": a.source} for a in p.aliases],
            "variant": variant, "rel_path": e.rel_path, "container": e.container, "codec": e.codec, "duration_s": e.duration_s,
            "status": e.status, "reviewed": e.reviewed, "entry_type": p.entry_type, "flags": p.flags,
        })
    return {"schema_version": SCHEMA_VERSION, "source": "ISLRTC Indian Sign Language Dictionary (HEVC re-encode), labels from file names",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "assets": assets}


def write_review_csv(entries: List[Entry], path: Path, defaults: Optional[Dict[str, str]] = None) -> int:
    """Rows a human must look at: anything not a clean source label, plus every term that has several live signs."""
    live = [e for e in entries if e.status in ("source_label", "needs_review", "verified")]
    groups: Dict[str, List[Entry]] = {}
    for e in live:
        groups.setdefault(e.parsed.label_norm, []).append(e)
    rows = []
    for e in entries:
        p = e.parsed
        siblings = len(groups.get(p.label_norm, [])) - 1 if e in live else 0
        issues = [f for f in p.flags if f in REVIEW_FLAGS or f in ("synonym_list", "english_tag")]
        if e.status == "duplicate":
            issues.append(f"duplicate_of:{e.duplicate_of}")
        if e.status in ("unreadable", "concept"):
            issues.append(e.status)
        if siblings > 0:
            issues.append(f"multiple_signs:{siblings + 1}")
        if e.status == "source_label" and siblings == 0 and not issues:
            continue
        rows.append({"asset_id": e.asset_id, "rel_path": e.rel_path, "label": p.label, "label_norm": p.label_norm, "status": e.status,
                     "issues": ";".join(issues), "variant": p.variant or "", "sense_qualifier": p.qualifier or "",
                     "siblings": siblings, "auto_default": "default" if (defaults or {}).get(p.label_norm) == e.asset_id else "", "decision": "", "label_override": "", "aliases_add": "", "notes": ""})
    rows.sort(key=lambda r: (r["label_norm"], r["variant"], r["rel_path"]))
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["asset_id"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def audit(entries: List[Entry], data_dir: Path) -> Dict[str, object]:
    by_status: Dict[str, int] = {}
    codecs: Dict[str, int] = {}
    for e in entries:
        by_status[e.status] = by_status.get(e.status, 0) + 1
        codecs[e.codec or "not_probed"] = codecs.get(e.codec or "not_probed", 0) + 1
    live = [e for e in entries if e.status in ("source_label", "needs_review", "verified")]
    groups: Dict[str, List[Entry]] = {}
    for e in live:
        groups.setdefault(e.parsed.label_norm, []).append(e)
    alias_keys: Dict[str, set] = {}
    for e in live:
        for a in e.parsed.aliases:
            if a.verified:
                alias_keys.setdefault(normalize_label(a.text), set()).add(e.asset_id)
    all_keys = set(groups) | set(alias_keys)
    on_disk_missing = [e.rel_path for e in entries if not (data_dir / e.rel_path).is_file()]
    return {
        "total_video_files": len(entries),
        "by_folder": {f: sum(1 for e in entries if e.folder == f) for f in sorted({e.folder for e in entries})},
        "by_status": dict(sorted(by_status.items())),
        "codecs": dict(sorted(codecs.items(), key=lambda kv: -kv[1])),
        "unique_raw_labels": len({e.parsed.stem.lower() for e in entries}),
        "unique_normalized_headwords": len({e.parsed.label_norm for e in entries}),
        "unique_searchable_keys_live": len(all_keys),
        "headwords_with_multiple_live_signs": sum(1 for v in groups.values() if len(v) > 1),
        "assets_in_multi_sign_headwords": sum(len(v) for v in groups.values() if len(v) > 1),
        "alias_keys_shared_by_several_assets": sum(1 for v in alias_keys.values() if len(v) > 1),
        "explanation_or_concept_entries": by_status.get("concept", 0),
        "duplicates": by_status.get("duplicate", 0),
        "weak_or_missing_labels": sum(1 for e in entries if "weak_label" in e.parsed.flags),
        "missing_files_at_audit_time": len(on_disk_missing),
        "flag_counts": _flag_counts(entries),
        "label_metadata_source": "file names only (no Parquet/CSV label table is present in the download)",
    }


def _flag_counts(entries: Iterable[Entry]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for e in entries:
        for f in e.parsed.flags:
            out[f] = out.get(f, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def run_index(data_dir: Path, out_dir: Path, probe: bool = True, workers: int = 6, reviews: Optional[Path] = None,
              ffprobe: Optional[str] = None, progress=None, synonyms: Optional[Path] = None, yap_vocab: Optional[Path] = None) -> Dict[str, object]:
    """Full run. Writes islrtc_index.sqlite, manifest.json, approved_mappings.json (only from reviews), review_queue.csv, audit.json."""
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ISLRTC data directory not found: {data_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = scan(data_dir)
    assign_status(entries)

    db = sqlite3.connect(out_dir / "islrtc_index.sqlite")
    db.executescript(SCHEMA)
    probe_info = {"requested": probe}
    if probe:
        exe = ffprobe or shutil.which("ffprobe")
        if not exe:
            raise RuntimeError("ffprobe not found on PATH (needed for --probe; pass --no-probe to skip codec/duration)")
        probed, reused = probe_all(entries, data_dir, db, workers, exe, progress)
        probe_info.update(probed=probed, reused_from_cache=reused)
        apply_probe_status(entries)
        # A probe failure can reveal a duplicate's "original" is unreadable; keep duplicate status as is (it never plays anyway).

    review = ReviewResult(0, {}, [])
    if reviews and reviews.is_file():
        review = apply_reviews(entries, reviews)

    auto_select, auto_meta = compute_auto_defaults(entries, review.select)
    synonym_map, synonym_info = build_synonyms(entries, {**auto_select, **review.select}, synonyms, yap_vocab)
    write_sqlite(entries, db)
    db.close()
    (out_dir / "manifest.json").write_text(json.dumps(to_manifest(entries), ensure_ascii=False, indent=1), encoding="utf-8")
    mappings_path = out_dir / "approved_mappings.json"
    # Reviewer decisions win over automatic defaults. `auto_default` lists, per key, [chosen, *alternates] so the UI can say "default among N".
    mappings_path.write_text(json.dumps({"select": {**auto_select, **review.select}, "alias": {}, "split": {}, "auto_default": auto_meta, "synonyms": synonym_map}, indent=1,
                                        ensure_ascii=False), encoding="utf-8")
    review_rows = write_review_csv(entries, out_dir / "review_queue.csv", auto_select)
    report = audit(entries, data_dir)
    report.update(synonyms=synonym_info, auto_defaults=len(auto_select), reviewer_defaults=len(review.select), probe=probe_info, review_rows=review_rows, reviews_applied=review.applied, review_unknown_ids=review.unknown_ids,
                  data_dir_name=data_dir.name)
    (out_dir / "audit.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report
