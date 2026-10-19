"""Map cleaned sentences back to the original transcript, by matching WORDS in order. A heuristic, and labelled as one.

Why it exists: the cleaner joins cues into chunks and the sentence splitter then re-cuts them, so sentence boundaries differ from
the transcript's segment boundaries. A sentence therefore cannot simply inherit "its" segment.

What it does: walk the transcript's word stream (or, for segments without word timings, the segment's text split into words) with
a forward-only cursor and find where each sentence's words occur in order. A sentence may skip a few transcript words (a filler
the cleaner removed) but may not reorder anything. Timing is then READ from the matched material: the first and last matched
word's own times when the engine gave word times, otherwise the matched segments' own start/end. Nothing is interpolated,
spread evenly or estimated. A sentence that cannot be matched is reported as unaligned and keeps the cleaned chunk's interval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

MAX_GAP = 3  # transcript words that may be missing from the sentence between two matched words (removed fillers, markers)
SEARCH_WINDOW = 400  # how far ahead of the cursor a sentence may start (material removed by the cleaner)


def _norm(token: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", token.lower())


_JOINERS = re.compile(r"[-‐-―/.,]")  # hyphens, non-breaking hyphens, dashes, slashes, dots and commas (co-founder, 1.3, U.S., 150,000)


def _tokens(text: str) -> list[str]:
    """Words of a text, with a hyphenated or slashed word split into its parts. Whisper writes 'co-founder' as two timed words while the
    cleaned sentence keeps it as one, so both sides must be split the same way or the match breaks right there."""
    return [n for part in str(text).split() for n in (_norm(p) for p in _JOINERS.split(part)) if n]


@dataclass
class Alignment:
    status: str  # exact | approximate | unaligned
    start: Optional[float] = None
    end: Optional[float] = None
    timing_basis: Optional[str] = None  # word | segment | mixed
    segment_ids: Optional[list] = None
    skipped_transcript_words: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _stream(segments: list[dict]) -> list[tuple]:
    out = []
    for si, seg in enumerate(segments):
        if seg.get("words"):
            for w in seg["words"]:
                for n in _tokens(w["text"]):  # a hyphenated word becomes several tokens that share that word's own timing
                    out.append((n, si, w))
        else:
            for n in _tokens(seg["text"]):
                out.append((n, si, None))
    return out


def align_sentences(segments: list[dict], sentences: list[str]) -> list[Alignment]:
    """One Alignment per sentence, in order. `segments` are the canonical transcript segments (dicts)."""
    stream = _stream(segments)
    cursor = 0
    results: list[Alignment] = []
    for text in sentences:
        toks = _tokens(text)
        found = None
        if toks:
            for s in range(cursor, min(len(stream), cursor + SEARCH_WINDOW)):
                if stream[s][0] != toks[0]:
                    continue
                pos, last, skipped, ok = s, s, 0, True
                for i, tok in enumerate(toks):
                    q = pos if i == 0 else next((q for q in range(pos, min(len(stream), pos + 1 + MAX_GAP)) if stream[q][0] == tok), None)
                    if q is None or (i == 0 and stream[q][0] != tok):
                        ok = False
                        break
                    skipped += q - pos
                    last, pos = q, q + 1
                if ok:
                    found = (s, last, skipped)
                    break
        if not found:
            results.append(Alignment("unaligned"))
            continue
        first, last, skipped = found
        cursor = last + 1
        fw, lw = stream[first][2], stream[last][2]
        fs, ls = segments[stream[first][1]], segments[stream[last][1]]
        start = fw["start"] if fw else fs["start"]
        end = lw["end"] if lw else ls["end"]
        basis = "word" if fw and lw else "segment" if not fw and not lw else "mixed"
        ids = [segments[i]["id"] for i in range(stream[first][1], stream[last][1] + 1)]
        results.append(Alignment("exact" if skipped == 0 else "approximate", start, end, basis, ids, skipped))
    return results
