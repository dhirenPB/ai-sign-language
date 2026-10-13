"""Append-only JSONL cache of LLM candidates, so an interrupted or quota-limited run can resume without re-spending
requests. Only the LLM step is cached: rules and validation are cheap and re-run every time, so rule changes take
effect immediately."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from .models import Candidate
from .prompt import PROMPT_VERSION  # recorded with each entry; the key itself uses the full system prompt


def cache_key(model: str, system_prompt: str, user_text: str, n, settings: str = "") -> str:
    """Keyed on the model, the generation settings (reasoning effort, temperature...), the whole system prompt and the
    whole user message (so the surrounding group context counts), plus the sentence label inside that message. Editing the prompt, the context or the grouping can therefore never
    serve a stale answer."""
    raw = "|".join((model, settings, hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
                    hashlib.sha256(user_text.encode("utf-8")).hexdigest(), str(n)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResultStore:
    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self._data: dict[str, dict] = {}
        if self.path and self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self._data[rec["key"]] = rec
                except (ValueError, KeyError):
                    continue  # a half-written last line must not break resuming

    def get(self, key: str) -> Optional[Candidate]:
        rec = self._data.get(key)
        if not rec:
            return None
        status = rec.get("status") or ("uncertain" if rec.get("uncertain") else "ok")
        return Candidate(tokens=list(rec["tokens"]), uncertain=status != "ok", notes=rec.get("notes", ""), status=status,
                         uncertain_parts=list(rec.get("uncertain_parts", [])),
                         needs_human_review=bool(rec.get("needs_human_review", False)),
                         echoed_source_text=rec.get("echoed_source_text"))

    def put(self, key: str, cand: Candidate, model: str) -> None:
        rec = {"key": key, "tokens": cand.tokens, "uncertain": cand.uncertain, "status": cand.status,
               "uncertain_parts": cand.uncertain_parts, "needs_human_review": cand.needs_human_review,
               "echoed_source_text": cand.echoed_source_text, "notes": cand.notes, "model": model,
               "prompt_version": PROMPT_VERSION}
        self._data[key] = rec
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._data)
