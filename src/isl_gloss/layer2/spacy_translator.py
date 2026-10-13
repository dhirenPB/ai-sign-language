"""Layer 2A adapter: runs the reference-equivalent spaCy translator on one cleaned English sentence.

* The translation logic lives, unmodified, in `spacy_reference.py`. This module only wraps it: it never raises, records a
  status and an error, keeps the raw output, and carries the sentence id and metadata through.
* The input is the same cleaned English text that Layer 1 sent to the LLM. No LLM gloss ever enters this module.
* Dependencies are never downloaded here. If spaCy, the model or NLTK is missing the result says so and gives setup steps.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable, Optional

ADAPTER_VERSION = "1.0.0"
REFERENCE = {
    "repository": "https://github.com/krishnshyam/VirtualISLInterpreter",
    "path": "src/translate/spacy_rules.py",
    "commit": "f5d9484d9fea21d1f0a77711eda0483c75187c04",
    "original_sha256": "2154e50a6a24155b7de4584e74b6318e61426dc66eb08ae8a099ff9df01c47d4",
    "compatibility_changes": ["C1 lazy model load", "C2 translate_to_tokens uses the lazy loader"],
}
SETUP_HELP = ("Layer 2 needs spaCy, its small English model and NLTK. Install once, outside the application:\n"
              "  python -m pip install spacy nltk\n"
              "  python -m spacy download en_core_web_sm\n"
              "NLTK's WordNet data is NOT needed: the reference only imports nltk.corpus.wordnet and never reads it.")

OK, EMPTY, ERROR, UNAVAILABLE = "ok", "empty", "error", "unavailable"


class DependencyUnavailable(RuntimeError):
    pass


@dataclass
class SpacyResult:
    sentence_id: str
    source_text: str
    spacy_gloss: Optional[str]  # reference formatting: lower-cased lemmas joined by spaces; None when nothing was produced
    spacy_tokens: Optional[list]  # the same tokens before joining (a merged phrase stays one token and may contain spaces)
    spacy_status: str  # ok | empty | error | unavailable
    spacy_error: Optional[dict] = None
    group_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)  # timestamps, speaker, turn index... copied from Layer 1 untouched

    def to_dict(self) -> dict[str, Any]:
        return {"sentence_id": self.sentence_id, "group_id": self.group_id, "source_text": self.source_text,
                "spacy_gloss": self.spacy_gloss, "spacy_tokens": self.spacy_tokens, "spacy_status": self.spacy_status,
                "spacy_error": self.spacy_error, **self.metadata}


def _dist_version(*names: str) -> Optional[str]:
    for n in names:
        try:
            return metadata.version(n)
        except metadata.PackageNotFoundError:
            continue
    return None


def dependency_versions() -> dict[str, Any]:
    """Versions that define the baseline. Reads package metadata only, so it works when spaCy is not installed."""
    return {"python": platform.python_version(), "spacy": _dist_version("spacy"), "en_core_web_sm": _dist_version("en-core-web-sm", "en_core_web_sm"),
            "nltk": _dist_version("nltk"), "adapter_version": ADAPTER_VERSION, "reference": REFERENCE}


class SpacyTranslator:
    """`translate_tokens` can be injected (tests, or an alternative backend); by default the reference port is imported
    lazily on first use, so merely creating this object never needs spaCy."""

    def __init__(self, translate_tokens: Optional[Callable[[str], list]] = None):
        self._fn = translate_tokens
        self._unavailable: Optional[str] = None

    # ------------------------------------------------------------------------------------------------------
    def _backend(self) -> Callable[[str], list]:
        if self._fn is not None:
            return self._fn
        if self._unavailable:
            raise DependencyUnavailable(self._unavailable)
        try:
            from . import spacy_reference as ref
            ref.get_nlp()  # fail here, once, with a clear reason, if the model is missing
        except ImportError as exc:
            self._unavailable = f"Python package not installed: {exc}"
            raise DependencyUnavailable(self._unavailable) from exc
        except OSError as exc:  # spacy.load raises OSError (E050) when the model is not installed
            self._unavailable = f"spaCy model en_core_web_sm is not available: {exc}"
            raise DependencyUnavailable(self._unavailable) from exc

        def run(text: str) -> list:
            # same formula as the reference translate_text(): lower-cased token texts
            return [t.text.lower() for t in ref.translate_to_tokens(text)]

        self._fn = run
        return run

    def available(self) -> tuple[bool, Optional[str]]:
        try:
            self._backend()
            return True, None
        except DependencyUnavailable as exc:
            return False, f"{exc}\n{SETUP_HELP}"

    # ------------------------------------------------------------------------------------------------------
    def translate(self, sentence_id: str, text: Optional[str], group_id: Optional[str] = None,
                  metadata: Optional[dict] = None) -> SpacyResult:
        """Translate one sentence. Never raises: every failure becomes a status and an error record."""
        text = "" if text is None else text
        base = dict(sentence_id=sentence_id, source_text=text, group_id=group_id, metadata=dict(metadata or {}))
        if not isinstance(text, str):
            return SpacyResult(**{**base, "source_text": str(text)}, spacy_gloss=None, spacy_tokens=None, spacy_status=ERROR,
                               spacy_error={"kind": "bad_input", "message": f"source text is {type(text).__name__}, not str"})
        if not text.strip():
            return SpacyResult(**base, spacy_gloss=None, spacy_tokens=None, spacy_status=EMPTY,
                               spacy_error={"kind": "empty_source", "message": "The cleaned source sentence is empty; nothing was translated."})
        try:
            tokens = self._backend()(text)
        except DependencyUnavailable as exc:
            return SpacyResult(**base, spacy_gloss=None, spacy_tokens=None, spacy_status=UNAVAILABLE,
                               spacy_error={"kind": "dependency_missing", "message": str(exc), "setup": SETUP_HELP})
        except Exception as exc:  # the reference code can raise on unusual parses; one sentence must not stop the run
            return SpacyResult(**base, spacy_gloss=None, spacy_tokens=None, spacy_status=ERROR,
                               spacy_error={"kind": type(exc).__name__, "message": str(exc)[:300], "sentence_id": sentence_id,
                                            "source_text": text[:200]})
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            return SpacyResult(**base, spacy_gloss=None, spacy_tokens=None, spacy_status=ERROR,
                               spacy_error={"kind": "malformed_output", "message": "the translator did not return a list of strings"})
        gloss = " ".join(tokens)  # reference formatting: not re-cased, not re-tokenised
        if not gloss.strip():
            return SpacyResult(**base, spacy_gloss="", spacy_tokens=tokens, spacy_status=EMPTY,
                               spacy_error={"kind": "empty_output", "message": "The translator returned no tokens."})
        return SpacyResult(**base, spacy_gloss=gloss, spacy_tokens=tokens, spacy_status=OK)


def environment_report() -> dict[str, Any]:
    return {**dependency_versions(), "executable": sys.executable}
