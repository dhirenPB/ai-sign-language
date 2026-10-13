"""Layer 2: an independent rule-based (spaCy) candidate, compared with the Layer 1 LLM candidate.

Layer 1 does not import this package; Layer 2 only reads saved Layer 1 results.
"""

from .candidate_comparator import Comparison, compare_candidates
from .pipeline import Layer2Config, run_layer2
from .selection import Selection, SelectionPolicy, select
from .spacy_translator import SpacyResult, SpacyTranslator

__all__ = ["Comparison", "compare_candidates", "Layer2Config", "run_layer2", "Selection", "SelectionPolicy", "select",
           "SpacyResult", "SpacyTranslator"]
