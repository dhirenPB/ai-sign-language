from transcript_cleaner.config import CleaningConfig
from transcript_cleaner.pipeline import process_bytes


def run(text, name="sample.txt", **cfg_kwargs):
    """Process text (str or bytes) through the full pipeline and return the ProcessResult."""
    data = text if isinstance(text, bytes) else text.encode("utf-8")
    return process_bytes(data, name, CleaningConfig(**cfg_kwargs))


def texts(result):
    return [c.text for c in result.chunks]


def flag_codes(chunk):
    return [f.code for f in chunk.uncertainty_flags]
