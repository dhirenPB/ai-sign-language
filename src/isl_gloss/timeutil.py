"""Tiny timestamp helper (HH:MM:SS.mmm -> seconds)."""

from __future__ import annotations


def seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)
