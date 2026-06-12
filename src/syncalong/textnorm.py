"""Shared text normalization for lyric and transcript words.

Both sides of the aligner must normalize identically so fuzzy matching
compares like with like — keep this the single source of truth.
"""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and accents, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()
