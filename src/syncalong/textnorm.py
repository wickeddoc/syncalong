"""Shared text normalization for lyric and transcript words.

Both sides of the aligner must normalize identically so fuzzy matching
compares like with like — keep this the single source of truth.
"""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and accents, and collapse whitespace.

    Apostrophes are deleted so contractions stay one token ("don't" →
    "dont"); every other punctuation mark becomes a word boundary so
    hyphenated compounds split the way Whisper transcribes them ("déjà-vu" →
    "deja vu"). Both sides of the aligner must call this so fuzzy matching
    compares like with like.

    Args:
        text: Arbitrary input text.

    Returns:
        The normalised text.
    """
    text = unicodedata.normalize("NFKD", text)
    # Drop combining accent marks — deleting, not spacing, so accented
    # letters mid-word don't split the word.
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
