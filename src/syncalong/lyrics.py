"""Parse a plain-text lyrics file into structured lines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from syncalong.textnorm import normalize


@dataclass
class LyricLine:
    """A single line from the lyrics file."""

    index: int              # 0-based line position in the file
    raw: str                # Original text as it appeared
    words: list[str]        # Normalized, split tokens for matching
    is_blank: bool = False  # True for blank / instrumental lines


def lyrics_prompt(lines: list[LyricLine], max_chars: int = 600) -> str:
    """
    Build a Whisper ``initial_prompt`` from the original lyric text.

    Feeding the known lyrics to the decoder biases transcription toward
    the correct words. Whisper only keeps a limited number of prompt
    tokens (and keeps the *last* ones), so we pass just the beginning of
    the song, truncated at a word boundary.
    """
    text = " ".join(line.raw for line in lines if not line.is_blank)
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars + 1)
    if cut == -1:
        cut = max_chars
    return text[:cut]


def parse_lyrics(path: Path) -> list[LyricLine]:
    """
    Read a lyrics text file and return a list of LyricLine objects.

    Blank lines are preserved (they represent instrumental breaks) but
    contain no words to match.

    Section headers like ``[Chorus]`` or ``(Bridge)`` are stripped.
    """
    raw_text = path.read_text(encoding="utf-8")
    lines: list[LyricLine] = []

    for idx, raw in enumerate(raw_text.splitlines()):
        stripped = raw.strip()

        # Detect section headers: [Chorus], [Verse 1], (Bridge), etc.
        if re.fullmatch(r"[\[\(].*[\]\)]", stripped):
            # Keep as blank line so spacing is preserved
            lines.append(LyricLine(index=idx, raw=stripped, words=[], is_blank=True))
            continue

        if not stripped:
            lines.append(LyricLine(index=idx, raw="", words=[], is_blank=True))
            continue

        norm = normalize(stripped)
        words = norm.split() if norm else []

        lines.append(
            LyricLine(
                index=idx,
                raw=stripped,
                words=words,
                is_blank=len(words) == 0,
            )
        )

    return lines
