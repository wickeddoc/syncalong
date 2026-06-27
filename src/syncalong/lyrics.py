"""Parse a plain-text lyrics file into structured lines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from syncalong.textnorm import normalize


@dataclass
class LyricLine:
    """A single line from the lyrics file.

    Attributes:
        index: 0-based line position in the file.
        raw: Original text as it appeared.
        words: Normalized, split tokens used for matching.
        is_blank: ``True`` for blank or instrumental/section-header lines.
    """

    index: int
    raw: str
    words: list[str]
    is_blank: bool = False


def lyrics_prompt(lines: list[LyricLine], max_chars: int = 600) -> str:
    """Build a Whisper ``initial_prompt`` from the original lyric text.

    Feeding the known lyrics to the decoder biases transcription toward the
    correct words. Whisper keeps only a limited number of prompt tokens
    (and keeps the *last* ones), so just the beginning of the song is passed,
    truncated at a word boundary.

    Args:
        lines: Parsed lyric lines.
        max_chars: Maximum prompt length in characters.

    Returns:
        The prompt text (possibly truncated at a word boundary).
    """
    text = " ".join(line.raw for line in lines if not line.is_blank)
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars + 1)
    if cut == -1:
        cut = max_chars
    return text[:cut]


def parse_lyrics_text(text: str) -> list[LyricLine]:
    """Parse raw lyrics text into structured lines.

    Blank lines are preserved (they represent instrumental breaks) but carry
    no words to match. Section headers like ``[Chorus]`` or ``(Bridge)`` are
    treated as blank lines so spacing is preserved.

    Args:
        text: The full lyrics as one string, lines separated by newlines.

    Returns:
        One :class:`LyricLine` per input line, in order.
    """
    lines: list[LyricLine] = []
    for idx, raw in enumerate(text.splitlines()):
        stripped = raw.strip()

        if re.fullmatch(r"[\[\(].*[\]\)]", stripped):
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


def parse_lyrics(path: Path) -> list[LyricLine]:
    """Read a lyrics text file and parse it into structured lines.

    Thin wrapper over :func:`parse_lyrics_text` that reads the file first.

    Args:
        path: Path to a UTF-8 plain-text lyrics file.

    Returns:
        One :class:`LyricLine` per line in the file, in order.
    """
    return parse_lyrics_text(path.read_text(encoding="utf-8"))
