"""syncalong — forced alignment of lyrics to audio, exporting LRC.

This package is both a CLI (``syncalong`` console script) and an importable
library. The high-level entry point is :func:`align`; lower-level building
blocks are re-exported for advanced use.
"""

from __future__ import annotations

from syncalong.align import align_lyrics_to_transcript
from syncalong.formatter import format_lrc
from syncalong.lyrics import (
    LyricLine,
    lyrics_prompt,
    parse_lyrics,
    parse_lyrics_text,
)
from syncalong.pipeline import AlignmentResult, align, align_to_lrc
from syncalong.transcribe import Transcriber, WordTimestamp, transcribe_audio
from syncalong.vocal_separator import separate

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # High-level facade
    "align",
    "align_to_lrc",
    "AlignmentResult",
    # Transcription
    "Transcriber",
    "transcribe_audio",
    "WordTimestamp",
    # Lyrics
    "parse_lyrics",
    "parse_lyrics_text",
    "lyrics_prompt",
    "LyricLine",
    # Low-level aligner / formatter / vocals
    "align_lyrics_to_transcript",
    "format_lrc",
    "separate",
]
