"""High-level alignment pipeline: lyrics + audio → timestamped result."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from syncalong.align import align_lyrics_to_transcript
from syncalong.formatter import format_lrc
from syncalong.lyrics import LyricLine, lyrics_prompt, parse_lyrics, parse_lyrics_text
from syncalong.transcribe import Transcriber

LyricsInput = Union[str, Path, list[LyricLine]]
AudioInput = Union[str, Path]


@dataclass
class AlignmentResult:
    """Result of aligning lyrics to an audio file.

    Attributes:
        timed_lines: One ``(LyricLine, timestamp_or_None)`` pair per lyric
            line, in order. The timestamp (seconds) is the line's start time,
            or ``None`` if it could not be aligned.
        lrc: The same alignment rendered as an LRC document.
        matched: Number of lines that received a timestamp.
        total: Total number of lines.
    """

    timed_lines: list[tuple[LyricLine, float | None]]
    lrc: str
    matched: int
    total: int


def _coerce_lyrics(lyrics: object) -> list[LyricLine]:
    """Normalise the ``lyrics`` argument into parsed lyric lines.

    Args:
        lyrics: A :class:`~pathlib.Path` to a lyrics file, the raw lyrics as a
            ``str``, or an already-parsed ``list[LyricLine]``.

    Returns:
        Parsed lyric lines.

    Raises:
        TypeError: If ``lyrics`` is none of the accepted types.
    """
    if isinstance(lyrics, Path):
        return parse_lyrics(lyrics)
    if isinstance(lyrics, str):
        return parse_lyrics_text(lyrics)
    if isinstance(lyrics, list):
        return lyrics
    raise TypeError(
        "lyrics must be a Path, str, or list[LyricLine]; "
        f"got {type(lyrics).__name__}"
    )


def align(
    lyrics: LyricsInput,
    audio: AudioInput,
    *,
    transcriber: Transcriber | None = None,
    model_name: str = "base",
    language: str | None = None,
    use_lyrics_prompt: bool = True,
    threshold: float = 55.0,
    separate_vocals: bool = False,
) -> AlignmentResult:
    """Align lyrics to an audio file and return a structured result.

    Args:
        lyrics: A :class:`~pathlib.Path` to a lyrics file, the raw lyrics as a
            ``str``, or an already-parsed ``list[LyricLine]``.
        audio: Path to the audio file (``str`` or :class:`~pathlib.Path`).
        transcriber: A reusable :class:`~syncalong.transcribe.Transcriber`.
            When ``None``, a transient one is built from ``model_name``.
        model_name: Whisper model to load when ``transcriber`` is ``None``.
        language: BCP-47 language code, or ``None`` to auto-detect.
        use_lyrics_prompt: Whether to bias Whisper with the lyrics text.
        threshold: Minimum fuzzy score (0–100) to accept a word match.
        separate_vocals: When ``True``, isolate vocals with Demucs first.

    Returns:
        An :class:`AlignmentResult`.

    Raises:
        TypeError: If ``lyrics`` is not a ``Path``, ``str``, or list.
        ModuleNotFoundError: If ``separate_vocals`` is ``True`` but ``demucs``
            is not installed.
    """
    lyric_lines = _coerce_lyrics(lyrics)
    audio_path = Path(str(audio))

    if transcriber is None:
        transcriber = Transcriber(model_name)

    prompt = lyrics_prompt(lyric_lines) if use_lyrics_prompt else None
    transcript = transcriber.transcribe(
        audio_path,
        language=language,
        initial_prompt=prompt,
        separate_vocals=separate_vocals,
    )

    timed_lines = align_lyrics_to_transcript(
        lyric_lines, transcript, threshold=threshold
    )
    lrc = format_lrc(timed_lines)
    matched = sum(1 for _, ts in timed_lines if ts is not None)
    return AlignmentResult(
        timed_lines=timed_lines,
        lrc=lrc,
        matched=matched,
        total=len(timed_lines),
    )


def align_to_lrc(lyrics: LyricsInput, audio: AudioInput, **kwargs: Any) -> str:
    """Align lyrics to audio and return just the LRC document.

    Convenience wrapper: equivalent to ``align(lyrics, audio, **kwargs).lrc``.

    Args:
        lyrics: See :func:`align`.
        audio: See :func:`align`.
        **kwargs: Forwarded verbatim to :func:`align`.

    Returns:
        The LRC document as a string.
    """
    return align(lyrics, audio, **kwargs).lrc
