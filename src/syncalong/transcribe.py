"""Transcribe audio with OpenAI Whisper and extract word-level timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from syncalong.textnorm import normalize


@dataclass
class WordTimestamp:
    """A single word with its start/end time in seconds."""

    word: str       # Normalized text
    raw: str        # Original text from Whisper
    start: float    # Seconds
    end: float      # Seconds


def _build_transcribe_options(
    *,
    language: str | None,
    initial_prompt: str | None,
) -> dict:
    opts: dict = {
        "word_timestamps": True,
        # Music is a classic trigger for Whisper's repetition/hallucination
        # loops, especially with repeated choruses — don't condition each
        # window on the previously decoded text.
        "condition_on_previous_text": False,
    }
    if language:
        opts["language"] = language
    if initial_prompt:
        opts["initial_prompt"] = initial_prompt
    return opts


def transcribe_audio(
    audio_path: Path,
    *,
    model_name: str = "base",
    language: str | None = None,
    initial_prompt: str | None = None,
) -> list[WordTimestamp]:
    """
    Run Whisper on *audio_path* and return word-level timestamps.

    Parameters
    ----------
    audio_path : Path
        Audio file (any format ffmpeg can decode).
    model_name : str
        Whisper model size: tiny | base | small | medium | large.
    language : str or None
        BCP-47 language code. ``None`` = auto-detect.
    initial_prompt : str or None
        Text to bias the decoder with (e.g. the song's lyrics).

    Returns
    -------
    list[WordTimestamp]
        Ordered list of every word Whisper recognised, with timing.
    """
    import whisper  # Heavy import — keep lazy

    model = whisper.load_model(model_name)

    transcribe_opts = _build_transcribe_options(
        language=language,
        initial_prompt=initial_prompt,
    )

    result = model.transcribe(str(audio_path), **transcribe_opts)

    words: list[WordTimestamp] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            raw = w.get("word", "").strip()
            if not raw:
                continue
            norm = normalize(raw)
            if not norm:
                continue
            words.append(
                WordTimestamp(
                    word=norm,
                    raw=raw,
                    start=w["start"],
                    end=w["end"],
                )
            )

    return words
