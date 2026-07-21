"""Transcribe audio with OpenAI Whisper and extract word-level timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from syncalong.textnorm import normalize


@dataclass
class WordTimestamp:
    """A single recognised word with its timing.

    Attributes:
        word: Normalized text (for matching).
        raw: Original text from Whisper.
        start: Start time in seconds.
        end: End time in seconds.
    """

    word: str
    raw: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for the transcription wire format.

        Returns:
            A mapping with ``word``, ``raw``, ``start``, and ``end`` keys.
        """
        return {
            "word": self.word,
            "raw": self.raw,
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WordTimestamp:
        """Rebuild a :class:`WordTimestamp` from its serialized form.

        Args:
            data: A mapping with ``word``, ``raw``, ``start``, ``end`` keys.

        Returns:
            The reconstructed :class:`WordTimestamp`.
        """
        return cls(
            word=data["word"],
            raw=data["raw"],
            start=float(data["start"]),
            end=float(data["end"]),
        )


def _build_transcribe_options(
    *,
    language: str | None,
    initial_prompt: str | None,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
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


def _extract_words(result: dict[str, Any]) -> list[WordTimestamp]:
    """Flatten Whisper's segmented result into word timestamps.

    Args:
        result: The dict returned by ``model.transcribe(...)``.

    Returns:
        Normalised word timestamps, in order; words that normalise to empty
        are skipped.
    """
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
                WordTimestamp(word=norm, raw=raw, start=w["start"], end=w["end"])
            )
    return words


class Transcriber:
    """Reusable Whisper transcriber that loads its model once.

    Loading a Whisper model is expensive, so a long-running consumer (e.g. a
    jukebox processing many songs) should create one ``Transcriber`` and call
    :meth:`transcribe` repeatedly rather than calling
    :func:`transcribe_audio` per song.

    Args:
        model_name: Whisper model size or variant (``tiny``, ``base``,
            ``small``, ``medium``, ``large``, ``turbo``, ``small.en``,
            ``large-v3``, ...). Ignored when ``model`` is provided.
        model: A preloaded Whisper model object. When given it is used as-is
            and ``model_name`` is not loaded — useful for sharing an
            already-loaded model or injecting a test double.
        device: Torch device to load the model on (e.g. ``cuda``); ``None``
            lets Whisper choose. Ignored when ``model`` is provided.
    """

    def __init__(
        self, model_name: str = "base", *, model: Any = None, device: str | None = None
    ):
        """Initialize the transcriber, loading a Whisper model if none is given.

        Args:
            model_name: Whisper model size or variant to load when ``model`` is
                ``None``. See the class docstring for accepted values.
            model: A preloaded Whisper model to use as-is instead of loading one.
            device: Torch device to load the model on (e.g. ``cuda``); ``None``
                lets Whisper choose. Ignored when ``model`` is provided.
        """
        injected = model is not None
        if model is None:
            # Heavy import, kept lazy so `import syncalong` never pulls in
            # whisper/torch. It is also deliberately absent at type-check time
            # in CI, hence the ignore.
            import whisper  # type: ignore[import-not-found]

            if device is None:
                model = whisper.load_model(model_name)
            else:
                model = whisper.load_model(model_name, device=device)
        self.model_name = None if injected else model_name
        self._model: Any = model

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        separate_vocals: bool = False,
    ) -> list[WordTimestamp]:
        """Transcribe one audio file into word-level timestamps.

        Args:
            audio_path: Audio file (any format ffmpeg can decode).
            language: BCP-47 language code, or ``None`` to auto-detect.
            initial_prompt: Text to bias the decoder with (e.g. the lyrics).
            separate_vocals: When ``True``, isolate vocals with Demucs first.

        Returns:
            Every recognised word, in order, with start/end times in seconds.

        Raises:
            ModuleNotFoundError: If ``separate_vocals`` is ``True`` but the
                optional ``demucs`` dependency is not installed.
        """
        if separate_vocals:
            audio_path = self._separate_vocals(Path(audio_path))
        opts = _build_transcribe_options(
            language=language, initial_prompt=initial_prompt
        )
        result = self._model.transcribe(str(audio_path), **opts)
        return _extract_words(result)

    @staticmethod
    def _separate_vocals(audio_path: Path) -> Path:
        import importlib.util

        if importlib.util.find_spec("demucs") is None:
            raise ModuleNotFoundError(
                "separate_vocals=True requires the optional 'demucs' dependency. "
                "Install it with: pip install syncalong[vocal-separation]"
            )
        from syncalong.vocal_separator import separate

        return separate(audio_path)


def transcribe_audio(
    audio_path: Path,
    *,
    model_name: str = "base",
    language: str | None = None,
    initial_prompt: str | None = None,
    separate_vocals: bool = False,
) -> list[WordTimestamp]:
    """Transcribe one audio file, loading the model for this call only.

    Convenience wrapper around :class:`Transcriber` for one-shot use. A
    consumer that transcribes many files should instantiate a
    :class:`Transcriber` once and reuse it instead.

    Args:
        audio_path: Audio file (any format ffmpeg can decode).
        model_name: Whisper model size or variant.
        language: BCP-47 language code, or ``None`` to auto-detect.
        initial_prompt: Text to bias the decoder with (e.g. the lyrics).
        separate_vocals: When ``True``, isolate vocals with Demucs first.

    Returns:
        Ordered word timestamps.
    """
    return Transcriber(model_name).transcribe(
        audio_path,
        language=language,
        initial_prompt=initial_prompt,
        separate_vocals=separate_vocals,
    )
