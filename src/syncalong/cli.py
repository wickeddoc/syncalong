"""Command-line interface and main orchestration."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from syncalong.transcribe import transcribe_audio
from syncalong.lyrics import parse_lyrics, lyrics_prompt
from syncalong.align import align_lyrics_to_transcript
from syncalong.formatter import format_lrc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syncalong",
        description=(
            "Align a plain-text lyrics file to an audio recording and "
            "output timestamped lyrics in LRC format."
        ),
    )
    parser.add_argument(
        "lyrics",
        type=Path,
        help="Path to a plain-text file containing the song lyrics (one line per lyric line).",
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to the audio file (wav, mp3, flac, ogg, …).",
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        metavar="MODEL",
        help=(
            "Whisper model name: tiny, base, small, medium, large, turbo, "
            "or a variant like small.en (English-only, often more accurate "
            "for English) or large-v3. Larger = more accurate but slower. "
            "(default: base)"
        ),
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Language code (e.g. 'en', 'de'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--separate-vocals",
        action="store_true",
        help="Pre-process audio with Demucs to isolate vocals (requires 'demucs' extra).",
    )
    parser.add_argument(
        "--no-lyrics-prompt",
        action="store_true",
        help=(
            "Don't feed the lyrics to Whisper as a decoding prompt. "
            "The prompt usually improves accuracy but can occasionally "
            "cause hallucinated text."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=55.0,
        help="Minimum fuzzy-match score (0–100) to accept a word alignment. (default: 55)",
    )
    return parser


def _demucs_available() -> bool:
    return importlib.util.find_spec("demucs") is not None


def resolve_audio_path(audio: Path, separate_vocals: bool) -> Path:
    """Optionally run vocal separation; return the path to use for transcription."""
    if not separate_vocals:
        return audio

    # vocal_separator only shells out to demucs, so importing it never
    # fails — check for the demucs package itself.
    if not _demucs_available():
        print(
            "ERROR: --separate-vocals requires the 'demucs' package.\n"
            "Install it with:  pip install syncalong[vocal-separation]",
            file=sys.stderr,
        )
        sys.exit(1)

    from syncalong.vocal_separator import separate

    print("Separating vocals with Demucs …", file=sys.stderr)
    return separate(audio)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- Validate inputs ---------------------------------------------------
    if not args.lyrics.is_file():
        parser.error(f"Lyrics file not found: {args.lyrics}")
    if not args.audio.is_file():
        parser.error(f"Audio file not found: {args.audio}")

    # --- Parse lyrics ------------------------------------------------------
    print(f"Reading lyrics from {args.lyrics} …", file=sys.stderr)
    lyric_lines = parse_lyrics(args.lyrics)
    if not lyric_lines:
        print("ERROR: Lyrics file is empty or contains no text.", file=sys.stderr)
        sys.exit(1)

    # --- Optional vocal separation -----------------------------------------
    audio_path = resolve_audio_path(args.audio, args.separate_vocals)

    # --- Transcribe audio --------------------------------------------------
    print(
        f"Transcribing with Whisper ({args.model}) …  "
        "(this may take a while on CPU)",
        file=sys.stderr,
    )
    prompt = None if args.no_lyrics_prompt else lyrics_prompt(lyric_lines)
    word_timestamps = transcribe_audio(
        audio_path,
        model_name=args.model,
        language=args.language,
        initial_prompt=prompt,
    )
    if not word_timestamps:
        print("ERROR: Whisper returned no words.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Got {len(word_timestamps)} words from transcription.",
        file=sys.stderr,
    )

    # --- Align lyrics to transcript ----------------------------------------
    print("Aligning lyrics …", file=sys.stderr)
    timed_lines = align_lyrics_to_transcript(
        lyric_lines,
        word_timestamps,
        threshold=args.threshold,
    )

    # --- Output LRC --------------------------------------------------------
    lrc = format_lrc(timed_lines)
    sys.stdout.write(lrc)
    print(
        f"\nDone — {sum(1 for _, t in timed_lines if t is not None)}"
        f"/{len(timed_lines)} lines matched.",
        file=sys.stderr,
    )
