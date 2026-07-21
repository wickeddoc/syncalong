"""Command-line interface and main orchestration."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from syncalong.align import align_lyrics_to_transcript
from syncalong.formatter import format_lrc
from syncalong.lyrics import lyrics_prompt, parse_lyrics
from syncalong.transcribe import transcribe_audio


def build_parser() -> argparse.ArgumentParser:
    """Build the ``syncalong`` command-line argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
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
        help=(
            "Path to a plain-text file containing the song lyrics "
            "(one line per lyric line)."
        ),
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to the audio file (wav, mp3, flac, ogg, …).",
    )
    parser.add_argument(
        "-m",
        "--model",
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
        "-l",
        "--language",
        default=None,
        help="Language code (e.g. 'en', 'de'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--separate-vocals",
        action="store_true",
        help=(
            "Pre-process audio with Demucs to isolate vocals "
            "(requires 'demucs' extra)."
        ),
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
        help=(
            "Minimum fuzzy-match score (0–100) to accept a word alignment. "
            "(default: 55)"
        ),
    )
    parser.add_argument(
        "--server",
        default=None,
        metavar="URL",
        help=(
            "Transcribe on a remote syncalong server instead of locally "
            "(e.g. http://gpu-box:8000). Falls back to $SYNCALONG_SERVER."
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        metavar="TOKEN",
        help="Bearer token for the remote server. Falls back to $SYNCALONG_TOKEN.",
    )
    return parser


def _demucs_available() -> bool:
    """Report whether the optional ``demucs`` package is importable.

    Returns:
        ``True`` if ``demucs`` is installed, else ``False``.
    """
    return importlib.util.find_spec("demucs") is not None


def _whisper_available() -> bool:
    """Report whether the optional ``openai-whisper`` package is importable.

    Returns:
        ``True`` if ``whisper`` is installed, else ``False``.
    """
    return importlib.util.find_spec("whisper") is not None


def resolve_audio_path(audio: Path, separate_vocals: bool) -> Path:
    """Optionally run vocal separation and return the path to transcribe.

    Args:
        audio: The user-supplied audio path.
        separate_vocals: Whether to isolate vocals with Demucs first.

    Returns:
        ``audio`` unchanged, or the path to the isolated vocals.

    Raises:
        SystemExit: If ``separate_vocals`` is requested but ``demucs`` is not
            installed.
    """
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
    """Parse arguments and run the full align-to-LRC pipeline.

    Writes the LRC document to stdout and progress messages to stderr.

    Args:
        argv: Argument vector to parse; defaults to ``sys.argv`` when ``None``.
    """
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

    server = args.server or os.environ.get("SYNCALONG_SERVER")
    token = args.token or os.environ.get("SYNCALONG_TOKEN")
    prompt = None if args.no_lyrics_prompt else lyrics_prompt(lyric_lines)

    # --- Transcribe (remote server or local Whisper) -----------------------
    if server:
        if args.model != "base":
            print(
                "note: -m/--model is ignored in remote mode; the server "
                "chooses the model.",
                file=sys.stderr,
            )
        print(f"Transcribing on remote server {server} …", file=sys.stderr)
        from syncalong.remote import RemoteTranscriber

        transcriber = RemoteTranscriber(server, token=token)
        try:
            word_timestamps = transcriber.transcribe(
                args.audio,
                language=args.language,
                initial_prompt=prompt,
                separate_vocals=args.separate_vocals,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        audio_path = resolve_audio_path(args.audio, args.separate_vocals)
        if not _whisper_available():
            print(
                "ERROR: local transcription requires Whisper.\n"
                "Install it with:  pip install syncalong[whisper]\n"
                "…or transcribe on a GPU server with --server URL.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"Transcribing with Whisper ({args.model}) …  "
            "(this may take a while on CPU)",
            file=sys.stderr,
        )
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


def serve_main(argv: list[str] | None = None) -> None:
    """Run the syncalong transcription server (``syncalong-serve``).

    Loads the Whisper model once, then serves it over HTTP so thin clients can
    transcribe without a local GPU. Requires the ``server`` extra.

    Args:
        argv: Argument vector to parse; defaults to ``sys.argv`` when ``None``.
    """
    try:
        import uvicorn

        from syncalong.server import create_app
    except ImportError:
        print(
            "The transcription server requires extra dependencies.\n"
            "Install them with:  pip install syncalong[server]",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    parser = argparse.ArgumentParser(
        prog="syncalong-serve",
        description="Serve Whisper transcription over HTTP for syncalong clients.",
    )
    parser.add_argument(
        "-m", "--model", default="base", help="Whisper model name (default: base)."
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)."
    )
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000).")
    parser.add_argument(
        "--device", default=None, help="Torch device, e.g. 'cuda' (default: auto)."
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Require this bearer token. Falls back to $SYNCALONG_TOKEN.",
    )
    parser.add_argument(
        "--no-vocal-separation",
        action="store_true",
        help="Reject separate_vocals requests (don't run Demucs).",
    )
    args = parser.parse_args(argv)

    token = args.token or os.environ.get("SYNCALONG_TOKEN")

    from syncalong.transcribe import Transcriber

    print(f"Loading Whisper model '{args.model}' …", file=sys.stderr)
    transcriber = Transcriber(args.model, device=args.device)
    app = create_app(
        transcriber,
        token=token,
        allow_vocal_separation=not args.no_vocal_separation,
    )
    print(f"Serving on http://{args.host}:{args.port} …", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)
