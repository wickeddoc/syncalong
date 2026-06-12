# CLAUDE.md

## Project Overview

**syncalong** is a Python CLI tool that performs forced alignment of plain-text lyrics against an audio file and outputs timestamped lyrics in LRC format. It uses OpenAI's Whisper for word-level speech-to-text, then a dynamic-programming sequence aligner to map transcript words back onto the original lyric lines.

## Architecture

The package lives in `src/syncalong/` with seven modules forming a linear pipeline:

```
audio file ─► transcribe.py ─► align.py ─► formatter.py ─► LRC stdout
                                  ▲
lyrics file ─► lyrics.py ─────────┘
```

- `cli.py` — Argparse CLI and orchestration. Entry point: `main()`. Registered as `syncalong` console script.
- `lyrics.py` — Parses a plain-text lyrics file into `LyricLine` dataclasses. Detects section headers like `[Chorus]` and `(Bridge)` as blank lines. Also builds the Whisper `initial_prompt` from the lyric text (`lyrics_prompt()`).
- `textnorm.py` — Shared `normalize()` (lowercase, strip punctuation/accents, collapse whitespace) used by both `lyrics.py` and `transcribe.py` so the two sides of the aligner normalize identically.
- `transcribe.py` — Wraps `whisper.load_model()` and `model.transcribe()` with `word_timestamps=True`, `condition_on_previous_text=False`, and an optional lyrics `initial_prompt`. Returns a list of `WordTimestamp` dataclasses. Heavy import of whisper is kept lazy.
- `align.py` — Core algorithm. Needleman–Wunsch-style DP over the flat lyric word list × transcript word list. Uses `rapidfuzz.fuzz.ratio` for fuzzy word scoring (falls back to `difflib`). Enforces monotonic alignment so repeated sections don't cross-match. Interpolates timestamps linearly for unmatched lines between matched anchors. Public API: `align_lyrics_to_transcript()`.
- `formatter.py` — Converts `list[tuple[LyricLine, float | None]]` to LRC string. Timestamps formatted as `[mm:ss.xx]`.
- `vocal_separator.py` — Optional module, only imported when `--separate-vocals` is passed. Shells out to `demucs` via subprocess to isolate vocals before transcription.

## Key Design Decisions

- **Forced alignment, not open recognition.** We already know the lyrics; the aligner just needs to find *when* each word occurs. This is more robust than STT → fuzzy search.
- **Monotonic DP alignment.** The mapping must be order-preserving — word 5 in the lyrics can't match a transcript word that comes before word 4's match. This prevents repeated choruses from collapsing onto the same audio segment.
- **Fuzzy matching threshold.** Default 55 (configurable via `--threshold`). Below this score, a word pair is treated as non-matching. This tolerates Whisper mishearing ("runnin" vs "running") without accepting garbage matches.
- **Gap interpolation.** If lines A and C are matched but B is not, B gets a linearly interpolated timestamp. This avoids gaps in the output without requiring every word to match.

## Commands

```bash
# Install in editable mode (with test dependencies)
pip install -e ".[dev]"

# Install with optional vocal separation
pip install -e ".[vocal-separation]"

# Run tests
python -m pytest tests/ -v

# Run the tool
syncalong lyrics.txt song.mp3
syncalong lyrics.txt song.wav -m medium --separate-vocals > output.lrc
```

## Dependencies

- `openai-whisper` — speech recognition with word-level timestamps
- `rapidfuzz` — fast fuzzy string matching for the aligner
- `demucs` (optional) — vocal isolation from Meta

System dependency: `ffmpeg` must be installed for Whisper audio decoding.

## Testing

Tests are in `tests/test_core.py`. They exercise the text pipeline (parsing, scoring, DP alignment, formatting) with synthetic `WordTimestamp` data — no audio files or Whisper model needed. Run with `pytest`.

Test classes:
- `TestParseLyrics` — lyrics file parsing, blank lines, section headers, punctuation, normalization
- `TestLyricsPrompt` — Whisper prompt building and truncation
- `TestTranscribeOptions` — transcription option defaults and CLI prompt flag
- `TestWordScore` — fuzzy similarity scoring
- `TestDPAlign` — DP alignment with exact matches, extra transcript words, fuzzy matches, empty inputs
- `TestAlignLyricsToTranscript` — end-to-end alignment, repeated-chorus monotonicity, gap interpolation
- `TestResolveAudioPath` — vocal-separation guard and missing-demucs error
- `TestLRCFormatter` — timestamp conversion, rounding carry, LRC output

## Code Conventions

- Python 3.9+ with `from __future__ import annotations` for modern type hints.
- Dataclasses for structured data (`LyricLine`, `WordTimestamp`).
- Text normalization lives in `textnorm.py` and is shared by `lyrics.py` and `transcribe.py` — never reimplement it locally; both sides must normalize identically.
- Status/progress messages go to stderr; only LRC output goes to stdout.
- Whisper is imported lazily inside `transcribe_audio()` to keep CLI startup fast.
