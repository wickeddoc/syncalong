# CLAUDE.md

## Project Overview

**syncalong** is a Python library and CLI tool that performs forced alignment of plain-text lyrics against an audio file and outputs timestamped lyrics in LRC format. It uses OpenAI's Whisper for word-level speech-to-text, then a dynamic-programming sequence aligner to map transcript words back onto the original lyric lines. The same pipeline is usable two ways: the `syncalong` console script, or the importable package API (`syncalong.align()`, `syncalong.Transcriber`, …) for embedding in other projects.

## Architecture

The package lives in `src/syncalong/` with eight modules forming a linear pipeline:

```
audio file ─► transcribe.py ─► align.py ─► formatter.py ─► LRC stdout
                                  ▲
lyrics file ─► lyrics.py ─────────┘
```

- `cli.py` — Argparse CLI and orchestration. Entry point: `main()`. Registered as `syncalong` console script.
- `lyrics.py` — Parses a plain-text lyrics file into `LyricLine` dataclasses. Detects section headers like `[Chorus]` and `(Bridge)` as blank lines. Also builds the Whisper `initial_prompt` from the lyric text (`lyrics_prompt()`).
- `textnorm.py` — Shared `normalize()` (lowercase, strip accents, collapse whitespace; apostrophes deleted so contractions stay one token, other punctuation becomes a word boundary so hyphenated compounds split) used by both `lyrics.py` and `transcribe.py` so the two sides of the aligner normalize identically.
- `transcribe.py` — The `Transcriber` class loads a Whisper model once (`whisper.load_model()`) and exposes `.transcribe()` (with `word_timestamps=True`, `condition_on_previous_text=False`, and an optional lyrics `initial_prompt`), returning `WordTimestamp` dataclasses. A long-running caller reuses one `Transcriber`; `transcribe_audio()` is a thin one-shot wrapper over it (unchanged API). The heavy whisper import stays lazy inside `Transcriber.__init__` (reached only when no model is injected), so `import syncalong` never pulls in whisper.
- `align.py` — Core algorithm. Needleman–Wunsch-style DP over the flat lyric word list × transcript word list. Uses `rapidfuzz.fuzz.ratio` for fuzzy word scoring (falls back to `difflib`; scores are `lru_cache`d since song vocabularies repeat heavily). Enforces monotonic alignment so repeated sections don't cross-match. Interpolates timestamps linearly for unmatched lines between matched anchors, and extrapolates lines before the first / after the last anchor from the transcript's start and end times. Public API: `align_lyrics_to_transcript()`.
- `formatter.py` — Converts `list[tuple[LyricLine, float | None]]` to LRC string. Timestamps formatted as `[mm:ss.xx]`.
- `vocal_separator.py` — Optional module, only imported when `--separate-vocals` is passed. Shells out to `demucs` via subprocess to isolate vocals before transcription. Progress streams to stderr (stdout stays clean for LRC); the temp output dir is removed at process exit via `atexit`.
- `pipeline.py` — High-level library facade. `align()` runs the whole lyrics+audio → `AlignmentResult` pipeline (`.timed_lines`, `.lrc`, `.matched`, `.total`); its `lyrics` argument is type-driven (`Path` = read a file, `str` = lyrics text, `list[LyricLine]` = already parsed). `align_to_lrc()` returns just the LRC string. `separate_vocals=True` raises `ModuleNotFoundError` when demucs is absent (library-appropriate, vs the CLI's `sys.exit`). Both are re-exported from the package root alongside the building blocks; the CLI keeps its own orchestration so its stderr/exit behavior is unchanged.

## Key Design Decisions

- **Forced alignment, not open recognition.** We already know the lyrics; the aligner just needs to find *when* each word occurs. This is more robust than STT → fuzzy search.
- **Monotonic DP alignment.** The mapping must be order-preserving — word 5 in the lyrics can't match a transcript word that comes before word 4's match. This prevents repeated choruses from collapsing onto the same audio segment.
- **Fuzzy matching threshold.** Default 55 (configurable via `--threshold`). Below this score, a word pair is treated as non-matching. This tolerates Whisper mishearing ("runnin" vs "running") without accepting garbage matches.
- **Gap interpolation and edge extrapolation.** If lines A and C are matched but B is not, B gets a linearly interpolated timestamp. Unmatched lines before the first match / after the last are extrapolated using the transcript's start and end as virtual anchors. Untagged lines are dropped by many LRC players, so every sung line should get a timestamp as long as at least one line matched.
- **Library-first, CLI-preserving.** The package exposes a curated public API (`__init__.py` + `__all__`, 14 names) and ships a `py.typed` marker. A reusable `Transcriber` lets a long-running consumer (e.g. a jukebox) load the Whisper model once; `align()` returns structured per-line timestamps (`AlignmentResult`), not just an LRC blob. The CLI is a thin layer over the same building blocks and its behavior is unchanged.

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

### Library use

```python
import syncalong
from pathlib import Path

tx = syncalong.Transcriber(model_name="base")   # load the model once, reuse it
# lyrics: Path → read a file; str → lyrics text; list[LyricLine] → pre-parsed
res = syncalong.align(Path("lyrics.txt"), "song.mp3", transcriber=tx)
res.timed_lines         # [(LyricLine, 12.3), ...]   per-line timestamps
res.lrc                 # "[00:12.30] ..."           formatted LRC document
res.matched, res.total  # 28, 30

# one-liner when you only want the LRC text:
lrc = syncalong.align_to_lrc(Path("lyrics.txt"), "song.mp3", transcriber=tx)
```

## Dependencies

- `openai-whisper` — speech recognition with word-level timestamps
- `rapidfuzz` — fast fuzzy string matching for the aligner
- `demucs` (optional) — vocal isolation from Meta

System dependency: `ffmpeg` must be installed for Whisper audio decoding.

## Testing

Tests are in `tests/test_core.py`. They exercise the text pipeline (parsing, scoring, DP alignment, formatting) and the library API with synthetic `WordTimestamp` data and an injected fake Whisper model — no audio files or real Whisper model needed. Run with `pytest`.

Test classes:
- `TestParseLyrics` / `TestParseLyricsText` — lyrics file and in-memory text parsing, blank lines, section headers, punctuation, normalization
- `TestLyricsPrompt` — Whisper prompt building and truncation
- `TestTranscribeOptions` — transcription option defaults and CLI prompt flag
- `TestTranscriber` — model reuse via an injected fake model (no real Whisper load), option forwarding, and `transcribe_audio` delegation
- `TestWordScore` — fuzzy similarity scoring
- `TestDPAlign` — DP alignment with exact matches, extra transcript words, fuzzy matches, empty inputs
- `TestAlignLyricsToTranscript` — end-to-end alignment, repeated-chorus monotonicity, gap interpolation, edge extrapolation
- `TestAlignFacade` — `align()` / `AlignmentResult` end-to-end: all three `lyrics` input forms, prompt toggle, bad-type `TypeError`, and the demucs-missing `ModuleNotFoundError`
- `TestVocalSeparator` — demucs orchestration with a faked subprocess (vocals path, cleanup registration, failure)
- `TestResolveAudioPath` — CLI vocal-separation guard and missing-demucs error
- `TestLRCFormatter` — timestamp conversion, rounding carry, LRC output
- `TestPublicAPI` — package re-exports / `__all__`, and the `import syncalong` does-not-load-whisper guarantee (checked in a subprocess)
- `TestPackaging` — the `py.typed` marker is present

## Code Conventions

- Python 3.9+ with `from __future__ import annotations` for modern type hints.
- Dataclasses for structured data (`LyricLine`, `WordTimestamp`).
- Text normalization lives in `textnorm.py` and is shared by `lyrics.py` and `transcribe.py` — never reimplement it locally; both sides must normalize identically.
- Status/progress messages go to stderr; only LRC output goes to stdout.
- Whisper is imported lazily inside `Transcriber.__init__()` (reached only when no model is injected) to keep CLI startup fast and ensure `import syncalong` never loads whisper.
- The public API is curated in `src/syncalong/__init__.py` via explicit re-exports + `__all__`; the package ships a `py.typed` marker, so keep type hints on the public surface correct (`pyright src/` should stay clean).
- All classes and functions use Google-style docstrings (`Args:` / `Returns:` / `Raises:` / `Attributes:` as applicable).
