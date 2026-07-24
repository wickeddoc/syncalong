# CLAUDE.md

## Project Overview

**syncalong** is a Python library and CLI tool that performs forced alignment of plain-text lyrics against an audio file and outputs timestamped lyrics in LRC format. It uses OpenAI's Whisper for word-level speech-to-text, then a dynamic-programming sequence aligner to map transcript words back onto the original lyric lines. The same pipeline is usable two ways: the `syncalong` console script, or the importable package API (`syncalong.align()`, `syncalong.Transcriber`, …) for embedding in other projects.

## Architecture

The package lives in `src/syncalong/` with ten modules forming a linear pipeline:

```
audio file ─► transcribe.py ─► align.py ─► formatter.py ─► LRC stdout
                                  ▲
lyrics file ─► lyrics.py ─────────┘
```

- `cli.py` — Argparse CLI and orchestration. Entry point: `main()`. Registered as `syncalong` console script.
- `lyrics.py` — Parses a plain-text lyrics file into `LyricLine` dataclasses. Detects section headers like `[Chorus]` and `(Bridge)` as blank lines. Also builds the Whisper `initial_prompt` from the lyric text (`lyrics_prompt()`).
- `textnorm.py` — Shared `normalize()` (lowercase, strip accents, collapse whitespace; apostrophes deleted so contractions stay one token, other punctuation becomes a word boundary so hyphenated compounds split) used by both `lyrics.py` and `transcribe.py` so the two sides of the aligner normalize identically.
- `transcribe.py` — The `Transcriber` class loads a Whisper model once (`whisper.load_model()`) and exposes `.transcribe()` (with `word_timestamps=True`, `condition_on_previous_text=False`, and an optional lyrics `initial_prompt`), returning `WordTimestamp` dataclasses. A long-running caller reuses one `Transcriber`; `transcribe_audio()` is a thin one-shot wrapper over it (unchanged API). The heavy whisper import stays lazy inside `Transcriber.__init__` (reached only when no model is injected), so `import syncalong` never pulls in whisper.
- `remote.py` — stdlib-only `RemoteTranscriber`, same `.transcribe()` interface as `Transcriber`, but uploads audio to a syncalong server over HTTP (`urllib`) and returns the word timestamps it computes. Lets a thin client (no Whisper, no GPU) drive the same `align()` pipeline.
- `server.py` — FastAPI `create_app` + `/health` + `/transcribe`, model loaded lazily/once. Only imported by `syncalong.cli.serve_main` and the `syncalong-serve` console script (the `server` extra); never imported by the package root, so `import syncalong` never pulls in fastapi.
- `align.py` — Core algorithm. Needleman–Wunsch-style DP over the flat lyric word list × transcript word list. Uses `rapidfuzz.fuzz.ratio` for fuzzy word scoring (falls back to `difflib`; scores are `functools.cache`d since song vocabularies repeat heavily). Enforces monotonic alignment so repeated sections don't cross-match. Interpolates timestamps linearly for unmatched lines between matched anchors, and extrapolates lines before the first / after the last anchor from the transcript's start and end times. Public API: `align_lyrics_to_transcript()`.
- `formatter.py` — Converts `list[tuple[LyricLine, float | None]]` to LRC string. Timestamps formatted as `[mm:ss.xx]`.
- `vocal_separator.py` — Optional module, only imported when `--separate-vocals` is passed. Shells out to `demucs` via subprocess to isolate vocals before transcription. Progress streams to stderr (stdout stays clean for LRC). The temp output dir is reclaimed per call: `Transcriber.transcribe()` calls `cleanup_separation()` right after transcribing (critical for the long-running server — on a tmpfs `/tmp` leaked stems accumulate in RAM), failures clean up immediately, and an `atexit` backstop covers one-shot CLI runs and crashes.
- `pipeline.py` — High-level library facade. `align()` runs the whole lyrics+audio → `AlignmentResult` pipeline (`.timed_lines`, `.lrc`, `.matched`, `.total`); its `lyrics` argument is type-driven (`Path` = read a file, `str` = lyrics text, `list[LyricLine]` = already parsed). `align_to_lrc()` returns just the LRC string. `separate_vocals=True` raises `ModuleNotFoundError` when demucs is absent (library-appropriate, vs the CLI's `sys.exit`). Both are re-exported from the package root alongside the building blocks; the CLI keeps its own orchestration so its stderr/exit behavior is unchanged.

## Key Design Decisions

- **Forced alignment, not open recognition.** We already know the lyrics; the aligner just needs to find *when* each word occurs. This is more robust than STT → fuzzy search.
- **Monotonic DP alignment.** The mapping must be order-preserving — word 5 in the lyrics can't match a transcript word that comes before word 4's match. This prevents repeated choruses from collapsing onto the same audio segment.
- **Fuzzy matching threshold.** Default 55 (configurable via `--threshold`). Below this score, a word pair is treated as non-matching. This tolerates Whisper mishearing ("runnin" vs "running") without accepting garbage matches.
- **Gap interpolation and edge extrapolation.** If lines A and C are matched but B is not, B gets a linearly interpolated timestamp. Unmatched lines before the first match / after the last are extrapolated using the transcript's start and end as virtual anchors. Untagged lines are dropped by many LRC players, so every sung line should get a timestamp as long as at least one line matched.
- **Library-first, CLI-preserving.** The package exposes a curated public API (`__init__.py` + `__all__`, 15 names) and ships a `py.typed` marker. A reusable `Transcriber` lets a long-running consumer (e.g. a jukebox) load the Whisper model once; `align()` returns structured per-line timestamps (`AlignmentResult`), not just an LRC blob. The CLI is a thin layer over the same building blocks and its behavior is unchanged.

## Commands

```bash
# Install dev + docs tooling in editable mode
pip install -e ".[dev,docs]"

# Install with optional vocal separation
pip install -e ".[vocal-separation]"

# Quality gate (mirrors CI — all four must pass)
python -m pytest tests/ -v
ruff check .
black --check .
pyright src

# Build & preview the docs site
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs build --strict # what CI runs

# Build the package
python -m build       # sdist + wheel into ./dist

# Run the tool
syncalong lyrics.txt song.mp3
syncalong lyrics.txt song.wav -m medium --separate-vocals > output.lrc

# Remote transcription: run Whisper on a GPU box, keep the client thin
syncalong-serve --model medium --host 0.0.0.0 --device cuda   # on the GPU box
syncalong lyrics.txt song.mp3 --server http://gpu-box:8000 > song.lrc
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

Runtime (core, always installed):

- `rapidfuzz` — fast fuzzy string matching for the aligner

`openai-whisper` is **not** a core dependency — it moved to the `whisper`
extra so a bare `pip install syncalong` stays thin and torch-free. Install it
locally, or run Whisper on a separate GPU box via the `server` extra and talk
to it with `RemoteTranscriber`/`--server` (see `remote.py`/`server.py` above).

Optional-dependency groups (`pyproject.toml`):

- `whisper` — `openai-whisper` (local transcription; pulls in PyTorch)
- `server` — `openai-whisper`, `fastapi`, `uvicorn[standard]`,
  `python-multipart` (the `syncalong-serve` GPU-side server)
- `vocal-separation` — `demucs`, `torchcodec` (vocal isolation from Meta;
  `torchcodec` is Demucs's audio I/O backend on modern torchaudio — separation
  fails to save without it)
- `dev` — `pytest`, `ruff`, `black`, `pyright`, `build`, `twine`, plus
  `fastapi`/`httpx`/`uvicorn`/`python-multipart` for testing `server.py`
  torch-free
- `docs` — `mkdocs-material`, `mkdocstrings[python]`

System dependency: `ffmpeg` must be installed wherever Whisper actually runs
(the local machine with the `whisper` extra, or the `server` box).

## Release, Packaging & Documentation

- **Versioning is tag-driven.** `setuptools-scm` derives the version from the
  latest git tag — there is **no** hardcoded version in the source. Tag `v0.1.0`
  → version `0.1.0`; commits after a tag → `0.1.1.devN+g<hash>`. `__version__`
  is read at runtime from installed metadata (`importlib.metadata.version`).
  To release, bump `CHANGELOG.md`, then `git tag -a vX.Y.Z && git push --tags`.
- **CI** (`.github/workflows/ci.yml`, on push/PR): a `quality` job (ruff, black
  `--check`, `pyright src`), a `test` matrix (Python 3.9–3.13), and a strict
  `docs` build. CI is intentionally **torch-free** — the test suite injects a
  fake Whisper model, and pyright resolves everything except the lazy
  `import whisper` (which carries a `# type: ignore[import-not-found]`). This
  keeps CI fast and avoids torch/Python-version flakiness.
- **Release** (`.github/workflows/release.yml`, on `v*.*.*` tag): builds the
  sdist + wheel, publishes to PyPI via **Trusted Publishing (OIDC)** — no stored
  token — then creates a GitHub Release with the artifacts attached. PyPI must
  have a trusted publisher configured for this repo + `release.yml` + the `pypi`
  environment.
- **Docs** are MkDocs Material + mkdocstrings (`mkdocs.yml`, `docs/`), published
  on ReadTheDocs (`.readthedocs.yaml`). The API reference is generated from the
  Google-style docstrings; mkdocstrings reads the source statically from `src/`
  (via `paths: [src]`), so the docs build never installs whisper/torch either.
- **License:** MIT (`LICENSE`), declared with the PEP 639 SPDX expression
  `license = "MIT"` in `pyproject.toml`.
- **Copyrighted audio never ships.** The git-ignored `test/` directory holds
  local audio fixtures; the sdist file list (via setuptools-scm) only includes
  git-tracked files, so nothing under `test/` can leak into a release.

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
- The public API is curated in `src/syncalong/__init__.py` via explicit re-exports + `__all__`; the package ships a `py.typed` marker, so keep type hints on the public surface correct (`pyright src/` should stay clean). `__version__` is sourced from installed metadata, not hardcoded.
- All classes and functions use Google-style docstrings (`Args:` / `Returns:` / `Raises:` / `Attributes:` as applicable). Ruff's pydocstyle rules (google convention) enforce their presence.
- Formatting is **black** and linting is **ruff** (both configured in `pyproject.toml`, line length 88); run `black .` / `ruff check --fix .` before committing. The full quality gate — `pytest`, `ruff check .`, `black --check .`, `pyright src` — must stay green and is enforced in CI.
