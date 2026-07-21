# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Remote transcription (server/client).** `syncalong-serve` (FastAPI) runs
  Whisper on a GPU box; a stdlib-only `RemoteTranscriber` on a thin client
  uploads audio and aligns locally. New `--server`/`--token` CLI flags
  (also `$SYNCALONG_SERVER` / `$SYNCALONG_TOKEN`), a `server` optional extra,
  and optional shared-token auth. Vocal separation runs server-side via the
  `separate_vocals` request flag.

### Changed

- **BREAKING:** `openai-whisper` moved from a core dependency to the new
  `whisper` extra. `pip install syncalong` is now thin (torch-free); install
  `syncalong[whisper]` for local transcription, or `syncalong[server]` to run
  the server. Existing local CLI use now needs `pip install "syncalong[whisper]"`.

## [0.1.0] — 2026-07-19

First public release.

### Added

- Forced alignment of plain-text lyrics to audio, exported as LRC.
- Command-line tool (`syncalong` console script) with options for model size,
  language, fuzzy-match threshold, optional vocal separation, and lyrics
  prompting.
- Library API: `align()` returning a structured `AlignmentResult`,
  `align_to_lrc()` convenience wrapper, and a reusable `Transcriber` that loads
  the Whisper model once for batch processing.
- Type-driven `lyrics` argument (`Path` = file, `str` = text,
  `list[LyricLine]` = pre-parsed).
- Optional Demucs-based vocal separation via the `vocal-separation` extra.
- `py.typed` marker and a curated, typed public API.
- Google-style docstrings across the codebase and a MkDocs Material +
  mkdocstrings documentation site.

[Unreleased]: https://github.com/wickeddoc/syncalong/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wickeddoc/syncalong/releases/tag/v0.1.0
