# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
