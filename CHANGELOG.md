# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] — 2026-07-23

The 2.0 release introduces a **client/server architecture**. syncalong's only
GPU-heavy stage — Whisper transcription (plus optional Demucs) — can now run on
a separate machine while a thin, torch-free client parses lyrics, aligns, and
writes LRC. To make that split possible, Whisper is no longer a core dependency.

### Upgrading from 1.x

- **Local CLI / library users:** `pip install syncalong` is now thin and
  **torch-free**, so it no longer transcribes on its own. Reinstall with the new
  extra to keep transcribing locally: `pip install "syncalong[whisper]"`.
  Nothing else about local use changes — the same commands and the same
  `align()` / `Transcriber` API keep working. The CLI now exits with an install
  hint if you run it locally without the `whisper` extra.
- **New thin-client + server users:** run `syncalong-serve` on a GPU box
  (`pip install "syncalong[server]"`) and point clients at it with `--server URL`
  (CLI) or [`RemoteTranscriber`](https://syncalong.readthedocs.io/en/latest/reference/remote/)
  (library). The client needs neither Whisper nor ffmpeg. See the
  [remote transcription guide](https://syncalong.readthedocs.io/en/latest/remote/).

### Added

- **Transcription server (`syncalong-serve` / `syncalong.server`).** A FastAPI
  app exposing `GET /health` and `POST /transcribe`, loading the Whisper model
  once and reusing it for every request. Installed by the new `server` extra.
  `create_app()` is importable for embedding the server in a larger ASGI app.
- **`RemoteTranscriber` (stdlib-only client).** A drop-in replacement for
  `Transcriber` that uploads audio to the server over HTTP and returns the word
  timestamps it computes; alignment still runs locally on the client. Added to
  the public API (`syncalong.RemoteTranscriber`).
- **Remote-mode CLI flags.** `--server URL` transcribes on a remote server
  instead of locally, and `--token` supplies a bearer token; both fall back to
  `$SYNCALONG_SERVER` / `$SYNCALONG_TOKEN`. In remote mode `-m/--model` is
  ignored (the server chooses the model).
- **Server-side vocal separation.** The `separate_vocals` request flag runs
  Demucs on the server; `syncalong-serve --no-vocal-separation` rejects it.
- **Optional shared-token auth.** Setting a token (flag or `$SYNCALONG_TOKEN`)
  requires `Authorization: Bearer <token>`; wrong/missing tokens get HTTP 401.
- **Documentation.** A remote transcription guide, an HTTP API contract, a
  `syncalong.server` API-reference page, and a CPU-vs-GPU benchmarks page backed
  by a reusable `benchmarks/` harness.

### Changed

- **BREAKING:** `openai-whisper` moved from a core dependency to the new
  `whisper` extra. `pip install syncalong` is now thin (torch-free); install
  `syncalong[whisper]` for local transcription, or `syncalong[server]` to run
  the server. Existing local CLI use now needs `pip install "syncalong[whisper]"`.

### Fixed

- **Vocal separation on current torch.** The `vocal-separation` extra now pulls
  in `torchcodec`, which recent torchaudio requires to save Demucs output —
  without it, `--separate-vocals` failed at write time
  (`TorchCodec is required for save_with_torchcodec`).

## [1.0.0] — 2026-07-21

First public release — the initial version published to PyPI.

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

## [0.1.0] — 2026-07-19

Initial development tag — never published to PyPI. The feature set above was
first released publicly as 1.0.0.

[Unreleased]: https://github.com/wickeddoc/syncalong/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/wickeddoc/syncalong/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/wickeddoc/syncalong/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/wickeddoc/syncalong/releases/tag/v0.1.0
