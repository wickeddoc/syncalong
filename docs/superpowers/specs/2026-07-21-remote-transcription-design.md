# Remote transcription (server/client) — design

**Status:** Approved (brainstorming) — ready for implementation planning
**Date:** 2026-07-21
**Author:** wickeddoc (with Claude)

## Problem

A jukebox machine holds all the music but is low-powered and has no GPU.
Running Whisper (and optionally Demucs) locally there is slow or infeasible.
A separate machine has a GPU. We want to split syncalong so the GPU-heavy work
runs on the powerful machine while the jukebox keeps the music, the lyrics, and
the cheap work.

## Key insight

syncalong's pipeline has exactly **one** GPU-heavy stage —
`Transcriber.transcribe()` (Whisper model load + `model.transcribe()`), plus the
optional Demucs vocal-separation pre-step. Everything else (lyrics parsing, the
Needleman–Wunsch alignment, LRC formatting) is cheap CPU work. And `align()`
already accepts an injectable `transcriber=` argument.

So the split is clean: run the transcriber remotely. A `RemoteTranscriber` that
satisfies the same `.transcribe()` interface drops into the existing seam, and
the rest of the pipeline runs unchanged on the jukebox.

## Decisions (settled during brainstorming)

1. **Split at the `Transcriber` boundary** — thin GPU service. The server does
   only transcription (+ optional Demucs) and returns word timestamps; the
   jukebox runs alignment/formatting locally.
2. **Shipped package feature** — first-class, documented, tested; server behind
   an optional `[server]` extra; client stays in core.
3. **Demucs runs server-side**, driven by a per-request `separate_vocals` flag —
   all GPU work stays on the server; the jukebox never needs demucs/torch.
4. **Server stack: FastAPI + uvicorn.** Client uses the Python **stdlib**
   (`urllib`) only — no new core dependency.
5. **Synchronous request/response** — the connection stays open until word
   timestamps return. No job queue (YAGNI for a home jukebox).
6. **Optional shared-token auth** — a bearer token checked only if one is
   configured; off by default for a trusted LAN.
7. **Whisper becomes an optional extra** (`[whisper]`) so the jukebox install is
   thin and torch-free. (Confirmed: accept the v0.1.0 behavior change.)
8. **Unify the Demucs path** — `separate_vocals` becomes part of the
   `.transcribe()` interface; `align()` just forwards it. (Confirmed.)

## Architecture

```
JUKEBOX  (pip install syncalong — thin, torch-free)     GPU BOX  (pip install syncalong[server])
──────────────────────────────────────────────────     ──────────────────────────────────────
lyrics.py  parse + build prompt          (local)
remote.py  RemoteTranscriber.transcribe(song.mp3) ─POST /transcribe─►  server.py  (FastAPI)
                                                     multipart form:      ├─ optional Demucs (GPU)
                                                       audio (file)       ├─ Whisper .transcribe (GPU)
                                                       language?          └─ one reused Transcriber
                                                       initial_prompt?       (model loaded once at startup)
                                                       separate_vocals?
           list[WordTimestamp]  ◄────── 200 JSON {words:[…], model} ───  serialize
align.py   align_lyrics_to_transcript   (local, cheap DP)
formatter  format_lrc  →  song.lrc
```

`RemoteTranscriber` satisfies the **same `.transcribe()` interface** as
`Transcriber`, so `align(lyrics, audio, transcriber=RemoteTranscriber(url))`
works unchanged, and so does every other consumer of the pipeline.

## The `.transcribe()` interface (unified)

Both transcribers implement:

```python
def transcribe(
    self,
    audio_path: Path,
    *,
    language: str | None = None,
    initial_prompt: str | None = None,
    separate_vocals: bool = False,
) -> list[WordTimestamp]: ...
```

- **`Transcriber`** (local): when `separate_vocals=True`, lazily import
  `syncalong.vocal_separator` and run Demucs locally before Whisper. Raise
  `ModuleNotFoundError` with an install hint if demucs is absent.
- **`RemoteTranscriber`**: forward `separate_vocals` to the server as a form
  field; the server runs Demucs.

`pipeline.align()` drops its own `_separate_vocals` branch and simply forwards
`separate_vocals=` to `transcriber.transcribe()`. One code path; both
transcribers do the right thing.

> The **CLI keeps its own orchestration** (`resolve_audio_path` in `cli.py`,
> with its `sys.exit` UX) unchanged. Local Demucs therefore lives in two places
> intentionally: the CLI layer (exit-based UX) and the library/pipeline layer
> (exception-based). This is consistent with the project's "library raises, CLI
> exits" split.

## New / changed components

### New: `src/syncalong/server.py`

- A FastAPI app built by a `create_app(transcriber=None, *, token=None,
  allow_vocal_separation=True)` **factory** (plus a module-level `app` for
  `uvicorn syncalong.server:app`). Injecting `transcriber` enables torch-free
  tests with a fake.
- Endpoints:
  - `GET /health` → `{"status": "ok", "model": "<name>"}`. No auth.
  - `POST /transcribe` → multipart/form-data:
    - `audio`: uploaded file (streamed to a temp file, ffmpeg-decodable).
    - `language`: optional str.
    - `initial_prompt`: optional str.
    - `separate_vocals`: optional bool (default false).
    - Response `200`: `{"words": [{"word","raw","start","end"}, …],
      "model": "<name>"}`.
- Loads **one** `Transcriber` at startup and reuses it (matches the
  "load the model once" design). Whisper stays lazily imported inside
  `Transcriber`, so importing `server.py` is torch-free.
- Optional auth dependency: if a token is configured, require
  `Authorization: Bearer <token>` (constant-time compare); else allow all.
- `serve_main(argv=None)` entry point parses CLI args and calls
  `uvicorn.run(...)`. It performs a **guarded import** of fastapi/uvicorn and
  prints `pip install syncalong[server]` if they're missing.

### New: `src/syncalong/remote.py`

- `RemoteTranscriber(base_url, *, token=None, timeout=300.0)` — **stdlib
  `urllib` only**. Implements `.transcribe(...)`:
  - Reads the audio bytes, builds a multipart/form-data body by hand, POSTs to
    `{base_url}/transcribe` with the optional `Authorization: Bearer` header.
  - Deserializes the JSON `words` array into `list[WordTimestamp]`.
  - Maps failures to clear exceptions (see Error handling).
- Imports `WordTimestamp` from `transcribe.py` (safe — that import does not pull
  whisper; the whisper import is inside `Transcriber.__init__`).

### Changed: `src/syncalong/transcribe.py`

- Add `separate_vocals: bool = False` to `Transcriber.transcribe()` and to the
  `transcribe_audio()` wrapper (forwarded). Local separation delegates to
  `vocal_separator.separate()` via a lazy import + a demucs-availability check.
- Add an optional `device: str | None = None` to `Transcriber.__init__`,
  forwarded to `whisper.load_model(model_name, device=device)` (so the server
  can pin `cuda`). Default `None` preserves current behavior.
- Add wire (de)serialization for the boundary type: `WordTimestamp.to_dict()`
  and `WordTimestamp.from_dict(d)` (classmethod). Used by both server and
  client so they agree on the shape.

### Changed: `src/syncalong/pipeline.py`

- Remove the local `_separate_vocals` branch; forward `separate_vocals=` to
  `transcriber.transcribe()`. `align()`'s public signature is unchanged.
- The demucs-missing `ModuleNotFoundError` now surfaces from
  `Transcriber.transcribe()` (same exception type and message intent).

### Changed: `src/syncalong/cli.py`

- **Client remote mode:** add `--server URL` and `--token TOKEN` (also honor
  `SYNCALONG_SERVER` / `SYNCALONG_TOKEN` env vars). When `--server` is set,
  build a `RemoteTranscriber` and transcribe through it; `--separate-vocals`
  and `--no-lyrics-prompt` continue to work (separation happens server-side).
  In remote mode, `-m/--model` is server-decided — passing it prints a stderr
  note that it's ignored.
- The existing local path is otherwise unchanged. If `--server` is not set and
  whisper is not installed, transcription fails with a clear
  `pip install syncalong[whisper]` message.

### Changed: `src/syncalong/__init__.py`

- Re-export `RemoteTranscriber`; add it to `__all__`. (Server internals are
  **not** part of the curated public API.) Keep the "`import syncalong` never
  loads whisper" guarantee.

### Changed: `pyproject.toml`

- Move `openai-whisper>=20231117` out of `dependencies` into a new
  `whisper` extra. Core `dependencies` becomes just `rapidfuzz>=3.0`.
- New extras:
  - `whisper = ["openai-whisper>=20231117"]`
  - `server = ["openai-whisper>=20231117", "fastapi>=0.110", "uvicorn[standard]>=0.29", "python-multipart>=0.0.9"]`
  - `vocal-separation = ["demucs>=4.0"]` (unchanged)
- `dev` gains `fastapi>=0.110` and `httpx>=0.27` (for Starlette's `TestClient`)
  so server/client tests run in CI **without torch**.
- New console script: `syncalong-serve = "syncalong.server:serve_main"`.

## Wire format

`POST /transcribe` — request: multipart/form-data
| field | type | required | notes |
|---|---|---|---|
| `audio` | file | yes | any ffmpeg-decodable format |
| `language` | text | no | BCP-47 code; auto-detect if omitted |
| `initial_prompt` | text | no | lyrics-derived Whisper bias prompt |
| `separate_vocals` | text `"true"/"false"` | no | default false |

Response `200`:
```json
{ "words": [ { "word": "hello", "raw": "Hello", "start": 1.2, "end": 1.6 } ],
  "model": "base" }
```
Error responses: `{ "detail": "<message>" }` with an appropriate status code.

## Data flow (remote)

1. Jukebox parses lyrics (local), builds the prompt (local, lyrics-derived),
   reads audio bytes.
2. `RemoteTranscriber` POSTs the multipart form (+ optional bearer token) to the
   GPU box.
3. Server: optional Demucs → Whisper (reused model) → words → JSON.
4. Jukebox deserializes to `list[WordTimestamp]`, runs
   `align_lyrics_to_transcript` (local) → `format_lrc` → writes `song.lrc`.

## Privacy note (documented honestly)

The lyrics **file** and the **entire alignment** stay on the jukebox. However,
with `use_lyrics_prompt=True` (the default), Whisper's `initial_prompt` is built
*from the lyrics* and **is sent to the server** to bias decoding. To send
**audio only** (zero lyric text over the wire), use `use_lyrics_prompt=False`
(CLI `--no-lyrics-prompt`), at some accuracy cost. This trade-off will be
documented on the remote-mode docs page.

## Error handling

| Situation | Behavior |
|---|---|
| Server unreachable / DNS / connection refused | `RemoteTranscriber` raises a clear error naming the URL |
| Non-2xx response | raise with the HTTP status + server `detail` message |
| Bad/missing token (server has one configured) | server returns `401`; client surfaces it |
| `separate_vocals=True`, server lacks demucs | server returns `400` with a "install `syncalong[vocal-separation]` on the server" hint |
| Local mode, whisper not installed | `ModuleNotFoundError` → `pip install syncalong[whisper]` hint |
| `syncalong-serve` run without the `[server]` extra | guarded import prints `pip install syncalong[server]` and exits non-zero |
| Client timeout on a large file/model | configurable `timeout`; documented; raise a clear timeout error |

## Testing (stays CI torch-free)

- **Server** (`server.py`): Starlette `TestClient` + `create_app(transcriber=
  FakeTranscriber())` — no real Whisper. Assert: `/health`; `/transcribe` returns
  the serialized words; `language`/`initial_prompt`/`separate_vocals` are plumbed
  into the injected transcriber; auth accept/reject; the demucs-missing error.
- **Client** (`remote.py`): `RemoteTranscriber` against a **stubbed `urlopen`**.
  Assert: correct URL/method, multipart fields present, bearer header when a
  token is set, JSON → `list[WordTimestamp]` round-trip, HTTP/connection errors
  mapped to clear exceptions.
- **Round-trip**: `WordTimestamp.to_dict()`/`from_dict()` invariance.
- **Pipeline remote path**: inject a fake transcriber into `align()` and assert
  `separate_vocals` is forwarded and no local Demucs runs.
- **CLI**: `--server` selects `RemoteTranscriber`; `--token`/env plumbed;
  `-m/--model` note in remote mode.
- **Packaging/public API**: `RemoteTranscriber` is re-exported and in `__all__`;
  `import syncalong` still does not load whisper (existing subprocess test).
- Add `fastapi` + `httpx` to `dev`; the `server` extra (with torch) is **not**
  required in CI — everything is faked.

## Docs & release

- New docs page: "Remote transcription (server/client)" — install matrix,
  `syncalong-serve` usage, `--server` client usage, the privacy note, auth,
  and a systemd/`uvicorn` deployment snippet. Add to `mkdocs.yml` nav.
- API reference: `RemoteTranscriber` (mkdocstrings picks it up from `src/`).
- `CHANGELOG.md`: note the new feature **and** the breaking change (whisper
  moved to the `[whisper]` extra; bare `pip install syncalong` no longer
  includes local transcription).
- README: short "GPU server / thin client" section.

## Out of scope (YAGNI for v1)

- Async job queue / polling / progress streaming.
- Per-request model selection (server loads one configured model).
- TLS in-process (document a reverse proxy if exposing beyond the LAN).
- Multi-tenant auth / user management / rate limiting.
- Batch/library pre-processing orchestration on the jukebox.
- Client dependency on `requests`/`httpx` (stdlib `urllib` only).

## Module boundaries summary

| Unit | Does | Used by | Depends on |
|---|---|---|---|
| `remote.RemoteTranscriber` | audio bytes → words over HTTP | jukebox / `align()` | stdlib `urllib`, `WordTimestamp` |
| `server` | HTTP → Demucs?/Whisper → words JSON | GPU box | fastapi, uvicorn, `Transcriber` |
| `transcribe.Transcriber` | audio → words (local, optional Demucs) | server / `align()` | whisper (lazy), vocal_separator (lazy) |
| `WordTimestamp.to_dict/from_dict` | wire (de)serialization | server + client | none |
| `pipeline.align()` | orchestrate lyrics + transcriber → result | library consumers | any transcriber implementing `.transcribe()` |
