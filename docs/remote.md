# Remote transcription (server/client)

syncalong's only GPU-heavy stage is transcription (Whisper, plus optional
Demucs). You can run that stage on a powerful machine and keep the music,
lyrics, and the cheap alignment on a low-powered client — e.g. a jukebox with
all the audio but no GPU. A GPU transcribes 6–37× faster than CPU depending on
the model — see [Benchmarks](benchmarks.md).

!!! info "New in syncalong 2.0"
    Remote transcription — the `syncalong-serve` server, `RemoteTranscriber`,
    and the `--server` CLI flag — is available **from version 2.0 onward**.
    syncalong 1.x transcribes locally only.

## Install matrix

| Machine | Install | Gets |
| --- | --- | --- |
| **Client** (jukebox) | `pip install syncalong` | parsing, alignment, LRC, and `RemoteTranscriber` — no PyTorch |
| **Client, local Whisper too** | `pip install "syncalong[whisper]"` | the above + local `Transcriber` |
| **Server** (GPU box) | `pip install "syncalong[server]"` | Whisper + the FastAPI server |
| **Server, with vocal isolation** | `pip install "syncalong[server,vocal-separation]"` | + Demucs |

`ffmpeg` is required on the **server** (Whisper decodes audio there).

## Start the server (GPU box)

```bash
syncalong-serve --model medium --host 0.0.0.0 --port 8000 --device cuda
```

Options: `--model`, `--host`, `--port`, `--device` (e.g. `cuda`), `--token`
(require a bearer token; falls back to `$SYNCALONG_TOKEN`),
`--no-vocal-separation` (reject `separate_vocals` requests). The Whisper model
loads once at startup and is reused for every request.

You can also run it with any ASGI server:

```bash
SYNCALONG_MODEL=medium uvicorn syncalong.server:app --host 0.0.0.0 --port 8000
```

## Transcribe from the client (jukebox)

CLI:

```bash
syncalong lyrics.txt song.mp3 --server http://gpu-box:8000 > song.lrc
# with a token and server-side vocal isolation:
syncalong lyrics.txt song.mp3 --server http://gpu-box:8000 \
    --token "$SYNCALONG_TOKEN" --separate-vocals > song.lrc
```

`--server` also reads `$SYNCALONG_SERVER`. In remote mode the **server**
chooses the Whisper model, so `-m/--model` is ignored.

Library:

```python
import syncalong

tx = syncalong.RemoteTranscriber("http://gpu-box:8000", token="…")
res = syncalong.align("lyrics.txt", "song.mp3", transcriber=tx)
print(res.lrc)
```

`RemoteTranscriber` implements the same interface as `Transcriber`, so it drops
into `align(transcriber=…)` unchanged — alignment still runs locally on the
client.

## What travels over the network

The audio file is uploaded to the server; the server returns only word
timestamps (`{word, raw, start, end}`). Your lyrics **file** and the whole
alignment stay on the client.

!!! note "Privacy: the lyrics prompt"
    By default Whisper is biased with a prompt built **from your lyrics**, and
    that prompt is sent to the server. To send **audio only** — zero lyric text
    over the wire — pass `--no-lyrics-prompt` (CLI) or `use_lyrics_prompt=False`
    (library), at some accuracy cost.

## HTTP API

`RemoteTranscriber` is just one client for a small, stable HTTP contract — you
can drive the server from `curl`, another language, or your own client. The
server ([`syncalong.server`](reference/server.md)) exposes two endpoints.

### `GET /health`

Liveness probe. Returns the loaded model name (`null` until the model is loaded
— it loads lazily on the first `/transcribe` unless preloaded by
`syncalong-serve`). Never requires auth.

```json
{ "status": "ok", "model": "medium" }
```

### `POST /transcribe`

`multipart/form-data` upload. Fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `audio` | file | **yes** | The audio file to transcribe. |
| `language` | text | no | BCP-47 language code; omit to auto-detect. |
| `initial_prompt` | text | no | Text to bias the decoder (syncalong sends the lyrics prompt here). |
| `separate_vocals` | text `true`/`false` | no | Isolate vocals with Demucs first (default `false`). |

Send `Authorization: Bearer <token>` when the server is started with a token.

On success (`200`) the body is the transcript — a list of word timestamps plus
the model that produced them:

```json
{
  "words": [
    { "word": "hello", "raw": "Hello", "start": 12.34, "end": 12.71 }
  ],
  "model": "medium"
}
```

`word` is the normalized token used for matching; `raw` is Whisper's original
text; `start`/`end` are seconds. The client rebuilds `WordTimestamp` records
from this and aligns locally.

Error responses carry a JSON `{"detail": "…"}` body:

| Status | When |
| --- | --- |
| `400` | `separate_vocals` requested but disabled (`--no-vocal-separation`) or Demucs isn't installed on the server. |
| `401` | A token is configured and the request's bearer token is missing or wrong. |

A one-shot `curl` equivalent of the client:

```bash
curl -sS http://gpu-box:8000/transcribe \
    -H "Authorization: Bearer $SYNCALONG_TOKEN" \
    -F "audio=@song.mp3" \
    -F "language=en" \
    -F "separate_vocals=false"
```

!!! tip "Embedding the server"
    `syncalong.server.create_app(...)` returns the FastAPI app so you can mount
    it in a larger ASGI application or configure the token, model, device, and
    vocal-separation policy in code. See the
    [`syncalong.server` reference](reference/server.md).

## Auth & exposure

Set a shared token to require `Authorization: Bearer <token>`:

```bash
syncalong-serve --token "$(openssl rand -hex 16)"
```

There is no TLS in-process. On a trusted LAN, bind to a private interface. If
you expose the server beyond the LAN, put it behind a TLS-terminating reverse
proxy (Caddy, nginx) and keep the token set.
