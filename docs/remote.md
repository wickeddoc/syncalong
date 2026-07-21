# Remote transcription (server/client)

syncalong's only GPU-heavy stage is transcription (Whisper, plus optional
Demucs). You can run that stage on a powerful machine and keep the music,
lyrics, and the cheap alignment on a low-powered client — e.g. a jukebox with
all the audio but no GPU.

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

## Auth & exposure

Set a shared token to require `Authorization: Bearer <token>`:

```bash
syncalong-serve --token "$(openssl rand -hex 16)"
```

There is no TLS in-process. On a trusted LAN, bind to a private interface. If
you expose the server beyond the LAN, put it behind a TLS-terminating reverse
proxy (Caddy, nginx) and keep the token set.
