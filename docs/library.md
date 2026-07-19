# Library guide

`syncalong` is a library first: the CLI is a thin layer over the same public
API you can import. `import syncalong` is cheap — the heavy Whisper import is
deferred until a model is actually loaded — so importing the package in a
larger application (a jukebox, a tagging tool, a batch job) costs almost
nothing until you transcribe.

```python
import syncalong
```

The full public surface is documented in the [API reference](reference/index.md).
This page shows the common patterns.

## Align a single song

[`align()`](reference/pipeline.md#syncalong.pipeline.align) runs the whole
pipeline and returns a structured
[`AlignmentResult`](reference/pipeline.md#syncalong.pipeline.AlignmentResult):

```python
import syncalong
from pathlib import Path

tx = syncalong.Transcriber(model_name="base")   # load the model once

res = syncalong.align(Path("lyrics.txt"), "song.mp3", transcriber=tx)

print(res.lrc)            # the LRC document, as a string
res.timed_lines           # [(LyricLine, 12.3), (LyricLine, None), ...]
res.matched, res.total    # e.g. (28, 30) — lines that got a timestamp / all lines
```

### The `lyrics` argument is type-driven

`align()` decides what to do with `lyrics` based on its type:

| You pass… | syncalong… |
| --- | --- |
| `pathlib.Path` | reads and parses that lyrics file |
| `str` | parses it as raw lyrics **text** (not a path) |
| `list[LyricLine]` | uses your already-parsed lines as-is |

!!! warning "`str` means lyrics text, not a filename"
    Passing `"lyrics.txt"` as a plain string treats the *characters*
    `lyrics.txt` as the song's lyrics. To read a file, wrap it in `Path`:
    `align(Path("lyrics.txt"), ...)`.

`audio` is always a path — a `str` or `Path` pointing at the audio file.

### Keyword options

`align()` mirrors the CLI:

| Argument | Meaning | Default |
| --- | --- | --- |
| `transcriber` | A pre-loaded [`Transcriber`](reference/transcribe.md#syncalong.transcribe.Transcriber) to reuse. If `None`, one is built from `model_name`. | `None` |
| `model_name` | Whisper model to load when `transcriber` is `None`. | `"base"` |
| `language` | BCP-47 language code, or `None` to auto-detect. | `None` |
| `use_lyrics_prompt` | Bias Whisper with the lyrics text. | `True` |
| `threshold` | Minimum fuzzy score (0–100) to accept a word match. | `55.0` |
| `separate_vocals` | Isolate vocals with Demucs first (needs the extra). | `False` |

## Just the LRC string

When you only want the LRC text,
[`align_to_lrc()`](reference/pipeline.md#syncalong.pipeline.align_to_lrc) is a
convenience wrapper — equivalent to `align(...).lrc` and it accepts the same
keyword arguments:

```python
lrc = syncalong.align_to_lrc(Path("lyrics.txt"), "song.mp3", transcriber=tx)
```

## Batch-process an album (reuse the model)

Loading a Whisper model takes seconds, so a long-running job should create one
[`Transcriber`](reference/transcribe.md#syncalong.transcribe.Transcriber) and
reuse it across every song:

```python
import syncalong
from pathlib import Path

tx = syncalong.Transcriber(model_name="small")   # loaded once for the whole run

for audio in sorted(Path("album/").glob("*.flac")):
    lyrics = audio.with_suffix(".txt")
    lrc = syncalong.align_to_lrc(lyrics, audio, transcriber=tx)
    audio.with_suffix(".lrc").write_text(lrc, encoding="utf-8")
    print(f"wrote {audio.stem}.lrc")
```

## Lyrics already in memory

If your lyrics come from a database or tag rather than a file, pass them as a
plain string:

```python
text = "I walk a lonely road\nThe only one that I have ever known\n"
res = syncalong.align(text, "song.mp3", transcriber=tx)
```

## Working with the result

[`AlignmentResult.timed_lines`](reference/pipeline.md#syncalong.pipeline.AlignmentResult)
is a list of `(LyricLine, timestamp_or_None)` pairs, in order. Each
[`LyricLine`](reference/lyrics.md#syncalong.lyrics.LyricLine) keeps its original
`raw` text, its normalized `words`, and an `is_blank` flag for
instrumental/section lines. This lets you build your own UI instead of parsing
LRC back out:

```python
res = syncalong.align(Path("lyrics.txt"), "song.mp3", transcriber=tx)

for line, ts in res.timed_lines:
    if line.is_blank:
        continue
    when = f"{ts:6.2f}s" if ts is not None else "  --  "
    print(f"{when}  {line.raw}")

print(f"{res.matched}/{res.total} lines aligned")
```

## Assembling the pipeline yourself

For full control you can call the building blocks directly instead of
`align()`. This is what the facade does internally:

```python
from pathlib import Path
from syncalong import (
    Transcriber,
    parse_lyrics,
    lyrics_prompt,
    align_lyrics_to_transcript,
    format_lrc,
)

lines = parse_lyrics(Path("lyrics.txt"))
tx = Transcriber(model_name="base")

transcript = tx.transcribe(
    Path("song.mp3"),
    initial_prompt=lyrics_prompt(lines),   # bias Whisper toward the known words
)

timed = align_lyrics_to_transcript(lines, transcript, threshold=55.0)
lrc = format_lrc(timed)
```

You could, for example, swap in your own transcript (any
`list[WordTimestamp]`) and reuse only the aligner and formatter.

## Injecting a Whisper model

`Transcriber` accepts a preloaded model via the keyword-only `model` argument.
Pass a model you already loaded to share it, or a fake object in tests — no
Whisper download required:

```python
from syncalong import Transcriber

tx = Transcriber(model=my_preloaded_whisper_model)
```

When `model` is given, `model_name` is ignored and Whisper is never loaded.

## Errors to expect

| Situation | Raised |
| --- | --- |
| `lyrics` is not a `Path`, `str`, or `list` | `TypeError` |
| `separate_vocals=True` but Demucs isn't installed | `ModuleNotFoundError` |
| Audio file can't be decoded | error from Whisper/ffmpeg |

Unlike the CLI (which prints to stderr and calls `sys.exit`), the library raises
exceptions so your application stays in control of error handling.
