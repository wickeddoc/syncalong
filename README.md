# syncalong

[![PyPI version](https://img.shields.io/pypi/v/syncalong.svg)](https://pypi.org/project/syncalong/)
[![Python versions](https://img.shields.io/pypi/pyversions/syncalong.svg)](https://pypi.org/project/syncalong/)
[![Documentation](https://readthedocs.org/projects/syncalong/badge/?version=latest)](https://syncalong.readthedocs.io/en/latest/)
[![CI](https://github.com/wickeddoc/syncalong/actions/workflows/ci.yml/badge.svg)](https://github.com/wickeddoc/syncalong/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library and CLI tool that aligns plain-text lyrics to an audio file
and outputs timestamped lyrics in [LRC format](https://en.wikipedia.org/wiki/LRC_(file_format)).

It uses OpenAI's [Whisper](https://github.com/openai/whisper) to transcribe
the audio with word-level timestamps, then runs a dynamic-programming
sequence alignment to map those words back onto your lyrics.

📖 **Full documentation:** <https://syncalong.readthedocs.io/>

## Prerequisites

- **Python 3.9+**
- **ffmpeg** — required by Whisper for audio decoding.
  Install via your package manager (`apt install ffmpeg`, `brew install ffmpeg`, etc.)

## Installation

```bash
pip install syncalong
```

This installs a thin, torch-free client (lyrics parsing, alignment, LRC
output). Local transcription needs the `whisper` extra:
`pip install "syncalong[whisper]"`. To transcribe on a separate GPU server
instead, see [Remote transcription](#remote-transcription-no-local-gpu) below.

Or from a checkout, in editable / development mode:

```bash
pip install -e ".[dev,docs]"
```

### Optional: vocal separation

For better results on studio recordings (where background music may
confuse the speech model), install with the `vocal-separation` extra:

```bash
pip install "syncalong[vocal-separation]"
```

This adds [Demucs](https://github.com/facebookresearch/demucs), which
isolates the vocal track before transcription.

### Remote transcription (no local GPU)

Run Whisper on a GPU machine and keep the audio + lyrics on a thin client:

```bash
# on the GPU box
pip install "syncalong[server]"
syncalong-serve --model medium --host 0.0.0.0 --device cuda

# on the client (torch-free `pip install syncalong`)
syncalong lyrics.txt song.mp3 --server http://gpu-box:8000 > song.lrc
```

See the [remote transcription guide](https://syncalong.readthedocs.io/en/latest/remote/).

## Usage

```bash
syncalong LYRICS_FILE AUDIO_FILE [options]
```

### Examples

Basic usage — outputs LRC to stdout:

```bash
syncalong song.txt song.mp3
```

Save to a file:

```bash
syncalong song.txt song.mp3 > song.lrc
```

Use a larger Whisper model for better accuracy:

```bash
syncalong song.txt song.wav -m medium
```

Isolate vocals first (recommended for studio tracks):

```bash
syncalong song.txt song.flac --separate-vocals
```

Specify language (skips auto-detection):

```bash
syncalong lied.txt lied.mp3 -l de
```

### Options

| Flag | Description | Default |
|---|---|---|
| `-m, --model` | Whisper model: `tiny`, `base`, `small`, `medium`, `large`, `turbo`, or variants like `small.en` (English-only) and `large-v3`. | `base` |
| `-l, --language` | Language code (e.g. `en`, `de`, `ja`). Auto-detected if omitted. | auto |
| `--separate-vocals` | Run Demucs to isolate vocals before transcription. | off |
| `--no-lyrics-prompt` | Don't feed the lyrics to Whisper as a decoding prompt. | off |
| `--threshold` | Minimum fuzzy-match score (0–100) to accept a word alignment. | 55 |

### Lyrics file format

One lyric line per text line. Blank lines are preserved as instrumental
breaks. Section headers in brackets (like `[Chorus]` or `(Bridge)`) are
detected and kept as structural markers.

```text
[Verse 1]
I walk a lonely road
The only one that I have ever known

[Chorus]
My shadow's the only one that walks beside me
```

### Output format

Standard LRC with `[mm:ss.xx]` timestamps:

```
[00:12.34] I walk a lonely road
[00:15.67] The only one that I have ever known
```

## Use as a library

`syncalong` is also an importable package, so you can embed alignment in your
own code or batch-process a whole album while loading the Whisper model only
once. `import syncalong` is cheap — the heavy Whisper import is deferred until a
model is actually loaded.

### Align a single song

```python
import syncalong
from pathlib import Path

tx = syncalong.Transcriber(model_name="base")   # load the model once

res = syncalong.align(Path("lyrics.txt"), "song.mp3", transcriber=tx)

print(res.lrc)            # the LRC document, as a string
res.timed_lines           # [(LyricLine, 12.3), (LyricLine, None), ...]
res.matched, res.total    # e.g. (28, 30) — lines that got a timestamp / all lines
```

`align()` accepts `lyrics` as a `pathlib.Path` (read from a file), a `str` of
lyrics text, or a pre-parsed `list[LyricLine]`; `audio` is a path (`str` or
`pathlib.Path`). Other keyword arguments: `transcriber` (pass a pre-loaded
`Transcriber` to reuse across songs), plus the CLI-mirroring `model_name`,
`language`, `use_lyrics_prompt`, `threshold`, and `separate_vocals`. If you omit
`transcriber`, `align()` loads a model itself from `model_name`.

### Batch-process an album (reuse the model)

Loading a Whisper model takes seconds, so a long-running job should create one
`Transcriber` and reuse it. `align_to_lrc()` is a convenience wrapper returning
just the LRC string (equivalent to `align(...).lrc`):

```python
import syncalong
from pathlib import Path

tx = syncalong.Transcriber(model_name="small")   # loaded once for the whole run

for audio in Path("album/").glob("*.flac"):
    lyrics = audio.with_suffix(".txt")
    lrc = syncalong.align_to_lrc(lyrics, audio, transcriber=tx)
    audio.with_suffix(".lrc").write_text(lrc, encoding="utf-8")
```

Lyrics already in memory? Pass them as a plain string instead of a path:

```python
text = "I walk a lonely road\nThe only one that I have ever known\n"
res = syncalong.align(text, "song.mp3", transcriber=tx)
```

> **Note:** `separate_vocals=True` requires the optional `vocal-separation`
> extra (`pip install syncalong[vocal-separation]`). Passing it without demucs
> installed raises `ModuleNotFoundError`.

## How it works

1. **Transcribe** — Whisper runs on the audio file with `word_timestamps=True`,
   producing a list of `(word, start_time, end_time)` tuples. The beginning of
   your lyrics is passed to the decoder as an `initial_prompt` to bias it
   toward the correct words (disable with `--no-lyrics-prompt`), and
   conditioning on previously decoded text is turned off to avoid the
   repetition loops Whisper is prone to on music.

2. **Normalize** — Both the lyrics and the transcript are lowercased and
   stripped of punctuation so that matching is case/punctuation-insensitive.

3. **Align** — A Needleman–Wunsch-style dynamic programming algorithm finds
   the optimal monotonic mapping between lyric words and transcript words.
   Word similarity is scored with rapidfuzz (Levenshtein ratio), so minor
   mishearings by Whisper are tolerated.

4. **Map to lines** — Each lyric line receives the timestamp of its earliest
   matched word. Small gaps between matched lines are filled with linear
   interpolation, and unmatched lines before the first match or after the
   last one are extrapolated from the transcript's start and end times, so
   every sung line gets a tag.

5. **Format** — The timed lines are printed in LRC format to stdout.

## Tips for best results

- **Use `--separate-vocals`** on any track with significant instrumentation.
- **Bump up the model size** (`-m small` or `-m medium`) if alignment is poor.
  For English songs the `.en` variants (`-m small.en`) are often more
  accurate at the same speed. The `large` model is the most accurate but is
  slow without a GPU; `turbo` is a good speed/quality compromise.
- **Check your lyrics** — extra or missing words relative to the actual
  performance will reduce alignment quality. The closer the lyrics match
  what's actually sung, the better.
- **GPU acceleration** — if you have a CUDA-capable GPU and PyTorch is
  installed with CUDA support, Whisper will use it automatically.

## Documentation

Full guides and the auto-generated API reference live at
**<https://syncalong.readthedocs.io/>**:

- [Installation](https://syncalong.readthedocs.io/en/latest/installation/)
- [CLI guide](https://syncalong.readthedocs.io/en/latest/cli/)
- [Library guide](https://syncalong.readthedocs.io/en/latest/library/)
- [How it works](https://syncalong.readthedocs.io/en/latest/how-it-works/)
- [API reference](https://syncalong.readthedocs.io/en/latest/reference/)

## Contributing

Contributions are welcome. Install the dev tooling with
`pip install -e ".[dev,docs]"` and make sure the quality gate passes before
opening a PR:

```bash
pytest
ruff check .
black --check .
pyright src
```

See the [contributing guide](https://syncalong.readthedocs.io/en/latest/contributing/)
for the full workflow, including how releases are cut from git tags.

## License

Released under the [MIT License](LICENSE).
