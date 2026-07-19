# syncalong

**Forced alignment of plain-text lyrics to an audio file, exported as
timestamped [LRC](https://en.wikipedia.org/wiki/LRC_(file_format)).**

`syncalong` transcribes your audio with OpenAI's
[Whisper](https://github.com/openai/whisper) (word-level timestamps), then runs
a dynamic-programming sequence aligner to map those words back onto the lyrics
you already have. The result is a `.lrc` file where every line carries the time
it is sung — ready for karaoke players, media apps, or a jukebox.

It works two ways from the same pipeline:

- a **command-line tool** (`syncalong` console script), and
- an **importable Python library** (`syncalong.align()`, `syncalong.Transcriber`, …).

<div class="grid cards" markdown>

- :material-console: **[CLI guide](cli.md)**
  Align a song from your terminal, one command.

- :material-language-python: **[Library guide](library.md)**
  Embed alignment in your own app; reuse the Whisper model across a whole album.

- :material-cog: **[How it works](how-it-works.md)**
  The transcribe → normalize → align → format pipeline, explained.

- :material-book-open-variant: **[API reference](reference/index.md)**
  Every public function and dataclass, generated from the docstrings.

</div>

## Quickstart

```bash
pip install syncalong
syncalong song.txt song.mp3 > song.lrc
```

Or from Python:

```python
import syncalong
from pathlib import Path

lrc = syncalong.align_to_lrc(Path("song.txt"), "song.mp3")
print(lrc)
# [00:12.34] I walk a lonely road
# [00:15.67] The only one that I have ever known
```

!!! note "Prerequisite: ffmpeg"
    Whisper decodes audio with [ffmpeg](https://ffmpeg.org/), so it must be
    installed and on your `PATH`. See [Installation](installation.md).

## Why forced alignment?

You already know the lyrics — you don't need open-ended speech recognition, you
need to know *when* each line is sung. syncalong treats the known lyrics as
ground truth and only solves the timing problem, which is far more robust than
transcribing blind and fuzzy-searching for words. Repeated choruses don't
collapse onto each other because the alignment is strictly order-preserving, and
minor mishearings ("runnin'" vs "running") are tolerated by fuzzy word scoring.
