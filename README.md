# syncalong

A CLI tool that aligns plain-text lyrics to an audio file and outputs
timestamped lyrics in [LRC format](https://en.wikipedia.org/wiki/LRC_(file_format)).

It uses OpenAI's [Whisper](https://github.com/openai/whisper) to transcribe
the audio with word-level timestamps, then runs a dynamic-programming
sequence alignment to map those words back onto your lyrics.

## Prerequisites

- **Python 3.9+**
- **ffmpeg** — required by Whisper for audio decoding.
  Install via your package manager (`apt install ffmpeg`, `brew install ffmpeg`, etc.)

## Installation

```bash
# From the project directory:
pip install .

# Or in editable / development mode:
pip install -e .
```

### Optional: vocal separation

For better results on studio recordings (where background music may
confuse the speech model), install with the `vocal-separation` extra:

```bash
pip install .[vocal-separation]
```

This adds [Demucs](https://github.com/facebookresearch/demucs), which
isolates the vocal track before transcription.

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
| `-m, --model` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) | `base` |
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
   interpolation.

5. **Format** — The timed lines are printed in LRC format to stdout.

## Tips for best results

- **Use `--separate-vocals`** on any track with significant instrumentation.
- **Bump up the model size** (`-m small` or `-m medium`) if alignment is poor.
  The `large` model is the most accurate but is slow without a GPU.
- **Check your lyrics** — extra or missing words relative to the actual
  performance will reduce alignment quality. The closer the lyrics match
  what's actually sung, the better.
- **GPU acceleration** — if you have a CUDA-capable GPU and PyTorch is
  installed with CUDA support, Whisper will use it automatically.
