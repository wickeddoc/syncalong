# CLI guide

The `syncalong` console script aligns a lyrics file to an audio file and writes
LRC to standard output. Progress messages go to standard error, so redirecting
stdout gives you a clean `.lrc` file.

```bash
syncalong LYRICS_FILE AUDIO_FILE [options]
```

## Examples

Align a song and print LRC to the terminal:

```bash
syncalong song.txt song.mp3
```

Save the result to a file (progress still shows on stderr):

```bash
syncalong song.txt song.mp3 > song.lrc
```

Use a larger Whisper model for better accuracy:

```bash
syncalong song.txt song.wav -m medium
```

Isolate vocals first — recommended for studio tracks with heavy instrumentation
(requires the [`vocal-separation` extra](installation.md#optional-vocal-separation)):

```bash
syncalong song.txt song.flac --separate-vocals
```

Skip Whisper's language auto-detection by naming the language:

```bash
syncalong lied.txt lied.mp3 -l de
```

## Options

| Flag | Description | Default |
| --- | --- | --- |
| `-m, --model` | Whisper model: `tiny`, `base`, `small`, `medium`, `large`, `turbo`, or variants like `small.en` (English-only) and `large-v3`. Larger is more accurate but slower. | `base` |
| `-l, --language` | Language code (e.g. `en`, `de`, `ja`). Auto-detected if omitted. | auto |
| `--separate-vocals` | Run Demucs to isolate vocals before transcription. | off |
| `--no-lyrics-prompt` | Don't feed the lyrics to Whisper as a decoding prompt. | off |
| `--threshold` | Minimum fuzzy-match score (0–100) to accept a word alignment. | `55` |
| `--server` | Transcribe on a remote syncalong server instead of locally (**new in 2.0**); falls back to `$SYNCALONG_SERVER`. | local |
| `--token` | Bearer token for the remote server; falls back to `$SYNCALONG_TOKEN`. | none |

Run `syncalong --help` for the authoritative list.

!!! info "Remote mode is new in 2.0"
    `--server` and `--token` (and the `syncalong-serve` server) are available
    from **syncalong 2.0** onward; 1.x is local-only. See
    [Remote transcription](remote.md).

## Lyrics file format

One lyric line per text line. Blank lines are preserved as instrumental breaks.
Section headers wrapped in brackets or parentheses (like `[Chorus]` or
`(Bridge)`) are detected and kept as structural markers — they aren't matched
against the audio.

```text
[Verse 1]
I walk a lonely road
The only one that I have ever known

[Chorus]
My shadow's the only one that walks beside me
```

!!! tip "Match the performance, not the booklet"
    The closer the lyrics are to what is actually sung — right number of
    repeats, ad-libs included or excluded — the better the alignment. Extra or
    missing words reduce quality.

## Output format

Standard LRC with `[mm:ss.xx]` timestamps, written to stdout:

```text
[00:12.34] I walk a lonely road
[00:15.67] The only one that I have ever known
```

Every sung line receives a timestamp as long as at least one line matched:
lines between two matches are linearly interpolated, and lines before the first
or after the last match are extrapolated from the transcript's start and end.
See [How it works](how-it-works.md) for the details.

## Tuning for better results

- **Use `--separate-vocals`** on any track with significant instrumentation.
- **Bump the model size** (`-m small` or `-m medium`) if alignment is poor. For
  English songs the `.en` variants (`-m small.en`) are often more accurate at
  the same speed. `large` is the most accurate but slow without a GPU; `turbo`
  is a good speed/quality compromise.
- **Lower/raise `--threshold`** if too few / too many words match. The default
  of 55 tolerates mishearings without accepting garbage.
- **GPU acceleration** — if PyTorch is installed with CUDA support, Whisper uses
  the GPU automatically. See [Benchmarks](benchmarks.md) for real CPU-vs-GPU
  timings per model.

## Exit behavior

`syncalong` exits non-zero and prints an error to stderr when the lyrics or
audio file is missing, the lyrics file is empty, Whisper returns no words, or
`--separate-vocals` is requested without Demucs installed.
