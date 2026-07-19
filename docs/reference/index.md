# API reference

`syncalong` exposes a small, curated public API. Everything below is
re-exported from the package root, so you can write `syncalong.align(...)`
rather than reaching into submodules.

The package ships a [`py.typed`](https://peps.python.org/pep-0561/) marker, so
type checkers (mypy, pyright) resolve these signatures in your own code.

## Where to start

| If you want to… | Use |
| --- | --- |
| Align one song and get a structured result | [`align()`](pipeline.md#syncalong.pipeline.align) → [`AlignmentResult`](pipeline.md#syncalong.pipeline.AlignmentResult) |
| Get just the LRC string | [`align_to_lrc()`](pipeline.md#syncalong.pipeline.align_to_lrc) |
| Load the Whisper model once and reuse it | [`Transcriber`](transcribe.md#syncalong.transcribe.Transcriber) |
| Parse a lyrics file/text yourself | [`parse_lyrics()`](lyrics.md#syncalong.lyrics.parse_lyrics) / [`parse_lyrics_text()`](lyrics.md#syncalong.lyrics.parse_lyrics_text) |
| Run the aligner on your own transcript | [`align_lyrics_to_transcript()`](align.md#syncalong.align.align_lyrics_to_transcript) |
| Render timed lines to LRC | [`format_lrc()`](formatter.md#syncalong.formatter.format_lrc) |

## Public names

The complete public surface (`syncalong.__all__`):

- **High-level facade:** `align`, `align_to_lrc`, `AlignmentResult`
- **Transcription:** `Transcriber`, `transcribe_audio`, `WordTimestamp`
- **Lyrics:** `parse_lyrics`, `parse_lyrics_text`, `lyrics_prompt`, `LyricLine`
- **Low-level building blocks:** `align_lyrics_to_transcript`, `format_lrc`, `separate`

Use the navigation to browse each module in detail.
