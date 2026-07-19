# How it works

syncalong is a linear pipeline. The lyrics file and the audio file enter from
opposite ends and meet at the aligner:

```
audio file ─► transcribe ─► align ─► format ─► LRC
                              ▲
lyrics file ─► parse ─────────┘
```

## 1. Transcribe

[Whisper](https://github.com/openai/whisper) runs on the audio with
`word_timestamps=True`, producing a list of
[`WordTimestamp`](reference/transcribe.md#syncalong.transcribe.WordTimestamp)
records — each a word plus its start/end time in seconds.

Two decisions make this robust on music:

- **Lyrics as a decoding prompt.** The beginning of your lyrics is passed to
  Whisper as an `initial_prompt`, biasing the decoder toward the correct words.
  (Disable with `--no-lyrics-prompt` / `use_lyrics_prompt=False`.)
- **No conditioning on previous text.** `condition_on_previous_text=False`
  avoids the repetition/hallucination loops Whisper is prone to on songs with
  repeated choruses.

## 2. Normalize

Both the lyrics and the transcript pass through the **same**
[`normalize()`](reference/textnorm.md#syncalong.textnorm.normalize) function, so
the two sides of the aligner compare like with like. Normalization lowercases,
strips accents, and collapses whitespace. Apostrophes are deleted so
contractions stay a single token (`don't` → `dont`), while other punctuation
becomes a word boundary so hyphenated compounds split the way Whisper
transcribes them (`déjà-vu` → `deja vu`).

Sharing one normalizer is a deliberate design rule: if the lyric side and the
transcript side normalized differently, fuzzy matching would silently degrade.

## 3. Align

A Needleman–Wunsch-style
[dynamic-programming aligner](reference/align.md#syncalong.align.align_lyrics_to_transcript)
finds the optimal **monotonic** mapping between the flat list of lyric words and
the flat list of transcript words. Word similarity is scored with
[rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) (a Levenshtein ratio, 0–100,
cached because song vocabularies repeat heavily); pairs scoring below the
`threshold` (default 55) are treated as non-matches.

Two properties matter:

- **Order preservation.** Lyric word 5 can't match a transcript word that comes
  before word 4's match. This is what stops repeated choruses from collapsing
  onto the same audio segment.
- **Fuzzy tolerance.** A mishearing like `runnin'` vs `running` still scores
  well above threshold, so minor Whisper errors don't break the line.

The complexity is `O(N·M)` in the number of lyric and transcript words — a few
milliseconds for a typical song.

## 4. Map to lines and fill gaps

Each lyric line takes the timestamp of its **earliest matched word**. Then two
passes ensure every sung line gets a time, because untagged lines are dropped by
many LRC players:

- **Interpolation.** An unmatched line sitting between two matched lines gets a
  linearly interpolated timestamp.
- **Extrapolation.** Unmatched lines before the first match or after the last
  one are extrapolated using the transcript's start and end times as virtual
  anchors.

As long as at least one line matched, every non-blank line comes out timed.

## 5. Format

Finally the timed lines are rendered as an LRC document by
[`format_lrc()`](reference/formatter.md#syncalong.formatter.format_lrc), with
`[mm:ss.xx]` tags (rounded to centiseconds, carrying correctly across the
minute boundary). Blank lines are preserved to keep the song's visual
structure.

## Why forced alignment instead of recognition?

Because you already have the lyrics. Open-ended speech-to-text on music is
error-prone; treating the known lyrics as ground truth and solving only the
*timing* problem is far more reliable. syncalong never has to guess *what* is
sung — only *when*.
