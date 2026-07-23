# Benchmarks

Transcription (Whisper) is syncalong's only heavy stage — lyrics parsing,
alignment, and LRC formatting take milliseconds. So these numbers measure
**Whisper speed across model sizes on CPU vs GPU**, driven through the real
syncalong pipeline.

The short version: on a GPU every model transcribes a 4-minute song in
single-digit seconds; on CPU it ranges from a few seconds to well over a minute,
and the gap widens sharply with model size.

!!! note "Test setup"
    - **Audio:** one ~4-minute (247 s) studio rock recording with a full band
      mix — a deliberately hard case (lots of instrumentation over the vocal).
      Real audio; not distributed with the project.
    - **Machine:** AMD Ryzen 9 9950X3D (16C/32T), NVIDIA RTX 5090 (32 GB),
      60 GB RAM.
    - **Stack:** PyTorch 2.11 (CUDA 13), openai-whisper 20250625, Python 3.14.
    - **Method:** the real `Transcriber` + aligner with the lyrics prompt on and
      `word_timestamps=True`. GPU = fastest of two timed passes
      (CUDA-synchronized); CPU = a single pass. Weights are pre-cached, so load
      times are disk reads, not downloads.
    - **Reproduce:** `python benchmarks/benchmark.py AUDIO LYRICS` — see
      [`benchmarks/`](https://github.com/wickeddoc/syncalong/tree/master/benchmarks).

## CPU vs GPU

Wall-clock time to transcribe the whole 247-second track, and the GPU speedup:

| Model | CPU time | GPU time | GPU speedup |
|---|--:|--:|--:|
| `tiny` | 3.2 s | 0.5 s | 6.4× |
| `base` | 7.2 s | 1.1 s | 6.5× |
| `small` | 30.1 s | 2.5 s | 12.0× |
| `medium` | 79.5 s | 4.8 s | 16.6× |
| `large` | 76.1 s | 3.0 s | 25.4× |
| `turbo` | 69.6 s | 1.9 s | 36.6× |

The bigger the model, the more a GPU pays off: `tiny` is ~6× faster on the GPU,
but `large`/`turbo` are 25–37× faster. On this (high-end) CPU even `large` stays
faster than real time — but on a typical laptop CPU the `medium`/`large` rows
would be several times slower, often slower than real time. That is exactly the
case [remote transcription](remote.md) (run Whisper on a GPU box, keep a thin
client) is built for.

## GPU, per model

Real numbers on the RTX 5090 — transcription time, speed as a multiple of the
audio's real-time duration, peak VRAM, and one-time load:

| Model | Transcribe | Speed | Peak VRAM | Load |
|---|--:|--:|--:|--:|
| `tiny` | 0.5 s | 465× realtime | 0.3 GB | 0.2 s |
| `base` | 1.1 s | 217× | 0.5 GB | 0.4 s |
| `small` | 2.5 s | 98× | 1.5 GB | 1.1 s |
| `medium` | 4.8 s | 51× | 4.6 GB | 3.0 s |
| `large` | 3.0 s | 82× | 9.4 GB | 5.8 s |
| `turbo` | 1.9 s | 128× | 4.9 GB | 2.5 s |

A few things worth noting:

- **Transcribe time tracks how much is decoded, not just model size.** `large`
  here is *faster* than `medium` because it decoded this audio in fewer steps —
  parameter count sets a rough floor, but the content actually decoded dominates.
  `turbo` (a distilled `large-v3`) is the sweet spot: near-`large` quality at a
  fraction of the time.
- **VRAM** ranges from 0.3 GB (`tiny`) to 9.4 GB (`large`), so even `large` fits
  on modest GPUs; the 32 GB card is nowhere near saturated.
- **Load time** (weights → GPU) is a one-time cost per process. A long-running
  job should reuse one
  [`Transcriber`](library.md#batch-process-an-album-reuse-the-model) so it's paid
  once, not per song.

## Accuracy note

Across every model and device above, alignment matched the same **33 of 40**
lyric lines — so for this track the smaller/faster models placed lines just as
well as `large`. Forced alignment is forgiving: the lyrics are known, so the
aligner only needs roughly-right word timings, which even `tiny` provides.
Harder audio (denser mixes, heavier accents, ad-libs) is where larger models pull
ahead — see [tuning for better results](cli.md#tuning-for-better-results).

!!! warning "Numbers are hardware- and track-specific"
    These are one machine and one song — treat them as *ratios and orders of
    magnitude*, not guarantees. A different CPU/GPU, a longer track, or another
    genre will shift the absolute times. Re-run `benchmarks/benchmark.py` on your
    own hardware and audio for numbers that match your setup.
