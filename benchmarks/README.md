# Benchmarks

`benchmark.py` measures syncalong's transcription speed by driving the **real**
pipeline (`syncalong.Transcriber` + the aligner) over one audio file for each
Whisper model and device, then printing a Markdown table. The published numbers
in [`docs/benchmarks.md`](../docs/benchmarks.md) are produced by this script.

## Requirements

- `pip install -e ".[whisper]"` (Whisper + PyTorch) and **ffmpeg** on `PATH`.
- A CUDA GPU (plus a CUDA-enabled PyTorch build) for the `cuda` device rows;
  CPU-only machines can still run `--devices cpu`.

## Usage

```bash
python benchmarks/benchmark.py AUDIO LYRICS [options]
```

For example, the full CPU-vs-GPU matrix used in the docs:

```bash
python benchmarks/benchmark.py \
    "test/song.mp3" "test/song.txt" \
    --models tiny base small medium large turbo \
    --devices cpu cuda \
    --markdown results.md
```

Options:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--models` | Whisper models to sweep. | `tiny base small medium large turbo` |
| `--devices` | Devices to sweep (`cpu`, `cuda`). | `cpu cuda` |
| `--repeats` | Timed transcription passes per config; the fastest is kept. | `1` |
| `--no-prompt` | Don't bias Whisper with the lyrics prompt. | off |
| `--no-prefetch` | Skip the untimed weight-download pass. | off |
| `--markdown` | Also write the table to this file. | — |

### Reported metrics

- **Load (s)** — time to construct the `Transcriber` (load weights to the device).
- **Transcribe (s)** — wall time of `.transcribe()` (CUDA-synchronized).
- **Speed** — audio duration ÷ transcribe time (multiples of real time; higher
  is faster).
- **Peak VRAM** — `torch.cuda.max_memory_allocated()` for `cuda` rows.
- **Matched** — lyric lines that received a timestamp, over the total, as a
  quick quality signal for the speed/accuracy trade-off.

## Methodology notes

- To keep CPU and GPU numbers comparable, **run one device at a time** — a
  concurrent CPU sweep and GPU sweep contend for the CPU and distort the CPU
  timings.
- Weights are prefetched (untimed) before the matrix so `Load (s)` reflects a
  disk read, not a download. Re-run once to warm the OS file cache if you want
  the steadiest load times.
- Numbers are hardware-specific — the script records CPU/GPU/torch/Whisper
  versions as HTML comments at the top of the table so results stay traceable.

## Audio is never committed

Pass a **local** file (e.g. one under the git-ignored `test/` directory). Only
the numeric results are published; no audio or lyrics leave your machine.
