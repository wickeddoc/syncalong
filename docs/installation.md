# Installation

## Requirements

- **Python 3.9 or newer**
- **[ffmpeg](https://ffmpeg.org/)** — must be installed wherever Whisper
  actually runs: the local machine, if using the `whisper` extra, or the GPU
  `server` box. It is a system package, not a Python one:

    ```bash
    # Debian/Ubuntu
    sudo apt install ffmpeg
    # macOS (Homebrew)
    brew install ffmpeg
    # Windows (winget)
    winget install ffmpeg
    ```

    A thin client that only talks to a remote server (`--server` /
    `RemoteTranscriber`) needs neither ffmpeg nor Whisper installed locally.

The first time you run a given Whisper model, its weights are downloaded
automatically and cached (typically under `~/.cache/whisper`).

## Install from PyPI

```bash
pip install syncalong
```

This installs the library and the `syncalong` CLI with a **thin** dependency
set ([rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) only) — enough to
parse lyrics, align a transcript, write LRC, and talk to a remote server.

!!! info "New in syncalong 2.0"
    The thin, torch-free install — and the `whisper` extra needed for local
    transcription below — arrived in **2.0**. In syncalong 1.x,
    `pip install syncalong` bundles Whisper and transcribes locally out of the
    box. Remote transcription (`--server`, `syncalong-serve`) is 2.0+ as well.

## Local transcription (Whisper)

To transcribe **on this machine**, add the `whisper` extra:

```bash
pip install "syncalong[whisper]"
```

!!! info "Whisper pulls in PyTorch"
    `openai-whisper` depends on PyTorch, a large download. With a CUDA-capable
    GPU, install a matching PyTorch build first (see the
    [PyTorch install guide](https://pytorch.org/get-started/locally/)) and
    Whisper will use the GPU automatically.

!!! tip "No GPU on this machine?"
    Run transcription on a separate GPU box instead — see
    [Remote transcription](remote.md). The client stays torch-free.

## Optional: vocal separation

For studio recordings where background music can confuse the speech model,
install the `vocal-separation` extra. It adds
[Demucs](https://github.com/facebookresearch/demucs), which isolates the vocal
track before transcription:

```bash
pip install "syncalong[vocal-separation]"
```

Then pass `--separate-vocals` (CLI) or `separate_vocals=True` (library).

!!! info "Demucs needs ffmpeg and torchcodec"
    Demucs decodes and writes audio through
    [torchcodec](https://github.com/pytorch/torchcodec) (pulled in by this extra)
    and needs [ffmpeg](https://ffmpeg.org/) on `PATH` — the same ffmpeg Whisper
    already requires. Separation uses the GPU automatically when one is available;
    see [Benchmarks](benchmarks.md#vocal-separation-demucs) for CPU-vs-GPU timings.

## Install for development

Clone the repository and install in editable mode with the `dev` and `docs`
extras:

```bash
git clone https://github.com/wickeddoc/syncalong.git
cd syncalong
pip install -e ".[dev,docs]"
```

The `dev` extra provides pytest, ruff, black, pyright, build, and twine; `docs`
provides MkDocs Material and mkdocstrings. See [Contributing](contributing.md)
for the full workflow.

## Verify

```bash
syncalong --help
python -c "import syncalong; print(syncalong.__version__)"
```
