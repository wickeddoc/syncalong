# Installation

## Requirements

- **Python 3.9 or newer**
- **[ffmpeg](https://ffmpeg.org/)** — required by Whisper to decode audio. It is
  a system package, not a Python one:

    ```bash
    # Debian/Ubuntu
    sudo apt install ffmpeg
    # macOS (Homebrew)
    brew install ffmpeg
    # Windows (winget)
    winget install ffmpeg
    ```

The first time you run a given Whisper model, its weights are downloaded
automatically and cached (typically under `~/.cache/whisper`).

## Install from PyPI

```bash
pip install syncalong
```

This installs the library, the `syncalong` command-line tool, and the core
dependencies ([openai-whisper](https://github.com/openai/whisper) and
[rapidfuzz](https://github.com/rapidfuzz/RapidFuzz)).

!!! info "Whisper pulls in PyTorch"
    `openai-whisper` depends on PyTorch, which is a large download. If you have a
    CUDA-capable GPU, install a matching PyTorch build first (see the
    [PyTorch install guide](https://pytorch.org/get-started/locally/)) and
    Whisper will use the GPU automatically.

## Optional: vocal separation

For studio recordings where background music can confuse the speech model,
install the `vocal-separation` extra. It adds
[Demucs](https://github.com/facebookresearch/demucs), which isolates the vocal
track before transcription:

```bash
pip install "syncalong[vocal-separation]"
```

Then pass `--separate-vocals` (CLI) or `separate_vocals=True` (library).

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
