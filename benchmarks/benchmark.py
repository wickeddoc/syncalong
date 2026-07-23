r"""Benchmark syncalong transcription speed across Whisper models and devices.

Runs the real syncalong pipeline (:class:`syncalong.Transcriber` plus
:func:`syncalong.align_lyrics_to_transcript`) on one audio file for every
requested model and device, reporting model-load time, transcription wall time,
speed relative to the audio's real-time duration, peak GPU memory (on CUDA), and
the resulting lyric-line match rate.

Audio is never committed to the repository — pass a local file, e.g. one under
the git-ignored ``test/`` directory. Only the numeric results are published.

Example:
    python benchmarks/benchmark.py \\
        "test/song.mp3" "test/song.txt" \\
        --models tiny base small medium large turbo \\
        --devices cpu cuda \\
        --markdown benchmarks/results.md
"""

from __future__ import annotations

import argparse
import gc
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import whisper

from syncalong import (
    Transcriber,
    align_lyrics_to_transcript,
    lyrics_prompt,
    parse_lyrics,
)

DEFAULT_MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]


@dataclass
class Result:
    """One (model, device) benchmark measurement.

    Attributes:
        model: Whisper model name.
        device: Torch device the model ran on.
        load_s: Seconds to load the model.
        transcribe_s: Transcription wall time (best of the repeats), in seconds.
        speed: Audio duration divided by ``transcribe_s`` (multiples of real time).
        peak_vram_gb: Peak CUDA memory in GB, or ``None`` on CPU.
        matched: Lyric lines that received a timestamp.
        total: Total lyric lines.
        error: Error message if the run failed, else ``None``.
    """

    model: str
    device: str
    load_s: float = 0.0
    transcribe_s: float = 0.0
    speed: float = 0.0
    peak_vram_gb: float | None = None
    matched: int = 0
    total: int = 0
    error: str | None = None


def cpu_name() -> str:
    """Return the CPU model name.

    Returns:
        The ``model name`` from ``/proc/cpuinfo`` when available, else a
        platform fallback.
    """
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def hardware_lines() -> list[str]:
    """Return human-readable hardware/software identification lines.

    Returns:
        Description strings for the report header.
    """
    lines = [
        f"CPU: {cpu_name()}",
        f"Python: {platform.python_version()}",
        f"torch: {torch.__version__}",
        f"whisper: {getattr(whisper, '__version__', 'unknown')}",
    ]
    if torch.cuda.is_available():
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    return lines


def audio_duration_s(path: Path) -> float:
    """Return the audio duration in seconds, as Whisper decodes it.

    Args:
        path: Path to the audio file.

    Returns:
        Duration in seconds (decoded samples / 16 kHz).
    """
    samples = whisper.load_audio(str(path))
    return len(samples) / whisper.audio.SAMPLE_RATE


def run_config(
    model: str,
    device: str,
    audio: Path,
    lines: list,
    prompt: str | None,
    duration_s: float,
    repeats: int,
) -> Result:
    """Benchmark a single model/device combination.

    Args:
        model: Whisper model name.
        device: ``"cpu"`` or ``"cuda"``.
        audio: Audio file path.
        lines: Parsed lyric lines, used for the match rate.
        prompt: Whisper initial prompt, or ``None``.
        duration_s: Audio duration in seconds.
        repeats: How many transcription passes to time (the fastest is kept).

    Returns:
        The populated :class:`Result`.
    """
    res = Result(model=model, device=device)
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        transcriber = Transcriber(model, device=device)
        res.load_s = time.perf_counter() - t0

        times: list[float] = []
        words: list = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            words = transcriber.transcribe(audio, initial_prompt=prompt)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        res.transcribe_s = min(times)
        res.speed = duration_s / res.transcribe_s if res.transcribe_s else 0.0

        if device == "cuda":
            res.peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

        timed = align_lyrics_to_transcript(lines, words)
        res.matched = sum(1 for _, ts in timed if ts is not None)
        res.total = len(timed)

        del transcriber, words
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # record and keep going with the next config
        res.error = f"{type(exc).__name__}: {exc}"
    return res


def match_rate(
    transcriber: Transcriber,
    audio: Path,
    lines: list,
    prompt: str | None,
) -> tuple[int, int]:
    """Transcribe audio and return the (matched, total) lyric-line counts.

    Args:
        transcriber: A loaded transcriber.
        audio: Audio file to transcribe.
        lines: Parsed lyric lines.
        prompt: Whisper initial prompt, or ``None``.

    Returns:
        ``(matched, total)`` lyric lines.
    """
    words = transcriber.transcribe(audio, initial_prompt=prompt)
    timed = align_lyrics_to_transcript(lines, words)
    matched = sum(1 for _, ts in timed if ts is not None)
    return matched, len(timed)


def prefetch_demucs(model: str = "htdemucs") -> None:
    """Download the Demucs model weights (untimed), if not already cached.

    Args:
        model: Demucs pretrained model name.
    """
    try:
        from demucs.pretrained import get_model

        get_model(model)
    except Exception as exc:  # optional dependency or offline
        print(f"note: demucs prefetch skipped: {exc}", file=sys.stderr)


def run_separation(
    audio: Path, device: str
) -> tuple[float, Path | None, Path, str | None]:
    """Time one Demucs vocal-separation pass on a device.

    Args:
        audio: Mixed audio file.
        device: ``"cpu"`` or ``"cuda"`` (passed to demucs ``-d``).

    Returns:
        ``(seconds, vocals_path, outdir, error)``. ``vocals_path`` is ``None``
        and ``error`` is set when the run fails; the caller removes ``outdir``.
    """
    outdir = Path(tempfile.mkdtemp(prefix="syncalong_bench_demucs_"))
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-d",
        device,
        "-o",
        str(outdir),
        str(audio),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return elapsed, None, outdir, tail[-1] if tail else f"exit {proc.returncode}"
    vocals = next(iter(outdir.rglob("vocals.wav")), None)
    if vocals is None:
        return elapsed, None, outdir, "no vocals.wav produced"
    return elapsed, vocals, outdir, None


def render_separation(
    sep_rows: list[tuple[str, float | None, str | None]],
    match_rows: list[tuple[str, tuple[int, int], tuple[int, int]]],
    duration_s: float,
) -> str:
    """Render the vocal-separation benchmark as Markdown.

    Args:
        sep_rows: ``(device, seconds_or_None, error_or_None)`` per device.
        match_rows: ``(model, original_counts, vocals_counts)`` per model.
        duration_s: Audio duration in seconds.

    Returns:
        A Markdown string.
    """
    out = [f"<!-- {line} -->" for line in hardware_lines()]
    out.append(f"<!-- Audio duration: {duration_s:.1f} s -->")
    out += ["", "### Separation time", ""]
    out += ["| Device | Time | Speed | Speedup vs CPU |", "|---|--:|--:|--:|"]
    cpu_t = next((t for d, t, _ in sep_rows if d == "cpu" and t), None)
    for device, secs, err in sep_rows:
        if err or not secs:
            out.append(f"| {device} | — | — | {err or 'n/a'} |")
            continue
        if device == "cpu":
            speedup = "1×"
        elif cpu_t:
            speedup = f"{cpu_t / secs:.1f}×"
        else:
            speedup = "—"
        out.append(
            f"| {device} | {secs:.1f} s | {duration_s / secs:.1f}×RT | {speedup} |"
        )
    if match_rows:
        out += ["", "### Match rate — original mix vs isolated vocals", ""]
        out += ["| Model | Original | Vocals-only |", "|---|--:|--:|"]
        for model, (mo, to), (mv, tv) in match_rows:
            out.append(f"| `{model}` | {mo}/{to} | {mv}/{tv} |")
    return "\n".join(out) + "\n"


def run_separation_benchmark(
    audio: Path,
    lines: list,
    prompt: str | None,
    duration_s: float,
    devices: list[str],
    models: list[str],
    markdown: Path | None,
) -> None:
    """Benchmark Demucs separation speed and its effect on the match rate.

    Args:
        audio: Mixed audio file.
        lines: Parsed lyric lines.
        prompt: Whisper initial prompt, or ``None``.
        duration_s: Audio duration in seconds.
        devices: Devices to time separation on.
        models: Whisper models for the match-rate comparison.
        markdown: Optional path to also write the table to.
    """
    prefetch_demucs()
    sep_rows: list[tuple[str, float | None, str | None]] = []
    outdirs: list[Path] = []
    vocals: Path | None = None
    for device in devices:
        print(f"→ separating on {device} …", file=sys.stderr, flush=True)
        secs, path, outdir, err = run_separation(audio, device)
        outdirs.append(outdir)
        if err:
            print(f"  FAILED: {err}", file=sys.stderr)
        else:
            print(f"  {secs:.1f}s  ({duration_s / secs:.1f}×RT)", file=sys.stderr)
        sep_rows.append((device, None if err else secs, err))
        if path is not None and (vocals is None or device == "cuda"):
            vocals = path

    match_rows: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    if vocals is not None:
        tdevice = "cuda" if torch.cuda.is_available() else "cpu"
        for model in models:
            print(f"→ match rate: {model} on {tdevice} …", file=sys.stderr, flush=True)
            transcriber = Transcriber(model, device=tdevice)
            original = match_rate(transcriber, audio, lines, prompt)
            isolated = match_rate(transcriber, vocals, lines, prompt)
            del transcriber
            gc.collect()
            if tdevice == "cuda":
                torch.cuda.empty_cache()
            match_rows.append((model, original, isolated))
            print(
                f"  original {original[0]}/{original[1]}"
                f"  vocals {isolated[0]}/{isolated[1]}",
                file=sys.stderr,
            )

    table = render_separation(sep_rows, match_rows, duration_s)
    print(table)
    if markdown:
        markdown.write_text(table, encoding="utf-8")
        print(f"wrote {markdown}", file=sys.stderr)
    for outdir in outdirs:
        shutil.rmtree(outdir, ignore_errors=True)


def render(results: list[Result], duration_s: float) -> str:
    """Render results as a Markdown table with a hardware/duration header.

    Args:
        results: Collected measurements.
        duration_s: Audio duration in seconds.

    Returns:
        A Markdown string.
    """
    head = [f"<!-- {line} -->" for line in hardware_lines()]
    head.append(f"<!-- Audio duration: {duration_s:.1f} s -->")
    head.append("")
    rows = [
        "| Model | Device | Load (s) | Transcribe (s) | Speed | Peak VRAM | Matched |",
        "|---|---|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        if r.error:
            rows.append(f"| `{r.model}` | {r.device} | — | — | — | — | {r.error} |")
            continue
        vram = f"{r.peak_vram_gb:.1f} GB" if r.peak_vram_gb is not None else "—"
        rows.append(
            f"| `{r.model}` | {r.device} | {r.load_s:.1f} | {r.transcribe_s:.1f} "
            f"| {r.speed:.1f}×RT | {vram} | {r.matched}/{r.total} |"
        )
    return "\n".join(head + rows) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the benchmark matrix, and print a Markdown table.

    Args:
        argv: Argument vector; defaults to ``sys.argv`` when ``None``.
    """
    parser = argparse.ArgumentParser(description="Benchmark syncalong transcription.")
    parser.add_argument("audio", type=Path, help="Audio file to transcribe.")
    parser.add_argument("lyrics", type=Path, help="Lyrics text file (match rate).")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-prompt", action="store_true", help="No lyrics prompt.")
    parser.add_argument(
        "--no-prefetch", action="store_true", help="Skip untimed weight prefetch."
    )
    parser.add_argument(
        "--separation",
        action="store_true",
        help="Benchmark Demucs vocal separation instead of the model matrix.",
    )
    parser.add_argument("--markdown", type=Path, default=None, help="Write table here.")
    args = parser.parse_args(argv)

    devices = list(args.devices)
    if "cuda" in devices and not torch.cuda.is_available():
        print("note: CUDA unavailable; skipping GPU runs.", file=sys.stderr)
        devices = [d for d in devices if d != "cuda"]

    lines = parse_lyrics(args.lyrics)
    prompt = None if args.no_prompt else lyrics_prompt(lines)
    duration = audio_duration_s(args.audio)

    print(f"Audio: {args.audio.name}  ({duration:.1f} s)", file=sys.stderr)
    for line in hardware_lines():
        print(f"  {line}", file=sys.stderr)

    if args.separation:
        run_separation_benchmark(
            args.audio, lines, prompt, duration, devices, args.models, args.markdown
        )
        return

    if not args.no_prefetch:
        for model in args.models:
            print(f"prefetching weights (untimed): {model} …", file=sys.stderr)
            whisper.load_model(model, device="cpu")
            gc.collect()

    results: list[Result] = []
    for device in devices:
        for model in args.models:
            print(f"→ {model} on {device} …", file=sys.stderr, flush=True)
            res = run_config(
                model, device, args.audio, lines, prompt, duration, args.repeats
            )
            if res.error:
                print(f"  FAILED: {res.error}", file=sys.stderr)
            else:
                print(
                    f"  load {res.load_s:.1f}s  transcribe {res.transcribe_s:.1f}s"
                    f"  {res.speed:.1f}×RT  match {res.matched}/{res.total}",
                    file=sys.stderr,
                )
            results.append(res)

    table = render(results, duration)
    print(table)
    if args.markdown:
        args.markdown.write_text(table, encoding="utf-8")
        print(f"wrote {args.markdown}", file=sys.stderr)


if __name__ == "__main__":
    main()
