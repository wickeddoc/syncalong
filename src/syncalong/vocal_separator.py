"""Optional vocal isolation using Meta's Demucs.

Separating vocals from the instrumental track dramatically improves
alignment accuracy on studio recordings where background music would
otherwise confuse the speech model.

The stems Demucs produces are large full-length WAVs written to a temp
directory; callers that outlive a single run (e.g. the transcription
server) must release each one with :func:`cleanup_separation` as soon as
the vocals have been transcribed — the ``atexit`` registration is only a
backstop for one-shot processes and crashes.

This module is only imported when ``--separate-vocals`` is passed.
"""

from __future__ import annotations

import atexit
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Maps each vocals path returned by separate() to the temp dir that holds it,
# so cleanup_separation() can remove exactly the right directory.
_separation_dirs: dict[Path, Path] = {}


def separate(audio_path: Path) -> Path:
    """Run Demucs on an audio file and return the isolated vocals path.

    Demucs writes to a temporary directory and this returns the path to the
    produced ``vocals.wav``. The directory is removed at process exit via
    ``atexit`` as a backstop, but long-running callers should reclaim it
    earlier with :func:`cleanup_separation` once the vocals are no longer
    needed. On failure the directory is removed immediately.

    Args:
        audio_path: The mixed audio file to separate.

    Returns:
        Path to the isolated ``vocals.wav``.

    Raises:
        RuntimeError: If Demucs exits with a non-zero status.
        FileNotFoundError: If no vocals file is produced.
    """
    outdir = Path(tempfile.mkdtemp(prefix="syncalong_demucs_"))
    # The intermediate WAVs are large — this backstop removes them when the
    # process exits, but a long-running caller must not wait for it: on a
    # tmpfs /tmp every leaked stem stays resident in RAM.
    atexit.register(shutil.rmtree, outdir, ignore_errors=True)

    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-o",
        str(outdir),
        str(audio_path),
    ]

    # Let demucs progress stream to the user's terminal (it can run for
    # minutes); redirect its stdout to stderr so ours stays clean for LRC.
    result = subprocess.run(
        cmd,
        stdout=sys.stderr,
        text=True,
    )

    if result.returncode != 0:
        shutil.rmtree(outdir, ignore_errors=True)
        raise RuntimeError(
            f"Demucs failed (exit {result.returncode}) — see output above."
        )

    # Demucs output structure: <outdir>/htdemucs/<stem_name>/vocals.wav
    # The exact model dir name can vary, so we glob for the vocals file.
    vocals_candidates = list(outdir.rglob("vocals.wav"))
    if not vocals_candidates:
        contents = list(outdir.rglob("*"))
        shutil.rmtree(outdir, ignore_errors=True)
        raise FileNotFoundError(
            f"Demucs did not produce a vocals.wav under {outdir}.\n"
            f"Contents: {contents}"
        )

    vocals_path = vocals_candidates[0]
    _separation_dirs[vocals_path] = outdir
    print(f"  Isolated vocals: {vocals_path}", file=sys.stderr)
    return vocals_path


def cleanup_separation(vocals_path: Path) -> None:
    """Remove the temporary Demucs directory behind a separated vocals file.

    :func:`separate` defers cleanup to process exit, which suits the one-shot
    CLI but leaks per-call stems in a long-running process (on a tmpfs
    ``/tmp`` they accumulate in RAM). Call this as soon as the vocals have
    been transcribed. Paths not produced by :func:`separate` are ignored, and
    the ``atexit`` backstop stays harmless afterwards.

    Args:
        vocals_path: A path previously returned by :func:`separate`.
    """
    outdir = _separation_dirs.pop(vocals_path, None)
    if outdir is not None:
        shutil.rmtree(outdir, ignore_errors=True)
