"""Optional vocal isolation using Meta's Demucs.

Separating vocals from the instrumental track dramatically improves
alignment accuracy on studio recordings where background music would
otherwise confuse the speech model.

This module is only imported when ``--separate-vocals`` is passed.
"""

from __future__ import annotations

import atexit
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def separate(audio_path: Path) -> Path:
    """Run Demucs on an audio file and return the isolated vocals path.

    Demucs writes to a temporary directory (removed at process exit via
    ``atexit``) and this returns the path to the produced ``vocals.wav``.

    Args:
        audio_path: The mixed audio file to separate.

    Returns:
        Path to the isolated ``vocals.wav``.

    Raises:
        RuntimeError: If Demucs exits with a non-zero status.
        FileNotFoundError: If no vocals file is produced.
    """
    outdir = Path(tempfile.mkdtemp(prefix="syncalong_demucs_"))
    # The intermediate WAVs are large — remove them when the process exits
    # (the vocals are only needed until transcription finishes).
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
        raise RuntimeError(
            f"Demucs failed (exit {result.returncode}) — see output above."
        )

    # Demucs output structure: <outdir>/htdemucs/<stem_name>/vocals.wav
    # The exact model dir name can vary, so we glob for the vocals file.
    vocals_candidates = list(outdir.rglob("vocals.wav"))
    if not vocals_candidates:
        raise FileNotFoundError(
            f"Demucs did not produce a vocals.wav under {outdir}.\n"
            f"Contents: {list(outdir.rglob('*'))}"
        )

    vocals_path = vocals_candidates[0]
    print(f"  Isolated vocals: {vocals_path}", file=sys.stderr)
    return vocals_path
