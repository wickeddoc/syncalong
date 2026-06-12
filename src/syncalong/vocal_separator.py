"""
Optional vocal isolation using Meta's Demucs.

Separating vocals from the instrumental track dramatically improves
alignment accuracy on studio recordings where background music would
otherwise confuse the speech model.

This module is only imported when ``--separate-vocals`` is passed.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def separate(audio_path: Path) -> Path:
    """
    Run Demucs on *audio_path* and return the path to the isolated vocals.

    Demucs writes its output to a temp directory and we return the path to
    the ``vocals.wav`` file.

    Raises
    ------
    RuntimeError
        If Demucs exits with a non-zero status.
    FileNotFoundError
        If the expected vocals file is not produced.
    """
    outdir = Path(tempfile.mkdtemp(prefix="syncalong_demucs_"))

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-o", str(outdir),
        str(audio_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Demucs failed (exit {result.returncode}):\n{result.stderr}"
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
