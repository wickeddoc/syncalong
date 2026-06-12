"""Format aligned lyrics as LRC (standard timed lyrics format)."""

from __future__ import annotations

from syncalong.lyrics import LyricLine


def _seconds_to_lrc(seconds: float) -> str:
    """
    Convert seconds to LRC timestamp: ``[mm:ss.xx]``.

    Examples
    --------
    >>> _seconds_to_lrc(0.0)
    '[00:00.00]'
    >>> _seconds_to_lrc(73.456)
    '[01:13.46]'
    """
    # Round to centiseconds first so 59.999 carries into the minute
    # instead of formatting as "60.00" seconds.
    centiseconds = round(seconds * 100)
    minutes, cs = divmod(centiseconds, 6000)
    return f"[{minutes:02d}:{cs // 100:02d}.{cs % 100:02d}]"


def format_lrc(
    timed_lines: list[tuple[LyricLine, float | None]],
) -> str:
    """
    Produce an LRC-formatted string from aligned lyrics.

    Lines with no timestamp are emitted without a time tag (some LRC
    players will just display them statically).  Blank lines become
    empty timed lines to preserve the song's visual structure.
    """
    parts: list[str] = []

    for line, ts in timed_lines:
        if ts is not None:
            tag = _seconds_to_lrc(ts)
            text = line.raw if not line.is_blank else ""
            parts.append(f"{tag} {text}")
        else:
            # No timestamp available — emit raw text without tag
            if line.is_blank:
                parts.append("")
            else:
                parts.append(line.raw)

    return "\n".join(parts) + "\n"
