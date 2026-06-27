"""Format aligned lyrics as LRC (standard timed lyrics format)."""

from __future__ import annotations

from syncalong.lyrics import LyricLine


def _seconds_to_lrc(seconds: float) -> str:
    """Convert seconds to an LRC timestamp tag ``[mm:ss.xx]``.

    Rounds to centiseconds first so e.g. 59.999 carries into the next minute
    instead of formatting as an invalid ``[00:60.00]``.

    Args:
        seconds: A non-negative time in seconds.

    Returns:
        The timestamp formatted as ``[mm:ss.xx]``.

    Examples:
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
    """Render aligned lyrics as an LRC document.

    Lines with no timestamp are emitted without a time tag (some players show
    them statically). Blank lines become empty timed lines to preserve the
    song's visual structure.

    Args:
        timed_lines: ``(LyricLine, timestamp_or_None)`` pairs in order.

    Returns:
        The LRC document, newline-terminated.
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
