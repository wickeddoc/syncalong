"""
Align lyrics to Whisper transcript using dynamic-programming sequence alignment.

The algorithm is a variant of the Needleman–Wunsch / Smith–Waterman family:
both the lyric word sequence and the transcript word sequence are in temporal
order, so we find a monotonic mapping that maximises the total fuzzy-match
score between paired words.

Complexity is O(N·M) where N = lyric words, M = transcript words.  For a
typical song (< 600 words each) this takes a few milliseconds.
"""

from __future__ import annotations

from functools import lru_cache

from syncalong.lyrics import LyricLine
from syncalong.transcribe import WordTimestamp


# ---------------------------------------------------------------------------
# Fuzzy word similarity
# ---------------------------------------------------------------------------

try:
    from rapidfuzz.fuzz import ratio as _rf_ratio

    def _ratio(a: str, b: str) -> float:
        return _rf_ratio(a, b)
except ImportError:
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100.0


@lru_cache(maxsize=None)
def _word_score(a: str, b: str) -> float:
    """Return a 0–100 fuzzy similarity score between two normalised words.

    Uses rapidfuzz for speed, falling back to difflib if unavailable. Cached:
    the DP loop scores every lyric×transcript pair and song vocabularies
    repeat heavily, so most pairs recur many times.

    Args:
        a: First normalised word.
        b: Second normalised word.

    Returns:
        Similarity in 0–100 (100 for an exact match).
    """
    if a == b:
        return 100.0
    return _ratio(a, b)


# ---------------------------------------------------------------------------
# DP alignment
# ---------------------------------------------------------------------------

def _dp_align(
    lyric_words: list[str],
    transcript_words: list[WordTimestamp],
    threshold: float,
) -> dict[int, int]:
    """Find the best monotonic alignment of lyric words to transcript words.

    Needleman–Wunsch-style DP where ``dp[i][j]`` is the best cumulative score
    aligning ``lyric[:i]`` to ``transcript[:j]``. Transitions: skip a
    transcript word, skip a lyric word (small penalty), or match a pair when
    its score clears ``threshold``.

    Args:
        lyric_words: Flattened, normalised lyric words in order.
        transcript_words: Transcript word timestamps in order.
        threshold: Minimum fuzzy score (0–100) for a pair to count as a match.

    Returns:
        Mapping ``{lyric_word_index: transcript_word_index}`` for every lyric
        word matched above ``threshold``.
    """
    n = len(lyric_words)
    m = len(transcript_words)

    if n == 0 or m == 0:
        return {}

    SKIP_LYRIC_PENALTY = -1.0  # Small penalty for skipping a lyric word

    # Use flat arrays for speed
    # dp[i*(m+1) + j]
    size = (n + 1) * (m + 1)
    dp = [0.0] * size
    # trace: 0 = none, 1 = skip transcript, 2 = skip lyric, 3 = match
    trace = [0] * size

    def idx(i: int, j: int) -> int:
        return i * (m + 1) + j

    for i in range(1, n + 1):
        # Applying skip-lyric penalty cumulatively
        dp[idx(i, 0)] = dp[idx(i - 1, 0)] + SKIP_LYRIC_PENALTY
        trace[idx(i, 0)] = 2

    for i in range(1, n + 1):
        lw = lyric_words[i - 1]
        for j in range(1, m + 1):
            tw = transcript_words[j - 1].word
            best = dp[idx(i, j - 1)]       # skip transcript word
            best_t = 1

            val = dp[idx(i - 1, j)] + SKIP_LYRIC_PENALTY
            if val > best:
                best = val
                best_t = 2

            score = _word_score(lw, tw)
            if score >= threshold:
                val = dp[idx(i - 1, j - 1)] + score
                if val > best:
                    best = val
                    best_t = 3

            dp[idx(i, j)] = best
            trace[idx(i, j)] = best_t

    # Traceback
    mapping: dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        t = trace[idx(i, j)]
        if t == 3:
            mapping[i - 1] = j - 1  # 0-based
            i -= 1
            j -= 1
        elif t == 2:
            i -= 1
        else:
            j -= 1

    # Handle remaining lyric words (all skipped)
    # i might still be > 0 but they have no match — that's fine.

    return mapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align_lyrics_to_transcript(
    lyric_lines: list[LyricLine],
    transcript: list[WordTimestamp],
    *,
    threshold: float = 55.0,
) -> list[tuple[LyricLine, float | None]]:
    """Align parsed lyrics to a word-level transcript.

    Args:
        lyric_lines: Parsed lyrics (from
            :func:`~syncalong.lyrics.parse_lyrics`).
        transcript: Word timestamps (from
            :func:`~syncalong.transcribe.transcribe_audio`).
        threshold: Minimum fuzzy score (0–100) to accept a match.

    Returns:
        One ``(LyricLine, timestamp_or_None)`` pair per lyric line. The
        timestamp (seconds) is the start time of the first matched word in the
        line, or ``None`` if no word in the line could be aligned.
    """
    # Flatten all lyric words into a single ordered list, keeping a back-
    # reference so we can map results back to lines.
    flat_words: list[str] = []
    word_to_line: list[int] = []      # flat index → index into lyric_lines

    for li, line in enumerate(lyric_lines):
        for w in line.words:
            flat_words.append(w)
            word_to_line.append(li)

    # Run DP alignment
    mapping = _dp_align(flat_words, transcript, threshold)

    # For each lyric line, find the earliest matched word's timestamp
    line_timestamps: dict[int, float] = {}
    for flat_idx, trans_idx in sorted(mapping.items()):
        li = word_to_line[flat_idx]
        ts = transcript[trans_idx].start
        if li not in line_timestamps:
            line_timestamps[li] = ts

    # Interpolate timestamps for unmatched non-blank lines that sit between
    # two matched lines — gives a rough estimate rather than leaving gaps.
    matched_indices = sorted(line_timestamps.keys())
    if matched_indices:
        _interpolate_gaps(lyric_lines, line_timestamps, matched_indices)
        _extrapolate_edges(
            lyric_lines,
            line_timestamps,
            matched_indices,
            t_min=transcript[0].start,
            t_max=transcript[-1].end,
        )

    return [
        (line, line_timestamps.get(i))
        for i, line in enumerate(lyric_lines)
    ]


def _interpolate_gaps(
    lines: list[LyricLine],
    timestamps: dict[int, float],
    matched: list[int],
) -> None:
    """Fill gaps between matched lines with linearly interpolated times.

    Args:
        lines: All lyric lines.
        timestamps: Map of line index → timestamp, mutated in place.
        matched: Sorted indices of lines that already have a timestamp.
    """
    for k in range(len(matched) - 1):
        start_li = matched[k]
        end_li = matched[k + 1]
        gap = end_li - start_li

        if gap <= 1:
            continue  # Adjacent — nothing to interpolate

        t_start = timestamps[start_li]
        t_end = timestamps[end_li]

        for offset in range(1, gap):
            li = start_li + offset
            if lines[li].is_blank or not lines[li].words:
                continue
            # Linear interpolation
            frac = offset / gap
            timestamps[li] = t_start + frac * (t_end - t_start)


def _extrapolate_edges(
    lines: list[LyricLine],
    timestamps: dict[int, float],
    matched: list[int],
    *,
    t_min: float,
    t_max: float,
) -> None:
    """Estimate timestamps for unmatched lines before the first / after the
    last matched line.

    No sung line should be left untagged, since untagged lines are dropped by
    many LRC players. The transcript's start and end act as virtual anchors at
    line index ``-1`` and ``len(lines)``.

    Args:
        lines: All lyric lines.
        timestamps: Map of line index → timestamp, mutated in place.
        matched: Sorted indices of lines that already have a timestamp.
        t_min: Transcript start time (virtual leading anchor).
        t_max: Transcript end time (virtual trailing anchor).
    """
    first, last = matched[0], matched[-1]

    gap = first - (-1)
    for li in range(first):
        if lines[li].is_blank or not lines[li].words:
            continue
        frac = (li + 1) / gap
        timestamps[li] = t_min + frac * (timestamps[first] - t_min)

    gap = len(lines) - last
    for li in range(last + 1, len(lines)):
        if lines[li].is_blank or not lines[li].words:
            continue
        frac = (li - last) / gap
        timestamps[li] = timestamps[last] + frac * (t_max - timestamps[last])
