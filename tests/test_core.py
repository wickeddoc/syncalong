"""Tests for lyrics parsing, alignment, and LRC formatting.

These tests don't require Whisper or audio files — they exercise the
text-processing pipeline with synthetic WordTimestamp data.

Run with: python -m pytest tests/
"""

from __future__ import annotations

import textwrap
import tempfile
from pathlib import Path

import pytest

from syncalong.lyrics import parse_lyrics, lyrics_prompt, LyricLine
from syncalong.textnorm import normalize
from syncalong.transcribe import WordTimestamp, _build_transcribe_options
from syncalong.align import align_lyrics_to_transcript, _dp_align, _word_score
from syncalong.formatter import format_lrc, _seconds_to_lrc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lyrics_file(text: str) -> Path:
    """Write text to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    tmp.write(textwrap.dedent(text))
    tmp.close()
    return Path(tmp.name)


def _make_transcript(words_and_times: list[tuple[str, float, float]]) -> list[WordTimestamp]:
    """Build a list of WordTimestamp from (word, start, end) tuples."""
    return [
        WordTimestamp(word=w.lower(), raw=w, start=s, end=e)
        for w, s, e in words_and_times
    ]


# ---------------------------------------------------------------------------
# Lyrics parsing
# ---------------------------------------------------------------------------

class TestParseLyrics:
    def test_basic(self):
        path = _make_lyrics_file("Hello world\nGoodbye moon\n")
        lines = parse_lyrics(path)
        assert len(lines) == 2
        assert lines[0].words == ["hello", "world"]
        assert lines[1].words == ["goodbye", "moon"]
        assert not lines[0].is_blank

    def test_blank_lines(self):
        path = _make_lyrics_file("Line one\n\nLine three\n")
        lines = parse_lyrics(path)
        assert len(lines) == 3
        assert lines[1].is_blank
        assert lines[1].words == []

    def test_section_headers(self):
        path = _make_lyrics_file("[Chorus]\nLa la la\n(Bridge)\nDum dum\n")
        lines = parse_lyrics(path)
        assert lines[0].is_blank  # [Chorus] treated as blank
        assert lines[0].raw == "[Chorus]"
        assert lines[2].is_blank  # (Bridge) treated as blank
        assert lines[1].words == ["la", "la", "la"]

    def test_normalize_strips_accents_and_case(self):
        assert normalize("Café déjà") == "cafe deja"

    def test_normalize_splits_hyphenated_words(self):
        # Whisper emits "deja vu" as two words — hyphens must become
        # word boundaries, not be deleted.
        assert normalize("déjà-vu") == "deja vu"

    def test_normalize_keeps_contractions_joined(self):
        assert normalize("don't") == "dont"
        assert normalize("don’t") == "dont"  # curly apostrophe

    def test_punctuation_stripping(self):
        path = _make_lyrics_file("Don't stop, believin'!\n")
        lines = parse_lyrics(path)
        # Punctuation is removed in normalized words
        assert "dont" in lines[0].words
        assert "stop" in lines[0].words
        assert "believin" in lines[0].words


# ---------------------------------------------------------------------------
# Lyrics prompt for Whisper biasing
# ---------------------------------------------------------------------------

class TestLyricsPrompt:
    def test_joins_non_blank_lines_with_original_text(self):
        path = _make_lyrics_file("[Verse 1]\nHello world\n\nDon't stop\n")
        lines = parse_lyrics(path)
        prompt = lyrics_prompt(lines)
        # Original text, blanks and section headers excluded
        assert prompt == "Hello world Don't stop"

    def test_truncates_at_word_boundary(self):
        path = _make_lyrics_file("alpha beta gamma delta\n")
        lines = parse_lyrics(path)
        prompt = lyrics_prompt(lines, max_chars=12)
        assert prompt == "alpha beta"

    def test_empty_lyrics_give_empty_prompt(self):
        path = _make_lyrics_file("\n\n")
        lines = parse_lyrics(path)
        assert lyrics_prompt(lines) == ""


# ---------------------------------------------------------------------------
# Whisper transcription options
# ---------------------------------------------------------------------------

class TestTranscribeOptions:
    def test_defaults_disable_conditioning(self):
        opts = _build_transcribe_options(language=None, initial_prompt=None)
        assert opts["word_timestamps"] is True
        assert opts["condition_on_previous_text"] is False
        assert "language" not in opts
        assert "initial_prompt" not in opts

    def test_language_and_prompt_passthrough(self):
        opts = _build_transcribe_options(language="de", initial_prompt="Hallo Welt")
        assert opts["language"] == "de"
        assert opts["initial_prompt"] == "Hallo Welt"

    def test_parser_accepts_all_whisper_model_names(self):
        from syncalong.cli import build_parser
        parser = build_parser()
        for model in ["turbo", "small.en", "large-v3", "base"]:
            args = parser.parse_args(["l.txt", "a.mp3", "-m", model])
            assert args.model == model

    def test_cli_flag_disables_lyrics_prompt(self):
        from syncalong.cli import build_parser
        parser = build_parser()
        assert parser.parse_args(["l.txt", "a.mp3"]).no_lyrics_prompt is False
        assert parser.parse_args(
            ["l.txt", "a.mp3", "--no-lyrics-prompt"]
        ).no_lyrics_prompt is True


# ---------------------------------------------------------------------------
# Word scoring
# ---------------------------------------------------------------------------

class TestWordScore:
    def test_identical(self):
        assert _word_score("hello", "hello") == 100.0

    def test_similar(self):
        score = _word_score("hello", "helo")
        assert score > 70.0

    def test_dissimilar(self):
        score = _word_score("hello", "xyz")
        assert score < 50.0


# ---------------------------------------------------------------------------
# DP alignment
# ---------------------------------------------------------------------------

class TestDPAlign:
    def test_exact_match(self):
        lyric = ["hello", "beautiful", "world"]
        transcript = _make_transcript([
            ("hello", 1.0, 1.5),
            ("beautiful", 2.0, 2.5),
            ("world", 3.0, 3.5),
        ])
        mapping = _dp_align(lyric, transcript, threshold=55.0)
        assert mapping == {0: 0, 1: 1, 2: 2}

    def test_extra_transcript_words(self):
        lyric = ["hello", "world"]
        transcript = _make_transcript([
            ("um", 0.5, 0.8),
            ("hello", 1.0, 1.5),
            ("uh", 1.8, 2.0),
            ("world", 3.0, 3.5),
        ])
        mapping = _dp_align(lyric, transcript, threshold=55.0)
        assert mapping[0] == 1  # "hello" → transcript[1]
        assert mapping[1] == 3  # "world" → transcript[3]

    def test_fuzzy_match(self):
        lyric = ["running", "through"]
        transcript = _make_transcript([
            ("runnin", 1.0, 1.5),
            ("through", 2.0, 2.5),
        ])
        mapping = _dp_align(lyric, transcript, threshold=55.0)
        assert 0 in mapping  # "running" ≈ "runnin"
        assert 1 in mapping

    def test_empty_inputs(self):
        assert _dp_align([], [], threshold=55.0) == {}
        transcript = _make_transcript([("hello", 1.0, 1.5)])
        assert _dp_align([], transcript, threshold=55.0) == {}


# ---------------------------------------------------------------------------
# Full alignment pipeline
# ---------------------------------------------------------------------------

class TestAlignLyricsToTranscript:
    def test_repeated_chorus_does_not_cross_match(self):
        # The same line sung twice must map to two different points in the
        # audio — monotonicity prevents the repeats from collapsing onto
        # the first occurrence.
        path = _make_lyrics_file("la la la\nsomething else\nla la la\n")
        lines = parse_lyrics(path)
        transcript = _make_transcript([
            ("la", 1.0, 1.2), ("la", 1.3, 1.5), ("la", 1.6, 1.8),
            ("something", 5.0, 5.5), ("else", 5.6, 6.0),
            ("la", 10.0, 10.2), ("la", 10.3, 10.5), ("la", 10.6, 10.8),
        ])
        result = align_lyrics_to_transcript(lines, transcript)
        assert result[0][1] == 1.0
        assert result[1][1] == 5.0
        assert result[2][1] == 10.0

    def test_unmatched_line_gets_interpolated_timestamp(self):
        path = _make_lyrics_file("hello world\nxylophone zebra\ngoodbye moon\n")
        lines = parse_lyrics(path)
        transcript = _make_transcript([
            ("hello", 0.0, 0.5), ("world", 1.0, 1.5),
            # "xylophone zebra" never appears in the transcript
            ("goodbye", 10.0, 10.5), ("moon", 11.0, 11.5),
        ])
        result = align_lyrics_to_transcript(lines, transcript)
        assert result[1][1] == pytest.approx(5.0)  # midpoint of 0.0 and 10.0

    def test_lines_before_first_match_are_extrapolated(self):
        path = _make_lyrics_file("xylophone zebra\nhello world\n")
        lines = parse_lyrics(path)
        transcript = _make_transcript([
            # Unmatched audio before the first matched lyric line
            ("blah", 2.0, 2.5),
            ("hello", 6.0, 6.5), ("world", 7.0, 7.5),
        ])
        result = align_lyrics_to_transcript(lines, transcript)
        # Halfway between transcript start (2.0) and the first anchor (6.0)
        assert result[0][1] == pytest.approx(4.0)
        assert result[1][1] == 6.0

    def test_lines_after_last_match_are_extrapolated(self):
        path = _make_lyrics_file("hello world\nxylophone zebra\n")
        lines = parse_lyrics(path)
        transcript = _make_transcript([
            ("hello", 1.0, 1.5), ("world", 2.0, 2.5),
            ("blah", 9.5, 10.0),
        ])
        result = align_lyrics_to_transcript(lines, transcript)
        assert result[0][1] == 1.0
        # Halfway between the last anchor (1.0) and transcript end (10.0)
        assert result[1][1] == pytest.approx(5.5)

    def test_no_matches_at_all_leaves_all_lines_untimed(self):
        path = _make_lyrics_file("xylophone zebra\n")
        lines = parse_lyrics(path)
        transcript = _make_transcript([("blah", 1.0, 1.5)])
        result = align_lyrics_to_transcript(lines, transcript)
        assert result[0][1] is None

    def test_simple_two_lines(self):
        path = _make_lyrics_file("hello world\ngoodbye moon\n")
        lines = parse_lyrics(path)
        transcript = _make_transcript([
            ("hello", 1.0, 1.5),
            ("world", 2.0, 2.5),
            ("goodbye", 4.0, 4.5),
            ("moon", 5.0, 5.5),
        ])
        result = align_lyrics_to_transcript(lines, transcript)
        assert len(result) == 2
        assert result[0][1] == 1.0   # First line starts at "hello"
        assert result[1][1] == 4.0   # Second line starts at "goodbye"


# ---------------------------------------------------------------------------
# Vocal separation (demucs faked — real runs need the optional extra)
# ---------------------------------------------------------------------------

class TestVocalSeparator:
    def _fake_run(self, returncode=0, make_vocals=True):
        import subprocess

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if make_vocals:
                outdir = Path(cmd[cmd.index("-o") + 1])
                stem_dir = outdir / "htdemucs" / "song"
                stem_dir.mkdir(parents=True)
                (stem_dir / "vocals.wav").write_bytes(b"")
            return subprocess.CompletedProcess(cmd, returncode)

        return fake_run, calls

    def test_returns_vocals_path_and_registers_cleanup(self, monkeypatch):
        from syncalong import vocal_separator as vs

        registered = []
        monkeypatch.setattr(
            vs.atexit, "register",
            lambda fn, *a, **kw: registered.append((fn, a, kw)),
        )
        fake_run, calls = self._fake_run()
        monkeypatch.setattr(vs.subprocess, "run", fake_run)

        vocals = vs.separate(Path("song.mp3"))
        assert vocals.name == "vocals.wav"
        assert vocals.is_file()

        # Demucs output must not pollute stdout (reserved for LRC), and
        # must not be captured (progress should stream to the user).
        _, kwargs = calls[0]
        assert not kwargs.get("capture_output")

        # The registered cleanup removes the temp dir
        assert registered
        for fn, a, kw in registered:
            fn(*a, **kw)
        assert not vocals.exists()

    def test_raises_on_demucs_failure(self, monkeypatch):
        from syncalong import vocal_separator as vs

        monkeypatch.setattr(vs.atexit, "register", lambda *a, **kw: None)
        fake_run, _ = self._fake_run(returncode=1, make_vocals=False)
        monkeypatch.setattr(vs.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError):
            vs.separate(Path("song.mp3"))


# ---------------------------------------------------------------------------
# CLI vocal-separation guard
# ---------------------------------------------------------------------------

class TestResolveAudioPath:
    def test_passthrough_without_separation(self):
        from syncalong import cli
        audio = Path("song.mp3")
        assert cli.resolve_audio_path(audio, separate_vocals=False) is audio

    def test_missing_demucs_exits_with_install_hint(self, monkeypatch, capsys):
        from syncalong import cli
        monkeypatch.setattr(cli, "_demucs_available", lambda: False)
        with pytest.raises(SystemExit):
            cli.resolve_audio_path(Path("song.mp3"), separate_vocals=True)
        err = capsys.readouterr().err
        assert "vocal-separation" in err


# ---------------------------------------------------------------------------
# LRC formatting
# ---------------------------------------------------------------------------

class TestLRCFormatter:
    def test_seconds_to_lrc(self):
        assert _seconds_to_lrc(0.0) == "[00:00.00]"
        assert _seconds_to_lrc(73.456) == "[01:13.46]"
        assert _seconds_to_lrc(600.0) == "[10:00.00]"

    def test_seconds_to_lrc_rounding_carries_into_minutes(self):
        # 59.999s must round up to a whole minute, never "[00:60.00]"
        assert _seconds_to_lrc(59.999) == "[01:00.00]"
        assert _seconds_to_lrc(119.996) == "[02:00.00]"

    def test_format_lrc(self):
        line1 = LyricLine(index=0, raw="Hello world", words=["hello", "world"])
        line2 = LyricLine(index=1, raw="Goodbye moon", words=["goodbye", "moon"])
        timed = [(line1, 1.0), (line2, 4.5)]
        lrc = format_lrc(timed)
        assert "[00:01.00] Hello world" in lrc
        assert "[00:04.50] Goodbye moon" in lrc

    def test_none_timestamp(self):
        line = LyricLine(index=0, raw="Mystery line", words=["mystery", "line"])
        lrc = format_lrc([(line, None)])
        assert "Mystery line" in lrc
        assert "[" not in lrc.split("Mystery")[0]  # No tag
