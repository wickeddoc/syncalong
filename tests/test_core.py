"""Tests for lyrics parsing, alignment, and LRC formatting.

These tests don't require Whisper or audio files — they exercise the
text-processing pipeline with synthetic WordTimestamp data.

Run with: python -m pytest tests/
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import pytest

from syncalong.align import _dp_align, _word_score, align_lyrics_to_transcript
from syncalong.formatter import _seconds_to_lrc, format_lrc
from syncalong.lyrics import LyricLine, lyrics_prompt, parse_lyrics, parse_lyrics_text
from syncalong.pipeline import AlignmentResult, align, align_to_lrc
from syncalong.textnorm import normalize
from syncalong.transcribe import (
    Transcriber,
    WordTimestamp,
    _build_transcribe_options,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lyrics_file(text: str) -> Path:
    """Write text to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(textwrap.dedent(text))
    return Path(tmp.name)


def _make_transcript(
    words_and_times: list[tuple[str, float, float]],
) -> list[WordTimestamp]:
    """Build a list of WordTimestamp from (word, start, end) tuples."""
    return [
        WordTimestamp(word=w.lower(), raw=w, start=s, end=e)
        for w, s, e in words_and_times
    ]


class _FakeModel:
    """Stand-in for a loaded Whisper model; records calls, returns canned result."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    def transcribe(self, audio, **opts):
        self.calls.append((audio, opts))
        return self._result


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


class TestParseLyricsText:
    def test_parses_text_without_a_file(self):
        lines = parse_lyrics_text("Hello world\nGoodbye moon\n")
        assert [ln.words for ln in lines] == [["hello", "world"], ["goodbye", "moon"]]
        assert not lines[0].is_blank

    def test_matches_parse_lyrics_on_same_content(self):
        text = "[Chorus]\nLa la la\n\nDon’t stop\n"
        path = _make_lyrics_file(text)
        from_file = [(ln.raw, ln.words, ln.is_blank) for ln in parse_lyrics(path)]
        from_text = [(ln.raw, ln.words, ln.is_blank) for ln in parse_lyrics_text(text)]
        assert from_file == from_text


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
        assert (
            parser.parse_args(["l.txt", "a.mp3", "--no-lyrics-prompt"]).no_lyrics_prompt
            is True
        )


# ---------------------------------------------------------------------------
# Transcriber class
# ---------------------------------------------------------------------------


class TestTranscriber:
    def test_injected_model_extracts_words_and_forwards_options(self):
        result = {
            "segments": [
                {
                    "words": [
                        {"word": " Hello", "start": 1.0, "end": 1.5},
                        {"word": " world", "start": 2.0, "end": 2.5},
                    ]
                }
            ]
        }
        fake = _FakeModel(result)
        tx = Transcriber(model=fake)
        words = tx.transcribe(Path("song.mp3"), language="en", initial_prompt="hi")
        assert [w.word for w in words] == ["hello", "world"]
        assert [w.start for w in words] == [1.0, 2.0]
        _, opts = fake.calls[0]
        assert opts["word_timestamps"] is True
        assert opts["condition_on_previous_text"] is False
        assert opts["language"] == "en"
        assert opts["initial_prompt"] == "hi"

    def test_reuses_same_model_across_calls(self):
        fake = _FakeModel({"segments": []})
        tx = Transcriber(model=fake)
        tx.transcribe(Path("a.mp3"))
        tx.transcribe(Path("b.mp3"))
        assert len(fake.calls) == 2

    def test_model_name_retained_on_load_path(self, monkeypatch):
        import sys
        import types

        import syncalong.transcribe as tr

        fake = _FakeModel({"segments": []})
        fake_whisper = types.ModuleType("whisper")
        # setattr (not attribute assignment) keeps type checkers happy about
        # adding a member to a ModuleType instance.
        setattr(fake_whisper, "load_model", lambda name: fake)  # noqa: B010
        monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
        tx = tr.Transcriber("medium")
        assert tx.model_name == "medium"
        assert tx._model is fake

    def test_model_name_none_when_model_injected(self):
        tx = Transcriber(model=_FakeModel({"segments": []}))
        assert tx.model_name is None

    def test_transcribe_audio_delegates_to_transcriber(self, monkeypatch):
        import syncalong.transcribe as tr

        captured = {}

        class FakeTranscriber:
            def __init__(self, model_name="base", *, model=None):
                captured["model_name"] = model_name

            def transcribe(
                self,
                audio_path,
                *,
                language=None,
                initial_prompt=None,
                separate_vocals=False,
            ):
                captured["args"] = (
                    audio_path,
                    language,
                    initial_prompt,
                    separate_vocals,
                )
                return [WordTimestamp("hi", "hi", 0.0, 0.5)]

        monkeypatch.setattr(tr, "Transcriber", FakeTranscriber)
        words = tr.transcribe_audio(
            Path("s.mp3"), model_name="small", language="de", initial_prompt="x"
        )
        assert captured["model_name"] == "small"
        assert captured["args"] == (Path("s.mp3"), "de", "x", False)
        assert words[0].word == "hi"

    def test_device_forwarded_to_load_model(self, monkeypatch):
        import sys
        import types

        import syncalong.transcribe as tr

        captured = {}
        fake = _FakeModel({"segments": []})
        fake_whisper = types.ModuleType("whisper")

        def load_model(name, device=None):
            captured["name"] = name
            captured["device"] = device
            return fake

        setattr(fake_whisper, "load_model", load_model)  # noqa: B010
        monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
        tr.Transcriber("small", device="cuda")
        assert captured == {"name": "small", "device": "cuda"}

    def test_separate_vocals_missing_demucs_raises(self, monkeypatch):
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        tx = Transcriber(model=_FakeModel({"segments": []}))
        with pytest.raises(ModuleNotFoundError):
            tx.transcribe(Path("song.mp3"), separate_vocals=True)

    def test_separate_vocals_runs_demucs_then_transcribes(self, monkeypatch):
        import importlib.util

        import syncalong.transcribe as tr

        monkeypatch.setattr(
            importlib.util, "find_spec", lambda name: object()
        )  # pretend demucs is installed
        fake_vs = __import__("types").ModuleType("syncalong.vocal_separator")
        setattr(fake_vs, "separate", lambda p: Path("/tmp/vocals.wav"))  # noqa: B010
        monkeypatch.setitem(
            __import__("sys").modules, "syncalong.vocal_separator", fake_vs
        )
        fake = _FakeModel({"segments": []})
        tx = tr.Transcriber(model=fake)
        tx.transcribe(Path("song.mp3"), separate_vocals=True)
        assert fake.calls[0][0] == "/tmp/vocals.wav"  # transcribed the vocals path


# ---------------------------------------------------------------------------
# WordTimestamp wire (de)serialization
# ---------------------------------------------------------------------------


class TestWordTimestampWire:
    def test_to_dict_shape(self):
        w = WordTimestamp(word="hi", raw="Hi", start=1.0, end=1.5)
        assert w.to_dict() == {"word": "hi", "raw": "Hi", "start": 1.0, "end": 1.5}

    def test_from_dict_rebuilds(self):
        d = {"word": "hi", "raw": "Hi", "start": "1.0", "end": "1.5"}
        w = WordTimestamp.from_dict(d)
        assert w == WordTimestamp(word="hi", raw="Hi", start=1.0, end=1.5)
        assert isinstance(w.start, float) and isinstance(w.end, float)

    def test_roundtrip(self):
        w = WordTimestamp(word="x", raw="X", start=0.25, end=0.75)
        assert WordTimestamp.from_dict(w.to_dict()) == w


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
        transcript = _make_transcript(
            [
                ("hello", 1.0, 1.5),
                ("beautiful", 2.0, 2.5),
                ("world", 3.0, 3.5),
            ]
        )
        mapping = _dp_align(lyric, transcript, threshold=55.0)
        assert mapping == {0: 0, 1: 1, 2: 2}

    def test_extra_transcript_words(self):
        lyric = ["hello", "world"]
        transcript = _make_transcript(
            [
                ("um", 0.5, 0.8),
                ("hello", 1.0, 1.5),
                ("uh", 1.8, 2.0),
                ("world", 3.0, 3.5),
            ]
        )
        mapping = _dp_align(lyric, transcript, threshold=55.0)
        assert mapping[0] == 1  # "hello" → transcript[1]
        assert mapping[1] == 3  # "world" → transcript[3]

    def test_fuzzy_match(self):
        lyric = ["running", "through"]
        transcript = _make_transcript(
            [
                ("runnin", 1.0, 1.5),
                ("through", 2.0, 2.5),
            ]
        )
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
        transcript = _make_transcript(
            [
                ("la", 1.0, 1.2),
                ("la", 1.3, 1.5),
                ("la", 1.6, 1.8),
                ("something", 5.0, 5.5),
                ("else", 5.6, 6.0),
                ("la", 10.0, 10.2),
                ("la", 10.3, 10.5),
                ("la", 10.6, 10.8),
            ]
        )
        result = align_lyrics_to_transcript(lines, transcript)
        assert result[0][1] == 1.0
        assert result[1][1] == 5.0
        assert result[2][1] == 10.0

    def test_unmatched_line_gets_interpolated_timestamp(self):
        path = _make_lyrics_file("hello world\nxylophone zebra\ngoodbye moon\n")
        lines = parse_lyrics(path)
        transcript = _make_transcript(
            [
                ("hello", 0.0, 0.5),
                ("world", 1.0, 1.5),
                # "xylophone zebra" never appears in the transcript
                ("goodbye", 10.0, 10.5),
                ("moon", 11.0, 11.5),
            ]
        )
        result = align_lyrics_to_transcript(lines, transcript)
        assert result[1][1] == pytest.approx(5.0)  # midpoint of 0.0 and 10.0

    def test_lines_before_first_match_are_extrapolated(self):
        path = _make_lyrics_file("xylophone zebra\nhello world\n")
        lines = parse_lyrics(path)
        transcript = _make_transcript(
            [
                # Unmatched audio before the first matched lyric line
                ("blah", 2.0, 2.5),
                ("hello", 6.0, 6.5),
                ("world", 7.0, 7.5),
            ]
        )
        result = align_lyrics_to_transcript(lines, transcript)
        # Halfway between transcript start (2.0) and the first anchor (6.0)
        assert result[0][1] == pytest.approx(4.0)
        assert result[1][1] == 6.0

    def test_lines_after_last_match_are_extrapolated(self):
        path = _make_lyrics_file("hello world\nxylophone zebra\n")
        lines = parse_lyrics(path)
        transcript = _make_transcript(
            [
                ("hello", 1.0, 1.5),
                ("world", 2.0, 2.5),
                ("blah", 9.5, 10.0),
            ]
        )
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
        transcript = _make_transcript(
            [
                ("hello", 1.0, 1.5),
                ("world", 2.0, 2.5),
                ("goodbye", 4.0, 4.5),
                ("moon", 5.0, 5.5),
            ]
        )
        result = align_lyrics_to_transcript(lines, transcript)
        assert len(result) == 2
        assert result[0][1] == 1.0  # First line starts at "hello"
        assert result[1][1] == 4.0  # Second line starts at "goodbye"


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
            vs.atexit,
            "register",
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


class TestCliRemote:
    def _lyrics_and_audio(self, tmp_path):
        lyr = tmp_path / "l.txt"
        lyr.write_text("hello world\ngoodbye moon\n", encoding="utf-8")
        aud = tmp_path / "a.mp3"
        aud.write_bytes(b"\x00")
        return lyr, aud

    def test_server_flag_uses_remote_transcriber(self, tmp_path, monkeypatch):
        import syncalong.remote as remote
        from syncalong import cli

        lyr, aud = self._lyrics_and_audio(tmp_path)
        captured = {}

        class FakeRemote:
            def __init__(self, base_url, *, token=None, timeout=300.0):
                captured["base_url"] = base_url
                captured["token"] = token

            def transcribe(
                self,
                audio_path,
                *,
                language=None,
                initial_prompt=None,
                separate_vocals=False,
            ):
                captured["separate_vocals"] = separate_vocals
                return [
                    WordTimestamp("hello", "hello", 1.0, 1.5),
                    WordTimestamp("world", "world", 2.0, 2.5),
                    WordTimestamp("goodbye", "goodbye", 4.0, 4.5),
                    WordTimestamp("moon", "moon", 5.0, 5.5),
                ]

        monkeypatch.setattr(remote, "RemoteTranscriber", FakeRemote)
        cli.main(
            [
                str(lyr),
                str(aud),
                "--server",
                "http://gpu:8000",
                "--token",
                "t",
                "--separate-vocals",
            ]
        )
        assert captured["base_url"] == "http://gpu:8000"
        assert captured["token"] == "t"
        assert captured["separate_vocals"] is True

    def test_server_from_env(self, tmp_path, monkeypatch):
        import syncalong.remote as remote
        from syncalong import cli

        lyr, aud = self._lyrics_and_audio(tmp_path)
        captured = {}

        class FakeRemote:
            def __init__(self, base_url, *, token=None, timeout=300.0):
                captured["base_url"] = base_url

            def transcribe(self, *a, **k):
                return [WordTimestamp("hello", "hello", 1.0, 1.5)]

        monkeypatch.setattr(remote, "RemoteTranscriber", FakeRemote)
        monkeypatch.setenv("SYNCALONG_SERVER", "http://env:9000")
        cli.main([str(lyr), str(aud)])
        assert captured["base_url"] == "http://env:9000"

    def test_local_without_whisper_shows_hint(self, tmp_path, monkeypatch, capsys):
        from syncalong import cli

        lyr, aud = self._lyrics_and_audio(tmp_path)
        monkeypatch.delenv("SYNCALONG_SERVER", raising=False)
        monkeypatch.setattr(cli, "_whisper_available", lambda: False)
        with pytest.raises(SystemExit):
            cli.main([str(lyr), str(aud)])
        assert "syncalong[whisper]" in capsys.readouterr().err


class TestServeMain:
    def test_builds_app_and_runs_uvicorn(self, monkeypatch):
        import uvicorn

        import syncalong.server as server
        import syncalong.transcribe as tr
        from syncalong import cli

        recorded = {}

        class FakeTranscriber:
            def __init__(self, model_name="base", *, device=None):
                recorded["model"] = model_name
                recorded["device"] = device

        monkeypatch.setattr(tr, "Transcriber", FakeTranscriber)

        fake_app = object()

        def fake_create_app(transcriber, *, token=None, allow_vocal_separation=True):
            recorded["token"] = token
            recorded["allow_sep"] = allow_vocal_separation
            return fake_app

        monkeypatch.setattr(server, "create_app", fake_create_app)

        def fake_run(app, host=None, port=None):
            recorded["run"] = (app, host, port)

        monkeypatch.setattr(uvicorn, "run", fake_run)

        cli.serve_main(
            [
                "-m",
                "small",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--token",
                "sek",
                "--no-vocal-separation",
            ]
        )
        assert recorded["model"] == "small"
        assert recorded["token"] == "sek"
        assert recorded["allow_sep"] is False
        assert recorded["run"] == (fake_app, "0.0.0.0", 9000)


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
        timed: list[tuple[LyricLine, float | None]] = [(line1, 1.0), (line2, 4.5)]
        lrc = format_lrc(timed)
        assert "[00:01.00] Hello world" in lrc
        assert "[00:04.50] Goodbye moon" in lrc

    def test_none_timestamp(self):
        line = LyricLine(index=0, raw="Mystery line", words=["mystery", "line"])
        lrc = format_lrc([(line, None)])
        assert "Mystery line" in lrc
        assert "[" not in lrc.split("Mystery")[0]  # No tag


# ---------------------------------------------------------------------------
# High-level pipeline facade
# ---------------------------------------------------------------------------


class TestAlignFacade:
    def _fake_transcriber(self):
        result = {
            "segments": [
                {
                    "words": [
                        {"word": " hello", "start": 1.0, "end": 1.5},
                        {"word": " world", "start": 2.0, "end": 2.5},
                        {"word": " goodbye", "start": 4.0, "end": 4.5},
                        {"word": " moon", "start": 5.0, "end": 5.5},
                    ]
                }
            ]
        }
        return Transcriber(model=_FakeModel(result))

    def test_returns_result_with_counts_and_lrc(self):
        res = align(
            "hello world\ngoodbye moon\n",
            "song.mp3",
            transcriber=self._fake_transcriber(),
        )
        assert isinstance(res, AlignmentResult)
        assert (res.matched, res.total) == (2, 2)
        assert [ts for _, ts in res.timed_lines] == [1.0, 4.0]
        assert "[00:01.00] hello world" in res.lrc

    def test_accepts_str_path_and_lines(self, tmp_path):
        p = tmp_path / "lyr.txt"
        p.write_text("hello world\ngoodbye moon\n", encoding="utf-8")
        lines = parse_lyrics_text("hello world\ngoodbye moon\n")
        results = [
            align(
                "hello world\ngoodbye moon\n",
                "s.mp3",
                transcriber=self._fake_transcriber(),
            ),
            align(p, "s.mp3", transcriber=self._fake_transcriber()),
            align(lines, "s.mp3", transcriber=self._fake_transcriber()),
        ]
        for r in results:
            assert [ts for _, ts in r.timed_lines] == [1.0, 4.0]

    def test_align_to_lrc_equals_align_lrc(self):
        lyrics = "hello world\ngoodbye moon\n"
        a = align_to_lrc(lyrics, "s.mp3", transcriber=self._fake_transcriber())
        b = align(lyrics, "s.mp3", transcriber=self._fake_transcriber()).lrc
        assert a == b

    def test_use_lyrics_prompt_toggles_prompt(self):
        fake_on = _FakeModel({"segments": []})
        align(
            "hello world\n",
            "s.mp3",
            transcriber=Transcriber(model=fake_on),
            use_lyrics_prompt=True,
        )
        assert fake_on.calls[0][1].get("initial_prompt") == "hello world"

        fake_off = _FakeModel({"segments": []})
        align(
            "hello world\n",
            "s.mp3",
            transcriber=Transcriber(model=fake_off),
            use_lyrics_prompt=False,
        )
        assert "initial_prompt" not in fake_off.calls[0][1]

    def test_rejects_bad_lyrics_type(self):
        with pytest.raises(TypeError):
            # 123 is deliberately the wrong type — verify the runtime guard.
            align(123, "s.mp3", transcriber=self._fake_transcriber())  # type: ignore[arg-type]

    def test_separate_vocals_without_demucs_raises(self, monkeypatch):
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        with pytest.raises(ModuleNotFoundError):
            align(
                "hello world\n",
                "s.mp3",
                transcriber=self._fake_transcriber(),
                separate_vocals=True,
            )

    def test_forwards_separate_vocals_to_transcriber(self):
        captured = {}

        class RecordingTranscriber:
            def transcribe(
                self,
                audio_path,
                *,
                language=None,
                initial_prompt=None,
                separate_vocals=False,
            ):
                captured["separate_vocals"] = separate_vocals
                return [WordTimestamp("hello", "hello", 1.0, 1.5)]

        align(
            "hello\n", "s.mp3", transcriber=RecordingTranscriber(), separate_vocals=True
        )
        assert captured["separate_vocals"] is True


class TestPublicAPI:
    def test_top_level_exports_present(self):
        import syncalong

        for name in [
            "align",
            "align_to_lrc",
            "AlignmentResult",
            "Transcriber",
            "RemoteTranscriber",
            "transcribe_audio",
            "WordTimestamp",
            "parse_lyrics",
            "parse_lyrics_text",
            "lyrics_prompt",
            "LyricLine",
            "align_lyrics_to_transcript",
            "format_lrc",
            "separate",
            "__version__",
        ]:
            assert name in syncalong.__all__, f"{name} missing from __all__"
            assert hasattr(syncalong, name), f"{name} not importable"

    def test_importing_syncalong_does_not_load_whisper(self):
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "-c",
                "import syncalong, sys; "
                "assert 'whisper' not in sys.modules; "
                "assert 'fastapi' not in sys.modules",
            ],
            check=True,
        )


class TestPackaging:
    def test_py_typed_marker_present(self):
        import pathlib

        import syncalong

        marker = pathlib.Path(syncalong.__file__).parent / "py.typed"
        assert marker.is_file()
