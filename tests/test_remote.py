"""Tests for the stdlib-only RemoteTranscriber HTTP client."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from syncalong import remote
from syncalong.remote import RemoteTranscriber
from syncalong.transcribe import WordTimestamp


class _FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, payload, capture):
    def fake_urlopen(request, timeout=None):
        capture["request"] = request
        capture["timeout"] = timeout
        return _FakeResponse(payload)

    monkeypatch.setattr(remote.urllib.request, "urlopen", fake_urlopen)


class TestRemoteTranscriber:
    def test_posts_and_parses_words(self, tmp_path, monkeypatch):
        audio = tmp_path / "song.mp3"
        audio.write_bytes(b"\x00\x01\x02")
        capture = {}
        _patch_urlopen(
            monkeypatch,
            {"words": [{"word": "hi", "raw": "Hi", "start": 0.1, "end": 0.4}]},
            capture,
        )

        tx = RemoteTranscriber("http://gpu:8000/")
        words = tx.transcribe(
            audio, language="en", initial_prompt="hi", separate_vocals=True
        )

        assert words == [WordTimestamp(word="hi", raw="Hi", start=0.1, end=0.4)]
        req = capture["request"]
        assert req.full_url == "http://gpu:8000/transcribe"
        assert req.method == "POST"
        body = req.data.decode("latin-1")
        assert 'name="language"' in body and "en" in body
        assert 'name="separate_vocals"' in body and "true" in body
        assert 'filename="song.mp3"' in body
        assert capture["timeout"] == 300.0  # default timeout forwarded to urlopen

    def test_sends_bearer_token(self, tmp_path, monkeypatch):
        audio = tmp_path / "s.mp3"
        audio.write_bytes(b"\x00")
        capture = {}
        _patch_urlopen(monkeypatch, {"words": []}, capture)

        RemoteTranscriber("http://gpu:8000", token="secret").transcribe(audio)

        assert capture["request"].headers["Authorization"] == "Bearer secret"

    def test_omits_none_fields_from_body(self, tmp_path, monkeypatch):
        audio = tmp_path / "s.mp3"
        audio.write_bytes(b"\x00")
        capture = {}
        _patch_urlopen(monkeypatch, {"words": []}, capture)

        # language and initial_prompt default to None
        RemoteTranscriber("http://gpu:8000").transcribe(audio)

        body = capture["request"].data.decode("latin-1")
        assert 'name="language"' not in body
        assert 'name="initial_prompt"' not in body
        # a non-None field is still present, and "None" is never serialized
        assert 'name="separate_vocals"' in body and "false" in body
        assert "None" not in body

    def test_http_error_becomes_runtimeerror(self, tmp_path, monkeypatch):
        audio = tmp_path / "s.mp3"
        audio.write_bytes(b"\x00")

        def raise_http(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                {},
                io.BytesIO(b'{"detail": "invalid or missing token"}'),
            )

        monkeypatch.setattr(remote.urllib.request, "urlopen", raise_http)
        with pytest.raises(RuntimeError) as excinfo:
            RemoteTranscriber("http://gpu:8000").transcribe(audio)
        message = str(excinfo.value)
        assert "HTTP 401" in message
        assert "invalid or missing token" in message  # server-provided detail

    def test_unreachable_becomes_runtimeerror(self, tmp_path, monkeypatch):
        audio = tmp_path / "s.mp3"
        audio.write_bytes(b"\x00")

        def raise_url(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(remote.urllib.request, "urlopen", raise_url)
        with pytest.raises(RuntimeError) as excinfo:
            RemoteTranscriber("http://gpu:8000").transcribe(audio)
        message = str(excinfo.value)
        assert "could not reach" in message
        assert "http://gpu:8000/transcribe" in message  # names the target URL
        assert "connection refused" in message  # includes the underlying reason
