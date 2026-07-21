"""Tests for the FastAPI transcription server (Whisper faked)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from syncalong.server import create_app
from syncalong.transcribe import WordTimestamp


class _FakeTranscriber:
    model_name = "fake"

    def __init__(self):
        self.calls = []

    def transcribe(
        self, audio_path, *, language=None, initial_prompt=None, separate_vocals=False
    ):
        self.calls.append(
            {
                "language": language,
                "initial_prompt": initial_prompt,
                "separate_vocals": separate_vocals,
            }
        )
        return [WordTimestamp(word="hello", raw="Hello", start=1.0, end=1.5)]


def _client(**kwargs):
    return TestClient(create_app(_FakeTranscriber(), **kwargs))


class TestHealth:
    def test_ok(self):
        resp = _client().get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "model": "fake"}


class TestTranscribeEndpoint:
    def test_returns_serialized_words(self):
        resp = _client().post(
            "/transcribe",
            files={"audio": ("song.mp3", b"\x00\x01")},
            data={"language": "en", "initial_prompt": "hi"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["words"] == [
            {"word": "hello", "raw": "Hello", "start": 1.0, "end": 1.5}
        ]
        assert body["model"] == "fake"

    def test_forwards_options_to_transcriber(self):
        fake = _FakeTranscriber()
        TestClient(create_app(fake)).post(
            "/transcribe",
            files={"audio": ("song.mp3", b"\x00")},
            data={"language": "de", "initial_prompt": "x", "separate_vocals": "true"},
        )
        assert fake.calls[0] == {
            "language": "de",
            "initial_prompt": "x",
            "separate_vocals": True,
        }


class TestAuth:
    def test_missing_token_rejected(self):
        resp = _client(token="secret").post(
            "/transcribe", files={"audio": ("s.mp3", b"\x00")}
        )
        assert resp.status_code == 401

    def test_correct_token_accepted(self):
        resp = _client(token="secret").post(
            "/transcribe",
            files={"audio": ("s.mp3", b"\x00")},
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200

    def test_no_token_configured_allows_all(self):
        resp = _client().post("/transcribe", files={"audio": ("s.mp3", b"\x00")})
        assert resp.status_code == 200


class TestVocalSeparationErrors:
    def test_demucs_missing_returns_400(self):
        class _NoDemucs(_FakeTranscriber):
            def transcribe(
                self,
                audio_path,
                *,
                language=None,
                initial_prompt=None,
                separate_vocals=False,
            ):
                if separate_vocals:
                    raise ModuleNotFoundError(
                        "install syncalong[vocal-separation] on the server"
                    )
                return super().transcribe(
                    audio_path, language=language, initial_prompt=initial_prompt
                )

        resp = TestClient(create_app(_NoDemucs())).post(
            "/transcribe",
            files={"audio": ("s.mp3", b"\x00")},
            data={"separate_vocals": "true"},
        )
        assert resp.status_code == 400
        assert "vocal-separation" in resp.json()["detail"]

    def test_disabled_separation_returns_400(self):
        resp = _client(allow_vocal_separation=False).post(
            "/transcribe",
            files={"audio": ("s.mp3", b"\x00")},
            data={"separate_vocals": "true"},
        )
        assert resp.status_code == 400
