# Remote Transcription (Server/Client) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a low-powered jukebox transcribe songs on a remote GPU box by adding a FastAPI transcription server and a stdlib-only `RemoteTranscriber` client that drops into syncalong's existing `align(transcriber=…)` seam.

**Architecture:** The only GPU-heavy stage (`Transcriber.transcribe()`, plus optional Demucs) moves behind an HTTP boundary. A FastAPI server (`syncalong[server]`) wraps a reused `Transcriber`; the thin client uploads audio and gets back `WordTimestamp` JSON, then runs the cheap alignment locally. `RemoteTranscriber` satisfies the same `.transcribe()` interface as `Transcriber`, so nothing downstream changes. Whisper moves to an optional `[whisper]` extra so the jukebox install is torch-free.

**Tech Stack:** Python 3.9+, FastAPI + uvicorn (server, optional extra), Python stdlib `urllib` (client), pytest + Starlette `TestClient` (tests, torch-free).

## Global Constraints

- Python **3.9+**; every module starts with `from __future__ import annotations`.
- Line length **88**; `black .` and `ruff check .` must pass. Ruff uses pydocstyle **google** convention — **public** (non-`_`) classes/functions need Google-style docstrings; `_`-prefixed functions are exempt. `tests/*` are exempt from all `D` rules.
- `pyright src` must stay clean.
- Status/progress → **stderr**; only LRC → **stdout**.
- `import syncalong` must **never** load `whisper` **or** `fastapi` (thin-client guarantee). Whisper stays lazily imported inside `Transcriber.__init__`; `server.py` is never imported by `__init__.py`.
- **Client (`remote.py`) uses the stdlib only** — no new core runtime dependency.
- Server tests **fake the model** — CI stays torch-free.
- Dataclasses for structured data; curated public API via `__all__`; ship `py.typed`.
- Quality gate to run before every commit: `python -m pytest tests/ -q && ruff check . && black --check . && pyright src`.

---

### Task 1: Packaging — whisper → extra, add `server` extra, dev tooling

**Files:**
- Modify: `pyproject.toml:42-60` (`dependencies` + `[project.optional-dependencies]`)

**Interfaces:**
- Consumes: nothing.
- Produces: extras `whisper`, `server`, `vocal-separation`; `dev` gains `fastapi`, `httpx`, `uvicorn`. Core runtime dep becomes `rapidfuzz` only.

> Setup/config task — no failing-test-first. Verified by install + the existing suite staying green.

- [ ] **Step 1: Edit `dependencies` and extras**

Replace the `dependencies` block (currently lines 42–45) with:

```toml
dependencies = [
    "rapidfuzz>=3.0",
]
```

Replace the `[project.optional-dependencies]` block (currently lines 47–60) with:

```toml
[project.optional-dependencies]
whisper = ["openai-whisper>=20231117"]
server = [
    "openai-whisper>=20231117",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "python-multipart>=0.0.9",
]
vocal-separation = ["demucs>=4.0"]
docs = [
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.24",
]
dev = [
    "pytest>=7.0",
    "ruff>=0.4",
    "black>=24.0",
    "pyright>=1.1",
    "build>=1.0",
    "twine>=5.0",
    "fastapi>=0.110",
    "httpx>=0.27",
    "uvicorn>=0.29",
]
```

- [ ] **Step 2: Reinstall the dev environment**

Run: `pip install -e ".[dev,docs]"`
Expected: succeeds; installs fastapi, httpx, uvicorn.

- [ ] **Step 3: Verify the toolchain imports and core stays thin**

Run: `python -c "import fastapi, httpx, uvicorn; print('ok')"`
Expected: prints `ok`.

Run: `python -c "import syncalong, sys; assert 'whisper' not in sys.modules and 'fastapi' not in sys.modules; print('thin ok')"`
Expected: prints `thin ok`.

- [ ] **Step 4: Run the existing suite (must stay green)**

Run: `python -m pytest tests/ -q && ruff check . && black --check . && pyright src`
Expected: PASS (no code changed yet).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: move whisper to [whisper] extra, add [server] extra and dev test deps"
```

---

### Task 2: Wire (de)serialization for `WordTimestamp`

**Files:**
- Modify: `src/syncalong/transcribe.py:12-26` (the `WordTimestamp` dataclass)
- Test: `tests/test_core.py` (new class `TestWordTimestampWire`)

**Interfaces:**
- Consumes: `WordTimestamp(word, raw, start, end)`.
- Produces:
  - `WordTimestamp.to_dict(self) -> dict[str, Any]` → `{"word","raw","start","end"}`
  - `WordTimestamp.from_dict(cls, data: dict[str, Any]) -> WordTimestamp`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_core.py` (after the imports; place the class near `TestTranscriber`):

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_core.py::TestWordTimestampWire -q`
Expected: FAIL — `AttributeError: 'WordTimestamp' object has no attribute 'to_dict'`.

- [ ] **Step 3: Add the methods**

In `src/syncalong/transcribe.py`, extend the `WordTimestamp` dataclass (after the `end: float` field, keeping the existing docstring/fields):

```python
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for the transcription wire format.

        Returns:
            A mapping with ``word``, ``raw``, ``start``, and ``end`` keys.
        """
        return {
            "word": self.word,
            "raw": self.raw,
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WordTimestamp:
        """Rebuild a :class:`WordTimestamp` from its serialized form.

        Args:
            data: A mapping with ``word``, ``raw``, ``start``, ``end`` keys.

        Returns:
            The reconstructed :class:`WordTimestamp`.
        """
        return cls(
            word=data["word"],
            raw=data["raw"],
            start=float(data["start"]),
            end=float(data["end"]),
        )
```

(`Any` is already imported in `transcribe.py`; `from __future__ import annotations` makes the unquoted `-> WordTimestamp` return annotation legal.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_core.py::TestWordTimestampWire -q && ruff check . && black --check . && pyright src`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/syncalong/transcribe.py tests/test_core.py
git commit -m "feat(transcribe): add WordTimestamp.to_dict/from_dict wire helpers"
```

---

### Task 3: `Transcriber` — `device` param, `separate_vocals`, and `transcribe_audio` forwarding

**Files:**
- Modify: `src/syncalong/transcribe.py` (`Transcriber.__init__`, `Transcriber.transcribe`, add `Transcriber._separate_vocals`, `transcribe_audio`)
- Test: `tests/test_core.py` (extend `TestTranscriber`; update the existing `test_transcribe_audio_delegates_to_transcriber`)

**Interfaces:**
- Consumes: `_build_transcribe_options`, `_extract_words`, `syncalong.vocal_separator.separate`.
- Produces:
  - `Transcriber.__init__(self, model_name="base", *, model=None, device=None)`
  - `Transcriber.transcribe(self, audio_path, *, language=None, initial_prompt=None, separate_vocals=False) -> list[WordTimestamp]`
  - `transcribe_audio(audio_path, *, model_name="base", language=None, initial_prompt=None, separate_vocals=False) -> list[WordTimestamp]`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_core.py` inside `class TestTranscriber` (new methods):

```python
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
```

Also **update the existing** `test_transcribe_audio_delegates_to_transcriber` so its `FakeTranscriber.transcribe` accepts the new keyword (otherwise `transcribe_audio` forwarding it raises `TypeError`). Change its `transcribe` signature and assertion:

```python
            def transcribe(
                self, audio_path, *, language=None, initial_prompt=None, separate_vocals=False
            ):
                captured["args"] = (audio_path, language, initial_prompt, separate_vocals)
                return [WordTimestamp("hi", "hi", 0.0, 0.5)]

        monkeypatch.setattr(tr, "Transcriber", FakeTranscriber)
        words = tr.transcribe_audio(
            Path("s.mp3"), model_name="small", language="de", initial_prompt="x"
        )
        assert captured["model_name"] == "small"
        assert captured["args"] == (Path("s.mp3"), "de", "x", False)
        assert words[0].word == "hi"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_core.py::TestTranscriber -q`
Expected: FAIL — new tests error on `separate_vocals`/`device`; the updated delegate test fails on the 4-tuple assertion.

- [ ] **Step 3: Implement the changes**

In `src/syncalong/transcribe.py`, replace `Transcriber.__init__` body's model-load branch and signature:

```python
    def __init__(
        self, model_name: str = "base", *, model: Any = None, device: str | None = None
    ):
        """Initialize the transcriber, loading a Whisper model if none is given.

        Args:
            model_name: Whisper model size or variant to load when ``model`` is
                ``None``. See the class docstring for accepted values.
            model: A preloaded Whisper model to use as-is instead of loading one.
            device: Torch device to load the model on (e.g. ``cuda``); ``None``
                lets Whisper choose. Ignored when ``model`` is provided.
        """
        injected = model is not None
        if model is None:
            # Heavy import, kept lazy so `import syncalong` never pulls in
            # whisper/torch. It is also deliberately absent at type-check time
            # in CI, hence the ignore.
            import whisper  # type: ignore[import-not-found]

            if device is None:
                model = whisper.load_model(model_name)
            else:
                model = whisper.load_model(model_name, device=device)
        self.model_name = None if injected else model_name
        self._model: Any = model
```

Replace `Transcriber.transcribe` with the `separate_vocals`-aware version and add the helper:

```python
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        separate_vocals: bool = False,
    ) -> list[WordTimestamp]:
        """Transcribe one audio file into word-level timestamps.

        Args:
            audio_path: Audio file (any format ffmpeg can decode).
            language: BCP-47 language code, or ``None`` to auto-detect.
            initial_prompt: Text to bias the decoder with (e.g. the lyrics).
            separate_vocals: When ``True``, isolate vocals with Demucs first.

        Returns:
            Every recognised word, in order, with start/end times in seconds.

        Raises:
            ModuleNotFoundError: If ``separate_vocals`` is ``True`` but the
                optional ``demucs`` dependency is not installed.
        """
        if separate_vocals:
            audio_path = self._separate_vocals(Path(audio_path))
        opts = _build_transcribe_options(
            language=language, initial_prompt=initial_prompt
        )
        result = self._model.transcribe(str(audio_path), **opts)
        return _extract_words(result)

    @staticmethod
    def _separate_vocals(audio_path: Path) -> Path:
        import importlib.util

        if importlib.util.find_spec("demucs") is None:
            raise ModuleNotFoundError(
                "separate_vocals=True requires the optional 'demucs' dependency. "
                "Install it with: pip install syncalong[vocal-separation]"
            )
        from syncalong.vocal_separator import separate

        return separate(audio_path)
```

Replace `transcribe_audio` to forward the flag:

```python
def transcribe_audio(
    audio_path: Path,
    *,
    model_name: str = "base",
    language: str | None = None,
    initial_prompt: str | None = None,
    separate_vocals: bool = False,
) -> list[WordTimestamp]:
    """Transcribe one audio file, loading the model for this call only.

    Convenience wrapper around :class:`Transcriber` for one-shot use. A
    consumer that transcribes many files should instantiate a
    :class:`Transcriber` once and reuse it instead.

    Args:
        audio_path: Audio file (any format ffmpeg can decode).
        model_name: Whisper model size or variant.
        language: BCP-47 language code, or ``None`` to auto-detect.
        initial_prompt: Text to bias the decoder with (e.g. the lyrics).
        separate_vocals: When ``True``, isolate vocals with Demucs first.

    Returns:
        Ordered word timestamps.
    """
    return Transcriber(model_name).transcribe(
        audio_path,
        language=language,
        initial_prompt=initial_prompt,
        separate_vocals=separate_vocals,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_core.py::TestTranscriber -q && ruff check . && black --check . && pyright src`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/syncalong/transcribe.py tests/test_core.py
git commit -m "feat(transcribe): add device + separate_vocals to Transcriber, forward from transcribe_audio"
```

---

### Task 4: `pipeline.align()` forwards `separate_vocals` (remove local Demucs branch)

**Files:**
- Modify: `src/syncalong/pipeline.py` (delete `_separate_vocals`; forward the flag in `align`)
- Test: `tests/test_core.py` (`TestAlignFacade` — add a forwarding test; the existing demucs test stays valid)

**Interfaces:**
- Consumes: any transcriber implementing `.transcribe(..., separate_vocals=...)` (Task 3 for local; Task 5 for remote).
- Produces: `align(...)` signature unchanged; `separate_vocals` now handled by the transcriber.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_core.py` inside `class TestAlignFacade`:

```python
    def test_forwards_separate_vocals_to_transcriber(self):
        captured = {}

        class RecordingTranscriber:
            def transcribe(
                self, audio_path, *, language=None, initial_prompt=None, separate_vocals=False
            ):
                captured["separate_vocals"] = separate_vocals
                return [WordTimestamp("hello", "hello", 1.0, 1.5)]

        align("hello\n", "s.mp3", transcriber=RecordingTranscriber(), separate_vocals=True)
        assert captured["separate_vocals"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_core.py::TestAlignFacade::test_forwards_separate_vocals_to_transcriber -q`
Expected: FAIL — current `align()` intercepts `separate_vocals` via `_separate_vocals` and never forwards it (`KeyError: 'separate_vocals'`).

- [ ] **Step 3: Implement**

In `src/syncalong/pipeline.py`:

1. **Delete** the entire `_separate_vocals` function (currently lines ~62–83).
2. In `align()`, **delete** the block:
   ```python
       if separate_vocals:
           audio_path = _separate_vocals(audio_path)
   ```
3. Change the transcribe call to forward the flag:
   ```python
       transcript = transcriber.transcribe(
           audio_path,
           language=language,
           initial_prompt=prompt,
           separate_vocals=separate_vocals,
       )
   ```

The `align` docstring's `Raises: ModuleNotFoundError` line stays accurate (the transcriber now raises it). No signature change.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_core.py::TestAlignFacade -q && ruff check . && black --check . && pyright src`
Expected: PASS — including the pre-existing `test_separate_vocals_without_demucs_raises` (it injects a fake-model `Transcriber`, so `align` forwards `separate_vocals=True` and `Transcriber._separate_vocals` raises `ModuleNotFoundError`).

- [ ] **Step 5: Commit**

```bash
git add src/syncalong/pipeline.py tests/test_core.py
git commit -m "refactor(pipeline): forward separate_vocals to the transcriber (unify demucs path)"
```

---

### Task 5: `RemoteTranscriber` — stdlib HTTP client

**Files:**
- Create: `src/syncalong/remote.py`
- Test: `tests/test_remote.py`

**Interfaces:**
- Consumes: `WordTimestamp.from_dict` (Task 2). Talks to `POST {base_url}/transcribe` (Task 6).
- Produces:
  - `RemoteTranscriber(base_url, *, token=None, timeout=300.0)` with `.model_name = None`
  - `RemoteTranscriber.transcribe(audio_path, *, language=None, initial_prompt=None, separate_vocals=False) -> list[WordTimestamp]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_remote.py`:

```python
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

    def test_sends_bearer_token(self, tmp_path, monkeypatch):
        audio = tmp_path / "s.mp3"
        audio.write_bytes(b"\x00")
        capture = {}
        _patch_urlopen(monkeypatch, {"words": []}, capture)

        RemoteTranscriber("http://gpu:8000", token="secret").transcribe(audio)

        assert capture["request"].headers["Authorization"] == "Bearer secret"

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
        with pytest.raises(RuntimeError, match="HTTP 401"):
            RemoteTranscriber("http://gpu:8000").transcribe(audio)

    def test_unreachable_becomes_runtimeerror(self, tmp_path, monkeypatch):
        audio = tmp_path / "s.mp3"
        audio.write_bytes(b"\x00")

        def raise_url(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(remote.urllib.request, "urlopen", raise_url)
        with pytest.raises(RuntimeError, match="could not reach"):
            RemoteTranscriber("http://gpu:8000").transcribe(audio)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_remote.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'syncalong.remote'`.

- [ ] **Step 3: Implement `src/syncalong/remote.py`**

```python
"""Remote transcription client for a syncalong server.

:class:`RemoteTranscriber` implements the same ``transcribe`` interface as
:class:`syncalong.transcribe.Transcriber`, but performs the work on a remote
GPU server over HTTP. It uses only the Python standard library, so a thin
client can ``pip install syncalong`` without pulling in Whisper/PyTorch.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from syncalong.transcribe import WordTimestamp


def _encode_multipart(
    fields: dict[str, str | None],
    filename: str,
    file_bytes: bytes,
) -> tuple[bytes, str]:
    """Encode text fields and one file as a multipart/form-data body.

    Args:
        fields: Text form fields; entries whose value is ``None`` are omitted.
        filename: Filename for the ``audio`` file part.
        file_bytes: Raw bytes of the audio file.

    Returns:
        A ``(body, boundary)`` tuple: the encoded body and the boundary token
        for the ``Content-Type`` header.
    """
    boundary = uuid.uuid4().hex
    marker = f"--{boundary}".encode()
    body = bytearray()
    for name, value in fields.items():
        if value is None:
            continue
        body += marker + b"\r\n"
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += value.encode() + b"\r\n"
    body += marker + b"\r\n"
    body += (
        f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'
    ).encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    body += file_bytes + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), boundary


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        text = exc.read().decode("utf-8")
    except Exception:
        return exc.reason or "unknown error"
    try:
        parsed = json.loads(text)
    except ValueError:
        return text or (exc.reason or "unknown error")
    if isinstance(parsed, dict) and "detail" in parsed:
        return str(parsed["detail"])
    return text


class RemoteTranscriber:
    """Transcribe audio via a remote syncalong server.

    Drop-in replacement for :class:`syncalong.transcribe.Transcriber` that sends
    the audio to a GPU server and returns the word timestamps it computes. The
    model is chosen by the server, so this class takes no ``model_name``.

    Args:
        base_url: Base URL of the server, e.g. ``http://gpu-box:8000``.
        token: Optional shared token sent as ``Authorization: Bearer <token>``.
        timeout: Socket timeout in seconds for the request.

    Attributes:
        base_url: The normalized server base URL (no trailing slash).
        token: The configured bearer token, if any.
        timeout: The socket timeout in seconds.
        model_name: Always ``None`` — the server decides the model.
    """

    def __init__(
        self, base_url: str, *, token: str | None = None, timeout: float = 300.0
    ):
        """Initialize the remote transcriber.

        Args:
            base_url: Base URL of the server, e.g. ``http://gpu-box:8000``.
            token: Optional shared token for ``Authorization: Bearer``.
            timeout: Socket timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.model_name: str | None = None

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        separate_vocals: bool = False,
    ) -> list[WordTimestamp]:
        """Transcribe one audio file on the remote server.

        Args:
            audio_path: Audio file to upload (any format the server's ffmpeg
                can decode).
            language: BCP-47 language code, or ``None`` to auto-detect.
            initial_prompt: Text to bias the decoder with (e.g. the lyrics).
            separate_vocals: Ask the server to isolate vocals (Demucs) first.

        Returns:
            Ordered word timestamps computed by the server.

        Raises:
            RuntimeError: If the server is unreachable or returns an error.
        """
        audio_path = Path(audio_path)
        fields: dict[str, str | None] = {
            "language": language,
            "initial_prompt": initial_prompt,
            "separate_vocals": "true" if separate_vocals else "false",
        }
        body, boundary = _encode_multipart(
            fields, audio_path.name, audio_path.read_bytes()
        )
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        url = f"{self.base_url}/transcribe"
        request = urllib.request.Request(
            url, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"syncalong server returned HTTP {exc.code}: {_error_detail(exc)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"could not reach syncalong server at {url}: {exc.reason}"
            ) from exc
        return [WordTimestamp.from_dict(word) for word in payload["words"]]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_remote.py -q && ruff check . && black --check . && pyright src`
Expected: PASS. (`pyright src` covers `remote.py`; it is stdlib-only.)

- [ ] **Step 5: Commit**

```bash
git add src/syncalong/remote.py tests/test_remote.py
git commit -m "feat(remote): add stdlib-only RemoteTranscriber HTTP client"
```

---

### Task 6: `server.py` — FastAPI transcription service

**Files:**
- Create: `src/syncalong/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `Transcriber` (Task 3), `WordTimestamp.to_dict` (Task 2).
- Produces:
  - `create_app(transcriber=None, *, token=None, model_name="base", device=None, allow_vocal_separation=True) -> FastAPI`
  - Module-level `app = create_app()`
  - Endpoints `GET /health`, `POST /transcribe` (used by Task 5's client and Task 7's `serve_main`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_server.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'syncalong.server'`.

- [ ] **Step 3: Implement `src/syncalong/server.py`**

```python
"""HTTP server exposing Whisper transcription for thin syncalong clients.

Run with the ``syncalong-serve`` console script (installed by the ``server``
extra) or ``uvicorn syncalong.server:app``. The heavy Whisper model is loaded
lazily on first use, so importing this module stays torch-free and testable
with an injected fake transcriber.
"""

from __future__ import annotations

import hmac
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from syncalong.transcribe import Transcriber


def create_app(
    transcriber: Any = None,
    *,
    token: str | None = None,
    model_name: str = "base",
    device: str | None = None,
    allow_vocal_separation: bool = True,
) -> FastAPI:
    """Build the FastAPI transcription application.

    Args:
        transcriber: A preloaded object implementing ``transcribe(...)``. When
            ``None``, a :class:`~syncalong.transcribe.Transcriber` is built
            lazily on the first request from ``model_name`` / ``device`` (or the
            ``SYNCALONG_MODEL`` / ``SYNCALONG_DEVICE`` env vars). Injecting a
            fake keeps tests torch-free.
        token: Optional shared token. When set (or ``SYNCALONG_TOKEN`` is set),
            requests must send ``Authorization: Bearer <token>``.
        model_name: Whisper model to load lazily when ``transcriber`` is None.
        device: Torch device for the lazily-loaded model (e.g. ``cuda``).
        allow_vocal_separation: When ``False``, reject ``separate_vocals``
            requests with HTTP 400.

    Returns:
        The configured :class:`fastapi.FastAPI` application.
    """
    app = FastAPI(title="syncalong transcription server")
    state: dict[str, Any] = {"transcriber": transcriber}

    def _get_transcriber() -> Any:
        if state["transcriber"] is None:
            state["transcriber"] = Transcriber(
                os.environ.get("SYNCALONG_MODEL", model_name),
                device=os.environ.get("SYNCALONG_DEVICE", device),
            )
        return state["transcriber"]

    def _check_auth(authorization: str | None) -> None:
        configured = token or os.environ.get("SYNCALONG_TOKEN")
        if not configured:
            return
        expected = f"Bearer {configured}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid or missing token")

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Report liveness and the configured model name."""
        return {
            "status": "ok",
            "model": getattr(state["transcriber"], "model_name", None),
        }

    @app.post("/transcribe")
    async def transcribe(
        audio: UploadFile = File(...),
        language: Optional[str] = Form(None),
        initial_prompt: Optional[str] = Form(None),
        separate_vocals: bool = Form(False),
        authorization: Optional[str] = Header(None),
    ) -> dict[str, Any]:
        """Transcribe an uploaded audio file into word timestamps."""
        _check_auth(authorization)
        if separate_vocals and not allow_vocal_separation:
            raise HTTPException(
                status_code=400, detail="vocal separation is disabled on this server"
            )
        suffix = Path(audio.filename or "audio").suffix
        data = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            words = _get_transcriber().transcribe(
                tmp_path,
                language=language,
                initial_prompt=initial_prompt,
                separate_vocals=separate_vocals,
            )
        except ModuleNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            tmp_path.unlink(missing_ok=True)
        return {
            "words": [word.to_dict() for word in words],
            "model": getattr(state["transcriber"], "model_name", None),
        }

    return app


app = create_app()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_server.py -q && ruff check . && black --check . && pyright src`
Expected: PASS. (`pyright src` resolves `fastapi` because it's in the `dev` extra from Task 1.)

Also confirm the thin-client guarantee still holds:
Run: `python -c "import syncalong, sys; assert 'fastapi' not in sys.modules; print('ok')"`
Expected: `ok` (importing `syncalong` must not import `server.py`/`fastapi`).

- [ ] **Step 5: Commit**

```bash
git add src/syncalong/server.py tests/test_server.py
git commit -m "feat(server): add FastAPI transcription service with optional token auth"
```

---

### Task 7: CLI — `--server`/`--token` remote mode, whisper guard, `syncalong-serve`

**Files:**
- Modify: `src/syncalong/cli.py` (imports, `build_parser`, `main`, add `_whisper_available`, add `serve_main`)
- Modify: `pyproject.toml:69-70` (`[project.scripts]`)
- Test: `tests/test_core.py` (new `TestCliRemote`, `TestServeMain`)

**Interfaces:**
- Consumes: `RemoteTranscriber` (Task 5), `create_app` (Task 6), `Transcriber` (Task 3).
- Produces: `_whisper_available() -> bool`; `serve_main(argv=None) -> None`; CLI flags `--server`, `--token`; console script `syncalong-serve`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_core.py`:

```python
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
                self, audio_path, *, language=None, initial_prompt=None, separate_vocals=False
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
            [str(lyr), str(aud), "--server", "http://gpu:8000", "--token", "t",
             "--separate-vocals"]
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
            ["-m", "small", "--host", "0.0.0.0", "--port", "9000",
             "--token", "sek", "--no-vocal-separation"]
        )
        assert recorded["model"] == "small"
        assert recorded["token"] == "sek"
        assert recorded["allow_sep"] is False
        assert recorded["run"] == (fake_app, "0.0.0.0", 9000)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_core.py::TestCliRemote tests/test_core.py::TestServeMain -q`
Expected: FAIL — `--server` is an unrecognized argument; `cli.serve_main` does not exist.

- [ ] **Step 3: Implement the CLI changes**

In `src/syncalong/cli.py`, add `import os` to the imports (after `import importlib.util`).

Add the two new parser arguments inside `build_parser()` (before `return parser`):

```python
    parser.add_argument(
        "--server",
        default=None,
        metavar="URL",
        help=(
            "Transcribe on a remote syncalong server instead of locally "
            "(e.g. http://gpu-box:8000). Falls back to $SYNCALONG_SERVER."
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        metavar="TOKEN",
        help="Bearer token for the remote server. Falls back to $SYNCALONG_TOKEN.",
    )
```

Add the whisper-availability helper (next to `_demucs_available`):

```python
def _whisper_available() -> bool:
    """Report whether the optional ``openai-whisper`` package is importable.

    Returns:
        ``True`` if ``whisper`` is installed, else ``False``.
    """
    return importlib.util.find_spec("whisper") is not None
```

In `main()`, replace the transcription section (currently lines ~155–170, from the `# --- Optional vocal separation` comment through the `transcribe_audio(...)` call) with:

```python
    server = args.server or os.environ.get("SYNCALONG_SERVER")
    token = args.token or os.environ.get("SYNCALONG_TOKEN")
    prompt = None if args.no_lyrics_prompt else lyrics_prompt(lyric_lines)

    # --- Transcribe (remote server or local Whisper) -----------------------
    if server:
        if args.model != "base":
            print(
                "note: -m/--model is ignored in remote mode; the server "
                "chooses the model.",
                file=sys.stderr,
            )
        print(f"Transcribing on remote server {server} …", file=sys.stderr)
        from syncalong.remote import RemoteTranscriber

        transcriber = RemoteTranscriber(server, token=token)
        try:
            word_timestamps = transcriber.transcribe(
                args.audio,
                language=args.language,
                initial_prompt=prompt,
                separate_vocals=args.separate_vocals,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        audio_path = resolve_audio_path(args.audio, args.separate_vocals)
        if not _whisper_available():
            print(
                "ERROR: local transcription requires Whisper.\n"
                "Install it with:  pip install syncalong[whisper]\n"
                "…or transcribe on a GPU server with --server URL.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"Transcribing with Whisper ({args.model}) …  "
            "(this may take a while on CPU)",
            file=sys.stderr,
        )
        word_timestamps = transcribe_audio(
            audio_path,
            model_name=args.model,
            language=args.language,
            initial_prompt=prompt,
        )
```

(The old `prompt = None if args.no_lyrics_prompt else lyrics_prompt(lyric_lines)` line that lived just above `transcribe_audio` is now folded into the block above — make sure it isn't duplicated.)

Add `serve_main` at the end of `cli.py`:

```python
def serve_main(argv: list[str] | None = None) -> None:
    """Run the syncalong transcription server (``syncalong-serve``).

    Loads the Whisper model once, then serves it over HTTP so thin clients can
    transcribe without a local GPU. Requires the ``server`` extra.

    Args:
        argv: Argument vector to parse; defaults to ``sys.argv`` when ``None``.
    """
    try:
        import uvicorn

        from syncalong.server import create_app
    except ImportError:
        print(
            "The transcription server requires extra dependencies.\n"
            "Install them with:  pip install syncalong[server]",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    parser = argparse.ArgumentParser(
        prog="syncalong-serve",
        description="Serve Whisper transcription over HTTP for syncalong clients.",
    )
    parser.add_argument(
        "-m", "--model", default="base", help="Whisper model name (default: base)."
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port (default: 8000)."
    )
    parser.add_argument(
        "--device", default=None, help="Torch device, e.g. 'cuda' (default: auto)."
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Require this bearer token. Falls back to $SYNCALONG_TOKEN.",
    )
    parser.add_argument(
        "--no-vocal-separation",
        action="store_true",
        help="Reject separate_vocals requests (don't run Demucs).",
    )
    args = parser.parse_args(argv)

    token = args.token or os.environ.get("SYNCALONG_TOKEN")

    from syncalong.transcribe import Transcriber

    print(f"Loading Whisper model '{args.model}' …", file=sys.stderr)
    transcriber = Transcriber(args.model, device=args.device)
    app = create_app(
        transcriber,
        token=token,
        allow_vocal_separation=not args.no_vocal_separation,
    )
    print(f"Serving on http://{args.host}:{args.port} …", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)
```

- [ ] **Step 4: Register the console script**

In `pyproject.toml`, update `[project.scripts]` (lines 69–70):

```toml
[project.scripts]
syncalong = "syncalong.cli:main"
syncalong-serve = "syncalong.cli:serve_main"
```

Run: `pip install -e ".[dev,docs]"`  (re-registers the new console script)
Expected: succeeds.

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_core.py::TestCliRemote tests/test_core.py::TestServeMain -q && ruff check . && black --check . && pyright src`
Expected: PASS.

Run: `syncalong-serve --help`
Expected: prints the server usage help (exit 0).

- [ ] **Step 6: Commit**

```bash
git add src/syncalong/cli.py pyproject.toml tests/test_core.py
git commit -m "feat(cli): add --server remote mode, whisper guard, and syncalong-serve"
```

---

### Task 8: Public API export + documentation

**Files:**
- Modify: `src/syncalong/__init__.py` (export `RemoteTranscriber`)
- Modify: `tests/test_core.py` (`TestPublicAPI` — add `RemoteTranscriber`; assert fastapi not loaded)
- Create: `docs/reference/remote.md`
- Create: `docs/remote.md`
- Modify: `mkdocs.yml` (nav)
- Modify: `docs/installation.md`, `docs/reference/index.md`, `README.md`, `CHANGELOG.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: `RemoteTranscriber` (Task 5).
- Produces: `syncalong.RemoteTranscriber` in `__all__`; docs.

- [ ] **Step 1: Write the failing test**

In `tests/test_core.py`, add `"RemoteTranscriber"` to the list in `TestPublicAPI.test_top_level_exports_present` (add it under the transcription names, e.g. after `"WordTimestamp"`). Then extend the whisper-guard test to also assert fastapi isn't loaded:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest "tests/test_core.py::TestPublicAPI::test_top_level_exports_present" -q`
Expected: FAIL — `RemoteTranscriber missing from __all__`.

- [ ] **Step 3: Export `RemoteTranscriber`**

In `src/syncalong/__init__.py`, add the import (after the `transcribe` import line):

```python
from syncalong.remote import RemoteTranscriber
```

And in `__all__`, under the `# Transcription` group, add `"RemoteTranscriber"`:

```python
    # Transcription
    "Transcriber",
    "RemoteTranscriber",
    "transcribe_audio",
    "WordTimestamp",
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_core.py::TestPublicAPI -q`
Expected: PASS.

- [ ] **Step 5: Write the docs pages**

Create `docs/reference/remote.md`:

```markdown
# `syncalong.remote`

Client-side transcription over HTTP.
[`RemoteTranscriber`](#syncalong.remote.RemoteTranscriber) is a drop-in
replacement for
[`Transcriber`](transcribe.md#syncalong.transcribe.Transcriber) that runs
Whisper on a remote GPU server, so a thin client needs neither Whisper nor a
GPU. It uses only the Python standard library.

::: syncalong.remote
```

Create `docs/remote.md`:

````markdown
# Remote transcription (server/client)

syncalong's only GPU-heavy stage is transcription (Whisper, plus optional
Demucs). You can run that stage on a powerful machine and keep the music,
lyrics, and the cheap alignment on a low-powered client — e.g. a jukebox with
all the audio but no GPU.

## Install matrix

| Machine | Install | Gets |
| --- | --- | --- |
| **Client** (jukebox) | `pip install syncalong` | parsing, alignment, LRC, and `RemoteTranscriber` — no PyTorch |
| **Client, local Whisper too** | `pip install "syncalong[whisper]"` | the above + local `Transcriber` |
| **Server** (GPU box) | `pip install "syncalong[server]"` | Whisper + the FastAPI server |
| **Server, with vocal isolation** | `pip install "syncalong[server,vocal-separation]"` | + Demucs |

`ffmpeg` is required on the **server** (Whisper decodes audio there).

## Start the server (GPU box)

```bash
syncalong-serve --model medium --host 0.0.0.0 --port 8000 --device cuda
```

Options: `--model`, `--host`, `--port`, `--device` (e.g. `cuda`), `--token`
(require a bearer token; falls back to `$SYNCALONG_TOKEN`),
`--no-vocal-separation` (reject `separate_vocals` requests). The Whisper model
loads once at startup and is reused for every request.

You can also run it with any ASGI server:

```bash
SYNCALONG_MODEL=medium uvicorn syncalong.server:app --host 0.0.0.0 --port 8000
```

## Transcribe from the client (jukebox)

CLI:

```bash
syncalong lyrics.txt song.mp3 --server http://gpu-box:8000 > song.lrc
# with a token and server-side vocal isolation:
syncalong lyrics.txt song.mp3 --server http://gpu-box:8000 \
    --token "$SYNCALONG_TOKEN" --separate-vocals > song.lrc
```

`--server` also reads `$SYNCALONG_SERVER`. In remote mode the **server**
chooses the Whisper model, so `-m/--model` is ignored.

Library:

```python
import syncalong

tx = syncalong.RemoteTranscriber("http://gpu-box:8000", token="…")
res = syncalong.align("lyrics.txt", "song.mp3", transcriber=tx)
print(res.lrc)
```

`RemoteTranscriber` implements the same interface as `Transcriber`, so it drops
into `align(transcriber=…)` unchanged — alignment still runs locally on the
client.

## What travels over the network

The audio file is uploaded to the server; the server returns only word
timestamps (`{word, raw, start, end}`). Your lyrics **file** and the whole
alignment stay on the client.

!!! note "Privacy: the lyrics prompt"
    By default Whisper is biased with a prompt built **from your lyrics**, and
    that prompt is sent to the server. To send **audio only** — zero lyric text
    over the wire — pass `--no-lyrics-prompt` (CLI) or `use_lyrics_prompt=False`
    (library), at some accuracy cost.

## Auth & exposure

Set a shared token to require `Authorization: Bearer <token>`:

```bash
syncalong-serve --token "$(openssl rand -hex 16)"
```

There is no TLS in-process. On a trusted LAN, bind to a private interface. If
you expose the server beyond the LAN, put it behind a TLS-terminating reverse
proxy (Caddy, nginx) and keep the token set.
````

- [ ] **Step 6: Wire the docs into the nav and update references**

In `mkdocs.yml`, add the guide page after `How it works` and the reference page under `API reference`:

```yaml
  - How it works: how-it-works.md
  - Remote (server/client): remote.md
```

```yaml
      - syncalong.transcribe: reference/transcribe.md
      - syncalong.remote: reference/remote.md
```

In `docs/reference/index.md`, add a row to the "Where to start" table and a bullet to the "Transcription" line:

```markdown
| Transcribe on a remote GPU server | [`RemoteTranscriber`](remote.md#syncalong.remote.RemoteTranscriber) |
```

```markdown
- **Transcription:** `Transcriber`, `RemoteTranscriber`, `transcribe_audio`, `WordTimestamp`
```

In `docs/installation.md`, replace the "Install from PyPI" info section so it reflects the thin-client default. Replace the paragraph + admonition after ````pip install syncalong```` with:

```markdown
This installs the library and the `syncalong` CLI with a **thin** dependency
set ([rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) only) — enough to
parse lyrics, align a transcript, write LRC, and talk to a remote server.

## Local transcription (Whisper)

To transcribe **on this machine**, add the `whisper` extra:

```bash
pip install "syncalong[whisper]"
```

!!! info "Whisper pulls in PyTorch"
    `openai-whisper` depends on PyTorch, a large download. With a CUDA-capable
    GPU, install a matching PyTorch build first (see the
    [PyTorch install guide](https://pytorch.org/get-started/locally/)) and
    Whisper will use the GPU automatically.

!!! tip "No GPU on this machine?"
    Run transcription on a separate GPU box instead — see
    [Remote transcription](remote.md). The client stays torch-free.
```

- [ ] **Step 7: Update CHANGELOG, README, and CLAUDE.md**

In `CHANGELOG.md`, replace the `## [Unreleased]` line with:

```markdown
## [Unreleased]

### Added

- **Remote transcription (server/client).** `syncalong-serve` (FastAPI) runs
  Whisper on a GPU box; a stdlib-only `RemoteTranscriber` on a thin client
  uploads audio and aligns locally. New `--server`/`--token` CLI flags
  (also `$SYNCALONG_SERVER` / `$SYNCALONG_TOKEN`), a `server` optional extra,
  and optional shared-token auth. Vocal separation runs server-side via the
  `separate_vocals` request flag.

### Changed

- **BREAKING:** `openai-whisper` moved from a core dependency to the new
  `whisper` extra. `pip install syncalong` is now thin (torch-free); install
  `syncalong[whisper]` for local transcription, or `syncalong[server]` to run
  the server. Existing local CLI use now needs `pip install "syncalong[whisper]"`.
```

In `README.md`, after the "Optional: vocal separation" subsection, add:

```markdown
### Remote transcription (no local GPU)

Run Whisper on a GPU machine and keep the audio + lyrics on a thin client:

```bash
# on the GPU box
pip install "syncalong[server]"
syncalong-serve --model medium --host 0.0.0.0 --device cuda

# on the client (torch-free `pip install syncalong`)
syncalong lyrics.txt song.mp3 --server http://gpu-box:8000 > song.lrc
```

See the [remote transcription guide](https://syncalong.readthedocs.io/en/latest/remote/).
```

Also update the README "Installation" note that a bare install pulls Whisper — change the sentence under the first `pip install syncalong` block to note that local transcription needs the `whisper` extra (`pip install "syncalong[whisper]"`) and point to the remote guide for GPU-server use.

In `CLAUDE.md`, add `remote.py` and `server.py` to the Architecture module list, note the `whisper`/`server` extras under Dependencies, and add the `syncalong-serve` / `--server` commands under Commands. (Concretely: a `remote.py` bullet — "stdlib-only `RemoteTranscriber`, same `.transcribe()` interface"; a `server.py` bullet — "FastAPI `create_app` + `/health` + `/transcribe`, model loaded lazily/once"; and an extras note that whisper is now optional.)

- [ ] **Step 8: Build docs and run the full gate**

Run: `mkdocs build --strict`
Expected: builds with no warnings.

Run: `python -m pytest tests/ -q && ruff check . && black --check . && pyright src`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/syncalong/__init__.py tests/test_core.py docs/ mkdocs.yml README.md CHANGELOG.md CLAUDE.md
git commit -m "feat: export RemoteTranscriber and document remote transcription"
```

---

## Self-Review

**Spec coverage** (each spec section → task):
- Thin GPU service / same `.transcribe()` seam → Tasks 5, 4.
- Whisper → `[whisper]` extra; `[server]` extra; dev fastapi/httpx/uvicorn → Task 1.
- Server-side Demucs via request flag → Tasks 3 (interface), 6 (endpoint + 400s).
- FastAPI + uvicorn; stdlib client → Tasks 6, 5.
- Synchronous request/response → Task 6 (blocking endpoint).
- Optional shared-token auth → Task 6 (`_check_auth`), Task 7 (client/server flags).
- Wire format + `to_dict`/`from_dict` → Task 2; endpoint contract → Task 6.
- `align()` forwards `separate_vocals` → Task 4.
- CLI `--server`/`--token`/env, `-m` note, whisper-missing hint, `syncalong-serve` → Task 7.
- Public API export; `import syncalong` loads neither whisper nor fastapi → Task 8.
- Error handling table (unreachable, non-2xx, 401, demucs-missing 400, whisper hint, serve without extra) → Tasks 5, 6, 7.
- Testing torch-free (fake model/transcriber, stubbed urlopen, TestClient) → Tasks 5, 6, 7.
- Docs + CHANGELOG breaking note + README + CLAUDE.md → Task 8.
- Privacy note documented honestly → Task 8 (`docs/remote.md`).

**Placeholder scan:** none — every code step shows complete code; every run step states the expected result.

**Type consistency:** `transcribe(self, audio_path, *, language=None, initial_prompt=None, separate_vocals=False) -> list[WordTimestamp]` is identical across `Transcriber` (Task 3), `RemoteTranscriber` (Task 5), and the fakes (Tasks 6–7). `to_dict`/`from_dict` (Task 2) are consumed with matching keys by server (Task 6) and client (Task 5). `create_app(...)` kwargs used in Task 7's `serve_main` match Task 6's definition. `_whisper_available()`/`serve_main()` defined and used in Task 7.
