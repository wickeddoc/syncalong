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
        audio: UploadFile = File(...),  # noqa: B008
        # Optional[...] (not `X | None`) is required here: FastAPI evaluates
        # these annotations at runtime via typing.get_type_hints() to build
        # its request model, and PEP 604's `X | None` runtime `|` operator on
        # types only exists on Python 3.10+, which would break this project's
        # declared `requires-python = ">=3.9"`.
        language: Optional[str] = Form(None),  # noqa: UP045
        initial_prompt: Optional[str] = Form(None),  # noqa: UP045
        separate_vocals: bool = Form(False),
        authorization: Optional[str] = Header(None),  # noqa: UP045
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
