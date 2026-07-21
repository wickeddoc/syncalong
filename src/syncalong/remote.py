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
    """Extract a server-provided error message from an HTTP error.

    Args:
        exc: The HTTP error raised by ``urlopen``.

    Returns:
        The server's ``detail`` string when the body is JSON with that key,
        otherwise the raw body text or the HTTP reason phrase.
    """
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
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
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
