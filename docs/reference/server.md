# `syncalong.server`

The FastAPI application that exposes Whisper transcription over HTTP for thin
[`RemoteTranscriber`](remote.md#syncalong.remote.RemoteTranscriber) clients.

Start it with the `syncalong-serve` console script or
`uvicorn syncalong.server:app`; both are installed by the `server` extra
(`pip install "syncalong[server]"`). This module is imported only on the server
side, so `import syncalong` never pulls in FastAPI.

[`create_app()`](#syncalong.server.create_app) builds and returns the app, so
you can mount it in a larger ASGI application or set the token, model, device,
and vocal-separation policy in code. For the wire protocol (endpoints, request
fields, response schema, and status codes), see the
[HTTP API](../remote.md#http-api) section of the remote guide.

!!! info "New in syncalong 2.0"
    The transcription server is available from **version 2.0** onward.

::: syncalong.server
