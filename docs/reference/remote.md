# `syncalong.remote`

Client-side transcription over HTTP.
[`RemoteTranscriber`](#syncalong.remote.RemoteTranscriber) is a drop-in
replacement for
[`Transcriber`](transcribe.md#syncalong.transcribe.Transcriber) that runs
Whisper on a remote GPU server, so a thin client needs neither Whisper nor a
GPU. It uses only the Python standard library.

!!! info "New in syncalong 2.0"
    `RemoteTranscriber` is available from **version 2.0** onward.

::: syncalong.remote
