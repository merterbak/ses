from fastapi import Form, UploadFile
from fastapi.responses import PlainTextResponse
from ... import SesError
from ...audio import WHISPER_SAMPLE_RATE, load_audio
from ...core import transcripts
from ...core.text import clean


def register(app, context):
    loader = context.loader

    def run(upload, model, language, response_format, temperature, prompt, task):
        loaded_model = loader.get(model or context.default_model("stt"))
        if loaded_model.kind != "stt":
            raise SesError(f"'{model}' is not a speech-to-text model, try 'whisper-base'")

        raw = upload.file.read()
        if not raw:
            raise SesError("uploaded file is empty")

        samples = load_audio(raw)
        duration = len(samples) / WHISPER_SAMPLE_RATE
        response_format = response_format.lower()

        with context.infer_lock:
            result = clean(
                loaded_model.engine.transcribe(
                    samples,
                    language=language or None,
                    task=task,
                    word_timestamps=response_format == "verbose_json",
                    temperature=temperature,
                    initial_prompt=prompt,
                )
            )

        if response_format == "json":
            return {"text": result["text"]}
        if response_format == "text":
            return PlainTextResponse(result["text"] + "\n")
        if response_format == "verbose_json":
            return transcripts.to_verbose_json(result, duration)
        if response_format == "srt":
            return PlainTextResponse(transcripts.to_srt(result.get("segments", [])))
        if response_format == "vtt":
            return PlainTextResponse(transcripts.to_vtt(result.get("segments", [])))
        raise SesError(
            f"unsupported response_format '{response_format}' "
            "(json, text, verbose_json, srt, vtt)"
        )

    @app.post("/v1/audio/transcriptions")
    def transcriptions(
        file: UploadFile,
        model: str = Form(""),
        language: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float | None = Form(None),
        prompt: str | None = Form(None),
    ):
        return run(file, model, language, response_format, temperature, prompt, "transcribe")

    @app.post("/v1/audio/translations")
    def translations(
        file: UploadFile,
        model: str = Form(""),
        response_format: str = Form("json"),
        temperature: float | None = Form(None),
        prompt: str | None = Form(None),
    ):
        return run(file, model, None, response_format, temperature, prompt, "translate")
