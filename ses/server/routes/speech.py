from fastapi.responses import Response, StreamingResponse
from ... import SesError
from ...audio import mp3_bytes, pcm16_bytes, resample, wav_bytes
from ...core.text import split_sentences
from ..schemas import SpeechRequest

MAX_INPUT_CHARS = 10_000
STREAM_SAMPLE_RATE = 24000


def register(app, context):
    loader = context.loader

    @app.post("/v1/audio/speech")
    def speech(request: SpeechRequest):
        if len(request.input) > MAX_INPUT_CHARS:
            raise SesError(f"input too long ({len(request.input)} chars, max {MAX_INPUT_CHARS})")

        model = loader.get(request.model or context.default_model("tts"))
        if model.kind != "tts":
            raise SesError(f"'{request.model}' is not a TTS model, try 'tts-english'")

        def synth(text):
            with context.infer_lock:
                return model.engine.synth(
                    text, voice=request.voice, speed=request.speed, lang=request.lang
                )

        if request.stream:

            def sentence_stream():
                for sentence in split_sentences(request.input):
                    samples, rate = synth(sentence)
                    if rate != STREAM_SAMPLE_RATE:
                        samples = resample(samples, rate, STREAM_SAMPLE_RATE)
                    yield pcm16_bytes(samples)

            return StreamingResponse(
                sentence_stream(),
                media_type="audio/pcm",
                headers={"X-Sample-Rate": str(STREAM_SAMPLE_RATE)},
            )

        samples, rate = synth(request.input)
        response_format = request.response_format.lower()
        if response_format == "wav":
            return Response(wav_bytes(samples, rate), media_type="audio/wav")
        if response_format == "pcm":
            return Response(
                pcm16_bytes(samples),
                media_type="audio/pcm",
                headers={"X-Sample-Rate": str(rate)},
            )
        if response_format == "mp3":
            return Response(mp3_bytes(samples, rate), media_type="audio/mpeg")
        raise SesError(f"unsupported response_format '{response_format}' (wav, pcm, mp3)")
