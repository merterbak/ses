class FasterWhisperEngine:
    kind = "stt"

    def __init__(self, model_dir):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(str(model_dir), device="auto", compute_type="auto")

    def transcribe(
        self,
        samples,
        language=None,
        task="transcribe",
        word_timestamps=False,
        temperature=None,
        initial_prompt=None,
    ):
        options = {"language": language, "task": task, "word_timestamps": word_timestamps}
        if temperature is not None:
            options["temperature"] = temperature
        if initial_prompt:
            options["initial_prompt"] = initial_prompt

        raw_segments, info = self.model.transcribe(samples, **options)

        segments = []
        for index, segment in enumerate(raw_segments):
            entry = {
                "id": index,
                "seek": segment.seek,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            }
            if word_timestamps and segment.words:
                entry["words"] = [
                    {
                        "word": word.word,
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                        "probability": word.probability,
                    }
                    for word in segment.words
                ]
            segments.append(entry)

        text = "".join(segment["text"] for segment in segments).strip()
        return {"text": text, "segments": segments, "language": info.language}
