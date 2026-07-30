import numpy as np
from .. import SesError
from ..audio import WHISPER_SAMPLE_RATE
from ..core.registry import PIPER_LANGUAGES

LANGUAGE_CODES = {name.lower(): code for code, name in PIPER_LANGUAGES.items()}


class MlxAudioSttEngine:
    kind = "stt"

    def __init__(self, model_dir):
        from pathlib import Path
        import mlx.core as mx
        from mlx_audio.stt.utils import load_model

        self.mx = mx
        self.model = load_model(Path(model_dir))

    def transcribe(
        self,
        samples,
        language=None,
        task="transcribe",
        word_timestamps=False,
        temperature=None,
        initial_prompt=None,
    ):
        if task == "translate":
            raise SesError(
                "this engine only transcribes: use a whisper model for --translate"
            )

        options = {}
        if language:
            options["language"] = language
        if temperature is not None:
            options["temperature"] = temperature

        audio = self.mx.array(np.asarray(samples, dtype=np.float32))
        try:
            result = self.model.generate(audio, **options)
        except Exception as error:
            raise SesError(f"mlx-audio failed to transcribe: {error}") from error

        return {
            "text": (getattr(result, "text", "") or "").strip(),
            "segments": self.segments_from(result),
            "language": self.language_of(result) or language,
        }

    def language_of(self, result):
        value = getattr(result, "language", None)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if not value:
            return None
        name = str(value).lower()
        return LANGUAGE_CODES.get(name, name)

    def segments_from(self, result):
        raw = getattr(result, "segments", None) or []
        segments = []
        for index, segment in enumerate(raw):
            if not isinstance(segment, dict):
                continue
            segments.append(
                {
                    "id": index,
                    "start": round(float(segment.get("start", 0.0)), 3),
                    "end": round(float(segment.get("end", 0.0)), 3),
                    "text": segment.get("text", ""),
                }
            )
        return segments


SAMPLE_RATE = WHISPER_SAMPLE_RATE
