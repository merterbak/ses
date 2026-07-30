import numpy as np
from .. import SesError

DEFAULT_SAMPLE_RATE = 24000
MIN_SPEED, MAX_SPEED = 0.5, 2.0
AUTO_LANGUAGE = "auto"

FRAMES_PER_SECOND = 12
CHARS_PER_SECOND = 15
TOKEN_HEADROOM = 4
MIN_TOKENS = 120
MAX_TOKENS = 1500
STABLE_TEMPERATURE = 0.7


def token_budget(text):
    expected_seconds = len(text) / CHARS_PER_SECOND
    budget = int(expected_seconds * FRAMES_PER_SECOND * TOKEN_HEADROOM)
    return max(MIN_TOKENS, min(budget, MAX_TOKENS))


class MlxAudioEngine:
    kind = "tts"

    def __init__(self, model_dir):
        from pathlib import Path
        from mlx_audio.tts.utils import load_model

        self.model = load_model(Path(model_dir))
        self.sample_rate = int(getattr(self.model, "sample_rate", DEFAULT_SAMPLE_RATE))

    def voices(self):
        speakers = getattr(self.model, "supported_speakers", None)
        if callable(getattr(self.model, "get_supported_speakers", None)):
            speakers = self.model.get_supported_speakers()
        return sorted(speakers) if speakers else ["default"]

    def synth(self, text, voice=None, speed=1.0, lang=None, instruct=None):
        text = (text or "").strip()
        if not text:
            raise SesError("nothing to say, input text is empty")

        options = {
            "speed": min(max(speed, MIN_SPEED), MAX_SPEED),
            "lang_code": lang or AUTO_LANGUAGE,
            "max_tokens": token_budget(text),
            "temperature": STABLE_TEMPERATURE,
            "verbose": False,
        }
        if voice and voice != "default":
            available = self.voices()
            options["voice"] = voice if voice in available else available[0]
        if instruct:
            options["instruct"] = instruct

        pieces = []
        rate = self.sample_rate
        try:
            for result in self.model.generate(text, **options):
                rate = int(getattr(result, "sample_rate", rate))
                pieces.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        except Exception as error:
            raise SesError(f"mlx-audio failed to synthesize: {error}") from error

        if not pieces:
            raise SesError("mlx-audio produced no audio")
        return np.concatenate(pieces), rate
