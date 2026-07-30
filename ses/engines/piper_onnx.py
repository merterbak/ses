import numpy as np
from .. import SesError

DEFAULT_SAMPLE_RATE = 22050
MIN_SPEED, MAX_SPEED = 0.5, 2.0


class PiperEngine:
    kind = "tts"

    def __init__(self, model_dir):
        from piper import PiperVoice

        weights = next(iter(sorted(model_dir.rglob("*.onnx"))), None)
        if weights is None:
            raise SesError(f"no .onnx voice found in {model_dir}")

        self.voice = PiperVoice.load(str(weights))
        self.name = weights.stem
        config = getattr(self.voice, "config", None)
        self.speaker_count = int(getattr(config, "num_speakers", 1) or 1)

    def voices(self):
        if self.speaker_count > 1:
            return [str(index) for index in range(self.speaker_count)]
        return [self.name]

    def synth(self, text, voice=None, speed=1.0, lang=None):
        text = (text or "").strip()
        if not text:
            raise SesError("nothing to say, input text is empty")

        pieces = []
        rate = DEFAULT_SAMPLE_RATE
        for chunk in self.voice.synthesize(text, syn_config=self.config(voice, speed)):
            rate = chunk.sample_rate
            pieces.append(np.asarray(chunk.audio_float_array, dtype=np.float32))

        if not pieces:
            raise SesError("piper produced no audio")
        return np.concatenate(pieces), rate

    def config(self, voice, speed):
        speaker_id = None
        if voice is not None and self.speaker_count > 1 and str(voice).isdigit():
            speaker_id = int(voice)

        length_scale = 1.0 / max(min(speed, MAX_SPEED), MIN_SPEED)
        try:
            from piper import SynthesisConfig

            return SynthesisConfig(length_scale=length_scale, speaker_id=speaker_id)
        except Exception:
            return None
