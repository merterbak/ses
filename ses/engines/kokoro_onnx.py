import re
import numpy as np
from .. import SesError

SAMPLE_RATE = 24000
MAX_PHONEMES = 500
PUNCTUATION = ".,!?;:"
MIN_SPEED, MAX_SPEED = 0.5, 2.0
DEFAULT_VOICE = "af_heart"

LANGUAGE_BY_PREFIX = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}


class KokoroEngine:
    kind = "tts"

    def __init__(self, model_dir):
        from kokoro_onnx import Kokoro

        weights = sorted(model_dir.glob("onnx/*.onnx")) or sorted(model_dir.glob("*.onnx"))
        if not weights:
            raise SesError(f"no .onnx model found in {model_dir}")

        voices_file = model_dir / "voices.npz"
        if not voices_file.is_file():
            raise SesError(f"voices.npz missing in {model_dir}, re-pull the model")

        self.kokoro = Kokoro(str(weights[0]), str(voices_file))
        self.input_types = {tensor.name: tensor.type for tensor in self.kokoro.sess.get_inputs()}

    def voices(self):
        return sorted(self.kokoro.voices.files)

    def synth(self, text, voice=DEFAULT_VOICE, speed=1.0, lang=None):
        from kokoro_onnx.trim import trim

        text = (text or "").strip()
        if not text:
            raise SesError("nothing to say, input text is empty")

        voice = voice or DEFAULT_VOICE
        if voice not in self.kokoro.voices:
            raise SesError(
                f"unknown voice '{voice}', run 'ses voices' to list the "
                f"{len(self.voices())} available"
            )

        speed = min(max(speed, MIN_SPEED), MAX_SPEED)
        lang = lang or LANGUAGE_BY_PREFIX.get(voice[0], "en-us")

        phonemes = self.kokoro.tokenizer.phonemize(text, lang)
        style = self.kokoro.get_voice_style(voice)

        pieces = []
        for batch in self.split_phonemes(phonemes):
            tokens = self.kokoro.tokenizer.tokenize(batch)[:MAX_PHONEMES]
            audio = self.run(tokens, style[len(tokens)], speed)
            trimmed, _ = trim(audio)
            pieces.append(trimmed)

        if not pieces:
            raise SesError("nothing to say, input produced no phonemes")
        return np.concatenate(pieces), SAMPLE_RATE

    def split_phonemes(self, phonemes):
        parts = re.split(rf"([{re.escape(PUNCTUATION)}])", phonemes)
        batches = []
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part in PUNCTUATION:
                current += part
            elif current and len(current) + len(part) + 1 > MAX_PHONEMES:
                batches.append(current)
                current = part
            else:
                current = f"{current} {part}".strip()
        if current:
            batches.append(current)
        return batches

    def run(self, tokens, style, speed):
        inputs = {}
        for name, declared_type in self.input_types.items():
            if name in ("input_ids", "tokens"):
                dtype = np.int64 if "int64" in declared_type else np.int32
                inputs[name] = np.array([[0, *tokens, 0]], dtype=dtype)
            elif name == "style":
                inputs[name] = np.asarray(style, dtype=np.float32)
            elif name == "speed":
                dtype = np.float32 if "float" in declared_type else np.int32
                inputs[name] = np.array([speed], dtype=dtype)

        output = self.kokoro.sess.run(None, inputs)[0]
        return np.ravel(np.asarray(output)).astype(np.float32)
