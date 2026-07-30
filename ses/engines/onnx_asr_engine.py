import numpy as np
from .. import SesError

MODEL_IDS = {
    "parakeet-tdt-0.6b-v3-onnx": "nemo-parakeet-tdt-0.6b-v3",
    "parakeet-tdt-0.6b-v2-onnx": "nemo-parakeet-tdt-0.6b-v2",
    "canary-1b-v2-onnx": "nemo-canary-1b-v2",
}


class OnnxAsrEngine:
    kind = "stt"

    def __init__(self, model_dir):
        from pathlib import Path
        import onnx_asr

        directory = Path(model_dir)
        name = self.model_id(directory)
        try:
            self.model = onnx_asr.load_model(
                name, str(directory), providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise SesError(f"onnx-asr could not load {directory.name}: {error}") from error

    def model_id(self, directory):
        import json

        source = directory.name
        manifest = directory / "manifest.json"
        if manifest.is_file():
            source = json.loads(manifest.read_text()).get("repo", source)

        for suffix, model_id in MODEL_IDS.items():
            if source.endswith(suffix):
                return model_id
        raise SesError(
            f"'{source}' isn't a model onnx-asr knows, "
            f"expected one of: {', '.join(MODEL_IDS)}"
        )

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
            raise SesError("this engine only transcribes: use a whisper model for --translate")

        from ..audio import WHISPER_SAMPLE_RATE

        audio = np.asarray(samples, dtype=np.float32)
        options = {"language": language} if language else {}
        try:
            text = self.model.recognize(audio, sample_rate=WHISPER_SAMPLE_RATE, **options)
        except TypeError:
            text = self.model.recognize(audio, sample_rate=WHISPER_SAMPLE_RATE)
        except Exception as error:
            raise SesError(f"onnx-asr failed to transcribe: {error}") from error

        return {"text": (text or "").strip(), "segments": [], "language": language}
