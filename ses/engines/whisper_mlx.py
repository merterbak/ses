import numpy as np

WARMUP_SAMPLES = 1600


class MlxWhisperEngine:
    kind = "stt"

    def __init__(self, model_dir):
        import mlx_whisper

        self.model_dir = str(model_dir)
        mlx_whisper.transcribe(
            np.zeros(WARMUP_SAMPLES, dtype=np.float32), path_or_hf_repo=self.model_dir
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
        import mlx_whisper

        options = {
            "path_or_hf_repo": self.model_dir,
            "language": language,
            "task": task,
            "word_timestamps": word_timestamps,
            "verbose": None,
        }
        if temperature is not None:
            options["temperature"] = temperature
        if initial_prompt:
            options["initial_prompt"] = initial_prompt

        result = mlx_whisper.transcribe(samples, **options)
        result["text"] = result.get("text", "").strip()
        return result
