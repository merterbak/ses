import inspect
from pathlib import Path
import numpy as np
from .. import SesError

MIN_SPEED = 0.5
MAX_SPEED = 2.0
NANO_WEIGHTS = "t3_nano_v1.safetensors"
MIN_CUDA_VRAM_GB = 6.0


class ChatterboxEngine:
    kind = "tts"

    def __init__(self, model_dir):
        try:
            import torch
            from chatterbox.tts_turbo import ChatterboxTurboTTS
        except ImportError as error:
            raise SesError(
                "chatterbox engine dependencies are missing, install chatterbox-tts"
            ) from error

        model_dir = Path(model_dir)
        self.device = self.select_device(torch)
        self.nano = any(model_dir.rglob(NANO_WEIGHTS))
        loader = ChatterboxTurboTTS.from_local
        if self.nano:
            try:
                supports_nano = "nano" in inspect.signature(loader).parameters
            except (TypeError, ValueError):
                supports_nano = False
            if not supports_nano:
                raise SesError(
                    "Chatterbox Nano weights found, but this chatterbox-tts "
                    "version does not support Nano, install a newer build"
                )
        try:
            self.model = self.load(loader, model_dir)
        except RuntimeError as error:
            if self.device not in {"cuda", "mps"} or not self.accelerator_error(error):
                raise SesError(f"chatterbox could not load on {self.device}: {error}") from error
            if self.device == "cuda":
                empty_cache = getattr(torch.cuda, "empty_cache", None)
                if callable(empty_cache):
                    empty_cache()
            self.device = "cpu"
            try:
                self.model = self.load(loader, model_dir)
            except Exception as retry_error:
                raise SesError(
                    f"chatterbox accelerator and CPU fallback both failed: {retry_error}"
                ) from retry_error

    def load(self, loader, model_dir):
        if self.nano:
            return loader(model_dir, self.device, nano=True)
        return loader(model_dir, self.device)

    @staticmethod
    def select_device(torch):
        if torch.cuda.is_available():
            try:
                total_bytes = torch.cuda.get_device_properties(0).total_memory
                if total_bytes / 1024**3 >= MIN_CUDA_VRAM_GB:
                    return "cuda"
            except (AttributeError, RuntimeError, TypeError):
                return "cuda"

        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def accelerator_error(error):
        message = str(error).lower()
        return any(
            marker in message
            for marker in ("out of memory", "cuda", "cudnn", "cublas", "mps")
        )

    def voices(self):
        return ["default"]

    def synth(self, text, voice="default", speed=1.0, lang=None):
        text = (text or "").strip()
        if not text:
            raise SesError("nothing to say, input text is empty")

        tensor = self.model.generate(text)
        audio = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        if not audio.size:
            raise SesError("chatterbox produced no audio")

        speed = self.normalize_speed(speed)
        if speed != 1.0:
            from scipy.signal import resample

            target_samples = max(1, round(audio.size / speed))
            audio = np.asarray(resample(audio, target_samples), dtype=np.float32)

        return audio, int(self.model.sr)

    @staticmethod
    def normalize_speed(speed):
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            return 1.0
        if not np.isfinite(speed) or speed <= 0:
            return 1.0
        return min(max(speed, MIN_SPEED), MAX_SPEED)
