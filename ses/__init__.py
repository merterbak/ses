from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ses")
except PackageNotFoundError:
    __version__ = "0.0.0"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11435
DEFAULT_TTS = "tts-english"
DEFAULT_STT = "whisper-base"


class SesError(Exception):
    pass


class ModelNotInstalled(SesError):
    def __init__(self, name, known):
        self.name = name
        self.known = known
        if known:
            message = f"model '{name}' is not installed. Run: ses pull {name}"
        elif "/" in name:
            message = (
                f"'{name}' is not installed, pull raw Hugging Face repos with: "
                f"ses pull {name} --engine "
                f"<mlx-whisper|faster-whisper|mlx-audio-stt|onnx-asr>"
            )
        else:
            message = f"unknown model '{name}', run 'ses search' to see the library"
        super().__init__(message)
