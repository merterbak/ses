import importlib
from .. import SesError

ENGINES = {
    "mlx-whisper": ("ses.engines.whisper_mlx", "MlxWhisperEngine"),
    "faster-whisper": ("ses.engines.whisper_ctranslate2", "FasterWhisperEngine"),
    "kokoro-onnx": ("ses.engines.kokoro_onnx", "KokoroEngine"),
    "piper": ("ses.engines.piper_onnx", "PiperEngine"),
    "mlx-audio": ("ses.engines.mlx_audio_tts", "MlxAudioEngine"),
    "mlx-audio-stt": ("ses.engines.mlx_audio_stt", "MlxAudioSttEngine"),
    "onnx-asr": ("ses.engines.onnx_asr_engine", "OnnxAsrEngine"),
    "transformers-stt": ("ses.engines.transformers_engine", "TransformersSttEngine"),
    "transformers-tts": ("ses.engines.transformers_engine", "TransformersTtsEngine"),
    "whisper-cpp": ("ses.engines.whisper_cpp", "WhisperCppEngine"),
    "vosk": ("ses.engines.vosk", "VoskEngine"),
    "chatterbox": ("ses.engines.chatterbox", "ChatterboxEngine"),
}

ENGINE_KINDS = {
    "mlx-whisper": "stt",
    "faster-whisper": "stt",
    "kokoro-onnx": "tts",
    "piper": "tts",
    "mlx-audio": "tts",
    "mlx-audio-stt": "stt",
    "onnx-asr": "stt",
    "transformers-stt": "stt",
    "transformers-tts": "tts",
    "whisper-cpp": "stt",
    "vosk": "stt",
    "chatterbox": "tts",
}

INSTALL_HINTS = {
    "transformers-stt": "pip install 'ses[transformers]'",
    "transformers-tts": "pip install 'ses[transformers]'",
    "kokoro-onnx": "pip install 'ses[kokoro]'",
    "mlx-audio": "pip install 'ses[mlx-audio]'  (Apple Silicon only)",
    "onnx-asr": "pip install 'ses[onnx-asr]'",
    "mlx-audio-stt": "pip install 'ses[mlx-audio]'  (Apple Silicon only)",
    "whisper-cpp": "pip install 'ses[whisper-cpp]'",
    "vosk": "pip install 'ses[vosk]'",
    "chatterbox": "pip install 'ses[chatterbox]'",
}


def load_engine(engine, model_dir):
    if engine not in ENGINES:
        raise SesError(f"unknown engine '{engine}'")

    module_path, class_name = ENGINES[engine]
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)(model_dir)
    except ImportError as error:
        hint = INSTALL_HINTS.get(engine)
        detail = f". Install it with: {hint}" if hint else f": {error}"
        raise SesError(f"engine '{engine}' isn't available here{detail}") from error
