import platform
import sys
from dataclasses import dataclass

APPLE_SILICON = "apple-silicon"
ANY = "any"


def is_apple_silicon():
    return sys.platform == "darwin" and platform.machine() == "arm64"


@dataclass
class Backend:
    engine: str
    repo: str
    size_mb: int
    files: tuple = None
    requires: str = ANY
    single_variant: bool = False
    url: str = None
    format: str = None
    required_ram_gb: float = None
    required_vram_gb: float = None
    recommended_vram_gb: float = None
    accelerators: tuple = None
    extra: str = None

    @property
    def supported(self):
        return self.requires == ANY or is_apple_silicon()

    @property
    def source(self):
        return self.url or f"https://huggingface.co/{self.repo}"


@dataclass
class ModelSpec:
    name: str
    kind: str
    description: str
    backends: tuple
    post_install: str = None
    aliases: tuple = ()
    languages: tuple = ()
    license: str = None
    github: str = None
    capabilities: tuple = ()

    def backend(self):
        for backend in self.backends:
            if backend.supported:
                return backend
        return None


@dataclass
class BrainSpec:
    name: str
    size_gb: float
    description: str


def whisper(
    size,
    mlx_repo,
    mlx_mb,
    ct2_repo,
    ct2_mb,
    description,
    aliases=(),
    languages=("multilingual",),
    required_ram_gb=None,
):
    return ModelSpec(
        name=f"whisper-{size}",
        kind="stt",
        description=description,
        backends=(
            Backend(
                "mlx-whisper",
                mlx_repo,
                mlx_mb,
                requires=APPLE_SILICON,
                format="MLX",
                required_ram_gb=required_ram_gb,
                accelerators=("apple",),
            ),
            Backend(
                "faster-whisper",
                ct2_repo,
                ct2_mb,
                format="CTranslate2",
                required_ram_gb=required_ram_gb,
                accelerators=("apple", "cpu", "cuda"),
            ),
        ),
        aliases=aliases,
        languages=languages,
        license="MIT",
        github="https://github.com/openai/whisper",
        capabilities=("transcribe", "translate", "word-timestamps"),
    )


def whisper_cpp(name, weight, size_mb, required_ram_gb, description, aliases=()):
    return ModelSpec(
        name=f"whisper-cpp-{name}",
        kind="stt",
        description=description,
        backends=(
            Backend(
                "whisper-cpp",
                "ggerganov/whisper.cpp",
                size_mb,
                files=(weight,),
                format="GGML Q5",
                required_ram_gb=required_ram_gb,
                accelerators=("apple", "cpu", "cuda"),
                extra="whisper-cpp",
            ),
        ),
        aliases=aliases,
        languages=("multilingual",),
        license="MIT",
        github="https://github.com/ggml-org/whisper.cpp",
        capabilities=("transcribe", "translate", "segment-timestamps"),
    )


def vosk_small(name, archive, size_mb, language, aliases=()):
    return ModelSpec(
        name=f"vosk-{name}",
        kind="stt",
        description=(
            f"Vosk small {language}, lightweight offline recognition for CPU and Windows"
        ),
        backends=(
            Backend(
                "vosk",
                archive.removesuffix(".zip"),
                size_mb,
                url=f"https://alphacephei.com/vosk/models/{archive}",
                format="Vosk/Kaldi ZIP",
                required_ram_gb=0.3,
                accelerators=("apple", "cpu", "cuda"),
                extra="vosk",
            ),
        ),
        aliases=aliases,
        languages=(language,),
        license="Apache-2.0",
        github="https://github.com/alphacep/vosk-api",
        capabilities=("transcribe", "word-timestamps"),
    )


KOKORO_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
KOKORO_VOICES = ("voices/*.bin", "config.json")

PIPER_REPO = "rhasspy/piper-voices"
PIPER_VOICE_MB = 61

PIPER_LANGUAGES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ka": "Georgian",
    "kk": "Kazakh",
    "ko": "Korean",
    "ku": "Kurdish",
    "lb": "Luxembourgish",
    "lv": "Latvian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "te": "Telugu",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def piper_voice(locale, language):
    return ModelSpec(
        name=f"tts-{language.lower()}",
        kind="tts",
        description=f"Piper {language}, natural {language} speech, runs everywhere",
        backends=(
            Backend(
                "piper",
                PIPER_REPO,
                PIPER_VOICE_MB,
                files=(f"{locale}/*/medium/*.onnx", f"{locale}/*/*/*.onnx"),
                single_variant=True,
                format="ONNX",
                required_ram_gb=0.3,
                accelerators=("apple", "cpu", "cuda"),
            ),
        ),
        aliases=(locale, f"piper-{locale}", language.lower()),
        languages=(language,),
        license="varies by voice",
        github="https://github.com/OHF-Voice/piper1-gpl",
        capabilities=("synthesize",),
    )


PIPER_MODELS = tuple(piper_voice(locale, language) for locale, language in PIPER_LANGUAGES.items())

SPEECH_MODELS = (
    ModelSpec(
        name="omnivoice",
        kind="tts",
        description=(
            "OmniVoice, multilingual speech from k2-fsa, 860k downloads upstream. "
            "Generates at about real time, so prefer kokoro for live replies "
            "(Apple Silicon)"
        ),
        backends=(
            Backend("mlx-audio", "mlx-community/OmniVoice-bf16", 1564, requires=APPLE_SILICON),
        ),
        aliases=("omni-voice",),
    ),
    ModelSpec(
        name="qwen-tts",
        kind="tts",
        description=(
            "Qwen3-TTS 0.6B, 2026's second most-downloaded TTS family. "
            "This is the Base build the community settled on; the CustomVoice "
            "ports over-generate badly (Apple Silicon)"
        ),
        backends=(
            Backend(
                "mlx-audio",
                "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
                1632,
                requires=APPLE_SILICON,
                format="MLX 4-bit",
                required_ram_gb=3.5,
                accelerators=("apple",),
                extra="mlx-audio",
            ),
        ),
        aliases=("qwen3-tts", "qwen-tts-base"),
        languages=("Chinese", "English", "Japanese", "Korean", "European languages"),
        license="Apache-2.0",
        github="https://github.com/QwenLM/Qwen3-TTS",
        capabilities=("synthesize",),
    ),
    ModelSpec(
        name="kokoro",
        kind="tts",
        description="Kokoro-82M v1.0, top-ranked open TTS, 54 voices, 8 languages",
        backends=(
            Backend(
                "kokoro-onnx",
                KOKORO_REPO,
                340,
                ("onnx/model.onnx",) + KOKORO_VOICES,
                format="ONNX",
                required_ram_gb=0.8,
                accelerators=("apple", "cpu", "cuda"),
                extra="kokoro",
            ),
        ),
        post_install="build_kokoro_voices",
        aliases=("kokoro-82m", "kokoro-v1.0"),
        languages=(
            "English",
            "Spanish",
            "French",
            "Hindi",
            "Italian",
            "Japanese",
            "Portuguese",
            "Chinese",
        ),
        license="Apache-2.0",
        github="https://github.com/hexgrad/kokoro",
        capabilities=("synthesize", "multi-voice"),
    ),
    ModelSpec(
        name="kokoro-q8",
        kind="tts",
        description="Kokoro-82M int8, 3.5x smaller download, near-identical voices",
        backends=(
            Backend(
                "kokoro-onnx",
                KOKORO_REPO,
                118,
                ("onnx/model_quantized.onnx",) + KOKORO_VOICES,
                format="ONNX int8",
                required_ram_gb=0.5,
                accelerators=("apple", "cpu", "cuda"),
                extra="kokoro",
            ),
        ),
        post_install="build_kokoro_voices",
        languages=(
            "English",
            "Spanish",
            "French",
            "Hindi",
            "Italian",
            "Japanese",
            "Portuguese",
            "Chinese",
        ),
        license="Apache-2.0",
        github="https://github.com/hexgrad/kokoro",
        capabilities=("synthesize", "multi-voice"),
    ),
    ModelSpec(
        name="chatterbox-turbo",
        kind="tts",
        description=((
            "Chatterbox Turbo 350M, voice cloning and paralinguistic tags. "
            "Measured at 16x slower than real time here, so it is for files, "
            "not live replies"
        )),
        backends=(
            Backend(
                "chatterbox",
                "ResembleAI/chatterbox-turbo",
                2850,
                files=(
                    "ve.safetensors",
                    "t3_turbo_v1.safetensors",
                    "s3gen_meanflow.safetensors",
                    "conds.pt",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                    "added_tokens.json",
                    "special_tokens_map.json",
                ),
                format="PyTorch safetensors",
                required_ram_gb=8.0,
                recommended_vram_gb=6.0,
                accelerators=("apple", "cpu", "cuda"),
                extra="chatterbox",
            ),
        ),
        aliases=("chatterbox", "chatterbox-350m"),
        languages=("English",),
        license="MIT",
        github="https://github.com/resemble-ai/chatterbox",
        capabilities=("synthesize", "expressive-tags"),
    ),
    ModelSpec(
        name="qwen-asr",
        kind="stt",
        description="Qwen3-ASR 0.6B, 2026's most-used new ASR, multilingual (Apple Silicon)",
        backends=(
            Backend(
                "mlx-audio-stt",
                "mlx-community/Qwen3-ASR-0.6B-8bit",
                1011,
                requires=APPLE_SILICON,
                format="MLX 8-bit",
                required_ram_gb=2.5,
                accelerators=("apple",),
                extra="mlx-audio",
            ),
        ),
        aliases=("qwen3-asr", "qwen-stt"),
        languages=("multilingual",),
        license="Apache-2.0",
        github="https://github.com/QwenLM/Qwen3-ASR",
        capabilities=("transcribe",),
    ),
    ModelSpec(
        name="qwen-asr-large",
        kind="stt",
        description="Qwen3-ASR 1.7B, the accurate sibling of qwen-asr, multilingual (Apple Silicon)",
        backends=(
            Backend(
                "mlx-audio-stt",
                "mlx-community/Qwen3-ASR-1.7B-8bit",
                2354,
                requires=APPLE_SILICON,
                format="MLX 8-bit",
                required_ram_gb=4.5,
                accelerators=("apple",),
                extra="mlx-audio",
            ),
        ),
        aliases=("qwen3-asr-1.7b", "qwen-asr-1.7b"),
        languages=("multilingual",),
        license="Apache-2.0",
        github="https://github.com/QwenLM/Qwen3-ASR",
        capabilities=("transcribe",),
    ),
    ModelSpec(
        name="nemotron-asr",
        kind="stt",
        description="NVIDIA Nemotron 3.5 streaming ASR 0.6B, built for live audio (Apple Silicon)",
        backends=(
            Backend(
                "mlx-audio-stt",
                "mlx-community/nemotron-3.5-asr-streaming-0.6b",
                1277,
                requires=APPLE_SILICON,
                format="MLX",
                required_ram_gb=3.0,
                accelerators=("apple",),
                extra="mlx-audio",
            ),
        ),
        aliases=("nemotron", "nemotron-3.5"),
        languages=("multilingual",),
        github="https://github.com/NVIDIA/NeMo",
        capabilities=("transcribe", "streaming"),
    ),
    ModelSpec(
        name="parakeet-v2",
        kind="stt",
        description="Parakeet TDT v2 via ONNX, English only, the fastest of the family",
        backends=(
            Backend("onnx-asr", "istupakov/parakeet-tdt-0.6b-v2-onnx", 2411),
        ),
        aliases=("parakeet-english",),
    ),
    ModelSpec(
        name="canary",
        kind="stt",
        description=(
            "NVIDIA Canary 1B v2 via ONNX, tops the open ASR leaderboard "
            "(5.63% WER) and runs on every platform. Its 25 languages are "
            "European only, so no Turkish; CPU-bound, so around real time"
        ),
        backends=(
            Backend(
                "onnx-asr",
                "istupakov/canary-1b-v2-onnx",
                4764,
                format="ONNX",
                required_ram_gb=8.0,
                accelerators=("apple", "cpu", "cuda"),
                extra="onnx-asr",
            ),
        ),
        aliases=("canary-1b", "canary-qwen"),
        languages=("25 European languages",),
        github="https://github.com/istupakov/onnx-asr",
        capabilities=("transcribe",),
    ),
    ModelSpec(
        name="parakeet",
        kind="stt",
        description=(
            "Parakeet TDT v3 via ONNX, leaderboard accuracy on every platform. "
            "English and 24 other European languages; no Turkish, no language flag"
        ),
        backends=(
            Backend(
                "onnx-asr",
                "istupakov/parakeet-tdt-0.6b-v3-onnx",
                2411,
                format="ONNX",
                required_ram_gb=4.5,
                accelerators=("apple", "cpu", "cuda"),
                extra="onnx-asr",
            ),
        ),
        aliases=("parakeet-v3", "parakeet-onnx", "parakeet-cross"),
        languages=("25 European languages",),
        license="CC-BY-4.0",
        github="https://github.com/istupakov/onnx-asr",
        capabilities=("transcribe",),
    ),
    whisper(
        "tiny",
        "mlx-community/whisper-tiny-mlx",
        71,
        "Systran/faster-whisper-tiny",
        75,
        "Whisper tiny, fastest, quick notes and commands",
        required_ram_gb=0.5,
    ),
    whisper(
        "base",
        "mlx-community/whisper-base-mlx",
        138,
        "Systran/faster-whisper-base",
        142,
        "Whisper base, good speed/accuracy balance (default)",
        required_ram_gb=1.0,
    ),
    whisper(
        "small",
        "mlx-community/whisper-small-mlx",
        461,
        "Systran/faster-whisper-small",
        464,
        "Whisper small, solid accuracy, still fast",
        required_ram_gb=2.0,
    ),
    whisper(
        "medium",
        "mlx-community/whisper-medium-mlx",
        1456,
        "Systran/faster-whisper-medium",
        1460,
        "Whisper medium, high accuracy, 99 languages",
        required_ram_gb=4.0,
    ),
    whisper(
        "large",
        "mlx-community/whisper-large-v3-mlx",
        2942,
        "Systran/faster-whisper-large-v3",
        2948,
        "Whisper large-v3, best accuracy",
        aliases=("whisper-large-v3",),
        required_ram_gb=6.0,
    ),
    whisper(
        "turbo",
        "mlx-community/whisper-large-v3-turbo",
        1540,
        "deepdml/faster-whisper-large-v3-turbo-ct2",
        1547,
        "Whisper large-v3-turbo, near large-v3 accuracy at 6x speed",
        aliases=("turbo", "whisper-large-v3-turbo"),
        required_ram_gb=4.0,
    ),
    whisper(
        "base.en",
        "mlx-community/whisper-base.en-mlx",
        138,
        "Systran/faster-whisper-base.en",
        141,
        "Whisper base.en, English-only, more accurate than base",
        languages=("English",),
        required_ram_gb=1.0,
    ),
    ModelSpec(
        name="whisper-distil-large-v3",
        kind="stt",
        description=(
            "Distil-Whisper large-v3, English long-form ASR distilled for much faster inference"
        ),
        backends=(
            Backend(
                "mlx-whisper",
                "mlx-community/distil-whisper-large-v3",
                1510,
                requires=APPLE_SILICON,
                format="MLX",
                required_ram_gb=4.0,
                accelerators=("apple",),
            ),
            Backend(
                "faster-whisper",
                "Systran/faster-distil-whisper-large-v3",
                1510,
                format="CTranslate2",
                required_ram_gb=4.0,
                accelerators=("apple", "cpu", "cuda"),
            ),
        ),
        aliases=("distil-whisper", "distil-large-v3"),
        languages=("English",),
        license="MIT",
        github="https://github.com/huggingface/distil-whisper",
        capabilities=("transcribe", "word-timestamps"),
    ),
    whisper_cpp(
        "tiny",
        "ggml-tiny-q5_1.bin",
        31,
        0.3,
        "whisper.cpp tiny Q5, smallest cross-platform Whisper build",
        aliases=("cpp-whisper-tiny",),
    ),
    whisper_cpp(
        "base",
        "ggml-base-q5_1.bin",
        57,
        0.4,
        "whisper.cpp base Q5, compact Windows/CPU transcription",
        aliases=("cpp-whisper-base",),
    ),
    whisper_cpp(
        "small",
        "ggml-small-q5_1.bin",
        181,
        0.9,
        "whisper.cpp small Q5, recommended CPU accuracy/speed balance",
        aliases=("cpp-whisper-small",),
    ),
    whisper_cpp(
        "medium",
        "ggml-medium-q5_0.bin",
        514,
        2.1,
        "whisper.cpp medium Q5, accurate multilingual CPU/GPU transcription",
        aliases=("cpp-whisper-medium",),
    ),
    whisper_cpp(
        "turbo",
        "ggml-large-v3-turbo-q5_0.bin",
        547,
        2.5,
        "whisper.cpp large-v3 Turbo Q5, fast high-accuracy cross-platform ASR",
        aliases=("cpp-whisper-turbo",),
    ),
    whisper_cpp(
        "large",
        "ggml-large-v3-q5_0.bin",
        1100,
        3.9,
        "whisper.cpp large-v3 Q5, highest-accuracy quantized Whisper option",
        aliases=("whisper-cpp-large-v3", "cpp-whisper-large"),
    ),
    vosk_small(
        "english",
        "vosk-model-small-en-us-0.15.zip",
        40,
        "English (US)",
        aliases=("vosk-en", "vosk-en-us"),
    ),
    vosk_small(
        "turkish",
        "vosk-model-small-tr-0.3.zip",
        35,
        "Turkish",
        aliases=("vosk-tr",),
    ),
    vosk_small(
        "german",
        "vosk-model-small-de-0.15.zip",
        45,
        "German",
        aliases=("vosk-de",),
    ),
    vosk_small(
        "spanish",
        "vosk-model-small-es-0.42.zip",
        39,
        "Spanish",
        aliases=("vosk-es",),
    ),
    vosk_small(
        "french",
        "vosk-model-small-fr-0.22.zip",
        41,
        "French",
        aliases=("vosk-fr",),
    ),
    vosk_small(
        "chinese",
        "vosk-model-small-cn-0.22.zip",
        42,
        "Chinese",
        aliases=("vosk-cn", "vosk-zh"),
    ),
)

BRAINS = (
    BrainSpec(
        "gemma4:e4b", 9.6, "Google's newest small model, the sweet spot for voice (default)"
    ),
    BrainSpec("qwen3.5:2b", 2.7, "newest Qwen, ultra-light, strong multilingual"),
    BrainSpec("llama3.2", 2.0, "Meta 3B, the most-pulled small assistant model"),
    BrainSpec("lfm2.5:8b-a1b", 5.2, "LiquidAI MoE, 1B active, snappiest mid-size replies"),
    BrainSpec("gpt-oss:20b", 14.0, "OpenAI's open flagship, best answers, needs ~24GB RAM"),
)

ALL_MODELS = SPEECH_MODELS + PIPER_MODELS

RECOMMENDED = {
    "whisper-turbo": "99 languages, 4x faster than audio, runs on every platform",
    "kokoro": "the most downloaded open voice model, 54 voices",
    "tts-english": "49 languages via Piper, 30x faster than audio, 61 MB",
}

FEATURED = (
    "whisper-turbo",
    "whisper-distil-large-v3",
    "whisper-cpp-small",
    "vosk-turkish",
    "parakeet",
    "canary",
    "whisper-base",
    "qwen-asr",
    "nemotron-asr",
    "kokoro",
    "chatterbox-turbo",
    "tts-english",
    "tts-turkish",
    "qwen-asr-large",
)

REGISTRY = {spec.name: spec for spec in ALL_MODELS}
_ALIASES = {}
for _spec in ALL_MODELS:
    for _alias in _spec.aliases:
        _ALIASES.setdefault(_alias, _spec.name)
_BRAINS_BY_NAME = {brain.name: brain for brain in BRAINS}


def resolve(name):
    name = name.strip().lower()
    if name in REGISTRY:
        return REGISTRY[name]
    if name in _ALIASES:
        return REGISTRY[_ALIASES[name]]
    return None


def resolve_brain(name):
    name = name.strip().lower()
    if name in _BRAINS_BY_NAME:
        return _BRAINS_BY_NAME[name]
    for brain in BRAINS:
        if brain.name.split(":")[0] == name:
            return brain
    return None
