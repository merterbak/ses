from . import DEFAULT_STT, DEFAULT_TTS, SesError
from .audio import WHISPER_SAMPLE_RATE, load_audio, play, record_for
from .core import paths
from .core.loader import ModelLoader
from .core.text import collapse

DEFAULT_VOICE = "af_heart"

KOKORO_PREFIXES = {
    "en": "a", "es": "e", "fr": "f", "hi": "h",
    "it": "i", "ja": "j", "pt": "p", "zh": "z",
    "english": "a", "spanish": "e", "french": "f", "hindi": "h",
    "italian": "i", "japanese": "j", "portuguese": "p", "chinese": "z",
}
LISTEN_SECONDS = 8.0
DICTATE_SECONDS = 15.0
MIN_SPEECH_SAMPLES = WHISPER_SAMPLE_RATE // 4


def model_for_language(language):
    if not language:
        return None

    from .core.registry import resolve

    wanted = language.strip().lower().replace("_", "-")
    for candidate in (wanted, wanted.split("-")[0]):
        spec = resolve(candidate)
        if spec is not None and spec.kind == "tts":
            return spec.name
    return None


def kokoro_voice_for(language, available):
    wanted = language.strip().lower().replace("_", "-")
    for candidate in (wanted, wanted.split("-")[0]):
        prefix = KOKORO_PREFIXES.get(candidate)
        if prefix:
            return next((voice for voice in sorted(available) if voice.startswith(prefix)), None)
    return None


def build_server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("ses")
    loader = ModelLoader()
    preferences = {"voice": DEFAULT_VOICE}

    def tts_model(name=None):
        model = loader.get(name or paths.default_for("tts") or DEFAULT_TTS, auto_pull=True)
        if model.kind != "tts":
            raise SesError(f"'{model.name}' is not a TTS model")
        return model

    def stt_model(name=None):
        model = loader.get(name or paths.default_for("stt") or DEFAULT_STT, auto_pull=True)
        if model.kind != "stt":
            raise SesError(f"'{model.name}' is not a speech-to-text model")
        return model

    def say(text, voice=None, speed=1.0, language=None):
        text = (text or "").strip()
        if not text:
            return "nothing to say"
        model = tts_model(model_for_language(language))
        chosen = voice or preferences["voice"]
        if language and not voice and model.engine_name == "kokoro-onnx":
            chosen = kokoro_voice_for(language, model.engine.voices()) or chosen
        samples, rate = model.engine.synth(text, voice=chosen, speed=speed)
        play(samples, rate)
        return f"spoke {len(samples) / rate:.1f}s of audio ({model.name})"

    def hear(seconds, language):
        model = stt_model()
        samples = record_for(seconds)
        if len(samples) < MIN_SPEECH_SAMPLES:
            return ""
        return collapse(
            model.engine.transcribe(samples, language=language)["text"]
        )

    @server.tool()
    def speak(text, voice=None, speed: float = 1.0, language=None):
        """Say text out loud through the user's speakers, locally.

        Use this to give the user a spoken summary or answer; it returns once
        the audio finishes. Pass `language` (for example 'tr' for Turkish) to
        speak non-English text with a voice that can, it downloads on demand.
        Speak in whatever language the user is writing in.
        """
        return say(text, voice, speed, language)

    @server.tool()
    def notify(message, language=None):
        """Speak a short alert, like 'the build finished' or 'I need approval'.

        Keep it to one brief sentence.
        """
        return say(message, language=language)

    @server.tool()
    def transcribe(path, language=None):
        """Transcribe an audio file (wav/mp3/m4a/flac/ogg) to text, locally."""
        return collapse(
            stt_model().engine.transcribe(load_audio(path), language=language)["text"]
        )

    @server.tool()
    def listen(seconds: int = LISTEN_SECONDS, language=None):
        """Record the microphone briefly and return what the user said."""
        return hear(seconds, language)

    @server.tool()
    def dictate(seconds: int = DICTATE_SECONDS, language=None):
        """Record a longer spoken instruction, so the user can talk instead of type."""
        return hear(seconds, language)

    @server.tool()
    def voice_mode(enabled: bool):
        """Turn auto read-aloud on or off.

        While it's on, the ses Claude Code hook asks for a spoken summary of
        each reply. The setting is saved in ~/.ses/voice_mode.json.
        """
        paths.write_json(paths.voice_mode_path(), {"enabled": bool(enabled)})
        return f"voice mode {'on' if enabled else 'off'}"

    @server.tool()
    def set_voice(voice):
        """Choose the default voice for speak and notify."""
        available = tts_model().engine.voices()
        if voice not in available:
            raise SesError(f"unknown voice '{voice}': see list_voices()")
        preferences["voice"] = voice
        return f"voice set to {voice}"

    @server.tool()
    def list_voices():
        """List the voice ids available to speak."""
        return tts_model().engine.voices()

    return server


def main():
    build_server().run()
