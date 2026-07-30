import io
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from .. import SesError

WHISPER_SAMPLE_RATE = 16000


def ffmpeg_install_hint():
    if sys.platform == "win32":
        return "winget install Gyan.FFmpeg"
    if sys.platform.startswith("linux"):
        return "sudo apt install ffmpeg"
    return "brew install ffmpeg"


def resample(samples, from_rate, to_rate):
    if from_rate == to_rate:
        return samples
    from scipy.signal import resample_poly

    common = math.gcd(from_rate, to_rate)
    return resample_poly(samples, to_rate // common, from_rate // common).astype(np.float32)


def decode_with_soundfile(source):
    import soundfile as sf

    data, rate = sf.read(source, dtype="float32", always_2d=True)
    return data.mean(axis=1), rate


def decode_with_ffmpeg(data, rate):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SesError(
            f"could not decode this audio format, install ffmpeg ({ffmpeg_install_hint()}) "
            "or provide wav/flac/ogg/mp3"
        )
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(rate), "pipe:1"],
        input=data,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr[-300:].decode(errors="replace")
        raise SesError(f"ffmpeg failed to decode audio: {detail}")
    return np.frombuffer(result.stdout, dtype=np.float32)


def load_audio(source, rate=WHISPER_SAMPLE_RATE):
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise SesError(f"file not found: {path}")
        try:
            samples, source_rate = decode_with_soundfile(str(path))
        except Exception:
            samples, source_rate = decode_with_ffmpeg(path.read_bytes(), rate), rate
    else:
        try:
            samples, source_rate = decode_with_soundfile(io.BytesIO(source))
        except Exception:
            samples, source_rate = decode_with_ffmpeg(source, rate), rate
    return resample(np.ascontiguousarray(samples, dtype=np.float32), source_rate, rate)


def save_audio(path, samples, rate):
    import soundfile as sf

    sf.write(str(path), samples, rate)


def wav_bytes(samples, rate):
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def pcm16_bytes(samples):
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def mp3_bytes(samples, rate):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SesError(
            f"mp3 output needs ffmpeg ({ffmpeg_install_hint()}), or use response_format=wav"
        )
    result = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-f",
            "f32le",
            "-ar",
            str(rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=samples.astype(np.float32).tobytes(),
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr[-300:].decode(errors="replace")
        raise SesError(f"ffmpeg mp3 encode failed: {detail}")
    return result.stdout


def play(samples, rate):
    if sys.platform == "darwin":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            save_audio(temp_path, samples, rate)
            subprocess.run(["afplay", str(temp_path)], check=False)
        finally:
            temp_path.unlink(missing_ok=True)
    else:
        import sounddevice as sd

        sd.play(samples, rate)
        sd.wait()
