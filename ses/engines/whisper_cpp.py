import math
import tempfile
import wave
from collections.abc import Mapping
from pathlib import Path
import numpy as np
from .. import SesError
from ..audio import WHISPER_SAMPLE_RATE

TICKS_PER_SECOND = 100.0


class WhisperCppEngine:
    kind = "stt"

    def __init__(self, model_dir):
        weights = sorted(Path(model_dir).rglob("*.bin"))
        if len(weights) != 1:
            detail = "none found" if not weights else f"found {len(weights)}"
            raise SesError(f"whisper.cpp needs exactly one .bin model in {model_dir} ({detail})")

        from pywhispercpp.model import Model

        self.model = Model(
            str(weights[0]),
            print_progress=False,
            print_realtime=False,
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
        audio = mono_audio(samples)
        detected_language = language or self.detect_language(audio)
        options = {"translate": task == "translate", "language": detected_language}
        if temperature is not None:
            options["temperature"] = temperature
        if initial_prompt:
            options["initial_prompt"] = initial_prompt

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            write_wav(temp_path, audio)
            raw_result = self.model.transcribe(str(temp_path), **options)
        except Exception as error:
            raise SesError(f"whisper.cpp failed to transcribe: {error}") from error
        finally:
            temp_path.unlink(missing_ok=True)

        if isinstance(raw_result, Mapping):
            detected_language = value(
                raw_result, "language", default=detected_language
            )
            raw_segments = value(raw_result, "segments", default=()) or ()
        else:
            raw_segments = raw_result or ()

        segments = [
            normalize_segment(segment, index)
            for index, segment in enumerate(raw_segments)
        ]
        text = " ".join(segment["text"] for segment in segments if segment["text"]).strip()
        return {"text": text, "segments": segments, "language": detected_language}

    def detect_language(self, audio):
        try:
            detected, _probabilities = self.model.auto_detect_language(audio)
            language, _confidence = detected
        except Exception as error:
            raise SesError(
                "whisper.cpp could not auto-detect the language, pass --language explicitly"
            ) from error
        if not language:
            raise SesError(
                "whisper.cpp could not auto-detect the language, pass --language explicitly"
            )
        return str(language)


def mono_audio(samples):
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1:
        raise SesError("whisper.cpp expects mono audio")
    if not audio.size:
        raise SesError("whisper.cpp received empty audio")
    return np.nan_to_num(audio, copy=True, nan=0.0, posinf=1.0, neginf=-1.0)


def write_wav(path, samples):
    audio = mono_audio(samples)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")

    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(WHISPER_SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def value(item, *names, default=None):
    for name in names:
        if isinstance(item, Mapping):
            if name in item:
                return item[name]
            continue
        try:
            found = getattr(item, name)
        except (AttributeError, TypeError):
            continue
        return found
    return default


def number(raw, default=0.0):
    try:
        result = float(raw)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def timestamp(item, direct_name, tick_name):
    direct = value(item, direct_name)
    if direct is not None:
        return max(0.0, number(direct))
    return max(0.0, number(value(item, tick_name)) / TICKS_PER_SECOND)


def probability(item):
    raw = value(item, "probability", "prob", "p")
    if raw is None:
        return None
    parsed = number(raw, default=math.nan)
    return parsed if math.isfinite(parsed) else None


def normalize_segment(raw, index):
    start = timestamp(raw, "start", "t0")
    end = max(start, timestamp(raw, "end", "t1"))
    entry = {
        "id": index,
        "start": round(start, 3),
        "end": round(end, 3),
        "text": str(value(raw, "text", default="") or "").strip(),
    }

    confidence = probability(raw)
    if confidence is not None:
        entry["probability"] = confidence

    return entry
