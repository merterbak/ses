import json
import math
from pathlib import Path
import numpy as np
from .. import SesError
from ..audio import WHISPER_SAMPLE_RATE

CHUNK_SECONDS = 0.25
CHUNK_SAMPLES = int(WHISPER_SAMPLE_RATE * CHUNK_SECONDS)


class VoskEngine:
    kind = "stt"

    def __init__(self, model_dir):
        from vosk import KaldiRecognizer, Model

        self.model_dir = self.find_model_dir(Path(model_dir))
        self.model = Model(str(self.model_dir))
        self.recognizer_type = KaldiRecognizer

    @staticmethod
    def find_model_dir(model_dir):
        if not model_dir.is_dir():
            return model_dir

        directories = [
            path for path in model_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
        ]
        return directories[0] if len(directories) == 1 else model_dir

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
            raise SesError("vosk can't translate: use a whisper model for --translate")

        recognizer = self.recognizer_type(self.model, WHISPER_SAMPLE_RATE)
        recognizer.SetWords(True)

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        segments = []
        last_end = 0.0

        for offset in range(0, audio.size, CHUNK_SAMPLES):
            stop = min(offset + CHUNK_SAMPLES, audio.size)
            chunk = self.pcm16(audio[offset:stop])
            if recognizer.AcceptWaveform(chunk):
                segment = self.segment_from_json(
                    recognizer.Result(),
                    len(segments),
                    last_end,
                    stop / WHISPER_SAMPLE_RATE,
                    word_timestamps,
                )
                if segment is not None:
                    segments.append(segment)
                    last_end = max(last_end, segment["end"])

        segment = self.segment_from_json(
            recognizer.FinalResult(),
            len(segments),
            last_end,
            audio.size / WHISPER_SAMPLE_RATE,
            word_timestamps,
        )
        if segment is not None:
            segments.append(segment)

        text = " ".join(segment["text"].strip() for segment in segments).strip()
        return {"text": text, "segments": segments, "language": language}

    @staticmethod
    def pcm16(samples):
        safe = np.nan_to_num(
            np.asarray(samples, dtype=np.float32),
            copy=True,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        np.clip(safe, -1.0, 1.0, out=safe)
        return (safe * 32767.0).astype("<i2").tobytes()

    @classmethod
    def segment_from_json(
        cls,
        raw_result,
        index,
        fallback_start,
        fallback_end,
        include_words,
    ):
        try:
            payload = json.loads(raw_result or "{}")
        except (TypeError, json.JSONDecodeError) as error:
            raise SesError("vosk returned invalid recognition JSON") from error
        if not isinstance(payload, dict):
            raise SesError("vosk returned invalid recognition JSON")

        words = []
        raw_words = payload.get("result")
        if isinstance(raw_words, list):
            for raw_word in raw_words:
                word = cls.word_from(raw_word)
                if word is not None:
                    words.append(word)

        text = str(payload.get("text") or "").strip()
        if not text and words:
            text = " ".join(word["word"] for word in words)
        if not text:
            return None

        if words:
            start = min(word["start"] for word in words)
            end = max(word["end"] for word in words)
        else:
            start = max(cls.finite_float(fallback_start, 0.0), 0.0)
            end = max(cls.finite_float(fallback_end, start), start)

        segment = {
            "id": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
        }
        if include_words and words:
            segment["words"] = words
        return segment

    @classmethod
    def word_from(cls, raw_word):
        if not isinstance(raw_word, dict):
            return None
        text = str(raw_word.get("word") or "").strip()
        if not text:
            return None

        start = max(cls.finite_float(raw_word.get("start"), 0.0), 0.0)
        end = max(cls.finite_float(raw_word.get("end"), start), start)
        word = {
            "word": text,
            "start": round(start, 3),
            "end": round(end, 3),
        }
        probability = cls.finite_float(raw_word.get("conf"), None)
        if probability is not None:
            word["probability"] = probability
        return word

    @staticmethod
    def finite_float(value, default):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default
