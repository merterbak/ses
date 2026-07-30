VERBOSE_SEGMENT_FIELDS = (
    "id",
    "seek",
    "start",
    "end",
    "text",
    "temperature",
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
)


def timestamp(seconds, decimal_mark):
    seconds = max(seconds, 0.0)
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    whole_seconds = int(seconds % 60)
    milliseconds = int(round(seconds % 1 * 1000))
    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{decimal_mark}{milliseconds:03d}"


def to_srt(segments):
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = timestamp(segment["start"], ",")
        end = timestamp(segment["end"], ",")
        blocks.append(f"{index}\n{start} --> {end}\n{segment['text'].strip()}")
    return "\n\n".join(blocks) + "\n"


def to_vtt(segments):
    blocks = ["WEBVTT"]
    for segment in segments:
        start = timestamp(segment["start"], ".")
        end = timestamp(segment["end"], ".")
        blocks.append(f"{start} --> {end}\n{segment['text'].strip()}")
    return "\n\n".join(blocks) + "\n"


def to_verbose_json(result, duration):
    segments = []
    for segment in result.get("segments", []):
        trimmed = {key: segment[key] for key in VERBOSE_SEGMENT_FIELDS if key in segment}
        if "words" in segment:
            trimmed["words"] = segment["words"]
        segments.append(trimmed)

    return {
        "task": "transcribe",
        "language": result.get("language"),
        "duration": round(duration, 3),
        "text": result.get("text", ""),
        "segments": segments,
    }
