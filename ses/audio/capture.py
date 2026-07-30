import collections
import numpy as np
from .codec import WHISPER_SAMPLE_RATE, resample

BLOCK_SECONDS = 0.03
CALIBRATION_BLOCKS = 10
PRE_ROLL_SECONDS = 0.6
NOISE_MULTIPLIER = 4.0
MIN_THRESHOLD = 0.012

NOISE_WINDOW_BLOCKS = 120
NOISE_PERCENTILE = 25
MAX_THRESHOLD = 0.06

RELEASE_RATIO = 0.5
MIN_UTTERANCE_SECONDS = 0.4

SILERO_WINDOW = 512
SILERO_PRE_ROLL_SECONDS = 0.4
SILERO_METER_THRESHOLD = 0.02


def input_sample_rate():
    import sounddevice as sd

    device = sd.query_devices(kind="input")
    return int(device["default_samplerate"])


def record_until_enter(rate=WHISPER_SAMPLE_RATE, prompt=""):
    import sounddevice as sd

    device_rate = input_sample_rate()
    blocks = []

    def on_audio(indata, _frames, _time, _status):
        blocks.append(indata.copy())

    with sd.InputStream(samplerate=device_rate, channels=1, dtype="float32", callback=on_audio):
        if prompt:
            print(prompt, end="", flush=True)
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print()

    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return resample(np.concatenate(blocks)[:, 0], device_rate, rate)


def record_for(seconds, rate=WHISPER_SAMPLE_RATE):
    import sounddevice as sd

    device_rate = input_sample_rate()
    recording = sd.rec(int(seconds * device_rate), samplerate=device_rate, channels=1, dtype="float32")
    sd.wait()
    return resample(recording[:, 0], device_rate, rate)


class SpeechSegmenter:
    def __init__(self, block_seconds, silence_ms=900, max_seconds=60.0):
        self.block_seconds = block_seconds
        self.silence_seconds = silence_ms / 1000
        self.max_seconds = max_seconds
        self.pre_roll = collections.deque(maxlen=max(1, int(PRE_ROLL_SECONDS / block_seconds)))
        self.noise_levels = collections.deque(maxlen=NOISE_WINDOW_BLOCKS)
        self.threshold = MIN_THRESHOLD
        self.speech = []
        self.speaking = False
        self.silent_for = 0.0
        self.speech_seconds = 0.0
        self.waited_for = 0.0
        self.level = 0.0

    @property
    def calibrating(self):
        return len(self.noise_levels) < CALIBRATION_BLOCKS

    def retune(self):
        floor = float(np.percentile(self.noise_levels, NOISE_PERCENTILE))
        self.threshold = min(max(NOISE_MULTIPLIER * floor, MIN_THRESHOLD), MAX_THRESHOLD)

    def feed(self, block):
        self.level = float(np.sqrt(np.mean(block * block)))

        if not self.speaking:
            self.noise_levels.append(self.level)
            self.pre_roll.append(block)
            if self.calibrating:
                return None
            self.retune()
            self.waited_for += self.block_seconds
            if self.level > self.threshold:
                self.speaking = True
                self.speech.extend(self.pre_roll)
            return None

        self.speech.append(block)
        self.speech_seconds += self.block_seconds
        quiet = self.level <= self.threshold * RELEASE_RATIO
        self.silent_for = self.silent_for + self.block_seconds if quiet else 0.0
        if self.silent_for >= self.silence_seconds or self.speech_seconds >= self.max_seconds:
            spoken = self.speech_seconds - self.silent_for
            utterance = self.take()
            return utterance if spoken >= MIN_UTTERANCE_SECONDS else None
        return None

    def take(self):
        utterance = np.concatenate(self.speech) if self.speech else np.zeros(0, dtype=np.float32)
        self.speech = []
        self.speaking = False
        self.silent_for = 0.0
        self.speech_seconds = 0.0
        self.waited_for = 0.0
        self.pre_roll.clear()
        return utterance


def open_blocks(block_seconds=BLOCK_SECONDS):
    import sounddevice as sd

    device_rate = input_sample_rate()
    block_frames = max(256, int(device_rate * block_seconds))
    stream = sd.InputStream(
        samplerate=device_rate, blocksize=block_frames, channels=1, dtype="float32"
    )
    return stream, device_rate, block_frames


def record_until_silence(
    rate=WHISPER_SAMPLE_RATE,
    silence_ms=900,
    max_seconds=60.0,
    wait_seconds=25.0,
    on_speech=None,
):
    stream, device_rate, block_frames = open_blocks()
    segmenter = SpeechSegmenter(block_frames / device_rate, silence_ms, max_seconds)
    announced = False

    with stream:
        while True:
            block = stream.read(block_frames)[0][:, 0]
            utterance = segmenter.feed(block)
            if utterance is not None:
                return resample(utterance, device_rate, rate)

            if segmenter.speaking and not announced:
                announced = True
                if on_speech:
                    on_speech()
            elif not segmenter.speaking and segmenter.waited_for > wait_seconds:
                return np.zeros(0, dtype=np.float32)


class SileroSegmenter:
    def __init__(self, block_seconds, silence_ms=800, max_seconds=30.0, threshold=0.5):
        from silero_vad import VADIterator, load_silero_vad

        self.block_seconds = block_seconds
        self.max_seconds = max_seconds
        self.iterator = VADIterator(
            load_silero_vad(),
            threshold=threshold,
            sampling_rate=WHISPER_SAMPLE_RATE,
            min_silence_duration_ms=silence_ms,
            speech_pad_ms=200,
        )
        self.pending = np.zeros(0, dtype=np.float32)
        self.pre_roll = np.zeros(0, dtype=np.float32)
        self.speech = []
        self.speaking = False
        self.speech_seconds = 0.0
        self.waited_for = 0.0
        self.level = 0.0
        self.threshold = SILERO_METER_THRESHOLD

    @property
    def calibrating(self):
        return False

    def feed(self, block):
        import torch

        block = np.asarray(block, dtype=np.float32)
        self.level = float(np.sqrt(np.mean(block * block))) if len(block) else 0.0
        self.pending = np.concatenate([self.pending, block])

        finished = None
        while len(self.pending) >= SILERO_WINDOW:
            window, self.pending = self.pending[:SILERO_WINDOW], self.pending[SILERO_WINDOW:]
            event = self.iterator(torch.from_numpy(window))
            seconds = SILERO_WINDOW / WHISPER_SAMPLE_RATE

            if self.speaking:
                self.speech.append(window)
                self.speech_seconds += seconds
            else:
                self.pre_roll = np.concatenate([self.pre_roll, window])
                keep = int(SILERO_PRE_ROLL_SECONDS * WHISPER_SAMPLE_RATE)
                self.pre_roll = self.pre_roll[-keep:]
                self.waited_for += seconds

            if event and "start" in event and not self.speaking:
                self.speaking = True
                self.speech = [self.pre_roll.copy()]
                self.speech_seconds = len(self.pre_roll) / WHISPER_SAMPLE_RATE
                self.pre_roll = np.zeros(0, dtype=np.float32)
            elif event and "end" in event and self.speaking:
                finished = self.take()
            elif self.speaking and self.speech_seconds >= self.max_seconds:
                finished = self.take()

        return finished

    def take(self):
        utterance = np.concatenate(self.speech) if self.speech else np.zeros(0, dtype=np.float32)
        self.speech = []
        self.speaking = False
        self.speech_seconds = 0.0
        self.waited_for = 0.0
        self.pre_roll = np.zeros(0, dtype=np.float32)
        if len(utterance) / WHISPER_SAMPLE_RATE < MIN_UTTERANCE_SECONDS:
            return np.zeros(0, dtype=np.float32)
        return utterance


def silero_available():
    import importlib.util

    return importlib.util.find_spec("silero_vad") is not None


def build_segmenter(block_seconds, silence_ms, max_seconds, learned=None):
    if learned is False or (learned is None and not silero_available()):
        return SpeechSegmenter(block_seconds, silence_ms, max_seconds)
    return SileroSegmenter(block_seconds, silence_ms, max_seconds)


def stream_utterances(
    rate=WHISPER_SAMPLE_RATE, silence_ms=800, max_seconds=30.0, stop=None,
    on_level=None, learned=None,
):
    stream, device_rate, block_frames = open_blocks()
    segmenter = build_segmenter(block_frames / device_rate, silence_ms, max_seconds, learned)

    with stream:
        while stop is None or not stop.is_set():
            utterance = segmenter.feed(stream.read(block_frames)[0][:, 0])
            if on_level:
                on_level(segmenter.level, segmenter.threshold, segmenter.speaking)
            if utterance is not None and len(utterance):
                yield resample(utterance, device_rate, rate)
