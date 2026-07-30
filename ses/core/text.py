import re

_BOUNDARY = re.compile(r'(?<=[.!?…])["\')\]]*\s+|\n{2,}')

MIN_SENTENCE_CHARS = 24

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"

THINKING = "thinking"
CONTENT = "content"

MAX_PHRASE_WORDS = 30
MIN_REPEATS = 3
MIN_REPEATED_WORDS = 8


class SentenceChunker:
    def __init__(self, min_chars=MIN_SENTENCE_CHARS):
        self.min_chars = min_chars
        self.buffer = ""

    def feed(self, text):
        self.buffer += text
        parts = _BOUNDARY.split(self.buffer)
        if len(parts) <= 1:
            return []

        self.buffer = parts.pop()
        sentences = []
        pending = ""
        for part in parts:
            pending = f"{pending} {part}".strip()
            if len(pending) >= self.min_chars:
                sentences.append(pending)
                pending = ""
        if pending:
            self.buffer = f"{pending} {self.buffer}".strip()
        return sentences

    def flush(self):
        remainder = self.buffer.strip()
        self.buffer = ""
        return remainder or None


def split_sentences(text, min_chars=MIN_SENTENCE_CHARS):
    chunker = SentenceChunker(min_chars)
    sentences = chunker.feed(text)
    remainder = chunker.flush()
    if remainder:
        sentences.append(remainder)
    return sentences


class ThinkingSplitter:
    def __init__(self):
        self.inside_thinking = False
        self.buffer = ""

    @property
    def channel(self):
        return THINKING if self.inside_thinking else CONTENT

    def feed(self, text):
        self.buffer += text
        pieces = []
        while True:
            tag = CLOSE_TAG if self.inside_thinking else OPEN_TAG
            index = self.buffer.find(tag)
            if index != -1:
                if index > 0:
                    pieces.append((self.channel, self.buffer[:index]))
                self.buffer = self.buffer[index + len(tag):]
                self.inside_thinking = not self.inside_thinking
                continue

            held_back = self.partial_tag_length(tag)
            ready = self.buffer[: len(self.buffer) - held_back] if held_back else self.buffer
            if ready:
                pieces.append((self.channel, ready))
                self.buffer = self.buffer[len(ready):]
            return pieces

    def partial_tag_length(self, tag):
        for length in range(min(len(tag) - 1, len(self.buffer)), 0, -1):
            if self.buffer[-length:] == tag[:length]:
                return length
        return 0

    def flush(self):
        if not self.buffer:
            return []
        remainder, self.buffer = self.buffer, ""
        return [(self.channel, remainder)]


def trailing_repeat(words):
    best = None
    for size in range(1, min(MAX_PHRASE_WORDS, len(words) // MIN_REPEATS) + 1):
        phrase = words[-size:]
        start = len(words) - size
        repeats = 1
        while start - size >= 0 and words[start - size : start] == phrase:
            repeats += 1
            start -= size
        if repeats < MIN_REPEATS or repeats * size < MIN_REPEATED_WORDS:
            continue
        if best is None or repeats * size > best[0]:
            best = (repeats * size, start + size)
    return best


def collapse(text):
    words = text.split()
    if len(words) < MIN_REPEATED_WORDS:
        return text
    found = trailing_repeat(words)
    if found is None:
        return text
    return " ".join(words[: found[1]])


def clean(result):
    text = result.get("text", "")
    collapsed = collapse(text)
    if collapsed == text:
        return result
    return {**result, "text": collapsed}
