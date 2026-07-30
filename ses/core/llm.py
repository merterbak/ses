import json
import os
from dataclasses import dataclass
import httpx
from .. import SesError
from .text import CONTENT, THINKING, ThinkingSplitter

OLLAMA_URL = "http://127.0.0.1:11434"
LMSTUDIO_URL = "http://127.0.0.1:1234"

OLLAMA = "ollama"
OPENAI = "openai"

NO_BRAIN_HINT = (
    "no local LLM found, start Ollama (`ollama serve`) or LM Studio's server "
    "(`lms server start`), or point SES_BRAIN_URL at any OpenAI-compatible server"
)


def normalize_url(url):
    url = url.strip().rstrip("/")
    return url if url.startswith("http") else f"http://{url}"


def responds(kind, base_url):
    path = "/api/tags" if kind == OLLAMA else "/v1/models"
    try:
        return httpx.get(base_url + path, timeout=1.5).status_code == 200
    except Exception:
        return False


@dataclass(frozen=True)
class Brain:
    kind: str
    base_url: str
    label: str

    def models(self):
        if self.kind == OLLAMA:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return [model["name"] for model in response.json().get("models", [])]
        response = httpx.get(f"{self.base_url}/v1/models", timeout=5)
        response.raise_for_status()
        return [model["id"] for model in response.json().get("data", [])]

    def loaded_models(self):
        if self.kind != OLLAMA:
            return None
        try:
            response = httpx.get(f"{self.base_url}/api/ps", timeout=3)
            response.raise_for_status()
            return [model.get("name", "") for model in response.json().get("models", [])]
        except Exception:
            return None

    def resolve_model(self, preferred):
        available = self.models()
        if preferred:
            if preferred in available:
                return preferred
            if self.kind == OLLAMA and f"{preferred}:latest" in available:
                return f"{preferred}:latest"
            listed = ", ".join(available[:6]) or "none"
            raise SesError(f"model '{preferred}' not available in {self.label} (has: {listed})")

        if not available:
            fix = "try: ollama pull llama3.2" if self.kind == OLLAMA else "download one in LM Studio"
            raise SesError(f"{self.label} is running but has no models, {fix}")
        return available[0]

    def chat(self, model, messages, timeout=300):
        if self.kind == OLLAMA:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "") or ""

        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": model, "messages": messages, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"].get("content") or ""

    def chat_stream(self, model, messages, timeout=300, think=True):
        if self.kind == OLLAMA:
            yield from self.stream_ollama(model, messages, timeout, think)
        else:
            yield from self.stream_openai(model, messages, timeout)

    def stream_ollama(self, model, messages, timeout, think, retrying=False):
        body = {"model": model, "messages": messages, "stream": True}
        if think and not retrying:
            body["think"] = True

        splitter = ThinkingSplitter()
        with httpx.stream("POST", f"{self.base_url}/api/chat", json=body, timeout=timeout) as response:
            if response.status_code == 400 and not retrying:
                yield from self.stream_ollama(model, messages, timeout, think, retrying=True)
                return
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise SesError(f"ollama: {event['error']}")

                message = event.get("message", {})
                if message.get("thinking"):
                    yield (THINKING, message["thinking"])
                if message.get("content"):
                    yield from splitter.feed(message["content"])
                if event.get("done"):
                    break
        yield from splitter.flush()

    def stream_openai(self, model, messages, timeout):
        splitter = ThinkingSplitter()
        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json={"model": model, "messages": messages, "stream": True},
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break

                delta = (json.loads(payload).get("choices") or [{}])[0].get("delta", {})
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    yield (THINKING, reasoning)
                if delta.get("content"):
                    yield from splitter.feed(delta["content"])
        yield from splitter.flush()


def detect():
    override = os.environ.get("SES_BRAIN_URL")
    if override:
        base_url = normalize_url(override)
        if responds(OLLAMA, base_url):
            return Brain(OLLAMA, base_url, "Ollama")
        if responds(OPENAI, base_url):
            return Brain(OPENAI, base_url, "OpenAI-compatible server")
        return None

    ollama_url = normalize_url(os.environ.get("OLLAMA_HOST", OLLAMA_URL))
    if responds(OLLAMA, ollama_url):
        return Brain(OLLAMA, ollama_url, "Ollama")
    if responds(OPENAI, LMSTUDIO_URL):
        return Brain(OPENAI, LMSTUDIO_URL, "LM Studio")
    return None


__all__ = ["Brain", "CONTENT", "NO_BRAIN_HINT", "OLLAMA", "OPENAI", "THINKING", "detect"]
