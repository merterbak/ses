<div align="center">

# 🎙️ ses

**Ollama for voice.** Pull, hot-swap and serve local speech models with one command, fully offline.


[![Platforms](https://img.shields.io/badge/macOS%20·%20Linux%20·%20Windows-supported-black?logo=apple)](#install)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](#install)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Models](https://img.shields.io/badge/80%20models-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](#the-models)

</div>

```bash
uv tool install ses

ses say "hello world"     # your speakers, model auto-pulled
ses listen --live         # live captions from your mic
ses talk                  # voice assistant on top of your Ollama model
ses serve                 # OpenAI-compatible API and a web playground
```

No model URLs, no config files, no Docker, no API keys.

## Why

Running speech models locally still means an afternoon of glue: hunt Hugging
Face for the right port of the right model, figure out which runtime your
machine can actually run, wire up audio formats, then redo all of it when you
switch models. Text models stopped being like this the day Ollama gave them
curated names, one `pull` and one API.

ses brings that to speech. It is a model manager, not another demo script:
`pull` a model by name, `ls` and `rm` what is installed, hot-swap without a
restart, and serve everything behind one API that keeps models warm.

One name gets the right backend for your machine. `ses pull whisper-base`
fetches MLX weights on Apple Silicon and CTranslate2 everywhere else, and your
commands never change. 49 languages to speak, 99 to transcribe. And through
MCP, Claude Code or Codex can speak and listen with the same local models.

## Install

macOS, Linux or Windows, Python 3.10+.

```bash
uv tool install ses             # core
uv tool install 'ses[default]'  # adds whisper.cpp, Vosk, Chatterbox, learned VAD
```

The only permission ses ever asks for is mic access, the first time you listen.

## Use it

```bash
$ ses say "The quick brown fox jumps over the lazy dog."
model kokoro not installed, pulling from onnx-community/Kokoro-82M-v1.0-ONNX (~340 MB)
✓ pulled kokoro
🔊

$ ses transcribe meeting.m4a -m whisper-turbo -f srt -o meeting.srt
✓ wrote meeting.srt

$ ses listen --live
🎧 live, whisper-turbo listening, Ctrl-C to stop
00:02  Let's start with the roadmap for next quarter.
00:09  Marta is taking the migration work.
```

`--live` keeps the mic open and prints each sentence the moment you finish it.
Boundaries come from voice activity detection; `ses[vad]` swaps the built-in
energy threshold for a learned detector that holds up in a noisy room.

## Give your coding agent a voice

This is the part I use every day. ses ships an MCP server, so any MCP client,
Claude Code, Codex, or your own agent, gets ears and a mouth that run locally:

```bash
uv tool install 'ses[mcp]'
claude mcp add ses -- ses mcp
```

That is the whole setup. Your agent now has eight tools:

- **`speak`** reads text out loud and returns when the audio ends. It takes a
  `language` argument, so an agent answering in Japanese picks a Japanese
  voice, downloading it on demand the first time.
- **`notify`** is the short version: a one-liner ping when a long build or test
  run finishes, so you can leave the desk.
- **`listen`** and **`dictate`** open the mic. Answer a question by speaking,
  or dictate a commit message instead of typing it.
- **`transcribe`** takes an audio file path, so "transcribe this voice memo
  and turn it into a ticket" works end to end.
- **`set_voice`**, **`list_voices`** and **`voice_mode`** let the agent switch
  voices and toggle spoken summaries without you touching a config file.

After that, things like *"summarize the diff and read it to me"* or *"tell me
out loud when the tests pass"* just work, in whatever language you write.
Nothing goes to a cloud API; the same local models serve every tool.

For Codex, put the same command in `~/.codex/config.toml` under
`[mcp_servers.ses]`.

## The models

80 curated models across 12 runtimes, 75 of them on every platform. Each one
was picked by download numbers first, then verified with a real recording. ⭐
marks the three ses recommends; they are also the three that run everywhere.

| speech to text | size | notes |
|---|---:|---|
| `whisper-turbo` ⭐ | 1.5 GB | large-v3 accuracy at a fraction of the cost |
| `whisper-tiny` … `whisper-large` | 71 MB – 2.9 GB | every size, 99 languages |
| `whisper-cpp-*` | 31 MB – 1.1 GB | GGML quantized, smallest downloads |
| `qwen-asr-large` | 2.3 GB | 5.76% WER, best Turkish in our tests (Apple Silicon) |
| `canary` · `parakeet` | 2.4 – 4.7 GB | open ASR leaderboard toppers, 25 European languages |
| `parakeet-v2` | 2.4 GB | English only, the fastest of the three |
| `vosk-*` | ~40 MB | six languages, tiny, CPU only |

| text to speech | size | notes |
|---|---:|---|
| `kokoro` ⭐ | 340 MB | most downloaded open voice model, 54 voices, 8 languages |
| `tts-<language>` ⭐ | ~61 MB | 49 languages via Piper, ~30× real time |
| `chatterbox-turbo` | 2.8 GB | voice cloning from a sample, English, slower than real time |
| `qwen-tts` · `omnivoice` | 1.6 GB | newer multilingual families (Apple Silicon) |

```bash
ses search german       # browse; any language name or code works
ses system              # what runs on this machine, and how fast
ses use stt whisper-turbo
```

The curated list is not the limit. Any Hugging Face repo in a format ses can
open installs directly, MLX, CTranslate2, ONNX and GGML out of the box, plain
PyTorch with `ses[transformers]`:

```bash
ses pull mlx-community/whisper-large-v3-mlx-4bit --engine mlx-whisper
ses pull facebook/mms-tts-tur --engine transformers-tts   # 1000-language MMS
```

## The assistant

```bash
ses talk
```

Mic in, Whisper to text, your Ollama or LM Studio model thinks, Kokoro speaks.
Hands free through voice activity detection, or `--push-to-talk` for Enter.
The voice starts on the first finished sentence while the model is still
writing. Three local models passing audio around your own machine.

ses does not run the thinking model itself, it borrows one from Ollama. `ses
pull` reaches those too, including any GGUF repo on Hugging Face:

```bash
ses pull llama3.2                                   # curated brain
ses pull hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
```

## The server

```bash
ses serve
# api         http://127.0.0.1:11435/v1
# playground  http://127.0.0.1:11435
```

Drop-in for any OpenAI SDK, just change the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11435/v1", api_key="ses")
client.audio.speech.create(model="kokoro", voice="af_heart",
                           input="Local models, cloud API shape.")
```

`/v1/audio/speech`, `/v1/audio/transcriptions` and `/translations` (json, srt,
vtt), `/v1/models`, plus streaming PCM and NDJSON chat. Models load on first
request, stay warm for `--keep-alive` (default 10m), least recently used is
evicted past `--max-loaded` (default 3).

