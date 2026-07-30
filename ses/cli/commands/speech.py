import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
import numpy as np
import typer
from ...audio import (
    WHISPER_SAMPLE_RATE,
    load_audio,
    mp3_bytes,
    play,
    record_for,
    record_until_enter,
    record_until_silence,
    save_audio,
    stream_utterances,
)
from ...core import transcripts
from ...core.text import clean, collapse
from ..ui import (
    console,
    default_stt,
    default_tts,
    error_console,
    fail,
    format_speed,
    load_model,
)

MIN_RECORDING_SAMPLES = WHISPER_SAMPLE_RATE // 4

CLIPBOARD_COMMANDS = {
    "darwin": ["pbcopy"],
    "win32": ["clip"],
}
LINUX_CLIPBOARD = ["xclip", "-selection", "clipboard"]


def register(app):
    app.command()(say)
    app.command()(transcribe)
    app.command()(listen)


def say(
    text: str = typer.Argument(..., help='Text to speak, or "-" to read stdin.'),
    model: str = typer.Option(None, "--model", "-m", help="TTS model (default: tts-english)"),
    voice: str = typer.Option(None, "--voice", "-v", help="Voice id (ses voices)"),
    speed: float = typer.Option(1.0, "--speed", "-s", min=0.5, max=2.0),
    lang: str = typer.Option(None, "--lang", help="Override language (e.g. en-us, fr-fr)"),
    out: Path = typer.Option(None, "--out", "-o", help="Write a .wav/.mp3 instead of playing."),
    no_play: bool = typer.Option(False, "--no-play", help="Don't play the audio."),
):
    """Speak text out loud, or render it to a file."""
    if text == "-":
        text = sys.stdin.read()
    if not text.strip():
        fail("no text to speak")

    model_handle = load_model(model or default_tts())
    if model_handle.kind != "tts":
        fail(f"'{model_handle.name}' is not a TTS model, try tts-english")

    started = time.time()
    with console.status(f"[bold]{voice}[/bold] speaking…", spinner="dots"):
        samples, rate = model_handle.engine.synth(text, voice=voice, speed=speed, lang=lang)

    took = time.time() - started
    duration = len(samples) / rate
    console.print(
        f"[dim]{duration:.1f}s of speech in {took:.1f}s, "
        f"{format_speed(duration, took)}[/dim]"
    )

    if out:
        if out.suffix.lower() == ".mp3":
            out.write_bytes(mp3_bytes(samples, rate))
        else:
            save_audio(out, samples, rate)
        console.print(f"[green]✓[/green] wrote {out}")
    elif not no_play:
        play(samples, rate)


def transcribe(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, help="Audio file (wav/mp3/m4a…)"),
    model: str = typer.Option(None, "--model", "-m", help="STT model (default: whisper-base)"),
    language: str = typer.Option(None, "--language", "-l", help="Language hint (e.g. en, tr)"),
    fmt: str = typer.Option("text", "--format", "-f", help="text | json | verbose_json | srt | vtt"),
    out: Path = typer.Option(None, "--out", "-o", help="Write the result to a file."),
    translate: bool = typer.Option(False, "--translate", help="Translate the speech to English."),
):
    """Transcribe an audio file."""
    model_handle = load_model(model or default_stt())
    if model_handle.kind != "stt":
        fail(f"'{model_handle.name}' is not a speech-to-text model, try whisper-base")

    samples = load_audio(file)
    duration = len(samples) / WHISPER_SAMPLE_RATE

    started = time.time()
    with console.status(f"transcribing {file.name} ({duration:.0f}s)…", spinner="dots"):
        result = clean(
            model_handle.engine.transcribe(
                samples,
                language=language,
                task="translate" if translate else "transcribe",
                word_timestamps=fmt == "verbose_json",
            )
        )
    took = time.time() - started

    rendered = render(result, fmt, duration)
    if out:
        out.write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
        console.print(
            f"[green]✓[/green] wrote {out} [dim]({duration:.0f}s audio in {took:.1f}s)[/dim]"
        )
    else:
        console.print(rendered)
        error_console.print(
            f"[dim]{duration:.0f}s of audio in {took:.1f}s, "
            f"{format_speed(duration, took, noun='audio')}[/dim]"
        )


def render(result, fmt, duration):
    if fmt == "text":
        return result["text"]
    if fmt == "json":
        return json.dumps({"text": result["text"]}, ensure_ascii=False, indent=2)
    if fmt == "verbose_json":
        return json.dumps(
            transcripts.to_verbose_json(result, duration), ensure_ascii=False, indent=2
        )
    if fmt == "srt":
        return transcripts.to_srt(result.get("segments", []))
    if fmt == "vtt":
        return transcripts.to_vtt(result.get("segments", []))
    fail(f"unknown format '{fmt}'")


def listen(
    model: str = typer.Option(None, "--model", "-m", help="STT model (default: whisper-base)"),
    seconds: float = typer.Option(None, "--seconds", "-t", help="Record N seconds, not until Enter."),
    auto: bool = typer.Option(False, "--auto", "-a", help="Hands-free: stop when you finish speaking."),
    live: bool = typer.Option(False, "--live", help="Keep listening, printing each sentence as you say it."),
    language: str = typer.Option(None, "--language", "-l"),
    copy: bool = typer.Option(False, "--copy", "-c", help="Copy the transcript to the clipboard."),
    out: Path = typer.Option(None, "--out", "-o", help="Also save the recording as wav."),
):
    """Record from the microphone and print what you said."""
    model_handle = load_model(model or default_stt())
    if model_handle.kind != "stt":
        fail(f"'{model_handle.name}' is not a speech-to-text model")

    if live:
        transcribe_live(model_handle, language, copy, out)
        return

    samples = record(seconds, auto)
    if len(samples) < MIN_RECORDING_SAMPLES:
        fail("recording too short")
    if out:
        save_audio(out, samples, WHISPER_SAMPLE_RATE)

    with console.status("transcribing…", spinner="dots"):
        text = collapse(
            model_handle.engine.transcribe(samples, language=language)["text"]
        )

    console.print(text)
    if copy:
        copy_to_clipboard(text)


def level_meter(meter, width=16):
    level, threshold = meter["level"], meter["threshold"]
    filled = 0 if threshold <= 0 else min(width, int(width * level / (threshold * 2)))
    bar = "█" * filled + "·" * (width - filled)
    if meter["speaking"]:
        return f"[red]●[/red] hearing you  [green]{bar}[/green]"
    return f"[dim]listening     {bar}[/dim]"


def transcribe_live(model_handle, language, copy, out):
    stop = threading.Event()
    incoming = queue.Queue()

    meter = {"level": 0.0, "threshold": 0.0, "speaking": False}

    def on_level(level, threshold, speaking):
        meter.update(level=level, threshold=threshold, speaking=speaking)

    def capture():
        try:
            for utterance in stream_utterances(stop=stop, on_level=on_level):
                incoming.put(utterance)
        except Exception as error:
            incoming.put(error)
        finally:
            incoming.put(None)

    worker = threading.Thread(target=capture, daemon=True)
    worker.start()

    console.print(
        f"[dim]🎧 live, [bold]{model_handle.name}[/bold] listening, Ctrl-C to stop[/dim]"
    )
    started = time.time()
    lines = []
    captured = []

    try:
        with console.status(level_meter(meter), spinner="dots") as status:
            while True:
                try:
                    item = incoming.get(timeout=0.15)
                except queue.Empty:
                    status.update(level_meter(meter))
                    continue

                if item is None:
                    break
                if isinstance(item, Exception):
                    stop.set()
                    fail(f"microphone failed: {item}")

                captured.append(item)
                status.update("[dim]transcribing…[/dim]")
                text = collapse(
                    model_handle.engine.transcribe(item, language=language)["text"]
                ).strip()
                status.update(level_meter(meter))
                if not text:
                    continue
                elapsed = int(time.time() - started)
                lines.append(text)
                console.print(f"[dim]{elapsed // 60:02d}:{elapsed % 60:02d}[/dim]  {text}")
    except KeyboardInterrupt:
        console.print()
    finally:
        stop.set()

    if not lines:
        console.print("[dim]heard nothing[/dim]")
        return

    transcript = " ".join(lines)
    if out and captured:
        save_audio(out, np.concatenate(captured), WHISPER_SAMPLE_RATE)
        console.print(f"[green]✓[/green] wrote {out}")
    if copy:
        copy_to_clipboard(transcript)


def record(seconds, auto):
    if seconds:
        console.print(f"[red]●[/red] recording {seconds:.0f}s…")
        return record_for(seconds)

    if auto:
        console.print("[dim]🎧 listening, speak, it stops when you finish[/dim]", end="\r")
        samples = record_until_silence(
            on_speech=lambda: console.print("[red]●[/red] hearing you…      ", end="\r")
        )
        if len(samples) == 0:
            fail("heard nothing")
        return samples

    return record_until_enter(prompt="\x1b[31m●\x1b[0m recording, press Enter to stop ")


def copy_to_clipboard(text):
    command = CLIPBOARD_COMMANDS.get(sys.platform, LINUX_CLIPBOARD)
    try:
        subprocess.run(command, input=text.encode(), check=True)
        error_console.print("[dim]copied to clipboard[/dim]")
    except (OSError, subprocess.CalledProcessError):
        error_console.print("[dim]--copy needs pbcopy/clip/xclip on this system[/dim]")
