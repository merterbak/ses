import json
import queue
import re
import threading
import typer
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from ... import SesError
from ...audio import (
    WHISPER_SAMPLE_RATE,
    play,
    record_until_enter,
    record_until_silence,
)
from ...core import conversations, llm, paths
from ...core.llm import THINKING
from ...core.text import SentenceChunker, collapse
from ..ui import console, default_stt, default_tts, fail, load_model

MIN_SPEECH_SAMPLES = WHISPER_SAMPLE_RATE // 4

VOICE_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Answer briefly and conversationally, "
    "in plain sentences without markdown, lists or code."
)
CHAT_SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and conversationally."

_MARKDOWN_NOISE = re.compile(r"[*_#`~]+|\[(.*?)\]\(.*?\)")


def register(app):
    app.command("talk")(talk)
    app.command("run", hidden=True)(talk)
    app.command()(chat)
    app.command()(history)


def speakable(text):
    without_markup = _MARKDOWN_NOISE.sub(lambda match: match.group(1) or "", text)
    return re.sub(r"\s+", " ", without_markup).strip()


def ollama_pull(base_url, name):
    import httpx

    console.print(f"[dim]brain[/dim] [bold]{name}[/bold] [dim]not in Ollama, pulling…[/dim]")
    try:
        with httpx.stream("POST", f"{base_url}/api/pull", json={"name": name}, timeout=None) as response:
            last_status = ""
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    fail(f"ollama pull failed: {event['error']}")
                status = event.get("status", "")
                if status and status != last_status:
                    last_status = status
                    console.print(f"[dim]  {status}[/dim]")
    except SesError:
        raise
    except Exception as error:
        fail(f"ollama pull {name} failed: {error}")


def connect(preferred, allow_pull=True):
    brain = llm.detect()
    if brain is None:
        fail(llm.NO_BRAIN_HINT)

    try:
        return brain, brain.resolve_model(preferred)
    except SesError:
        if allow_pull and brain.kind == llm.OLLAMA and preferred:
            ollama_pull(brain.base_url, preferred)
            return brain, brain.resolve_model(preferred)
        raise


class SpeechPlayer:
    def __init__(self, engine, voice, speed):
        self.engine = engine
        self.voice = voice
        self.speed = speed
        self.queue = queue.Queue()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while True:
            sentence = self.queue.get()
            if sentence is None:
                return
            if self.stopped.is_set():
                continue
            try:
                samples, rate = self.engine.synth(sentence, voice=self.voice, speed=self.speed)
                if not self.stopped.is_set():
                    play(samples, rate)
            except Exception:
                pass

    def say(self, sentence):
        spoken = speakable(sentence)
        if spoken:
            self.queue.put(spoken)

    def stop(self):
        self.stopped.set()

    def finish(self):
        self.queue.put(None)
        self.thread.join()


def talk(
    brain: str = typer.Option(None, "--brain", "-b", help="LLM to think with (default: ses use brain)."),
    voice: str = typer.Option("af_heart", "--voice", "-v"),
    stt: str = typer.Option(None, "--stt", help="STT model (default: whisper-base)"),
    tts: str = typer.Option(None, "--tts", help="TTS model (default: tts-english)"),
    speed: float = typer.Option(1.0, "--speed", "-s", min=0.5, max=2.0),
    push_to_talk: bool = typer.Option(
        False, "--push-to-talk", "-p", help="Press Enter to send each turn instead of auto-detecting."
    ),
    think: bool = typer.Option(
        True, "--think/--no-think", help="Show the model's reasoning (--no-think hides and skips it)."
    ),
    system: str = typer.Option(VOICE_SYSTEM_PROMPT, "--system", help="System prompt for the brain."),
):
    """Voice-chat with a local LLM: mic → Whisper → LLM → Kokoro."""
    connection, model = connect(brain or paths.default_for("brain"))
    ears = load_model(stt or default_stt())
    mouth = load_model(tts or default_tts())

    mode = "press Enter to send" if push_to_talk else "just speak, it detects when you finish"
    console.print(
        Panel.fit(
            f"[bold]🎙  ses talk[/bold]\n"
            f"[dim]ears[/dim] {ears.name}   [dim]brain[/dim] {model} · {connection.label}   "
            f"[dim]mouth[/dim] {mouth.name} ({voice})\n"
            f"[dim]{mode} · Ctrl+C to quit[/dim]",
            border_style="dim",
        )
    )

    messages = [{"role": "system", "content": system}]
    while True:
        try:
            samples = listen(push_to_talk)
        except KeyboardInterrupt:
            break
        if samples is None:
            continue
        if len(samples) < MIN_SPEECH_SAMPLES:
            console.print("[dim](too short, try again)[/dim]")
            continue

        with console.status("hearing…", spinner="dots"):
            heard = collapse(ears.engine.transcribe(samples)["text"]).strip()
        if not heard:
            console.print("[dim](heard nothing)[/dim]")
            continue

        console.print(f"[bold cyan]you[/bold cyan]  {heard}")
        messages.append({"role": "user", "content": heard})

        reply, interrupted = speak_reply(connection, model, messages, mouth, voice, speed, think)
        if interrupted:
            break
        messages.append({"role": "assistant", "content": reply})

    console.print("\n[dim]bye 👋[/dim]")


def listen(push_to_talk):
    if push_to_talk:
        return record_until_enter(prompt="\x1b[31m●\x1b[0m listening, Enter to send ")

    console.print("[dim]🎧 listening…[/dim]", end="\r")
    samples = record_until_silence(
        on_speech=lambda: console.print("[red]●[/red] hearing you…", end="\r")
    )
    if len(samples) == 0:
        console.print("[dim](quiet for a while, still here, just speak)[/dim]")
        return None
    return samples


def speak_reply(connection, model, messages, mouth, voice, speed, think):
    short_name = model.split(":")[0].split("/")[-1]
    warm = connection.loaded_models()
    if warm is not None and model not in warm:
        waiting_message = f"loading [bold]{model}[/bold] into memory, first use takes longer…"
    else:
        waiting_message = f"{short_name} is thinking…"

    status = console.status(f"[dim]{waiting_message}[/dim]", spinner="dots")
    status.start()

    player = SpeechPlayer(mouth.engine, voice, speed)
    chunker = SentenceChunker()
    reply = ""
    waiting = True
    showing_thoughts = False
    interrupted = False

    try:
        for channel, token in connection.chat_stream(model, messages, think=think):
            if channel == THINKING:
                if not think:
                    continue
                if waiting:
                    status.stop()
                    waiting = False
                if not showing_thoughts:
                    console.print("[dim italic]💭 ", end="")
                    showing_thoughts = True
                console.print(f"[dim italic]{token}[/dim italic]", end="")
                continue

            if waiting:
                status.stop()
                waiting = False
            if showing_thoughts:
                print()
                showing_thoughts = False
            if not reply:
                console.print(f"[bold magenta]{short_name}[/bold magenta]  ", end="")

            reply += token
            print(token, end="", flush=True)
            for sentence in chunker.feed(token):
                player.say(sentence)

        remainder = chunker.flush()
        if remainder:
            player.say(remainder)
    except KeyboardInterrupt:
        interrupted = True
        player.stop()
    except SesError as error:
        player.finish()
        fail(str(error))
    except Exception as error:
        player.finish()
        fail(f"{connection.label} request failed: {error}")
    finally:
        if waiting:
            status.stop()

    print()
    try:
        player.finish()
    except KeyboardInterrupt:
        interrupted = True
        player.stop()
    return reply, interrupted


def chat(
    message: str = typer.Argument(None, help="One-shot message; omit for an interactive session."),
    brain: str = typer.Option(None, "--brain", "-b", help="Ollama/LM Studio model."),
    cont: bool = typer.Option(False, "--continue", "-c", help="Continue the most recent conversation."),
    conversation: str = typer.Option(None, "--conversation", help="Continue a specific conversation."),
    speak: bool = typer.Option(False, "--speak", help="Also read replies aloud."),
    voice: str = typer.Option("af_heart", "--voice", "-v"),
    think: bool = typer.Option(
        True, "--think/--no-think", help="Show the model's reasoning (--no-think hides and skips it)."
    ),
    system: str = typer.Option(CHAT_SYSTEM_PROMPT, "--system", help="System prompt."),
):
    """Text chat with your local LLM, saved to history (see: ses history)."""
    connection, model = connect(brain or paths.default_for("brain"), allow_pull=False)
    current = open_conversation(conversation, cont, model)
    mouth = load_model(default_tts()) if speak else None

    def turn(user_text):
        run_chat_turn(connection, model, current, user_text, system, think, mouth, voice)

    if message:
        turn(message)
        console.print(f"[dim]saved to {current['id']}, resume with: ses chat -c[/dim]")
        return

    console.print(
        Panel.fit(
            f"[bold]💬 ses chat[/bold]  [dim]{current.get('title', 'New chat')}[/dim]\n"
            f"[dim]brain[/dim] {model}   [dim]id[/dim] {current['id']}\n"
            f"[dim]type your message · /exit to quit · history: ses history[/dim]",
            border_style="dim",
        )
    )
    for entry in current["messages"]:
        who = "[bold cyan]you[/bold cyan]" if entry["role"] == "user" else "[bold magenta]ai[/bold magenta]"
        console.print(f"{who}  {entry['content']}")

    while True:
        try:
            user_text = console.input("[bold cyan]you[/bold cyan]  ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_text.strip() in ("/exit", "/quit", "/q"):
            break
        if user_text.strip():
            turn(user_text)

    console.print(f"[dim]saved · resume with: ses chat --conversation {current['id']}[/dim]")


def open_conversation(conversation_id, continue_latest, model):
    if conversation_id:
        conversation = conversations.load(conversation_id)
        if conversation is None:
            fail(f"conversation '{conversation_id}' not found, see: ses history")
        return conversation

    if continue_latest:
        latest = conversations.latest_id()
        conversation = conversations.load(latest) if latest else None
        if conversation is not None:
            return conversation

    return conversations.create(brain=model)


def run_chat_turn(connection, model, conversation, user_text, system, think, mouth, voice):
    conversations.append(conversation, "user", user_text)
    messages = [{"role": "system", "content": system}] + [
        {"role": entry["role"], "content": entry["content"]} for entry in conversation["messages"]
    ]

    console.print(f"[bold magenta]{model.split(':')[0].split('/')[-1]}[/bold magenta]")
    reply = ""
    thoughts = ""

    try:
        with Live(console=console, refresh_per_second=12, vertical_overflow="visible") as live:
            for channel, token in connection.chat_stream(model, messages, think=think):
                if channel == THINKING:
                    if not think:
                        continue
                    thoughts += token
                else:
                    reply += token

                parts = []
                if think and thoughts:
                    parts.append(Text(f"💭 {thoughts.strip()}", style="dim italic"))
                parts.append(Markdown(reply or "…"))
                live.update(Group(*parts))
    except SesError as error:
        print()
        fail(str(error))

    console.print()
    conversations.append(conversation, "assistant", reply)

    if mouth is not None:
        spoken = speakable(reply)
        if spoken:
            samples, rate = mouth.engine.synth(spoken, voice=voice)
            play(samples, rate)


def history(
    action: str = typer.Argument(None, help="(empty) list · show <id> · rm <id>"),
    target: str = typer.Argument(None, help="conversation id for show/rm"),
    delete_all: bool = typer.Option(False, "--all", help="with rm: delete every conversation"),
):
    """List, show or delete saved conversations."""
    action = (action or "list").lower()

    if action == "list":
        list_conversations()
    elif action == "show":
        show_conversation(target)
    elif action == "rm":
        remove_conversations(target, delete_all)
    else:
        fail(f"unknown action '{action}', use: list, show <id>, rm <id>")


def list_conversations():
    saved = conversations.list_all()
    if not saved:
        console.print("no conversations yet, start one with [bold]ses chat[/bold]")
        return

    table = Table(box=None, header_style="bold dim", pad_edge=False)
    table.add_column("ID", style="dim")
    table.add_column("TITLE", style="bold", max_width=44)
    table.add_column("MSGS", justify="right")
    table.add_column("BRAIN", style="dim")
    table.add_column("UPDATED", style="dim")

    for entry in saved:
        table.add_row(
            entry["id"],
            entry["title"],
            str(entry["messages"]),
            (entry.get("brain") or "").split(":")[0],
            entry["updated"][:16].replace("T", " "),
        )
    console.print(table)
    console.print(
        "\n[dim]open with[/dim] ses history show <id>  "
        "[dim]· resume with[/dim] ses chat --conversation <id>"
    )


def show_conversation(conversation_id):
    if not conversation_id:
        fail("which one?, ses history show <id>")
    conversation = conversations.load(conversation_id)
    if conversation is None:
        fail(f"conversation '{conversation_id}' not found")

    console.print(
        Panel.fit(
            f"[bold]{conversation.get('title', 'chat')}[/bold]\n"
            f"[dim]{conversation['id']} · {conversation.get('brain') or '?'} · "
            f"{len(conversation['messages'])} messages[/dim]",
            border_style="dim",
        )
    )
    for entry in conversation["messages"]:
        who = (
            "[bold cyan]you[/bold cyan]"
            if entry["role"] == "user"
            else "[bold magenta]ai [/bold magenta]"
        )
        console.print(f"{who}  {entry['content']}\n")


def remove_conversations(conversation_id, delete_all):
    if delete_all:
        removed = sum(1 for entry in conversations.list_all() if conversations.delete(entry["id"]))
        console.print(f"[green]✓[/green] deleted {removed} conversation(s)")
        return
    if not conversation_id:
        fail("which one?, ses history rm <id>  (or --all)")
    if conversations.delete(conversation_id):
        console.print("[green]✓[/green] deleted")
    else:
        console.print(f"[dim]no such conversation: {conversation_id}[/dim]")
