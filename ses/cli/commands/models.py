import os
import time
from collections import Counter
import typer
from rich.markup import escape
from rich.table import Table
from ... import DEFAULT_PORT, DEFAULT_STT, DEFAULT_TTS, SesError
from ...core import catalog, download, hardware, llm, paths, store
from ...core.registry import ANY, BRAINS, REGISTRY, resolve, resolve_brain
from ..ui import (
    console,
    default_tts,
    download_progress,
    fail,
    format_size,
    load_model,
)

MB = 1024**2

FIT_STYLES = {
    "great": "[green]runs great[/green]",
    "ok": "[green]fits[/green]",
    "tight": "[yellow]tight[/yellow]",
    "toobig": "[red]too big[/red]",
    "unknown": "[dim], [/dim]",
    "unsupported": "[red]unsupported[/red]",
}

VOICE_LANGUAGES = {
    "a": "English (US)",
    "b": "English (UK)",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Portuguese (BR)",
    "z": "Chinese",
}


def register(app):
    app.command()(pull)
    app.command("ls")(list_models)
    app.command("list", hidden=True)(list_models)
    app.command()(search)
    app.command()(system)
    app.command()(rm)
    app.command()(use)
    app.command()(info)
    app.command()(voices)
    app.command()(ps)


def pull(
    model: str = typer.Argument(
        ..., help="Library name (ses search) or any HF repo id (org/name)."
    ),
    engine: str = typer.Option(
        None,
        "--engine",
        "-e",
        help="For raw HF repos; omitted when ses can safely detect the format.",
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even if installed."),
):
    """Download a model from Hugging Face into ~/.ses.

    Beyond the curated library, any compatible repo works:
    ses pull mlx-community/whisper-large-v3-mlx-4bit --engine mlx-whisper
    """
    spec = resolve(model)
    if spec is None and "/" in model:
        pull_custom_repo(model.strip(), engine, force)
        return
    if spec is None:
        brain = resolve_brain(model)
        if brain is not None:
            pull_brain(brain)
            return
        fail(f"unknown model '{model}', run [bold]ses search[/bold] to see the library")

    if store.is_installed(spec.name) and not force:
        console.print(
            f"[green]✓[/green] {spec.name} already installed (use --force to re-download)"
        )
        return

    backend = spec.backend()
    if backend is None:
        fail(f"'{spec.name}' isn't available for this platform yet: see [bold]ses search[/bold]")

    console.print(
        f"pulling [bold]{spec.name}[/bold] from {backend.source} "
        f"[dim](~{format_size(backend.size_mb * MB)} · {backend.engine})[/dim]"
    )
    try:
        with download_progress(f"pulling {spec.name}") as on_progress:
            directory = store.install(spec, force=force, on_progress=on_progress)
    except Exception as error:
        fail(f"pull failed: {error}")

    console.print(
        f"[green]✓[/green] pulled [bold]{spec.name}[/bold] "
        f"({format_size(store.dir_size(directory))}) → {directory}"
    )
    if backend.extra and not catalog.runtime_installed(backend.engine):
        install_command = escape(f"pip install 'ses[{backend.extra}]'")
        console.print(
            f"[yellow]runtime needed:[/yellow] {install_command} "
            f"[dim](then {backend.engine} can load this model)[/dim]"
        )


CUSTOM_REPO_ENGINES = (
    "mlx-whisper",
    "faster-whisper",
    "mlx-audio",
    "mlx-audio-stt",
    "onnx-asr",
    "transformers-stt",
    "transformers-tts",
)


def pull_custom_repo(repo, engine, force):
    from ...engines import ENGINE_KINDS

    store.validate_hf_repo(repo)
    if engine is None:
        console.print(f"[dim]inspecting {repo} model files…[/dim]")
        detected = catalog.inspect_hf_repo(repo)
        engine = detected.get("engine") if detected.get("pull") else None
        if engine:
            console.print(f"[dim]detected {engine} repository layout[/dim]")
        else:
            fail(
                f"ses couldn't safely detect a runnable format for '{repo}': "
                f"{detected['reason']}.\n"
                f"  ses pull {repo} --engine <{'|'.join(CUSTOM_REPO_ENGINES)}>\n\n"
                "Compatible ports use a specific layout:\n"
                "  mlx-whisper · mlx-audio-stt                 MLX STT ports (Apple Silicon)\n"
                "  faster-whisper                              CTranslate2 Whisper ports\n"
                "  onnx-asr                                    supported ONNX ASR exports\n\n"
                "A repository name alone doesn't define a runtime. For GGML, Vosk, Piper, "
                "Kokoro and Chatterbox use a verified curated name from [bold]ses search[/bold]."
            )
    if engine not in ENGINE_KINDS:
        fail(f"unknown engine '{engine}'. Pick one of: {', '.join(ENGINE_KINDS)}")
    if engine not in CUSTOM_REPO_ENGINES:
        fail(
            f"'{engine}' can't load an arbitrary repo yet, its models need extra setup "
            f"(Kokoro's voice pack, a Piper voice pair). Use a curated name: ses search"
        )

    if store.is_installed(repo) and not force:
        console.print(f"[green]✓[/green] {repo} already installed (use --force to re-download)")
        return

    if not download.is_loadable(repo):
        fail(
            f"'{repo}' ships no weights any ses engine can open, it looks like a "
            "plain PyTorch checkpoint.\n"
            "Look for an MLX, ONNX or CTranslate2 port of it instead, usually "
            "published by mlx-community, onnx-community or Systran."
        )

    console.print(f"pulling [bold]{repo}[/bold] [dim](raw repo · {engine})[/dim]")
    try:
        with download_progress(f"pulling {repo}") as on_progress:
            directory = store.install_custom(repo, engine, force=force, on_progress=on_progress)
    except Exception as error:
        fail(f"pull failed: {error}")

    console.print(
        f"[green]✓[/green] pulled [bold]{repo}[/bold] "
        f"({format_size(store.dir_size(directory))}): use it with -m '{repo}'"
    )


def pull_brain(spec):
    from .assistant import ollama_pull

    brain = llm.detect()
    if brain is None or brain.kind != llm.OLLAMA:
        fail(
            f"'{spec.name}' is an LLM brain, served by Ollama, "
            "install it from https://ollama.com and retry"
        )

    try:
        installed = set(brain.models())
    except Exception:
        installed = set()

    if spec.name in installed or f"{spec.name}:latest" in installed:
        console.print(f"[green]✓[/green] {spec.name} already in Ollama")
    else:
        console.print(
            f"pulling brain [bold]{spec.name}[/bold] [dim](~{spec.size_gb:g} GB · via Ollama)[/dim]"
        )
        ollama_pull(brain.base_url, spec.name)
        console.print(f"[green]✓[/green] pulled [bold]{spec.name}[/bold]")

    console.print(
        f"[dim]talk to it:[/dim] ses talk -b {spec.name}  "
        f"[dim]· make it default:[/dim] ses use brain {spec.name}"
    )


def list_models():
    """List installed models."""
    manifests = store.installed()
    if not manifests:
        console.print(
            "no models installed yet, try [bold]ses pull tts-english[/bold] "
            "or browse with [bold]ses search[/bold]"
        )
        return

    table = Table(box=None, header_style="bold dim", pad_edge=False)
    for column in ("NAME", "KIND", "ENGINE", "SIZE", "PULLED"):
        table.add_column(column, style="bold" if column == "NAME" else None)
    for manifest in manifests:
        table.add_row(
            manifest["name"],
            manifest.get("kind", "?"),
            manifest.get("engine", "?"),
            format_size(manifest.get("size_bytes", 0)),
            manifest.get("installed_at", "")[:10],
        )
    console.print(table)


def fit_cell(size_gb, ram_gb, backend=None):
    details = hardware.fit(
        size_gb,
        ram_gb,
        required_ram_gb=backend.required_ram_gb if backend else None,
        required_vram_gb=backend.required_vram_gb if backend else None,
        accelerators=backend.accelerators if backend else None,
    )
    return FIT_STYLES.get(details["level"], f"[dim]{details['label']}[/dim]")


def search(term: str = typer.Argument(None, help="Filter by name/description.")):
    """Browse the model library, with a fit check for your machine."""
    machine = hardware.system_info()
    ram_gb = machine["ram_gb"]
    console.print(
        f"[dim]your machine:[/dim] {machine['os']} · "
        f"{str(ram_gb) + ' GB RAM' if ram_gb else 'RAM unknown'} · "
        f"{machine.get('cpu_count') or '?'} threads · {machine['accelerator']}\n"
    )

    table = Table(box=None, header_style="bold dim", pad_edge=False)
    table.add_column("NAME", style="bold")
    table.add_column("KIND")
    table.add_column("ENGINE")
    table.add_column("SIZE", justify="right")
    table.add_column("RAM EST.", justify="right")
    table.add_column("FITS")
    table.add_column("", width=1)
    table.add_column("DESCRIPTION", max_width=52)

    for spec in REGISTRY.values():
        searchable = " ".join(
            (
                spec.name,
                spec.description,
                " ".join(spec.languages),
                spec.license or "",
                (
                    "windows linux cross-platform"
                    if any(backend.requires == ANY for backend in spec.backends)
                    else "apple silicon"
                ),
                " ".join(
                    f"{backend.engine} {backend.format or ''} "
                    f"{' '.join(backend.accelerators or ())}"
                    for backend in spec.backends
                ),
            )
        )
        if term and term.lower() not in searchable.lower():
            continue
        installed = "[green]✓[/green]" if store.is_installed(spec.name) else ""
        backend = spec.backend()
        if backend is None:
            table.add_row(
                f"[dim]{spec.name}[/dim]",
                spec.kind,
                "[dim], [/dim]",
                "[dim], [/dim]",
                "[dim], [/dim]",
                "[dim]n/a[/dim]",
                installed,
                f"[dim]{spec.description}, not for this platform yet[/dim]",
            )
        else:
            needed_ram = (
                f"~{backend.required_ram_gb:g} GB" if backend.required_ram_gb else "[dim]est.[/dim]"
            )
            table.add_row(
                spec.name,
                spec.kind,
                backend.engine,
                format_size(backend.size_mb * MB),
                needed_ram,
                fit_cell(backend.size_mb / 1024, ram_gb, backend),
                installed,
                spec.description,
            )
    console.print(table)
    print_brains(term, ram_gb)

    console.print(
        "\n[dim]install with[/dim] ses pull <name>  [dim](brains run via Ollama)[/dim]"
        "\n[dim]beyond the library, any compatible HF repo works:[/dim] "
        "ses pull <org/repo> --engine <engine>"
    )


def print_brains(term, ram_gb):
    brain = llm.detect()
    in_ollama = set()
    if brain is not None and brain.kind == llm.OLLAMA:
        try:
            in_ollama = set(brain.models())
        except Exception:
            in_ollama = set()

    table = Table(box=None, header_style="bold dim", pad_edge=False)
    table.add_column("BRAIN (LLM)", style="bold")
    table.add_column("SIZE", justify="right")
    table.add_column("FITS")
    table.add_column("", width=1)
    table.add_column("DESCRIPTION", max_width=52)

    shown = 0
    for spec in BRAINS:
        if term and term.lower() not in f"{spec.name} {spec.description}".lower():
            continue
        shown += 1
        installed = (
            "[green]✓[/green]"
            if spec.name in in_ollama or f"{spec.name}:latest" in in_ollama
            else ""
        )
        table.add_row(
            spec.name,
            f"{spec.size_gb:g} GB",
            fit_cell(spec.size_gb, ram_gb),
            installed,
            spec.description,
        )
    if shown:
        console.print()
        console.print(table)


def system():
    """Show your hardware and what it can run."""
    machine = hardware.system_info()
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("os", machine["os"])
    table.add_row("arch", machine["arch"])
    table.add_row("cpu threads", str(machine["cpu_count"] or "unknown"))
    table.add_row("memory", f"{machine['ram_gb']} GB" if machine["ram_gb"] else "unknown")
    table.add_row("accelerator", machine["accelerator"])
    if machine.get("gpu_name"):
        gpu = machine["gpu_name"]
        if machine.get("vram_gb"):
            gpu += f" · {machine['vram_gb']} GB VRAM"
        table.add_row("gpu", gpu)
    console.print(table)

    if machine["ram_gb"]:
        comfortable = machine["ram_gb"] * 0.65
        console.print(
            f"\n[dim]rule of thumb: models up to ~{comfortable:.0f} GB run comfortably here.[/dim]"
        )

    from ...engines import ENGINE_KINDS, INSTALL_HINTS

    engines = Table(box=None, pad_edge=False, title_style="")
    engines.add_column("engine", style="dim")
    engines.add_column("")
    engines.add_column("models")
    engines.add_column("speed here")
    engines.add_column("runtime")

    counts = Counter(backend.engine for spec in REGISTRY.values() for backend in spec.backends)
    console.print("\n[bold]what runs on this machine[/bold]")
    for engine, kind in ENGINE_KINDS.items():
        speed = hardware.engine_speed(engine)
        runs = speed is not None
        engines.add_row(
            engine,
            kind,
            str(counts.get(engine, 0)) if runs else ", ",
            hardware.speed_label(engine) if runs else "[dim]needs other hardware[/dim]",
            (
                "[green]installed[/green]"
                if catalog.runtime_installed(engine)
                else f"[dim]{escape(INSTALL_HINTS.get(engine, 'not installed'))}[/dim]"
            ),
        )
    console.print(engines)
    console.print(
        "\n[dim]speeds are estimates for planning; measure yours with[/dim] ses transcribe"
        "\n[dim]see per-model fit with[/dim] ses search"
    )


def rm(
    model: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
):
    """Remove an installed model."""
    spec = resolve(model)
    name = spec.name if spec else model
    if not store.is_installed(name):
        fail(f"model '{name}' is not installed")

    size = format_size(store.dir_size(paths.model_dir(name)))
    if not yes and not typer.confirm(f"remove {name} ({size})?"):
        raise typer.Exit()

    store.remove(name)
    console.print(f"[green]✓[/green] removed {name} (freed {size})")


def use(
    role: str = typer.Argument(None, help="stt | tts | brain"),
    model: str = typer.Argument(None, help="Model name, or 'default' to reset."),
):
    """Set the default model for a role: ses use stt whisper-small.

    'brain' is the LLM used by `ses talk`. Run with no arguments to see the
    current defaults.
    """
    builtin = {"stt": DEFAULT_STT, "tts": DEFAULT_TTS, "brain": "(first Ollama model)"}

    if role is None:
        print_defaults(builtin)
        return
    if role not in builtin:
        fail(f"unknown role '{role}'. Pick one of: stt, tts, brain")
    if model is None:
        console.print(f"{role} → [bold]{paths.default_for(role) or builtin[role]}[/bold]")
        return
    if model in ("default", "reset", "none"):
        paths.set_default(role, None)
        console.print(f"[green]✓[/green] {role} reset to built-in default")
        return

    if role in ("stt", "tts"):
        model = validate_speech_default(role, model)
    paths.set_default(role, model)
    console.print(f"[green]✓[/green] default {role} → [bold]{model}[/bold]")


def print_defaults(builtin):
    settings = paths.config()
    table = Table(box=None, header_style="bold dim", pad_edge=False)
    table.add_column("ROLE", style="bold")
    table.add_column("MODEL")
    table.add_column("", style="dim")

    for role, fallback in builtin.items():
        from_env = os.environ.get(f"SES_{role.upper()}")
        if from_env:
            table.add_row(role, from_env, f"from env SES_{role.upper()}")
        elif role in settings:
            table.add_row(role, settings[role], "set via ses use")
        else:
            table.add_row(role, fallback, "built-in default")

    console.print(table)
    console.print("\n[dim]change with[/dim] ses use <role> <model>")


def validate_speech_default(role, model):
    spec = resolve(model)
    manifest = store.manifest_for(model) if spec is None else None
    if spec is None and manifest is None:
        fail(
            f"unknown model '{model}', run [bold]ses search[/bold] "
            f"(or pull it first: ses pull {model} --engine …)"
        )

    kind = spec.kind if spec else manifest.get("kind")
    if kind != role:
        fail(f"'{model}' is a {kind} model, can't be the default {role}")

    name = spec.name if spec else model
    if not store.is_installed(name):
        console.print(f"[dim]tip: it isn't pulled yet, ses pull {name}[/dim]")
    return name


def info(model: str = typer.Argument(...)):
    """Show details for a model."""
    spec = resolve(model)
    manifest = store.manifest_for(spec.name if spec else model)
    if spec is None and manifest is None:
        fail(f"unknown model '{model}'")

    name = spec.name if spec else manifest["name"]
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("name", name)

    if spec:
        table.add_row("kind", spec.kind)
        source = manifest or {}
        backend = spec.backend()
        engine = source.get("engine") or (backend.engine if backend else "?")
        origin = source.get("source") or (backend.source if backend else "?")
        table.add_row("engine", engine)
        table.add_row("source", origin)
        if backend and backend.format:
            table.add_row("format", backend.format)
        if backend and backend.required_ram_gb:
            table.add_row("planning RAM", f"~{backend.required_ram_gb:g} GB")
        if backend and backend.recommended_vram_gb:
            table.add_row("GPU planning", f"~{backend.recommended_vram_gb:g} GB VRAM")
        if backend and backend.accelerators:
            table.add_row("hardware", ", ".join(backend.accelerators))
        if backend and backend.extra:
            installed_text = (
                "installed"
                if catalog.runtime_installed(engine)
                else (escape(f"pip install 'ses[{backend.extra}]'"))
            )
            table.add_row("runtime", installed_text)
        if spec.languages:
            table.add_row("languages", ", ".join(spec.languages))
        if spec.license:
            table.add_row("license", spec.license)
        if spec.github:
            table.add_row("project", spec.github)
        table.add_row("about", spec.description)

    if manifest:
        table.add_row("installed", manifest.get("installed_at", "?"))
        table.add_row("size", format_size(manifest.get("size_bytes", 0)))
        table.add_row("path", str(paths.model_dir(name)))
    else:
        table.add_row("installed", f"no, ses pull {name}")

    console.print(table)


def voices(
    model: str = typer.Option(None, "--model", "-m", help="TTS model (default: tts-english)"),
):
    """List a TTS model's voices."""
    loaded = load_model(model or default_tts())
    if loaded.kind != "tts":
        fail(f"'{loaded.name}' is not a TTS model")

    names = loaded.engine.voices()
    if all(is_kokoro_voice(name) for name in names):
        grouped = {}
        for name in names:
            grouped.setdefault(VOICE_LANGUAGES.get(name[0], "other"), []).append(name)
        for language, group in grouped.items():
            console.print(f"[bold]{language}[/bold]  [dim]({len(group)})[/dim]")
            console.print("  " + "  ".join(group))
    else:
        console.print(f"[bold]{loaded.name}[/bold]  [dim]({len(names)} voices)[/dim]")
        console.print("  " + "  ".join(names))

    console.print('\n[dim]use with[/dim] ses say -v <voice> "…"')


def is_kokoro_voice(name):
    return len(name) > 3 and name[2] == "_" and name[:2].isalpha()


def ps():
    """Show which models the running server has in memory."""
    import httpx

    url = server_url()
    try:
        response = httpx.get(f"{url}/api/ps", timeout=3)
        response.raise_for_status()
    except Exception:
        console.print(f"[dim]no ses server running at {url}, start one with[/dim] ses serve")
        return

    models = response.json().get("models", [])
    if not models:
        console.print("server is up, no models loaded")
        return

    table = Table(box=None, header_style="bold dim", pad_edge=False)
    table.add_column("NAME", style="bold")
    table.add_column("KIND")
    table.add_column("SIZE", justify="right")
    table.add_column("LOADED IN", justify="right")
    table.add_column("EXPIRES", style="dim")

    now = time.time()
    for model in models:
        expires_at = model.get("expires_at")
        expires = f"{max(0, int(expires_at - now)) // 60}m" if expires_at else "never"
        table.add_row(
            model["name"],
            model["kind"],
            format_size(model.get("size_bytes", 0)),
            f"{model.get('load_seconds', 0):.1f}s",
            expires,
        )
    console.print(table)


def server_url():
    from ... import DEFAULT_HOST

    host = os.environ.get("SES_HOST", DEFAULT_HOST)
    port = os.environ.get("SES_PORT", str(DEFAULT_PORT))
    return f"http://{host}:{port}"


__all__ = ["register", "SesError"]
