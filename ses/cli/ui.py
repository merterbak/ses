import threading
import time
from contextlib import contextmanager
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from .. import ModelNotInstalled, SesError
from ..core import paths, store
from ..core.loader import ModelLoader
from ..core.registry import resolve

console = Console()
error_console = Console(stderr=True)

GB = 1024**3
MB = 1024**2
LOAD_SECONDS_PER_GB = 2.0
MIN_LOAD_ESTIMATE = 1.5
MAX_ESTIMATED_PERCENT = 95


def fail(message):
    error_console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(1)


def format_size(num_bytes):
    if num_bytes >= GB:
        return f"{num_bytes / GB:.1f} GB"
    return f"{num_bytes / MB:.0f} MB"


def format_speed(audio_seconds, took, noun="speech"):
    if audio_seconds <= 0 or took <= 0:
        return ""
    per_second = took / audio_seconds
    if per_second <= 1:
        return f"a second of {noun} every {per_second:.2f}s"
    return f"{per_second:.1f}s of work per second of {noun}, slower than real time"


@contextmanager
def download_progress(label):
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(compact=True),
        console=console,
        transient=True,
    )
    with progress:
        task = progress.add_task(label, total=None)

        def on_progress(done, total):
            progress.update(task, completed=done, total=total)

        yield on_progress


def ensure_installed(name):
    requested = name.strip()
    spec = resolve(requested)
    canonical = spec.name if spec else (requested if "/" in requested else requested.lower())

    if store.is_installed(canonical):
        return canonical
    if spec is None:
        raise ModelNotInstalled(canonical, known=False)

    backend = spec.backend()
    if backend is None:
        raise SesError(f"'{canonical}' isn't available for this platform yet, see: ses search")

    console.print(
        f"[dim]model[/dim] [bold]{canonical}[/bold] [dim]not installed, pulling from[/dim] "
        f"{backend.repo} [dim](~{format_size(backend.size_mb * MB)})[/dim]"
    )
    with download_progress(f"pulling {canonical}") as on_progress:
        store.install(spec, on_progress=on_progress)
    console.print(f"[green]✓[/green] pulled [bold]{canonical}[/bold]")
    return canonical


def load_model(name, quiet=False):
    canonical = ensure_installed(name)
    loader = ModelLoader()
    if quiet:
        return loader.get(canonical)

    manifest = store.manifest_for(canonical) or {}
    size_gb = manifest.get("size_bytes", 0) / GB
    estimate = max(MIN_LOAD_ESTIMATE, size_gb * LOAD_SECONDS_PER_GB)

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold]loading {canonical} into memory"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("", total=100)
        done = threading.Event()

        def animate():
            started = time.time()
            while not done.wait(0.1):
                elapsed_percent = 100 * (time.time() - started) / estimate
                progress.update(task, completed=min(MAX_ESTIMATED_PERCENT, elapsed_percent))

        ticker = threading.Thread(target=animate, daemon=True)
        ticker.start()
        try:
            model = loader.get(canonical)
        finally:
            done.set()
            ticker.join()
        progress.update(task, completed=100)

    return model


def default_tts():
    from .. import DEFAULT_TTS

    return paths.default_for("tts") or DEFAULT_TTS


def default_stt():
    from .. import DEFAULT_STT

    return paths.default_for("stt") or DEFAULT_STT
