import os
import sys
import typer
from rich.markup import escape
from rich.panel import Panel
from ... import DEFAULT_HOST, DEFAULT_PORT, __version__
from ...core import store
from ...core.loader import ModelLoader, parse_duration
from ..ui import console, fail


def register(app):
    app.command()(serve)
    app.command()(mcp)


def serve(
    host: str = typer.Option(None, "--host", help=f"Bind address (default {DEFAULT_HOST})"),
    port: int = typer.Option(None, "--port", "-p", help=f"Port (default {DEFAULT_PORT})"),
    keep_alive: str = typer.Option(
        None, "--keep-alive", help="Unload idle models after this long (10m, 1h, -1 = never)."
    ),
    max_loaded: int = typer.Option(None, "--max-loaded", help="How many models to keep in memory."),
):
    """Start the local voice server: OpenAI-compatible API plus the playground."""
    import uvicorn
    from ...server import create_app, log_load_event

    host = host or os.environ.get("SES_HOST", DEFAULT_HOST)
    port = port or int(os.environ.get("SES_PORT", str(DEFAULT_PORT)))

    def on_load(event, name, seconds):
        if event == "loading":
            console.print(f"[dim]⏳ loading[/dim] [bold]{name}[/bold] [dim]into memory…[/dim]")
        else:
            console.print(f"[dim]✓ {name} ready in {seconds:.1f}s[/dim]")

    loader = ModelLoader(
        max_loaded=max_loaded,
        keep_alive=parse_duration(keep_alive, None) if keep_alive else None,
        on_event=on_load,
    )

    installed = [manifest["name"] for manifest in store.installed()]
    keep_alive_label = (
        "never" if loader.keep_alive < 0 else f"{int(loader.keep_alive) // 60}m"
    )
    console.print(
        Panel.fit(
            f"[bold]ses[/bold] [dim]v{__version__}, local voice server[/dim]\n\n"
            f"[dim]api[/dim]         http://{host}:{port}/v1\n"
            f"[dim]playground[/dim]  [bold]http://{host}:{port}[/bold]\n"
            f"[dim]models[/dim]      {', '.join(installed) if installed else 'none, ses pull tts-english'}\n"
            f"[dim]keep-alive[/dim]  {keep_alive_label} · max loaded {loader.max_loaded}",
            border_style="dim",
        )
    )

    uvicorn.run(create_app(loader), host=host, port=port, log_level="warning", access_log=False)

    _ = log_load_event


def mcp(
    stdio: bool = typer.Option(
        False, "--stdio", help="Serve even when stdin is a terminal."
    ),
):
    """Run ses as an MCP server, giving agents a voice.

    Wire it in:
      Claude Code:  claude mcp add ses -- ses mcp
      Codex (~/.codex/config.toml):
        [mcp_servers.ses]
        command = "ses"
        args = ["mcp"]

    Tools: speak, notify, listen, dictate, transcribe, voice_mode, set_voice,
    list_voices, all running locally.
    """
    try:
        from ...mcp import main as run_mcp_server
    except ImportError:
        fail("MCP support needs the 'mcp' package, install with: pip install 'ses[mcp]'")

    if sys.stdin.isatty() and not stdio:
        explain_mcp()
        return
    run_mcp_server()


def explain_mcp():
    console.print(
        Panel.fit(
            "[bold]ses mcp[/bold] speaks the Model Context Protocol over stdin and "
            "stdout.\nIt is meant to be launched by an agent, not run by hand.\n\n"
            "[bold]Claude Code[/bold]\n  claude mcp add ses -- ses mcp\n\n"
            "[bold]Codex[/bold]  [dim]~/.codex/config.toml[/dim]\n"
            "  " + escape("[mcp_servers.ses]") + "\n"
            "  command = \"ses\"\n"
            "  args = " + escape('["mcp"]') + "\n\n"
            "[dim]Once wired in, your agent can speak, listen and transcribe locally.\n"
            "To pipe protocol messages yourself, use:[/dim] ses mcp --stdio",
            title="not an interactive command",
            border_style="yellow",
        )
    )
