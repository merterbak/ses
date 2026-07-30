import sys
import typer
from rich.markup import escape
from .. import SesError, __version__
from .commands import register_all
from .ui import console, error_console

app = typer.Typer(
    name="ses",
    help="Ollama for voice, pull, hot-swap and serve local speech models.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)


def show_version(value):
    if value:
        console.print(f"ses {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(
        False, "--version", callback=show_version, is_eager=True, help="Show version and exit."
    ),
):
    pass


register_all(app)


def main():
    try:
        app()
    except SesError as error:
        error_console.print(f"[red]error:[/red] {escape(str(error))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
