import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .. import DEFAULT_STT, DEFAULT_TTS, ModelNotInstalled, SesError, __version__
from ..core import paths
from ..core.loader import ModelLoader
from .routes import register_all

REAP_INTERVAL_SECONDS = 30
UI_DIR = Path(__file__).parent / "ui"


@dataclass
class ServerContext:
    loader: ModelLoader
    infer_lock: threading.Lock
    version: str = __version__

    def default_model(self, kind):
        return paths.default_for(kind) or (DEFAULT_TTS if kind == "tts" else DEFAULT_STT)


def error_response(status, message, code):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "invalid_request_error", "code": code}},
    )


def log_load_event(event, name, seconds):
    if event == "loading":
        print(f"⏳ loading {name} into memory…", flush=True)
    else:
        print(f"✓ {name} ready in {seconds:.1f}s", flush=True)


def create_app(loader=None):
    loader = loader or ModelLoader(on_event=log_load_event)
    context = ServerContext(loader=loader, infer_lock=threading.Lock())
    stop_reaper = threading.Event()

    @asynccontextmanager
    async def lifespan(_app):
        def reap_idle_models():
            while not stop_reaper.wait(REAP_INTERVAL_SECONDS):
                loader.reap()

        threading.Thread(target=reap_idle_models, daemon=True, name="ses-reaper").start()
        yield
        stop_reaper.set()

    app = FastAPI(title="ses", version=__version__, lifespan=lifespan)
    app.state.loader = loader
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.exception_handler(ModelNotInstalled)
    async def handle_missing_model(_request, error):
        return error_response(404, str(error), "model_not_found")

    @app.exception_handler(SesError)
    async def handle_ses_error(_request, error):
        return error_response(400, str(error), "invalid_request_error")

    register_all(app, context)

    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")

    @app.get("/", response_class=HTMLResponse)
    def playground():
        return (UI_DIR / "index.html").read_text(encoding="utf-8")

    return app
