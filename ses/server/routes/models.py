from datetime import datetime
from ... import SesError
from ...core import catalog, llm, store

MAX_HF_LIMIT = 50


def register(app, context):
    loader = context.loader

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "version": context.version}

    @app.get("/api/version")
    def version():
        return {"version": context.version}

    @app.get("/v1/models")
    def list_models():
        models = []
        for manifest in store.installed():
            try:
                created = int(datetime.fromisoformat(manifest["installed_at"]).timestamp())
            except (KeyError, ValueError):
                created = 0
            models.append(
                {
                    "id": manifest["name"],
                    "object": "model",
                    "created": created,
                    "owned_by": "ses",
                    "ses": {
                        "kind": manifest.get("kind"),
                        "engine": manifest.get("engine"),
                        "repo": manifest.get("repo"),
                        "size_bytes": manifest.get("size_bytes", 0),
                    },
                }
            )
        return {"object": "list", "data": models}

    @app.get("/api/tags")
    def tags():
        return {"models": store.installed()}

    @app.get("/api/ps")
    def loaded():
        return {"models": loader.describe_loaded()}

    @app.get("/api/catalog")
    def model_catalog():
        return catalog.build(installed_brains())

    @app.get("/api/hf")
    def browse_hugging_face(task: str = "llm", window: str = "trending", limit: int = 25):
        return catalog.hf_models(task, window, min(max(limit, 1), MAX_HF_LIMIT))

    @app.get("/v1/audio/voices")
    def voices(model: str = ""):
        loaded_model = loader.get(model or context.default_model("tts"))
        if loaded_model.kind != "tts":
            raise SesError(f"'{model}' is not a TTS model")
        return {"model": loaded_model.name, "voices": loaded_model.engine.voices()}


def installed_brains():
    brain = llm.detect()
    if brain is None or brain.kind != llm.OLLAMA:
        return set()
    try:
        return set(brain.models())
    except Exception:
        return set()
