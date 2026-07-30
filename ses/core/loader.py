import os
import threading
import time
from dataclasses import dataclass
from .. import ModelNotInstalled, SesError
from ..engines import load_engine
from . import paths, store
from .registry import resolve

DEFAULT_KEEP_ALIVE = 600.0
DEFAULT_MAX_LOADED = 3


def parse_duration(value, default):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)

    text = value.strip().lower()
    if text in ("-1", "forever", "never"):
        return -1.0

    multiplier = 1
    if text.endswith("ms"):
        multiplier = 0.001
        text = text[:-2]
    elif text.endswith("h"):
        multiplier = 3600
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 60
        text = text[:-1]
    elif text.endswith("s"):
        text = text[:-1]

    try:
        return float(text.strip()) * multiplier
    except ValueError:
        raise SesError(f"cannot parse duration '{value}' (try '300', '10m', '1h', '-1')")


@dataclass
class LoadedModel:
    name: str
    kind: str
    engine_name: str
    engine: object
    loaded_at: float
    last_used: float
    load_seconds: float
    size_bytes: int


class ModelLoader:
    def __init__(self, max_loaded=None, keep_alive=None, on_event=None):
        self.max_loaded = max_loaded or int(os.environ.get("SES_MAX_LOADED", DEFAULT_MAX_LOADED))
        self.keep_alive = (
            keep_alive
            if keep_alive is not None
            else parse_duration(os.environ.get("SES_KEEP_ALIVE"), DEFAULT_KEEP_ALIVE)
        )
        self.on_event = on_event
        self._models = {}
        self._lock = threading.Lock()

    def get(self, name, auto_pull=False):
        requested = name.strip()
        spec = resolve(requested)
        canonical = spec.name if spec else (requested if "/" in requested else requested.lower())

        with self._lock:
            warm = self._models.get(canonical)
            if warm:
                warm.last_used = time.time()
                return warm

            if not store.is_installed(canonical):
                if spec is None:
                    raise ModelNotInstalled(canonical, known=False)
                if not auto_pull:
                    raise ModelNotInstalled(canonical, known=True)
                store.install(spec)

            model = self.load(canonical, spec)
            self._models[canonical] = model
            while len(self._models) > self.max_loaded:
                oldest = min(self._models, key=lambda name: self._models[name].last_used)
                del self._models[oldest]
            return model

    def load(self, canonical, spec):
        manifest = store.manifest_for(canonical) or {}
        backend = spec.backend() if spec else None
        engine_name = manifest.get("engine") or (backend.engine if backend else None)
        kind = manifest.get("kind") or (spec.kind if spec else None)
        if not engine_name or not kind:
            raise SesError(
                f"model '{canonical}' has a broken manifest. Try: ses pull {canonical} --force"
            )

        if self.on_event:
            self.on_event("loading", canonical, None)
        started = time.time()
        engine = load_engine(engine_name, paths.model_dir(canonical))
        elapsed = time.time() - started
        if self.on_event:
            self.on_event("loaded", canonical, elapsed)

        now = time.time()
        return LoadedModel(
            name=canonical,
            kind=kind,
            engine_name=engine_name,
            engine=engine,
            loaded_at=now,
            last_used=now,
            load_seconds=elapsed,
            size_bytes=int(manifest.get("size_bytes", 0)),
        )

    def unload(self, name):
        with self._lock:
            return self._models.pop(name, None) is not None

    def unload_all(self):
        with self._lock:
            self._models.clear()

    def reap(self):
        if self.keep_alive < 0:
            return []
        now = time.time()
        with self._lock:
            expired = [
                name
                for name, model in self._models.items()
                if now - model.last_used > self.keep_alive
            ]
            for name in expired:
                del self._models[name]
            return expired

    def describe_loaded(self):
        with self._lock:
            rows = []
            for model in self._models.values():
                row = {
                    "name": model.name,
                    "kind": model.kind,
                    "engine": model.engine_name,
                    "size_bytes": model.size_bytes,
                    "loaded_at": model.loaded_at,
                    "last_used": model.last_used,
                    "load_seconds": round(model.load_seconds, 2),
                }
                if self.keep_alive >= 0:
                    row["expires_at"] = model.last_used + self.keep_alive
                rows.append(row)
            return rows
