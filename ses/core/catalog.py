import importlib.util
import re
import time
from . import ecosystem, hardware, store
from .registry import BRAINS, FEATURED, REGISTRY

HF_API = "https://huggingface.co/api/models"
CACHE_SECONDS = 1800
FETCH_LIMIT = 50
GB = 1024

BYTES_PER_PARAM_4BIT = 0.65
BILLION = 1_000_000_000

TASKS = {
    "llm": "text-generation",
    "tts": "text-to-speech",
    "stt": "automatic-speech-recognition",
}

WINDOWS = {
    "trending": ("trendingScore", "downloads"),
    "month": ("downloads", "downloads"),
    "liked": ("likes", "likes"),
}

_PARAM_COUNT_IN_NAME = re.compile(r"(\d+(?:\.\d+)?)\s*([bm])(?![a-z0-9])", re.IGNORECASE)

_VARIANT_SUFFIX = re.compile(
    r"-(4bit|8bit|6bit|fp16|bf16|mlx|onnx|ct2|ctranslate2|q[45]_[a-z0-9]+)$",
    re.IGNORECASE,
)
MIN_MATCH_CHARS = 6
MAX_SHARED_NAMES = 5

_cache = {}

ENGINE_PACKAGES = {
    "chatterbox": "chatterbox",
    "faster-whisper": "faster_whisper",
    "kokoro-onnx": "kokoro_onnx",
    "mlx-audio-stt": "mlx_audio",
    "mlx-audio": "mlx_audio",
    "mlx-whisper": "mlx_whisper",
    "onnx-asr": "onnx_asr",
    "piper": "piper",
    "vosk": "vosk",
    "whisper-cpp": "pywhispercpp",
}


def runtime_installed(engine):
    package = ENGINE_PACKAGES.get(engine)
    if not package:
        return False
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def backend_fit(backend, ram_gb):
    if backend is None:
        return {"level": "unsupported", "label": "not on this platform"}
    return hardware.fit(
        backend.size_mb / GB,
        ram_gb,
        required_ram_gb=backend.required_ram_gb,
        required_vram_gb=backend.required_vram_gb,
        accelerators=backend.accelerators,
    )


def backend_payload(backend):
    return {
        "engine": backend.engine,
        "repo": backend.repo,
        "source": backend.source,
        "size_gb": round(backend.size_mb / GB, 2),
        "format": backend.format,
        "required_ram_gb": backend.required_ram_gb,
        "required_vram_gb": backend.required_vram_gb,
        "recommended_vram_gb": backend.recommended_vram_gb,
        "accelerators": list(backend.accelerators or ()),
        "extra": backend.extra,
        "platform_supported": backend.supported,
        "runtime_installed": runtime_installed(backend.engine),
    }


def repo_key(repo):
    name = repo.split("/")[-1].lower()
    previous = None
    while name != previous:
        previous = name
        name = _VARIANT_SUFFIX.sub("", name)
    return name


def curated_index():
    sharing = {}
    for spec in REGISTRY.values():
        for backend in spec.backends:
            sharing.setdefault(repo_key(backend.repo), []).append(spec.name)

    index = {}
    for key, names in sharing.items():
        if len(names) <= MAX_SHARED_NAMES:
            index[key] = min(names, key=len)
    return index


def curated_name_for(repo, index):
    key = repo_key(repo)
    if key in index:
        return index[key]
    best = None
    for known, name in index.items():
        if len(known) < MIN_MATCH_CHARS or len(key) < MIN_MATCH_CHARS:
            continue
        if known.startswith(key) or key.startswith(known):
            shared = min(len(known), len(key))
            if best is None or shared > best[0]:
                best = (shared, name)
    return best[1] if best else None


def entry_tags(entry):
    return {str(tag).lower() for tag in (entry.get("tags") or ())}


def entry_files(entry):
    files = set()
    for sibling in entry.get("siblings") or ():
        if isinstance(sibling, str):
            files.add(sibling)
        elif isinstance(sibling, dict):
            path = sibling.get("rfilename") or sibling.get("path")
            if path:
                files.add(str(path))
    return files


def detect_hf_engine(entry):
    repo = entry["id"]
    lowered = repo.lower()
    tags = entry_tags(entry)
    library = str(entry.get("library_name") or "").lower()
    files = entry_files(entry)

    is_ctranslate2 = "ctranslate2" in tags or library == "ctranslate2"
    has_ctranslate2_layout = {
        "model.bin",
        "config.json",
        "tokenizer.json",
    }.issubset(files)
    if is_ctranslate2 and has_ctranslate2_layout:
        return {
            "engine": "faster-whisper",
            "status": "compatible",
            "reason": "Verified offline CTranslate2 model, config and tokenizer layout",
            "pull": f"ses pull {repo} --engine faster-whisper",
        }

    is_mlx_audio = library in {"mlx-audio", "mlx_audio"} and (
        entry.get("pipeline_tag") == TASKS["stt"]
        or any(marker in lowered for marker in ("whisper", "-asr", "nemotron", "canary"))
    )
    if is_mlx_audio:
        if not hardware.is_apple_silicon():
            return {
                "engine": "mlx-audio-stt",
                "status": "other-platform",
                "reason": "MLX Audio repositories require Apple Silicon",
                "pull": None,
            }
        return {
            "engine": "mlx-audio-stt",
            "status": "compatible",
            "reason": "Repository declares the mlx-audio STT library",
            "pull": f"ses pull {repo} --engine mlx-audio-stt",
        }

    is_mlx_whisper = (
        lowered.startswith("mlx-community/")
        and "whisper" in lowered
        and "weights.npz" in files
        and library not in {"mlx-audio", "mlx_audio"}
    )
    if is_mlx_whisper:
        if not hardware.is_apple_silicon():
            return {
                "engine": "mlx-whisper",
                "status": "other-platform",
                "reason": "MLX repositories require Apple Silicon",
                "pull": None,
            }
        return {
            "engine": "mlx-whisper",
            "status": "compatible",
            "reason": "Verified mlx-whisper weights.npz layout",
            "pull": f"ses pull {repo} --engine mlx-whisper",
        }

    return {
        "engine": None,
        "status": "unknown-format",
        "reason": "No verified ses runtime for this repository layout",
        "pull": None,
    }


def inspect_hf_repo(repo):
    import httpx

    try:
        response = httpx.get(f"{HF_API}/{repo}", timeout=15, follow_redirects=True)
        response.raise_for_status()
        entry = response.json()
    except Exception as error:
        return {
            "engine": None,
            "status": "inspection-failed",
            "reason": f"Could not inspect the Hugging Face repository: {error}",
            "pull": None,
        }

    if not isinstance(entry, dict):
        return {
            "engine": None,
            "status": "inspection-failed",
            "reason": "Hugging Face returned invalid repository metadata",
            "pull": None,
        }
    entry.setdefault("id", repo)
    return detect_hf_engine(entry)


def hf_compatibility(entry, curated):
    if curated:
        spec = REGISTRY[curated]
        backend = spec.backend()
        return {
            "engine": backend.engine if backend else None,
            "status": "curated" if backend else "other-platform",
            "reason": "Verified curated model" if backend else "No backend for this platform",
            "pull": f"ses pull {curated}" if backend else None,
        }
    return detect_hf_engine(entry)


def parameter_count(entry):
    safetensors = entry.get("safetensors") or {}
    total = safetensors.get("total")
    if isinstance(total, (int, float)) and total > 0:
        return int(total)

    matches = _PARAM_COUNT_IN_NAME.findall(entry["id"].split("/")[-1])
    if not matches:
        return None
    largest = 0
    for amount, unit in matches:
        scale = BILLION if unit.lower() == "b" else 1_000_000
        largest = max(largest, float(amount) * scale)
    return int(largest) or None


def estimated_size_gb(parameters):
    if not parameters:
        return None
    return round(parameters * BYTES_PER_PARAM_4BIT / 1e9, 1)


def hf_models(task, window, limit=25):
    if task not in TASKS or window not in WINDOWS:
        return {"task": task, "window": window, "metric": "downloads", "models": []}

    sort_field, metric = WINDOWS[window]
    cached = _cache.get((task, window))
    if not cached or time.time() - cached["fetched_at"] >= CACHE_SECONDS:
        cached = {"fetched_at": time.time(), "models": fetch_hf(TASKS[task], sort_field)}
        _cache[(task, window)] = cached

    return {
        "task": task,
        "window": window,
        "metric": metric,
        "models": cached["models"][:limit],
    }


def fetch_hf(pipeline_tag, sort_field):
    import httpx

    ram_gb = hardware.total_ram_gb()
    try:
        response = httpx.get(
            HF_API,
            params={
                "pipeline_tag": pipeline_tag,
                "sort": sort_field,
                "direction": -1,
                "limit": FETCH_LIMIT,
                "expand[]": [
                    "safetensors",
                    "downloads",
                    "likes",
                    "library_name",
                    "pipeline_tag",
                    "tags",
                    "siblings",
                ],
            },
            timeout=10,
            follow_redirects=True,
        )
        response.raise_for_status()
        entries = response.json()
    except Exception:
        return []

    index = curated_index()
    models = []
    for entry in entries:
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        parameters = parameter_count(entry)
        curated = curated_name_for(entry["id"], index)
        compatibility = hf_compatibility(entry, curated)
        curated_backend = REGISTRY[curated].backend() if curated else None
        if curated_backend:
            size_gb = round(curated_backend.size_mb / GB, 2)
            fit = backend_fit(curated_backend, ram_gb)
            size_basis = "curated download"
        else:
            size_gb = estimated_size_gb(parameters)
            fit = hardware.fit(size_gb, ram_gb)
            size_basis = "estimated 4-bit weights" if size_gb else "unknown"
        models.append(
            {
                "id": entry["id"],
                "downloads": entry.get("downloads") or 0,
                "likes": entry.get("likes") or 0,
                "parameters": parameters,
                "size_gb": size_gb,
                "size_basis": size_basis,
                "fit": fit,
                "curated": curated,
                "engine": compatibility["engine"],
                "compatibility": compatibility,
                "pull": compatibility["pull"],
            }
        )
    return models


def build(installed_brains=None):
    ram_gb = hardware.total_ram_gb()
    installed_brains = installed_brains or set()

    featured_rank = {name: index for index, name in enumerate(FEATURED)}

    speech = []
    for spec in REGISTRY.values():
        backend = spec.backend()
        size_gb = round(backend.size_mb / GB, 2) if backend else None
        fit = backend_fit(backend, ram_gb)
        speech.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "description": spec.description,
                "size_gb": size_gb,
                "available": bool(backend),
                "engine": backend.engine if backend else None,
                "format": backend.format if backend else None,
                "source": backend.source if backend else None,
                "required_ram_gb": backend.required_ram_gb if backend else None,
                "required_vram_gb": backend.required_vram_gb if backend else None,
                "recommended_vram_gb": (
                    backend.recommended_vram_gb if backend else None
                ),
                "accelerators": list(backend.accelerators or ()) if backend else [],
                "extra": backend.extra if backend else None,
                "runtime_installed": runtime_installed(backend.engine) if backend else False,
                "languages": list(spec.languages),
                "license": spec.license,
                "github": spec.github,
                "capabilities": list(spec.capabilities),
                "backends": [backend_payload(item) for item in spec.backends],
                "installed": store.is_installed(spec.name),
                "featured": spec.name in featured_rank,
                "fit": fit,
                "pull": f"ses pull {spec.name}" if backend else None,
            }
        )

    speech.sort(
        key=lambda model: (
            featured_rank.get(model["name"], len(featured_rank)),
            model["size_gb"] or 0,
            model["name"],
        )
    )

    brains = [
        {
            "name": brain.name,
            "size_gb": brain.size_gb,
            "description": brain.description,
            "installed": brain.name in installed_brains
            or f"{brain.name}:latest" in installed_brains,
            "fit": hardware.fit(brain.size_gb, ram_gb),
            "pull": f"ses pull {brain.name}",
        }
        for brain in BRAINS
    ]

    return {
        "system": hardware.system_info(),
        "ecosystem": ecosystem.entries(),
        "curated": {"brains": brains, "speech": speech},
    }
