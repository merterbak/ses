import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .. import SesError, __version__
from . import download, paths

MANIFEST_NAME = paths.MANIFEST_NAME


def is_installed(model_name):
    return (paths.model_dir(model_name) / MANIFEST_NAME).is_file()


def manifest_for(model_name):
    return paths.read_json(paths.model_dir(model_name) / MANIFEST_NAME)


def installed():
    root = paths.models_dir()
    if not root.is_dir():
        return []
    manifests = []
    for directory in sorted(root.iterdir()):
        manifest = paths.read_json(directory / MANIFEST_NAME)
        if manifest:
            manifests.append(manifest)
    return manifests


def dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def remove(model_name):
    directory = paths.model_dir(model_name)
    if not directory.exists():
        raise SesError(f"model '{model_name}' is not installed")
    shutil.rmtree(directory)


def staging_dir(model_name):
    root = paths.models_dir()
    root.mkdir(parents=True, exist_ok=True)
    prefix = f".{paths.safe_dir_name(model_name)}-"
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


def cleanup_dir(directory):
    if directory.is_symlink():
        directory.unlink(missing_ok=True)
    elif directory.exists():
        shutil.rmtree(directory)


def commit_install(staging, target):
    backup = None
    if target.exists():
        backup = target.with_name(f".{target.name}-backup-{uuid.uuid4().hex}")
        os.replace(target, backup)

    try:
        os.replace(staging, target)
    except Exception:
        if backup is not None and backup.exists():
            os.replace(backup, target)
        raise
    else:
        if backup is not None:
            cleanup_dir(backup)


def write_manifest(directory, **fields):
    manifest = dict(
        fields,
        installed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ses_version=__version__,
        size_bytes=dir_size(directory),
    )
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def files_to_fetch(backend):
    if not backend.single_variant:
        return backend.files

    chosen = download.resolve_single_variant(backend.repo, backend.files)
    if not chosen:
        raise SesError(
            f"couldn't find a matching voice in {backend.repo}, "
            "check your connection, or pick another model with: ses search"
        )
    return chosen


def install(spec, force=False, on_progress=None):
    backend = spec.backend()
    if backend is None:
        raise SesError(
            f"'{spec.name}' has no backend for this platform yet, "
            "try a whisper-* model, they run everywhere"
        )

    target = paths.model_dir(spec.name)
    if is_installed(spec.name) and not force:
        return target

    staging = staging_dir(spec.name)
    try:
        if backend.url:
            download.archive(backend.url, staging, on_progress=on_progress)
        else:
            download.snapshot(
                backend.repo, staging, files_to_fetch(backend), on_progress=on_progress
            )

        if spec.post_install:
            hook = POST_INSTALL_HOOKS.get(spec.post_install)
            if hook is None:
                raise SesError(f"unknown post-install hook: {spec.post_install}")
            hook(staging)

        validate_layout(staging, backend.engine)
        write_manifest(
            staging,
            name=spec.name,
            kind=spec.kind,
            engine=backend.engine,
            repo=backend.repo,
            source=backend.source,
            format=backend.format,
        )
        commit_install(staging, target)
    except Exception:
        cleanup_dir(staging)
        raise
    return target


def install_custom(repo, engine, force=False, on_progress=None):
    from ..engines import ENGINE_KINDS

    validate_hf_repo(repo)
    if engine not in ENGINE_KINDS:
        raise SesError(f"unknown engine '{engine}'. Pick one of: {', '.join(ENGINE_KINDS)}")

    target = paths.model_dir(repo)
    if is_installed(repo) and not force:
        return target

    staging = staging_dir(repo)
    try:
        download.snapshot(repo, staging, None, on_progress=on_progress)
        validate_layout(staging, engine)
        write_manifest(
            staging,
            name=repo,
            kind=ENGINE_KINDS[engine],
            engine=engine,
            repo=repo,
            source=f"https://huggingface.co/{repo}",
            custom=True,
        )
        commit_install(staging, target)
    except Exception:
        cleanup_dir(staging)
        raise
    return target


def validate_hf_repo(repo):
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(repo)
        if repo.count("/") != 1:
            raise HFValidationError("expected an org/name repository id")
        paths.safe_dir_name(repo)
    except (HFValidationError, ValueError) as error:
        raise SesError(
            f"invalid Hugging Face repo '{repo}': expected a safe org/name id"
        ) from error


def validate_layout(directory, engine):
    def require(condition, expected):
        if not condition:
            raise SesError(
                f"download is not a valid {engine} model, missing {expected}"
            )

    if engine == "whisper-cpp":
        require(len(list(directory.rglob("*.bin"))) == 1, "exactly one GGML .bin")
    elif engine == "vosk":
        require(any(directory.rglob("final.mdl")), "final.mdl")
    elif engine == "chatterbox":
        required = (
            "ve.safetensors",
            "t3_turbo_v1.safetensors",
            "s3gen_meanflow.safetensors",
            "conds.pt",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
        )
        missing = [name for name in required if not (directory / name).is_file()]
        require(not missing, ", ".join(missing))
    elif engine == "faster-whisper":
        required = ("model.bin", "config.json", "tokenizer.json")
        missing = [name for name in required if not (directory / name).is_file()]
        require(not missing, ", ".join(missing))
    elif engine == "kokoro-onnx":
        require(any(directory.rglob("*.onnx")), "ONNX weights")
        require((directory / "voices.npz").is_file(), "voices.npz")
    elif engine == "piper":
        require(any(directory.rglob("*.onnx")), "ONNX voice weights")
        require(any(directory.rglob("*.onnx.json")), "ONNX voice config")
    elif engine == "onnx-asr":
        require(any(directory.rglob("*.onnx")), "ONNX weights")
    elif engine in {"mlx-whisper", "mlx-audio-stt", "mlx-audio"}:
        require((directory / "config.json").is_file(), "config.json")
        weights = (directory / "weights.npz").is_file() or any(
            directory.rglob("*.safetensors")
        )
        require(weights, "MLX weights")


def build_kokoro_voices(directory):
    import numpy as np

    voices_dir = directory / "voices"
    voice_files = sorted(voices_dir.glob("*.bin"))
    if not voice_files:
        raise SesError(f"no voice files found in {voices_dir}")

    styles = {}
    for voice_file in voice_files:
        values = np.fromfile(voice_file, dtype=np.float32)
        if values.size == 510 * 256:
            styles[voice_file.stem] = values.reshape(510, 1, 256)
    if not styles:
        raise SesError("voice files could not be parsed")
    np.savez(directory / "voices.npz", **styles)


POST_INSTALL_HOOKS = {"build_kokoro_voices": build_kokoro_voices}
