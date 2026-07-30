import fnmatch
import stat
import tempfile
import threading
import zipfile
from pathlib import Path
from .. import SesError

IGNORE_PATTERNS = [".gitattributes", "*.mp3"]
HF_API = "https://huggingface.co/api/models"


def matches(path, patterns):
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def remote_size(repo, allow_patterns=None):
    import httpx

    try:
        response = httpx.get(
            f"{HF_API}/{repo}/tree/main",
            params={"recursive": "true"},
            timeout=15,
            follow_redirects=True,
        )
        response.raise_for_status()
        listing = response.json()
    except Exception:
        return None

    total = 0
    for entry in listing:
        if entry.get("type") != "file":
            continue
        path = entry["path"]
        if matches(path, IGNORE_PATTERNS):
            continue
        if allow_patterns and not matches(path, allow_patterns):
            continue
        total += entry.get("size", 0)
    return total or None


def list_remote_files(repo, subpath=""):
    import httpx

    url = f"{HF_API}/{repo}/tree/main"
    if subpath:
        url = f"{url}/{subpath}"
    try:
        response = httpx.get(url, params={"recursive": "true"}, timeout=20, follow_redirects=True)
        response.raise_for_status()
        return [entry["path"] for entry in response.json() if entry.get("type") == "file"]
    except Exception:
        return []


def static_prefix(pattern):
    head = pattern.split("*", 1)[0]
    return head.rsplit("/", 1)[0] if "/" in head else ""


def resolve_single_variant(repo, patterns, sibling_suffixes=(".json",)):
    for pattern in patterns:
        available = list_remote_files(repo, static_prefix(pattern))
        if not available:
            continue

        found = sorted(path for path in available if matches(path, [pattern]))
        if not found:
            continue

        chosen = found[0]
        siblings = [
            path
            for path in available
            if any(path == chosen + suffix for suffix in sibling_suffixes)
        ]
        return (chosen, *siblings)
    return None


def snapshot(repo, dest, allow_patterns=None, on_progress=None):
    from huggingface_hub import snapshot_download

    patterns = list(allow_patterns) if allow_patterns else None
    total = remote_size(repo, allow_patterns) if on_progress else None

    if not (on_progress and total):
        snapshot_download(
            repo_id=repo,
            local_dir=dest,
            allow_patterns=patterns,
            ignore_patterns=IGNORE_PATTERNS,
        )
        return

    from tqdm.auto import tqdm

    lock = threading.Lock()
    downloaded = 0

    class AggregatingTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            nonlocal downloaded
            if n:
                with lock:
                    downloaded += n
                    on_progress(min(downloaded, total), total)
            return super().update(n)

    snapshot_download(
        repo_id=repo,
        local_dir=dest,
        allow_patterns=patterns,
        ignore_patterns=IGNORE_PATTERNS,
        tqdm_class=AggregatingTqdm,
    )
    on_progress(total, total)


def safe_zip_members(archive, dest):
    root = Path(dest).resolve()
    members = archive.infolist()
    for member in members:
        target = (root / member.filename).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise SesError(f"archive contains an unsafe path: {member.filename}") from error

        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise SesError(f"archive contains an unsupported symlink: {member.filename}")
    return members


def archive(url, dest, on_progress=None):
    import httpx

    temporary = tempfile.NamedTemporaryFile(prefix="ses-model-", suffix=".zip", delete=False)
    temp_path = Path(temporary.name)
    temporary.close()

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            raw_total = response.headers.get("content-length")
            total = int(raw_total) if raw_total and raw_total.isdigit() else None
            downloaded = 0
            with temp_path.open("wb") as output:
                for chunk in response.iter_bytes():
                    output.write(chunk)
                    downloaded += len(chunk)
                    if on_progress:
                        on_progress(downloaded, total)

        if on_progress and total:
            on_progress(total, total)

        try:
            with zipfile.ZipFile(temp_path) as bundle:
                members = safe_zip_members(bundle, dest)
                bundle.extractall(dest, members=members)
        except zipfile.BadZipFile as error:
            raise SesError(f"downloaded model is not a valid ZIP archive: {url}") from error
    finally:
        temp_path.unlink(missing_ok=True)


LOADABLE_SUFFIXES = (".onnx", ".npz", ".safetensors", ".bin", ".gguf", ".ggml", ".tflite")


def loadable_formats(repo):
    files = list_remote_files(repo)
    if not files:
        return None
    found = set()
    for path in files:
        lower = path.lower()
        for suffix in LOADABLE_SUFFIXES:
            if lower.endswith(suffix):
                found.add(suffix)
    return found


def is_loadable(repo):
    found = loadable_formats(repo)
    if found is None:
        return True
    return bool(found)
