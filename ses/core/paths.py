import json
import os
import re
from pathlib import Path, PureWindowsPath
from .. import SesError

MANIFEST_NAME = "manifest.json"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
WINDOWS_DEVICES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


def home():
    return Path(os.environ.get("SES_HOME", "~/.ses")).expanduser()


def models_dir():
    return home() / "models"


def conversations_dir():
    directory = home() / "conversations"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def config_path():
    return home() / "config.json"


def voice_mode_path():
    return home() / "voice_mode.json"


def safe_dir_name(model_name):
    name = str(model_name).strip()
    parts = name.split("/")
    if (
        not name
        or len(name) > 193
        or "\\" in name
        or PureWindowsPath(name).drive
        or len(parts) > 2
        or any(
            not SAFE_NAME.fullmatch(part)
            or part in {".", ".."}
            or part.startswith(("-", "."))
            or part.endswith(("-", "."))
            or ".." in part
            or "--" in part
            or part.split(".", 1)[0].lower() in WINDOWS_DEVICES
            for part in parts
        )
    ):
        raise SesError(
            f"invalid model name '{model_name}': use a library name or Hugging Face org/repo id"
        )
    return "--".join(parts)


def model_dir(model_name):
    root = models_dir().resolve()
    directory = (root / safe_dir_name(model_name)).resolve()
    try:
        directory.relative_to(root)
    except ValueError as error:
        raise SesError(f"model path escapes the ses model store: {model_name}") from error
    return directory


def read_json(path, fallback=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return fallback


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def config():
    return read_json(config_path(), {}) or {}


def set_default(role, model):
    settings = config()
    if model is None:
        settings.pop(role, None)
    else:
        settings[role] = model
    write_json(config_path(), settings)


def default_for(role):
    from_env = os.environ.get(f"SES_{role.upper()}")
    if from_env:
        return from_env
    return config().get(role)
