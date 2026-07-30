import csv
import os
import platform
import re
import shutil
import subprocess
from io import StringIO

GREAT_RATIO = 0.45
OK_RATIO = 0.65
TIGHT_RATIO = 0.9
NVIDIA_SMI_TIMEOUT_SECONDS = 2.0

OS_NAMES = {"Darwin": "macOS", "Linux": "Linux", "Windows": "Windows"}

APPLE, CUDA, CPU = "apple", "cuda", "cpu"

ENGINE_SPEED = {
    "mlx-whisper": {APPLE: 0.10, CUDA: None, CPU: None},
    "mlx-audio-stt": {APPLE: 0.05, CUDA: None, CPU: None},
    "mlx-audio": {APPLE: 0.10, CUDA: None, CPU: None},
    "faster-whisper": {APPLE: 0.30, CUDA: 0.04, CPU: 0.30},
    "kokoro-onnx": {APPLE: 0.25, CUDA: 0.08, CPU: 0.25},
    "piper": {APPLE: 0.05, CUDA: 0.05, CPU: 0.05},
    "onnx-asr": {APPLE: 0.65, CUDA: 0.10, CPU: 0.65},
    "whisper-cpp": {APPLE: 0.20, CUDA: 0.10, CPU: 0.60},
    "vosk": {APPLE: 0.10, CUDA: 0.10, CPU: 0.10},
    "chatterbox": {APPLE: 1.00, CUDA: 0.25, CPU: 4.00},
}


def total_ram_gb():
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, AttributeError, OSError):
        pass

    if platform.system() == "Windows":
        return windows_ram_gb()
    return None


def windows_ram_gb():
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return round(status.ullTotalPhys / 1024**3, 1)
    except Exception:
        return None


def cpu_count():
    try:
        count = os.cpu_count()
    except (OSError, RuntimeError):
        return None
    return count if isinstance(count, int) and count > 0 else None


def nvidia_gpu_info():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"name": None, "vram_gb": None}

    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"name": None, "vram_gb": None}

    if result.returncode != 0:
        return {"name": None, "vram_gb": None}

    gpus = []
    for row in csv.reader(StringIO(result.stdout)):
        if len(row) < 2:
            continue
        name = row[0].strip()
        memory_match = re.search(r"\d+(?:\.\d+)?", row[1])
        if not name or memory_match is None:
            continue
        vram_gb = round(float(memory_match.group()) / 1024, 1)
        gpus.append((vram_gb, name))

    if not gpus:
        return {"name": None, "vram_gb": None}
    vram_gb, name = max(gpus)
    return {"name": name, "vram_gb": vram_gb}


def is_apple_silicon():
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def has_cuda():
    return shutil.which("nvidia-smi") is not None


def accelerator():
    if is_apple_silicon():
        return "Apple Silicon GPU (MLX)"
    if has_cuda():
        return "NVIDIA GPU (CUDA)"
    return "CPU"


def accelerator_class():
    if is_apple_silicon():
        return APPLE
    return CUDA if has_cuda() else CPU


def engine_speed(engine, hardware=None):
    row = ENGINE_SPEED.get(engine)
    if row is None:
        return None
    return row.get(hardware or accelerator_class())


def speed_label(engine, hardware=None):
    seconds = engine_speed(engine, hardware)
    if seconds is None:
        return "not on this machine"
    if seconds >= 1:
        return f"{seconds:.1f}s per second, slower than real time"
    return f"~{1 / seconds:.0f}x faster than real time"


def system_info():
    gpu = nvidia_gpu_info()
    return {
        "os": OS_NAMES.get(platform.system(), platform.system()),
        "arch": platform.machine(),
        "ram_gb": total_ram_gb(),
        "apple_silicon": is_apple_silicon(),
        "accelerator": accelerator(),
        "accelerator_class": accelerator_class(),
        "cpu_count": cpu_count(),
        "gpu_name": gpu["name"],
        "vram_gb": gpu["vram_gb"],
    }


def capacity_fit(required_gb, available_gb, resource="RAM"):
    if not required_gb or not available_gb:
        return {"level": "unknown", "label": "unknown"}

    ratio = required_gb / available_gb
    if ratio <= GREAT_RATIO:
        return {"level": "great", "label": "runs great"}
    if ratio <= OK_RATIO:
        return {"level": "ok", "label": "fits"}
    if ratio <= TIGHT_RATIO:
        suffix = "close other apps" if resource == "RAM" else "close other GPU apps"
        return {"level": "tight", "label": f"tight, {suffix}"}
    label = "too big for your RAM" if resource == "RAM" else f"needs {required_gb:g} GB VRAM"
    return {"level": "toobig", "label": label}


def fit(
    size_gb,
    ram_gb=None,
    required_ram_gb=None,
    required_vram_gb=None,
    accelerators=None,
):
    if accelerators is not None:
        allowed = {accelerators} if isinstance(accelerators, str) else set(accelerators)
        current = accelerator_class()
        if current not in allowed:
            wanted = " / ".join(sorted(allowed)) or "another accelerator"
            return {"level": "unsupported", "label": f"needs {wanted}"}

    if ram_gb is None:
        ram_gb = total_ram_gb()

    ram_required = size_gb if required_ram_gb is None else required_ram_gb
    checks = [capacity_fit(ram_required, ram_gb)]

    if required_vram_gb:
        vram_gb = nvidia_gpu_info()["vram_gb"]
        vram_check = capacity_fit(required_vram_gb, vram_gb, resource="VRAM")
        if vram_gb is None:
            vram_check = {"level": "unknown", "label": "VRAM unknown"}
        checks.append(vram_check)

    priority = {
        "great": 0,
        "ok": 1,
        "tight": 2,
        "unknown": 3,
        "toobig": 4,
        "unsupported": 5,
    }
    return max(checks, key=lambda check: priority[check["level"]])
