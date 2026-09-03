"""Atomic persistence and provenance utilities for the main model."""
from __future__ import annotations
import hashlib, json, os, platform, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def runtime_identity(device: str) -> dict[str, Any]:
    return {"device": device, "device_class": torch.device(device).type, "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def source_identity(root: Path) -> dict[str, Any]:
    files = {str(path.relative_to(root)): file_sha256(path) for path in sorted((root / "flymodel" / "main").glob("*.py"))}
    return {"files": files, "sha256": canonical_sha256(files), "runtime_imports_legacy_packages": False}


def training_source_identity(root: Path) -> dict[str, Any]:
    names = ("data.py", "experiment.py", "metrics.py", "model.py")
    files = {f"flymodel/main/{name}": file_sha256(root / "flymodel" / "main" / name) for name in names}
    return {"files": files, "sha256": canonical_sha256(files), "runtime_imports_legacy_packages": False}
