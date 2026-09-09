from __future__ import annotations

import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app.inference.factory import create_localization_provider
from app.inference.locateanything import (
    LocateAnythingVisionLocalizationProvider,
)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> int:
    provider = create_localization_provider(dict(os.environ))
    if not isinstance(provider, LocateAnythingVisionLocalizationProvider):
        raise TypeError("Production provider selection did not produce LocateAnything")

    diagnostics = provider.runtime_diagnostics
    worker = provider.worker_path
    hf_home = Path(
        os.environ.get("HF_HOME", REPOSITORY_ROOT / "model-cache" / "huggingface")
    ).expanduser()
    model_cache = hf_home / "hub" / "models--nvidia--LocateAnything-3B"
    frontend_root = REPOSITORY_ROOT / "frontend"
    data_dir = Path(
        os.environ.get(
            "VERIFYVISION_DATA_DIR", REPOSITORY_ROOT / "backend" / "data" / "projects"
        )
    ).expanduser()
    report = {
        "runtime": "local_only",
        "provider": provider.name,
        "model": provider.model_name,
        "model_revision": provider.model_revision,
        "available": diagnostics.available,
        "platform": diagnostics.platform,
        "python": diagnostics.python_version,
        "torch": diagnostics.torch_version,
        "transformers": package_version("transformers"),
        "ultralytics": package_version("ultralytics"),
        "cuda_available": diagnostics.cuda_available,
        "cuda_runtime": diagnostics.cuda_version,
        "device": diagnostics.device,
        "gpu": diagnostics.gpu_name,
        "cuda_capability": diagnostics.compute_capability,
        "requested_dtype": diagnostics.requested_dtype,
        "selected_dtype": diagnostics.selected_dtype,
        "dtype_reason": diagnostics.dtype_reason,
        "model_state": provider.runtime_status,
        "worker_path": str(worker) if worker else None,
        "worker_available": bool(worker and worker.is_file()),
        "huggingface_cache": str(hf_home),
        "model_cache_available": model_cache.is_dir(),
        "project_data": str(data_dir),
        "frontend_dependencies_available": (frontend_root / "node_modules").is_dir(),
        "frontend_lock_available": (frontend_root / "package-lock.json").is_file(),
        "detail": diagnostics.detail,
    }
    print(json.dumps(report, indent=2))
    return 0 if diagnostics.available and report["worker_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
