from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from app.inference.locateanything import LocateAnythingVisionLocalizationProvider
from app.inference.provider import VisionLocalizationProvider

DEFAULT_LOCATEANYTHING_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"


class ProviderConfigurationError(ValueError):
    pass


def create_localization_provider(
    environment: Mapping[str, str] | None = None,
) -> VisionLocalizationProvider:
    """Create the local LocateAnything-3B provider."""

    values = os.environ if environment is None else environment
    worker_path_value = values.get("VERIFYVISION_LOCATEANYTHING_WORKER_PATH", "").strip()
    if not worker_path_value:
        repository_root = Path(__file__).resolve().parents[3]
        vendored_worker = repository_root / ".vendor" / "Eagle" / "Embodied" / "locateanything_worker.py"
        if vendored_worker.is_file():
            worker_path_value = str(vendored_worker)
    max_tokens_value = values.get("VERIFYVISION_LOCATEANYTHING_MAX_NEW_TOKENS", "8192")
    dtype_value = values.get("VERIFYVISION_LOCATEANYTHING_DTYPE", "auto").strip().casefold()
    dtype_value = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(dtype_value, dtype_value)
    try:
        max_new_tokens = int(max_tokens_value)
    except ValueError as error:
        raise ProviderConfigurationError("VERIFYVISION_LOCATEANYTHING_MAX_NEW_TOKENS must be an integer") from error

    try:
        return LocateAnythingVisionLocalizationProvider(
            model_name=values.get("VERIFYVISION_LOCATEANYTHING_MODEL", "nvidia/LocateAnything-3B"),
            model_revision=values.get(
                "VERIFYVISION_LOCATEANYTHING_REVISION",
                DEFAULT_LOCATEANYTHING_REVISION,
            ),
            device=values.get("VERIFYVISION_LOCATEANYTHING_DEVICE", "auto"),
            dtype=dtype_value,
            generation_mode=values.get("VERIFYVISION_LOCATEANYTHING_GENERATION_MODE", "hybrid"),
            max_new_tokens=max_new_tokens,
            worker_path=Path(worker_path_value) if worker_path_value else None,
        )
    except ValueError as error:
        raise ProviderConfigurationError(str(error)) from error


def validate_runtime_provider(provider: VisionLocalizationProvider) -> None:
    """Reject non-LocateAnything runtime providers."""

    if provider.name != "locateanything":
        raise ProviderConfigurationError(
            f"VerifyVision local runtime requires the locateanything provider; received {provider.name}"
        )
