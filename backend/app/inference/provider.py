from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from app.domain.models import BoundingBox


@dataclass(frozen=True, slots=True)
class LocalizationRequest:
    image_key: str
    image_path: Path
    image_width: int
    image_height: int
    prompt: str
    label_name: str


@dataclass(frozen=True, slots=True)
class LocalizationPrediction:
    prediction_id: UUID
    box: BoundingBox
    confidence: float | None = None
    label_hint: str | None = None


ProviderRuntimeStatus = Literal["not_loaded", "loading", "ready", "unavailable", "error"]


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    name: str
    model_name: str
    device: str
    runtime_status: ProviderRuntimeStatus
    available: bool
    platform: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    compute_capability: str | None = None
    requested_dtype: str | None = None
    selected_dtype: str | None = None
    dtype_reason: str | None = None
    detail: str | None = None


class LocalizationProviderError(Exception):
    """User-facing localization error."""


class ProviderUnavailableError(LocalizationProviderError):
    """The model cannot load on this system."""


class ProviderInputError(LocalizationProviderError):
    """The model cannot use the image or prompt."""


class ProviderResponseError(LocalizationProviderError):
    """The model output cannot become annotations."""


class ProviderInferenceError(LocalizationProviderError):
    """Inference failed after the model loaded."""


@runtime_checkable
class VisionLocalizationProvider(Protocol):
    """Interface between app services and localization models."""

    @property
    def name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def localize(self, request: LocalizationRequest) -> list[LocalizationPrediction]: ...


def describe_provider(provider: VisionLocalizationProvider) -> ProviderInfo:
    """Return optional runtime details."""

    diagnostics = getattr(provider, "runtime_diagnostics", None)
    status = getattr(provider, "runtime_status", "ready")
    return ProviderInfo(
        name=provider.name,
        model_name=provider.model_name,
        device=str(getattr(provider, "device", "not_applicable")),
        runtime_status=status,
        available=bool(getattr(diagnostics, "available", status != "unavailable")),
        platform=getattr(diagnostics, "platform", None),
        python_version=getattr(diagnostics, "python_version", None),
        torch_version=getattr(diagnostics, "torch_version", None),
        cuda_version=getattr(diagnostics, "cuda_version", None),
        gpu_name=getattr(diagnostics, "gpu_name", None),
        compute_capability=getattr(diagnostics, "compute_capability", None),
        requested_dtype=getattr(diagnostics, "requested_dtype", None),
        selected_dtype=getattr(diagnostics, "selected_dtype", None),
        dtype_reason=getattr(diagnostics, "dtype_reason", None),
        detail=getattr(diagnostics, "detail", None),
    )
