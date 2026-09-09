from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any, Protocol, cast
from uuid import NAMESPACE_URL, uuid5

from PIL import Image, UnidentifiedImageError

from app.domain.models import BoundingBox
from app.inference.decord_compat import install_image_only_decord_stub
from app.inference.provider import (
    LocalizationPrediction,
    LocalizationRequest,
    ProviderInferenceError,
    ProviderInputError,
    ProviderResponseError,
    ProviderRuntimeStatus,
    ProviderUnavailableError,
)

logger = logging.getLogger("uvicorn.error")

_BOX_PATTERN = re.compile(r"<box>\s*<(-?\d+)>\s*<(-?\d+)>\s*<(-?\d+)>\s*<(-?\d+)>\s*</box>")
_NO_BOX_PATTERN = re.compile(r"<box>\s*none\s*</box>", re.IGNORECASE)
_ANY_BOX_TAG = re.compile(r"<box\b.*?</box>", re.IGNORECASE | re.DOTALL)
_CUDA_DEVICE = re.compile(r"cuda(?::(\d+))?")
_VALID_DTYPES = {"auto", "bfloat16", "float16", "float32"}
_VALID_GENERATION_MODES = {"fast", "hybrid", "slow"}
_SUPPORTED_PLATFORMS = {"linux", "win32"}
_SUPPORTED_PYTHON = (3, 12)

# Expandable segments reduce CUDA memory fragmentation across image sizes.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


class LocateAnythingWorker(Protocol):
    def ground_multi(self, image: Image.Image, phrase: str, **kwargs: object) -> object: ...


WorkerClassLoader = Callable[[], type[LocateAnythingWorker]]
RuntimeLoader = Callable[[], ModuleType]


@dataclass(frozen=True, slots=True)
class LocateAnythingDiagnostics:
    image_key: str
    prompt: str
    raw_boxes: tuple[tuple[int, int, int, int], ...]
    normalized_boxes: tuple[BoundingBox, ...]


@dataclass(frozen=True, slots=True)
class LocateAnythingRuntimeDiagnostics:
    available: bool
    platform: str
    python_version: str
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    device: str
    compute_capability: str | None = None
    requested_dtype: str = "auto"
    selected_dtype: str | None = None
    dtype_reason: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class LocateAnythingDtypeSelection:
    requested: str
    selected: str
    compute_capability: tuple[int, int] | None
    reason: str

    @property
    def capability_label(self) -> str | None:
        if self.compute_capability is None:
            return None
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"


def select_locateanything_dtype(cuda: Any, device_index: int, requested: str) -> LocateAnythingDtypeSelection:
    """Choose a dtype from CUDA capabilities, not GPU names."""

    try:
        capability_value = cuda.get_device_capability(device_index)
        capability = (int(capability_value[0]), int(capability_value[1]))
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        capability = None
    supports_bfloat16 = False
    should_probe_bfloat16 = requested == "bfloat16" or (
        requested == "auto" and (capability is None or capability[0] >= 8)
    )
    if should_probe_bfloat16:
        try:
            supports_bfloat16 = bool(cuda.is_bf16_supported())
        except (AttributeError, RuntimeError, TypeError):
            supports_bfloat16 = False

    if requested == "bfloat16":
        if not supports_bfloat16:
            raise ProviderUnavailableError("BF16 was requested, but PyTorch does not report BF16 support for this GPU.")
        return LocateAnythingDtypeSelection(
            requested=requested,
            selected="bfloat16",
            compute_capability=capability,
            reason="explicit BF16 developer override; PyTorch reports device support",
        )
    if requested in {"float16", "float32"}:
        return LocateAnythingDtypeSelection(
            requested=requested,
            selected=requested,
            compute_capability=capability,
            reason=f"explicit {requested} developer override",
        )

    # Ampere added native BF16 acceleration. Older GPUs are faster with FP16.
    if capability is not None and capability[0] >= 8 and supports_bfloat16:
        return LocateAnythingDtypeSelection(
            requested=requested,
            selected="bfloat16",
            compute_capability=capability,
            reason="Ampere-or-newer CUDA capability with PyTorch BF16 support",
        )
    if capability is not None and capability[0] < 8:
        reason = "pre-Ampere CUDA device; FP16 selected for efficient native inference"
    elif not supports_bfloat16:
        reason = "PyTorch does not report BF16 support; FP16 selected"
    else:
        reason = "CUDA capability could not be inspected reliably; FP16 selected conservatively"
    return LocateAnythingDtypeSelection(
        requested=requested,
        selected="float16",
        compute_capability=capability,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class LocateAnythingTiming:
    image_key: str
    prompt: str
    model_loaded_this_request: bool
    model_load_ms: float
    image_decode_ms: float
    input_preparation_ms: float
    gpu_inference_ms: float
    output_parse_ms: float
    coordinate_normalization_ms: float
    prediction_build_ms: float
    total_ms: float
    completed_at: float

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "image_key": self.image_key,
            "prompt": self.prompt,
            "model_loaded_this_request": self.model_loaded_this_request,
            "model_load_ms": round(self.model_load_ms, 2),
            "image_decode_ms": round(self.image_decode_ms, 2),
            "input_preparation_ms": round(self.input_preparation_ms, 2),
            "gpu_inference_ms": round(self.gpu_inference_ms, 2),
            "output_parse_ms": round(self.output_parse_ms, 2),
            "coordinate_normalization_ms": round(self.coordinate_normalization_ms, 2),
            "prediction_build_ms": round(self.prediction_build_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


def normalize_locateanything_boxes(answer: str, image_width: int, image_height: int) -> list[BoundingBox]:
    """Convert NVIDIA's normalized 0..1000 x1/y1/x2/y2 tokens to pixel boxes."""

    if image_width <= 0 or image_height <= 0:
        raise ProviderInputError("Image dimensions must be positive for localization")
    if not isinstance(answer, str) or not answer.strip():
        raise ProviderResponseError("LocateAnything returned an empty response")

    matches = list(_BOX_PATTERN.finditer(answer))
    no_box_matches = list(_NO_BOX_PATTERN.finditer(answer))
    tagged_blocks = list(_ANY_BOX_TAG.finditer(answer))
    if tagged_blocks and len(matches) + len(no_box_matches) != len(tagged_blocks):
        raise ProviderResponseError("LocateAnything returned a malformed or non-box coordinate block")
    if matches and no_box_matches:
        raise ProviderResponseError("LocateAnything returned both detections and a no-object block")
    if not matches:
        if no_box_matches:
            return []
        if "<box" in answer.casefold() or "</box>" in answer.casefold():
            raise ProviderResponseError("LocateAnything returned a malformed box response")
        return []

    boxes: list[BoundingBox] = []
    for match in matches:
        x1, y1, x2, y2 = (int(value) for value in match.groups())
        coordinates = (x1, y1, x2, y2)
        if any(value < 0 or value > 1000 for value in coordinates):
            raise ProviderResponseError("LocateAnything returned coordinates outside its 0..1000 range")
        if x2 <= x1 or y2 <= y1:
            raise ProviderResponseError("LocateAnything returned a reversed or zero-area bounding box")

        pixel_x1 = min(float(image_width), max(0.0, x1 / 1000 * image_width))
        pixel_y1 = min(float(image_height), max(0.0, y1 / 1000 * image_height))
        pixel_x2 = min(float(image_width), max(0.0, x2 / 1000 * image_width))
        pixel_y2 = min(float(image_height), max(0.0, y2 / 1000 * image_height))
        box = BoundingBox(
            x=pixel_x1,
            y=pixel_y1,
            width=pixel_x2 - pixel_x1,
            height=pixel_y2 - pixel_y1,
        )
        if not box.fits_within(image_width, image_height):
            raise ProviderResponseError("LocateAnything returned a box outside the source image")
        boxes.append(box)
    return boxes


class LocateAnythingVisionLocalizationProvider:
    """Lazy adapter for NVIDIA's LocateAnything worker."""

    name = "locateanything"

    def __init__(
        self,
        *,
        model_name: str = "nvidia/LocateAnything-3B",
        model_revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        generation_mode: str = "hybrid",
        max_new_tokens: int = 8192,
        worker_path: Path | None = None,
        worker_class_loader: WorkerClassLoader | None = None,
        runtime_loader: RuntimeLoader | None = None,
        platform_name: str | None = None,
        python_version: tuple[int, int, int] | None = None,
    ) -> None:
        configured_device = device.strip().casefold()
        configured_dtype = dtype.strip().casefold()
        configured_mode = generation_mode.strip().casefold()
        if configured_device not in {"auto", "cuda"} and not _CUDA_DEVICE.fullmatch(configured_device):
            raise ValueError("LocateAnything device must be auto, cuda, or cuda:<index>")
        if configured_dtype not in _VALID_DTYPES:
            raise ValueError("LocateAnything dtype must be auto, bfloat16, float16, or float32")
        if configured_mode not in _VALID_GENERATION_MODES:
            raise ValueError("LocateAnything generation mode must be fast, hybrid, or slow")
        if not 1 <= max_new_tokens <= 8192:
            raise ValueError("LocateAnything max_new_tokens must be between 1 and 8192")

        self.model_name = model_name
        self.model_revision = model_revision.strip() if model_revision else None
        self._configured_device = configured_device
        self._configured_dtype = configured_dtype
        self._generation_mode = configured_mode
        self._max_new_tokens = max_new_tokens
        self._worker_path = worker_path
        self._worker_class_loader = worker_class_loader
        self._runtime_loader = runtime_loader or self._import_torch
        self._platform_name = platform_name or sys.platform
        self._python_version = python_version or (
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        )
        self._runtime: ModuleType | None = None
        self._runtime_diagnostics: LocateAnythingRuntimeDiagnostics | None = None
        self._dtype_selection: LocateAnythingDtypeSelection | None = None
        self._worker: LocateAnythingWorker | None = None
        self._resolved_device: str | None = None
        self._runtime_status: ProviderRuntimeStatus = "not_loaded"
        self._inference_lock = asyncio.Lock()
        self._last_diagnostics: LocateAnythingDiagnostics | None = None
        self._last_timing: LocateAnythingTiming | None = None

    @property
    def device(self) -> str:
        return self._resolved_device or self._configured_device

    @property
    def worker_path(self) -> Path | None:
        return self._worker_path

    @property
    def runtime_status(self) -> ProviderRuntimeStatus:
        return self._runtime_status

    @property
    def last_diagnostics(self) -> LocateAnythingDiagnostics | None:
        return self._last_diagnostics

    @property
    def last_timing(self) -> LocateAnythingTiming | None:
        return self._last_timing

    @property
    def runtime_diagnostics(self) -> LocateAnythingRuntimeDiagnostics:
        if self._runtime_diagnostics is None:
            self._runtime_diagnostics = self._inspect_runtime()
            if self._runtime_diagnostics.device.startswith("cuda"):
                self._resolved_device = self._runtime_diagnostics.device
            if not self._runtime_diagnostics.available:
                self._runtime_status = "unavailable"
        return self._runtime_diagnostics

    async def localize(self, request: LocalizationRequest) -> list[LocalizationPrediction]:
        # One call at a time avoids duplicate models and excess GPU memory use.
        async with self._inference_lock:
            return await asyncio.to_thread(self._localize_sync, request)

    def _localize_sync(self, request: LocalizationRequest) -> list[LocalizationPrediction]:
        total_started = perf_counter()
        model_loaded_this_request = self._worker is None
        stage_started = perf_counter()
        worker = self._get_or_load_worker()
        model_load_ms = (perf_counter() - stage_started) * 1000
        stage_started = perf_counter()
        try:
            with Image.open(request.image_path) as opened:
                opened.load()
                if opened.size != (request.image_width, request.image_height):
                    raise ProviderInputError("Stored image dimensions do not match project metadata")
                image = opened.convert("RGB")
        except ProviderInputError:
            raise
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ProviderInputError("The selected image could not be read for LocateAnything inference") from error
        image_decode_ms = (perf_counter() - stage_started) * 1000

        logger.info(
            "Running LocateAnything inference (image_id=%s, width=%d, height=%d, prompt=%r)",
            request.image_key,
            request.image_width,
            request.image_height,
            request.prompt,
        )
        stage_started = perf_counter()
        try:
            raw_result = worker.ground_multi(
                image,
                request.prompt,
                generation_mode=self._generation_mode,
                max_new_tokens=self._max_new_tokens,
                verbose=False,
            )
        except Exception as error:
            self._runtime_status = "error"
            logger.exception("LocateAnything inference failed")
            raise ProviderInferenceError(
                "LocateAnything inference failed; check backend logs for runtime details"
            ) from error
        finally:
            image.close()
        worker_call_ms = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        if not isinstance(raw_result, Mapping):
            raise ProviderResponseError("LocateAnything returned an unexpected response type")
        raw_answer = raw_result.get("answer")
        if not isinstance(raw_answer, str):
            raise ProviderResponseError("LocateAnything response did not contain a text answer")

        raw_boxes = tuple(tuple(int(value) for value in match.groups()) for match in _BOX_PATTERN.finditer(raw_answer))
        output_parse_ms = (perf_counter() - stage_started) * 1000
        logger.info("LocateAnything raw detections: %d", len(raw_boxes))
        logger.debug("LocateAnything raw boxes (0..1000 x1/y1/x2/y2): %s", raw_boxes)
        stage_started = perf_counter()
        try:
            boxes = normalize_locateanything_boxes(raw_answer, request.image_width, request.image_height)
        except ProviderResponseError:
            self._runtime_status = "error"
            logger.exception("LocateAnything output normalization failed")
            raise
        coordinate_normalization_ms = (perf_counter() - stage_started) * 1000
        self._last_diagnostics = LocateAnythingDiagnostics(
            image_key=request.image_key,
            prompt=request.prompt,
            raw_boxes=raw_boxes,
            normalized_boxes=tuple(boxes),
        )
        logger.debug(
            "VerifyVision normalized boxes (source-image pixels): %s",
            tuple(box.model_dump() for box in boxes),
        )
        stage_started = perf_counter()
        predictions = []
        for index, box in enumerate(boxes):
            identity = (
                f"{self.name}:{self.model_name}:{request.image_key}:{request.prompt}:"
                f"{index}:{box.x:.6f}:{box.y:.6f}:{box.width:.6f}:{box.height:.6f}"
            )
            predictions.append(
                LocalizationPrediction(
                    prediction_id=uuid5(NAMESPACE_URL, identity),
                    box=box,
                    confidence=None,
                    label_hint=request.label_name,
                )
            )
        prediction_build_ms = (perf_counter() - stage_started) * 1000
        worker_timings = raw_result.get("timings_ms")
        timing_values = worker_timings if isinstance(worker_timings, Mapping) else {}

        def timing_value(name: str) -> float:
            value = timing_values.get(name)
            return float(value) if isinstance(value, (int, float)) else 0.0

        input_preparation_ms = sum(
            timing_value(name)
            for name in (
                "message_build",
                "chat_template",
                "vision_info",
                "processor",
                "input_to_device",
                "dtype_cast",
            )
        )
        gpu_inference_ms = timing_value("model_generate")
        if not timing_values:
            gpu_inference_ms = worker_call_ms
        self._last_timing = LocateAnythingTiming(
            image_key=request.image_key,
            prompt=request.prompt,
            model_loaded_this_request=model_loaded_this_request,
            model_load_ms=model_load_ms,
            image_decode_ms=image_decode_ms,
            input_preparation_ms=input_preparation_ms,
            gpu_inference_ms=gpu_inference_ms,
            output_parse_ms=output_parse_ms + timing_value("result_packaging"),
            coordinate_normalization_ms=coordinate_normalization_ms,
            prediction_build_ms=prediction_build_ms,
            total_ms=(perf_counter() - total_started) * 1000,
            completed_at=perf_counter(),
        )
        self._runtime_status = "ready"
        logger.info(
            "LocateAnything timing: %s",
            self._last_timing.as_dict(),
        )
        logger.info("LocateAnything inference completed (detections=%d)", len(predictions))
        return predictions

    def _get_or_load_worker(self) -> LocateAnythingWorker:
        if self._worker is not None:
            return self._worker

        self._runtime_status = "loading"
        try:
            diagnostics = self.runtime_diagnostics
            if not diagnostics.available:
                raise ProviderUnavailableError(diagnostics.detail or "LocateAnything CUDA runtime is unavailable")
            runtime = self._runtime
            if runtime is None:
                raise ProviderUnavailableError("PyTorch runtime inspection did not complete")
            resolved_device = diagnostics.device
            resolved_dtype = self._resolve_dtype(runtime, resolved_device)
            if self._platform_name == "win32":
                install_image_only_decord_stub()
            worker_class = (
                self._worker_class_loader()
                if self._worker_class_loader is not None
                else self._load_official_worker_class()
            )
            model_source = self._resolve_model_source()
            logger.info(
                "Loading %s (revision=%s, device=%s, dtype=%s)...",
                self.model_name,
                self.model_revision or "unpinned",
                resolved_device,
                diagnostics.selected_dtype or self._configured_dtype,
            )
            self._resolved_device = resolved_device
            self._worker = worker_class(
                model_source,
                device=resolved_device,
                dtype=resolved_dtype,
            )
            self._runtime_status = "ready"
            logger.info("LocateAnything model ready")
            return self._worker
        except ProviderUnavailableError:
            if self._runtime_status != "unavailable":
                self._runtime_status = "error"
            logger.exception("LocateAnything model load failed")
            raise
        except Exception as error:
            self._runtime_status = "error"
            logger.exception("LocateAnything model load failed")
            raise ProviderUnavailableError(
                "LocateAnything could not start. Check the runtime, worker path, device, and backend log."
            ) from error

    def _resolve_model_source(self) -> str:
        if not self.model_revision:
            return self.model_name
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError

            try:
                return snapshot_download(
                    repo_id=self.model_name,
                    revision=self.model_revision,
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                return snapshot_download(repo_id=self.model_name, revision=self.model_revision)
        except Exception as error:
            raise ProviderUnavailableError(
                f"Pinned LocateAnything revision {self.model_revision} is unavailable in the "
                "configured Hugging Face cache and could not be downloaded"
            ) from error

    def _inspect_runtime(self) -> LocateAnythingRuntimeDiagnostics:
        python_version = ".".join(str(value) for value in self._python_version)
        default = {
            "platform": self._platform_name,
            "python_version": python_version,
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
            "device": self._configured_device,
            "requested_dtype": self._configured_dtype,
        }
        if self._platform_name not in _SUPPORTED_PLATFORMS:
            return LocateAnythingRuntimeDiagnostics(
                available=False,
                detail="LocateAnything requires Windows or Linux with an NVIDIA CUDA GPU.",
                **default,
            )
        if self._python_version[:2] != _SUPPORTED_PYTHON:
            return LocateAnythingRuntimeDiagnostics(
                available=False,
                detail=(
                    f"Python {python_version} is not supported. Create the VerifyVision environment with Python 3.12."
                ),
                **default,
            )
        try:
            runtime = self._runtime_loader()
        except ProviderUnavailableError as error:
            return LocateAnythingRuntimeDiagnostics(available=False, detail=str(error), **default)
        except Exception as error:
            logger.exception("PyTorch runtime inspection failed")
            return LocateAnythingRuntimeDiagnostics(
                available=False,
                detail=f"PyTorch could not be inspected: {type(error).__name__}",
                **default,
            )

        self._runtime = runtime
        cuda = cast(Any, runtime).cuda
        runtime_version = getattr(runtime, "version", None)
        torch_version = str(getattr(runtime, "__version__", "unknown"))
        cuda_version_value = getattr(runtime_version, "cuda", None)
        cuda_version = str(cuda_version_value) if cuda_version_value else None
        hip_version = getattr(runtime_version, "hip", None)
        inspected = {
            **default,
            "torch_version": torch_version,
            "cuda_version": cuda_version,
        }
        if hip_version:
            return LocateAnythingRuntimeDiagnostics(
                available=False,
                detail="AMD ROCm is not supported. LocateAnything requires NVIDIA CUDA.",
                **inspected,
            )
        if not cuda_version:
            return LocateAnythingRuntimeDiagnostics(
                available=False,
                detail="PyTorch has no CUDA support. Install the CUDA build with VerifyVision setup.",
                **inspected,
            )
        if not cuda.is_available():
            return LocateAnythingRuntimeDiagnostics(
                available=False,
                detail="CUDA is not available to PyTorch. Check the NVIDIA driver and CUDA-enabled PyTorch.",
                **inspected,
            )
        inspected["cuda_available"] = True

        match = _CUDA_DEVICE.fullmatch(self._configured_device)
        index = int(match.group(1)) if match and match.group(1) is not None else 0
        if index >= int(cuda.device_count()):
            return LocateAnythingRuntimeDiagnostics(
                available=False,
                detail=f"Configured CUDA device index {index} is not available.",
                **inspected,
            )
        device = f"cuda:{index}"
        try:
            gpu_name = str(cuda.get_device_name(index))
        except Exception as error:
            logger.exception("CUDA device inspection failed")
            return LocateAnythingRuntimeDiagnostics(
                available=False,
                detail=f"CUDA device {index} could not be inspected: {type(error).__name__}",
                **inspected,
            )
        try:
            dtype_selection = select_locateanything_dtype(cuda, index, self._configured_dtype)
        except ProviderUnavailableError as error:
            return LocateAnythingRuntimeDiagnostics(
                available=False,
                gpu_name=gpu_name,
                device=device,
                detail=str(error),
                **{key: value for key, value in inspected.items() if key not in {"gpu_name", "device"}},
            )
        self._dtype_selection = dtype_selection
        logger.info(
            "LocateAnything CUDA runtime: GPU=%s; CUDA capability=%s; requested dtype=%s; selected dtype=%s; reason=%s",
            gpu_name,
            dtype_selection.capability_label or "unknown",
            dtype_selection.requested,
            dtype_selection.selected,
            dtype_selection.reason,
        )
        dtype_diagnostics = {
            "compute_capability": dtype_selection.capability_label,
            "requested_dtype": dtype_selection.requested,
            "selected_dtype": dtype_selection.selected,
            "dtype_reason": dtype_selection.reason,
        }
        dependency_error = self._runtime_dependency_error()
        if dependency_error is not None:
            return LocateAnythingRuntimeDiagnostics(
                available=False,
                platform=self._platform_name,
                python_version=python_version,
                torch_version=torch_version,
                cuda_available=True,
                cuda_version=cuda_version,
                gpu_name=gpu_name,
                device=device,
                **dtype_diagnostics,
                detail=dependency_error,
            )
        return LocateAnythingRuntimeDiagnostics(
            available=True,
            platform=self._platform_name,
            python_version=python_version,
            torch_version=torch_version,
            cuda_available=True,
            cuda_version=cuda_version,
            gpu_name=gpu_name,
            device=device,
            **dtype_diagnostics,
        )

    def _runtime_dependency_error(self) -> str | None:
        if self._worker_class_loader is not None:
            return None
        if importlib.util.find_spec("transformers") is None:
            return "Transformers is not installed. Run VerifyVision setup."
        if self._worker_path is not None:
            if self._worker_path.resolve().is_file():
                return None
            return "The configured official LocateAnything worker file does not exist."
        try:
            worker_spec = importlib.util.find_spec("locateanything_worker")
        except (ImportError, ValueError):
            worker_spec = None
        if worker_spec is None:
            return "LocateAnything worker is missing. Run setup or set VERIFYVISION_LOCATEANYTHING_WORKER_PATH."
        return None

    def _resolve_dtype(self, runtime: ModuleType, device: str) -> object:
        if self._dtype_selection is None:
            index = int(device.split(":", 1)[1]) if ":" in device else 0
            self._dtype_selection = select_locateanything_dtype(cast(Any, runtime).cuda, index, self._configured_dtype)
        return getattr(runtime, self._dtype_selection.selected)

    def _load_official_worker_class(self) -> type[LocateAnythingWorker]:
        if self._worker_path is None:
            try:
                module = importlib.import_module("locateanything_worker")
            except ImportError as error:
                raise ProviderUnavailableError(
                    "LocateAnything worker is missing. Set VERIFYVISION_LOCATEANYTHING_WORKER_PATH "
                    "to the official NVlabs/Eagle worker."
                ) from error
        else:
            worker_path = self._worker_path.resolve()
            if not worker_path.is_file():
                raise ProviderUnavailableError("VERIFYVISION_LOCATEANYTHING_WORKER_PATH does not point to a file")
            spec = importlib.util.spec_from_file_location("_verifyvision_nvidia_locateanything_worker", worker_path)
            if spec is None or spec.loader is None:
                raise ProviderUnavailableError("NVIDIA's LocateAnything worker could not be loaded")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        worker_class = getattr(module, "LocateAnythingWorker", None)
        if worker_class is None:
            raise ProviderUnavailableError("The configured NVIDIA worker does not define LocateAnythingWorker")
        return cast(type[LocateAnythingWorker], worker_class)

    @staticmethod
    def _import_torch() -> ModuleType:
        try:
            return importlib.import_module("torch")
        except ImportError as error:
            raise ProviderUnavailableError(
                "PyTorch is not installed. Install a build matching the local CUDA runtime."
            ) from error
