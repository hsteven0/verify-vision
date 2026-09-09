from __future__ import annotations

import gc
import importlib.util
import math
import os
import platform
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from app.domain.models import BoundingBox
from app.training.models import (
    SUPPORTED_CHECKPOINTS,
    ClassDetectionMetrics,
    DetectionMetrics,
    DetectorPrediction,
    ModelChoice,
    TrainingAvailability,
    TrainingConfig,
    TrainingProgressPoint,
    TrainingRuntimeDetails,
    TrainingStage,
    TrainingTimings,
)


class TrainingDependencyUnavailableError(Exception):
    pass


class TrainingExecutionError(Exception):
    pass


class TrainingCancellationRequested(Exception):
    """Stops Ultralytics at a safe callback."""


@dataclass(frozen=True, slots=True)
class AdapterTrainingResult:
    best_checkpoint: Path | None
    last_checkpoint: Path | None
    validation: AdapterValidationResult | None = None
    runtime: TrainingRuntimeDetails | None = None
    timings: TrainingTimings | None = None
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class AdapterValidationResult:
    metrics: DetectionMetrics
    per_class: tuple[ClassDetectionMetrics, ...]


ProgressCallback = Callable[[TrainingProgressPoint], None]
CancellationCheck = Callable[[], bool]
StageCallback = Callable[[TrainingStage, str], None]
RuntimeCallback = Callable[[TrainingRuntimeDetails], None]


def cuda_training_supported(platform_name: str, cuda_available: bool) -> bool:
    """Return whether the strict local YOLO training policy is satisfied."""

    return platform_name in {"Windows", "Linux"} and cuda_available


class DetectorTrainingAdapter(Protocol):
    def availability(self) -> TrainingAvailability: ...

    def resolve_device(self, requested: str) -> str: ...

    def train(
        self,
        *,
        dataset_yaml: Path,
        config: TrainingConfig,
        output_dir: Path,
        device: str,
        class_names: list[str],
        on_progress: ProgressCallback,
        on_stage: StageCallback,
        on_runtime: RuntimeCallback,
        is_cancelled: CancellationCheck,
    ) -> AdapterTrainingResult: ...

    def validate(
        self,
        *,
        checkpoint: Path,
        dataset_yaml: Path,
        config: TrainingConfig,
        output_dir: Path,
        device: str,
        class_names: list[str],
    ) -> AdapterValidationResult: ...

    def predict(
        self,
        *,
        checkpoint: Path,
        image_path: Path,
        image_size: int,
        device: str,
        confidence: float,
    ) -> tuple[DetectorPrediction, ...]: ...


def _finite_probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or number > 1:
        return None
    return number


def metrics_from_mapping(values: Mapping[str, Any] | None) -> DetectionMetrics | None:
    if not values:
        return None

    def first(*keys: str) -> float | None:
        for key in keys:
            if key in values:
                result = _finite_probability(values[key])
                if result is not None:
                    return result
        return None

    metrics = DetectionMetrics(
        precision=first("metrics/precision(B)", "precision"),
        recall=first("metrics/recall(B)", "recall"),
        map50=first("metrics/mAP50(B)", "map50"),
        map50_95=first("metrics/mAP50-95(B)", "map", "map50_95"),
    )
    return metrics if any(value is not None for value in metrics.model_dump().values()) else None


def parse_validation_metrics(raw: Any, class_names: list[str]) -> AdapterValidationResult:
    box = getattr(raw, "box", None)
    if box is None:
        return AdapterValidationResult(DetectionMetrics(), ())

    overall = DetectionMetrics(
        precision=_finite_probability(getattr(box, "mp", None)),
        recall=_finite_probability(getattr(box, "mr", None)),
        map50=_finite_probability(getattr(box, "map50", None)),
        map50_95=_finite_probability(getattr(box, "map", None)),
    )
    raw_indices = getattr(box, "ap_class_index", [])
    indices = raw_indices.tolist() if hasattr(raw_indices, "tolist") else list(raw_indices)
    per_class: list[ClassDetectionMetrics] = []
    for position, class_id_value in enumerate(indices):
        try:
            class_id = int(class_id_value)
            precision, recall, map50, map50_95 = box.class_result(position)
        except (TypeError, ValueError, IndexError):
            continue
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"Class {class_id}"
        per_class.append(
            ClassDetectionMetrics(
                class_id=class_id,
                class_name=class_name,
                precision=_finite_probability(precision),
                recall=_finite_probability(recall),
                map50=_finite_probability(map50),
                map50_95=_finite_probability(map50_95),
            )
        )
    return AdapterValidationResult(overall, tuple(per_class))


class UltralyticsTrainingAdapter:
    """Lazy adapter for the Ultralytics Python API."""

    model_choices = [
        ModelChoice(checkpoint="yolo26n.pt", label="Nano", size="Smallest checkpoint"),
        ModelChoice(checkpoint="yolo26s.pt", label="Small", size="More capacity"),
        ModelChoice(checkpoint="yolo26m.pt", label="Medium", size="Largest offered here"),
    ]

    def availability(self) -> TrainingAvailability:
        installed = importlib.util.find_spec("ultralytics") is not None
        torch_installed = importlib.util.find_spec("torch") is not None
        platform_name = platform.system()
        supported_platform = platform_name in {"Windows", "Linux"}
        version = None
        if installed:
            with suppress(metadata.PackageNotFoundError):
                version = metadata.version("ultralytics")
        if not installed or not torch_installed:
            missing = "Ultralytics" if not installed else "PyTorch"
            reason = f"{missing} is not installed. Install the optional backend training extra."
            detected_device = "unavailable"
            gpu_name = None
            gpu_memory_gb = None
            torch_version = None
            cuda_version = None
        elif not supported_platform:
            reason = "Local training requires Windows or Linux with an NVIDIA CUDA GPU."
            detected_device = "unavailable"
            gpu_name = None
            gpu_memory_gb = None
            torch_version = None
            cuda_version = None
        else:
            import torch

            torch_version = str(torch.__version__)
            cuda_version = str(torch.version.cuda) if torch.version.cuda else None
            if cuda_training_supported(platform_name, torch.cuda.is_available()):
                detected_device = "cuda:0"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3,
                    2,
                )
                reason = None
            else:
                detected_device = "unavailable"
                gpu_name = None
                gpu_memory_gb = None
                reason = "Local training requires an NVIDIA CUDA GPU visible to PyTorch. CPU training is not supported."
        return TrainingAvailability(
            available=installed and torch_installed and supported_platform and reason is None,
            reason=reason,
            package_version=version,
            supported_models=self.model_choices,
            device_options=["auto", "cuda:0"] if reason is None else [],
            detected_device=detected_device,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            torch_version=torch_version,
            cuda_version=cuda_version,
        )

    def resolve_device(self, requested: str) -> str:
        availability = self.availability()
        if not availability.available:
            raise TrainingDependencyUnavailableError(availability.reason or "Training unavailable")
        if requested == "cpu":
            raise TrainingDependencyUnavailableError(
                "CPU training is not supported. Use Windows or Linux with an NVIDIA CUDA GPU."
            )
        if requested == "auto":
            return "cuda:0"
        if requested == "cuda:0" and self._detected_device() != "cuda:0":
            raise TrainingDependencyUnavailableError("CUDA GPU 0 is not available to PyTorch")
        return requested

    @staticmethod
    def _detected_device() -> str:
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "unavailable"

    @staticmethod
    def _ultralytics_device(device: str) -> str | int:
        return 0 if device == "cuda:0" else device

    def train(
        self,
        *,
        dataset_yaml: Path,
        config: TrainingConfig,
        output_dir: Path,
        device: str,
        class_names: list[str],
        on_progress: ProgressCallback,
        on_stage: StageCallback,
        on_runtime: RuntimeCallback,
        is_cancelled: CancellationCheck,
    ) -> AdapterTrainingResult:
        from ultralytics import YOLO

        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        model_loaded_at: float | None = None
        runtime_started_at: float | None = None
        dataloader_started_at: float | None = None
        validation_started_at: float | None = None
        validation_seconds = 0.0
        runtime: TrainingRuntimeDetails | None = None
        model: Any = None
        interrupted = False
        checkpoint = self._checkpoint_path(config.checkpoint)
        workers = self._default_workers()

        on_stage(TrainingStage.LOADING_MODEL, "Loading pretrained YOLO weights")
        try:
            model = YOLO(str(checkpoint))
            model_loaded_at = time.perf_counter()
            on_stage(TrainingStage.INITIALIZING_RUNTIME, "Initializing CUDA and mixed precision")

            def cancel_if_requested(_: Any) -> None:
                if is_cancelled():
                    raise TrainingCancellationRequested

            def preparing_dataloader(_: Any) -> None:
                nonlocal runtime_started_at, dataloader_started_at
                cancel_if_requested(None)
                runtime_started_at = time.perf_counter()
                dataloader_started_at = runtime_started_at
                on_stage(
                    TrainingStage.PREPARING_DATALOADER,
                    "Scanning labels and preparing the data loader",
                )

            def training_started(trainer: Any) -> None:
                nonlocal runtime
                cancel_if_requested(trainer)
                now = time.perf_counter()
                runtime = self._runtime_details(
                    trainer=trainer,
                    checkpoint=checkpoint,
                    fallback_workers=workers,
                    fallback_batch=config.batch_size,
                )
                on_runtime(runtime)
                on_stage(TrainingStage.TRAINING, "Training epoch 1")
                if dataloader_started_at is None:
                    return
                timings.dataloader_initialization_seconds = now - dataloader_started_at
                timings.startup_seconds = now - started

            def training_epoch_started(trainer: Any) -> None:
                cancel_if_requested(trainer)
                epoch = int(getattr(trainer, "epoch", 0)) + 1
                on_stage(TrainingStage.TRAINING, f"Training epoch {epoch} of {config.epochs}")

            def validation_started(_: Any) -> None:
                nonlocal validation_started_at
                cancel_if_requested(None)
                validation_started_at = time.perf_counter()
                on_stage(TrainingStage.VALIDATING, "Validating the current checkpoint")

            def validation_finished(_: Any) -> None:
                nonlocal validation_seconds, validation_started_at
                cancel_if_requested(None)
                if validation_started_at is not None:
                    validation_seconds += time.perf_counter() - validation_started_at
                    validation_started_at = None

            def checkpoint_started(_: Any) -> None:
                cancel_if_requested(None)
                on_stage(TrainingStage.SAVING_CHECKPOINT, "Saving a valid checkpoint")

            def report_epoch(trainer: Any) -> None:
                cancel_if_requested(trainer)
                on_progress(
                    TrainingProgressPoint(
                        epoch=min(
                            config.epochs,
                            int(getattr(trainer, "epoch", -1)) + 1,
                        ),
                        duration_seconds=self._finite_nonnegative(getattr(trainer, "epoch_time", None)),
                        training_loss=self._training_loss(getattr(trainer, "tloss", None)),
                        metrics=metrics_from_mapping(getattr(trainer, "metrics", None)),
                    )
                )

            timings = TrainingTimings(
                model_load_seconds=model_loaded_at - started,
            )
            model.add_callback("on_pretrain_routine_start", preparing_dataloader)
            model.add_callback("on_train_start", training_started)
            model.add_callback("on_train_epoch_start", training_epoch_started)
            model.add_callback("on_train_batch_start", cancel_if_requested)
            model.add_callback("on_train_batch_end", cancel_if_requested)
            model.add_callback("on_val_start", validation_started)
            model.add_callback("on_val_batch_start", cancel_if_requested)
            model.add_callback("on_val_batch_end", cancel_if_requested)
            model.add_callback("on_val_end", validation_finished)
            model.add_callback("on_model_save", checkpoint_started)
            model.add_callback("on_fit_epoch_end", report_epoch)
            model.add_callback("on_train_end", checkpoint_started)
            train_call_started = time.perf_counter()
            model.train(
                data=str(dataset_yaml),
                epochs=config.epochs,
                imgsz=config.image_size,
                batch=config.batch_size,
                device=self._ultralytics_device(device),
                project=str(output_dir),
                name="fit",
                exist_ok=True,
                seed=config.seed,
                deterministic=True,
                amp=True,
                cache=False,
                plots=True,
                workers=workers,
            )
            if runtime_started_at is not None:
                timings.runtime_initialization_seconds = runtime_started_at - train_call_started
            timings.validation_seconds = validation_seconds
            trainer = getattr(model, "trainer", None)
            best, last = self._trainer_checkpoints(trainer)
            raw_validation = getattr(model, "metrics", None)
            validation = parse_validation_metrics(raw_validation, class_names)
            return AdapterTrainingResult(
                best_checkpoint=best,
                last_checkpoint=last,
                validation=validation,
                runtime=runtime,
                timings=timings,
            )
        except TrainingCancellationRequested:
            interrupted = True
            trainer = getattr(model, "trainer", None) if model is not None else None
            best, last = self._trainer_checkpoints(trainer)
            timings = locals().get("timings", TrainingTimings())
            timings.validation_seconds = validation_seconds
            return AdapterTrainingResult(
                best_checkpoint=best,
                last_checkpoint=last,
                runtime=runtime,
                timings=timings,
                cancelled=True,
            )
        except Exception:
            interrupted = True
            raise
        finally:
            if interrupted and model is not None:
                self._close_trainer(getattr(model, "trainer", None))
            if model is not None:
                del model
            self._release_cuda_memory()

    @staticmethod
    def _default_workers() -> int:
        if os.name == "nt":
            # Windows avoids slow spawned workers. Linux uses a small worker pool.
            return 0
        return min(4, max(0, (os.cpu_count() or 1) - 1))

    @staticmethod
    def _checkpoint_path(checkpoint: str) -> Path | str:
        configured = os.getenv("VERIFYVISION_YOLO_CACHE_DIR")
        roots = [Path(configured)] if configured else []
        project_root = Path(__file__).resolve().parents[3]
        roots.extend((project_root, Path.cwd()))
        for root in roots:
            candidate = (root / checkpoint).resolve()
            if candidate.is_file():
                return candidate
        return checkpoint

    @staticmethod
    def _trainer_checkpoints(trainer: Any) -> tuple[Path | None, Path | None]:
        if trainer is None:
            return None, None
        best_value = getattr(trainer, "best", None)
        last_value = getattr(trainer, "last", None)
        best = Path(best_value) if best_value and Path(best_value).is_file() else None
        last = Path(last_value) if last_value and Path(last_value).is_file() else None
        return best, last

    @staticmethod
    def _finite_nonnegative(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    @classmethod
    def _training_loss(cls, values: Any) -> float | None:
        if not isinstance(values, Mapping):
            return None
        losses = [cls._finite_nonnegative(value) for value in values.values()]
        finite = [value for value in losses if value is not None]
        return sum(finite) if finite else None

    @staticmethod
    def _runtime_details(
        *,
        trainer: Any,
        checkpoint: Path | str,
        fallback_workers: int,
        fallback_batch: int,
    ) -> TrainingRuntimeDetails:
        import torch

        device = getattr(trainer, "device", None)
        device_index = getattr(device, "index", None) or 0
        gpu_name = torch.cuda.get_device_name(device_index) if torch.cuda.is_available() else None
        gpu_memory_gb = (
            round(torch.cuda.get_device_properties(device_index).total_memory / 1024**3, 2)
            if torch.cuda.is_available()
            else None
        )
        loader = getattr(trainer, "train_loader", None)
        return TrainingRuntimeDetails(
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            torch_version=str(torch.__version__),
            cuda_version=str(torch.version.cuda) if torch.version.cuda else None,
            amp_enabled=bool(getattr(trainer, "amp", False)),
            workers=int(getattr(loader, "num_workers", fallback_workers)),
            batch_size=int(getattr(trainer, "batch_size", fallback_batch)),
            cache_mode="reusable verified snapshot",
            checkpoint_source=str(checkpoint),
        )

    @staticmethod
    def _close_trainer(trainer: Any) -> None:
        if trainer is None:
            return
        for loader_name in ("train_loader", "test_loader"):
            loader = getattr(trainer, loader_name, None)
            if hasattr(loader, "close"):
                with suppress(Exception):
                    loader.close()
        with suppress(Exception):
            trainer.run_callbacks("teardown")

    @staticmethod
    def _release_cuda_memory() -> None:
        gc.collect()
        with suppress(ImportError, RuntimeError):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def validate(
        self,
        *,
        checkpoint: Path,
        dataset_yaml: Path,
        config: TrainingConfig,
        output_dir: Path,
        device: str,
        class_names: list[str],
    ) -> AdapterValidationResult:
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))
        raw = model.val(
            data=str(dataset_yaml),
            split="val",
            imgsz=config.image_size,
            batch=config.batch_size,
            device=self._ultralytics_device(device),
            project=str(output_dir),
            name="validation",
            exist_ok=True,
            plots=True,
            workers=0 if os.name == "nt" else 4,
        )
        return parse_validation_metrics(raw, class_names)

    def predict(
        self,
        *,
        checkpoint: Path,
        image_path: Path,
        image_size: int,
        device: str,
        confidence: float,
    ) -> tuple[DetectorPrediction, ...]:
        from ultralytics import YOLO

        results = YOLO(str(checkpoint)).predict(
            source=str(image_path),
            imgsz=image_size,
            conf=confidence,
            device=self._ultralytics_device(device),
            verbose=False,
        )
        if not results:
            return ()
        result = results[0]
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return ()
        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        predictions: list[DetectorPrediction] = []
        for edges, score_value, class_value in zip(coordinates, confidences, classes, strict=True):
            class_id = int(class_value)
            score = _finite_probability(score_value)
            if score is None or len(edges) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in edges)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            if isinstance(names, Mapping):
                class_name = str(names.get(class_id, f"Class {class_id}"))
            else:
                class_name = str(names[class_id]) if class_id < len(names) else f"Class {class_id}"
            predictions.append(
                DetectorPrediction(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=score,
                    box=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                )
            )
        return tuple(predictions)


assert tuple(choice.checkpoint for choice in UltralyticsTrainingAdapter.model_choices) == (SUPPORTED_CHECKPOINTS)
