from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from threading import Event, RLock
from time import perf_counter
from uuid import UUID, uuid4
from zipfile import ZipFile

from PIL import Image, ImageOps, UnidentifiedImageError

from app.exporting.common import assign_dataset_splits, training_projection
from app.exporting.models import DatasetSplit, ExportFormat, ExportOptions
from app.exporting.service import ExportService
from app.persistence.repository import ProjectRepository
from app.services.projects import MAX_IMAGE_BYTES
from app.training.adapter import (
    DetectorTrainingAdapter,
    TrainingDependencyUnavailableError,
    TrainingExecutionError,
    UltralyticsTrainingAdapter,
)
from app.training.models import (
    ACTIVE_TRAINING_STATUSES,
    DatasetSnapshot,
    DetectorPrediction,
    PredictionPreview,
    TrainingArtifact,
    TrainingAvailability,
    TrainingConfig,
    TrainingProgressPoint,
    TrainingReadiness,
    TrainingRun,
    TrainingRuntimeDetails,
    TrainingStage,
    TrainingStatus,
    TrainingTimings,
)
from app.training.repository import TrainingRunRepository

LOGGER = logging.getLogger(__name__)


class TrainingConflictError(Exception):
    pass


class TrainingValidationError(Exception):
    pass


class TrainingPredictionError(Exception):
    pass


class TrainingService:
    """Owns verified snapshots, one local worker, and persisted detector runs."""

    def __init__(
        self,
        projects: ProjectRepository,
        export_service: ExportService | None = None,
        adapter: DetectorTrainingAdapter | None = None,
    ) -> None:
        self.projects = projects
        self.exports = export_service or ExportService(projects)
        self.adapter = adapter or UltralyticsTrainingAdapter()
        self.runs = TrainingRunRepository(projects)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verifyvision-yolo")
        self._lock = RLock()
        self._cancel_events: dict[UUID, Event] = {}
        self._recover_interrupted_runs()

    def availability(self) -> TrainingAvailability:
        return self.adapter.availability()

    def readiness(self, project_id: UUID, config: TrainingConfig) -> TrainingReadiness:
        project = self.projects.get(project_id)
        options = ExportOptions(
            split=DatasetSplit.TRAIN_VAL,
            train_ratio=config.train_ratio,
            seed=config.seed,
        )
        preview = self.exports.validator.validate(project, options)
        blockers = [finding.message for finding in preview.findings if finding.severity == "error"]
        advisories = [finding.message for finding in preview.findings if finding.severity == "warning"]
        train_boxes = 0
        validation_boxes = 0
        projected = training_projection(project)
        if len(projected) >= 2:
            split = assign_dataset_splits(projected, options)
            train_boxes = sum(len(item.annotations) for item in split if item.split == "train")
            validation_boxes = sum(len(item.annotations) for item in split if item.split == "val")

        if preview.training_boxes == 0:
            blockers.append("The verified dataset has no eligible training annotations.")
        elif train_boxes == 0:
            blockers.append("The deterministic training split contains no eligible boxes.")
        if preview.validation_images and validation_boxes == 0:
            blockers.append("The deterministic validation split has no eligible boxes; change the split seed or ratio.")
        if 0 < preview.validation_images < 5:
            advisories.append("Fewer than five validation images make metrics unreliable.")
        return TrainingReadiness(
            ready=not blockers,
            preview=preview,
            train_boxes=train_boxes,
            validation_boxes=validation_boxes,
            blockers=blockers,
            advisories=advisories,
        )

    def create_run(self, project_id: UUID, config: TrainingConfig) -> TrainingRun:
        availability = self.adapter.availability()
        if not availability.available:
            raise TrainingDependencyUnavailableError(availability.reason or "Training is unavailable")
        readiness = self.readiness(project_id, config)
        if not readiness.ready:
            raise TrainingValidationError(readiness.blockers[0])

        with self._lock:
            active = next(
                (run for run in self.runs.list_all() if run.status in ACTIVE_TRAINING_STATUSES),
                None,
            )
            if active is not None:
                raise TrainingConflictError(
                    f"Training run {active.id} is already active; wait for it to finish or cancel it."
                )
            run = self.runs.save(
                TrainingRun(
                    project_id=project_id,
                    config=config,
                    total_epochs=config.epochs,
                )
            )
            cancel_event = Event()
            self._cancel_events[run.id] = cancel_event
            self._executor.submit(self._execute_run, run.id, project_id, cancel_event)
            return run

    def list_runs(self, project_id: UUID) -> list[TrainingRun]:
        self.projects.get(project_id)
        return self.runs.list(project_id)

    def get_run(self, project_id: UUID, run_id: UUID) -> TrainingRun:
        self.projects.get(project_id)
        return self.runs.get(project_id, run_id)

    def cancel(self, project_id: UUID, run_id: UUID) -> TrainingRun:
        with self._lock:
            run = self.get_run(project_id, run_id)
            if run.status is TrainingStatus.CANCELLING:
                raise TrainingConflictError("Training cancellation is already in progress")
            if run.status not in ACTIVE_TRAINING_STATUSES:
                raise TrainingConflictError(f"Training run is already {run.status.value}")
            now = datetime.now(UTC)
            run.cancel_requested = True
            run.cancel_requested_at = now
            run.status = TrainingStatus.CANCELLING
            run.stage = TrainingStage.CANCELLING
            run.stage_message = "Stopping after the current safe training operation"
            self.runs.save(run)
            self._cancel_events.setdefault(run.id, Event()).set()
            return run

    def artifact(self, project_id: UUID, run_id: UUID, kind: str) -> tuple[Path, str]:
        run = self.get_run(project_id, run_id)
        record = next((artifact for artifact in run.artifacts if artifact.kind == kind), None)
        if record is None:
            raise TrainingPredictionError(f"Artifact {kind} is not available for this run")
        root = self.runs.artifacts_dir(project_id, run_id).resolve()
        candidate = (root / record.filename).resolve()
        if candidate.parent != root or not candidate.is_file():
            raise TrainingPredictionError("Registered training artifact is missing")
        return candidate, record.filename

    def predict_project_image(
        self,
        project_id: UUID,
        run_id: UUID,
        image_id: UUID,
        confidence: float,
    ) -> PredictionPreview:
        project = self.projects.get(project_id)
        image = next((item for item in project.images if item.id == image_id), None)
        if image is None:
            raise TrainingPredictionError(f"Image {image_id} was not found")
        image_path = self.projects.image_path(project_id, image.storage_name)
        if not image_path.is_file():
            raise TrainingPredictionError(f"Image file for {image.filename} is missing")
        detections = self._predict(project_id, run_id, image_path, confidence)
        return PredictionPreview(
            source="project_image",
            filename=image.filename,
            width=image.width,
            height=image.height,
            image_id=image.id,
            detections=list(detections),
        )

    def predict_uploaded_image(
        self,
        project_id: UUID,
        run_id: UUID,
        filename: str,
        content: bytes,
        confidence: float,
    ) -> PredictionPreview:
        self.projects.get(project_id)
        if not content:
            raise TrainingPredictionError("Choose an image to preview")
        if len(content) > MAX_IMAGE_BYTES:
            raise TrainingPredictionError("Preview images must be 25 MB or smaller")
        preview_id = uuid4()
        directory = self.runs.predictions_dir(project_id, run_id) / str(preview_id)
        directory.mkdir(parents=True, exist_ok=False)
        destination = directory / "source.png"
        temporary = directory / "source.png.tmp"
        try:
            with Image.open(BytesIO(content)) as raw:
                raw.load()
                normalized = ImageOps.exif_transpose(raw).convert("RGB")
                width, height = normalized.size
                normalized.save(temporary, format="PNG", optimize=True)
            os.replace(temporary, destination)
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            shutil.rmtree(directory, ignore_errors=True)
            raise TrainingPredictionError("Uploaded preview is not a valid image") from error
        try:
            detections = self._predict(project_id, run_id, destination, confidence)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return PredictionPreview(
            id=preview_id,
            source="uploaded_image",
            filename=Path(filename).name or "preview.png",
            width=width,
            height=height,
            detections=list(detections),
        )

    def prediction_image(self, project_id: UUID, run_id: UUID, preview_id: UUID) -> Path:
        self.get_run(project_id, run_id)
        root = self.runs.predictions_dir(project_id, run_id).resolve()
        candidate = (root / str(preview_id) / "source.png").resolve()
        if candidate.parent.parent != root or not candidate.is_file():
            raise TrainingPredictionError(f"Prediction preview {preview_id} was not found")
        return candidate

    def shutdown(self) -> None:
        for event in self._cancel_events.values():
            event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _execute_run(self, run_id: UUID, project_id: UUID, cancel_event: Event) -> None:
        worker_started = perf_counter()
        try:
            if cancel_event.is_set():
                self._mark_cancelled(project_id, run_id)
                return
            run = self.get_run(project_id, run_id)
            started_at = datetime.now(UTC)
            run.status = TrainingStatus.PREPARING
            run.stage = TrainingStage.VALIDATING_DATASET
            run.stage_message = "Validating verified annotations and the train/validation split"
            run.started_at = started_at
            run.timings.request_to_worker_seconds = max(
                0.0,
                (started_at - run.created_at).total_seconds(),
            )
            self.runs.save(run)

            validation_started = perf_counter()
            readiness = self.readiness(project_id, run.config)
            run = self.get_run(project_id, run_id)
            run.timings.dataset_validation_seconds = perf_counter() - validation_started
            self.runs.save(run)
            if not readiness.ready:
                raise TrainingValidationError(readiness.blockers[0])
            if cancel_event.is_set():
                self._mark_cancelled(project_id, run_id)
                return

            self._record_stage(
                project_id,
                run_id,
                TrainingStage.PREPARING_DATASET,
                "Preparing an immutable verified dataset snapshot",
            )
            preparation_started = perf_counter()
            dataset_yaml, dataset_reused = self._create_snapshot(run, readiness)
            run = self.get_run(project_id, run_id)
            run.timings.dataset_preparation_seconds = perf_counter() - preparation_started
            run.timings.dataset_reused = dataset_reused
            self.runs.save(run)
            if cancel_event.is_set():
                self._mark_cancelled(project_id, run_id)
                return
            run.resolved_device = self.adapter.resolve_device(run.config.device)
            self.runs.save(run)

            result = self.adapter.train(
                dataset_yaml=dataset_yaml,
                config=run.config,
                output_dir=self.runs.work_dir(project_id, run_id),
                device=run.resolved_device,
                class_names=run.dataset.class_names if run.dataset else [],
                on_progress=lambda point: self._record_progress(project_id, run_id, point),
                on_stage=lambda stage, message: self._record_stage(project_id, run_id, stage, message),
                on_runtime=lambda runtime: self._record_runtime(project_id, run_id, runtime),
                is_cancelled=cancel_event.is_set,
            )
            self._merge_adapter_diagnostics(project_id, run_id, result.runtime, result.timings)
            if result.cancelled or cancel_event.is_set():
                self._save_cancelled_checkpoints(project_id, run_id, result)
                self._mark_cancelled(project_id, run_id)
                return

            run = self.get_run(project_id, run_id)
            best = result.best_checkpoint
            if best is None or not best.is_file():
                raise TrainingExecutionError("Ultralytics did not produce a best checkpoint")
            if result.validation is None:
                raise TrainingExecutionError("Ultralytics did not return final validation metrics")
            if cancel_event.is_set():
                self._save_cancelled_checkpoints(project_id, run_id, result)
                self._mark_cancelled(project_id, run_id)
                return

            self._record_stage(
                project_id,
                run_id,
                TrainingStage.SAVING_CHECKPOINT,
                "Preserving checkpoints and reproducibility details",
            )
            artifact_started = perf_counter()
            run = self.get_run(project_id, run_id)
            run.validation_metrics = result.validation.metrics
            run.per_class_metrics = list(result.validation.per_class)
            run.validation_image_count = readiness.preview.validation_images
            run.current_epoch = run.total_epochs
            run.artifacts.extend(self._register_checkpoints(run, result.best_checkpoint, result.last_checkpoint))
            self._write_run_summary(run)
            run = self.get_run(project_id, run_id)
            run.timings.artifact_save_seconds = perf_counter() - artifact_started
            run.status = TrainingStatus.COMPLETED
            run.stage = TrainingStage.COMPLETED
            run.stage_message = "Training and held-out validation completed"
            run.completed_at = datetime.now(UTC)
            if run.started_at is not None:
                run.timings.total_seconds = max(
                    0.0,
                    (run.completed_at - run.started_at).total_seconds(),
                )
            run.failure_reason = None
            self.runs.save(run)
        except (
            TrainingValidationError,
            TrainingDependencyUnavailableError,
            TrainingExecutionError,
        ) as error:
            self._mark_failed(project_id, run_id, str(error))
        except Exception as error:
            if cancel_event.is_set():
                self._mark_cancelled(project_id, run_id)
                return
            LOGGER.exception("Downstream training run %s failed", run_id)
            self._mark_failed(
                project_id,
                run_id,
                self._failure_message(error),
            )
        finally:
            LOGGER.debug(
                "Training worker for run %s released after %.3fs",
                run_id,
                perf_counter() - worker_started,
            )
            with self._lock:
                self._cancel_events.pop(run_id, None)

    def _create_snapshot(self, run: TrainingRun, readiness: TrainingReadiness) -> tuple[Path, bool]:
        options = ExportOptions(
            split=DatasetSplit.TRAIN_VAL,
            train_ratio=run.config.train_ratio,
            seed=run.config.seed,
        )
        project = self.projects.get(run.project_id)
        fingerprint_payload = {
            "schema_version": 1,
            "project_id": str(project.id),
            "project_updated_at": project.updated_at.isoformat(),
            "format": ExportFormat.YOLO.value,
            "split": options.split.value,
            "train_ratio": options.train_ratio,
            "seed": options.seed,
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        cache_root = self.projects.project_dir(run.project_id) / "training" / "dataset-cache"
        cache_dir = cache_root / fingerprint
        cache_archive = cache_dir / "dataset-snapshot.zip"
        cache_dataset = cache_dir / "dataset"
        manifest_path = cache_dir / "manifest.json"
        reused = False
        manifest: dict[str, object] | None = None
        if cache_archive.is_file() and (cache_dataset / "data.yaml").is_file():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    loaded.get("fingerprint") == fingerprint
                    and loaded.get("project_updated_at") == project.updated_at.isoformat()
                    and loaded.get("sha256") == self._sha256(cache_archive)
                    and isinstance(loaded.get("class_names"), list)
                ):
                    manifest = loaded
                    reused = True
            except (OSError, UnicodeError, ValueError, TypeError):
                manifest = None

        if manifest is None:
            if cache_dir.exists():
                self._remove_cache_directory(cache_root, cache_dir)
            temporary = cache_root / f".{fingerprint}.{uuid4().hex}.tmp"
            temporary.mkdir(parents=True, exist_ok=False)
            temporary_archive = temporary / "dataset-snapshot.zip"
            temporary_dataset = temporary / "dataset"
            temporary_dataset.mkdir()
            artifact = self.exports.export(run.project_id, ExportFormat.YOLO, options)
            try:
                shutil.copy2(artifact.path, temporary_archive)
                digest = self._sha256(temporary_archive)
                with ZipFile(temporary_archive) as archive:
                    for member in archive.infolist():
                        candidate = (temporary_dataset / member.filename).resolve()
                        if temporary_dataset.resolve() not in candidate.parents:
                            raise TrainingExecutionError("Dataset snapshot contains an unsafe path")
                    archive.extractall(temporary_dataset)
                if artifact.source_project_updated_at is None or not artifact.class_names:
                    raise TrainingExecutionError("YOLO snapshot is missing its project revision metadata")
                manifest = {
                    "schema_version": 1,
                    "fingerprint": fingerprint,
                    "sha256": digest,
                    "project_updated_at": artifact.source_project_updated_at.isoformat(),
                    "class_names": list(artifact.class_names),
                }
                (temporary / "manifest.json").write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                    newline="\n",
                )
                cache_root.mkdir(parents=True, exist_ok=True)
                os.replace(temporary, cache_dir)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            finally:
                artifact.cleanup()

            dataset_yaml = cache_dataset / "data.yaml"
            yaml_text = dataset_yaml.read_text(encoding="utf-8")
            absolute_path = json.dumps(str(cache_dataset.resolve()), ensure_ascii=False)
            dataset_yaml.write_text(
                yaml_text.replace("path: .", f"path: {absolute_path}", 1),
                encoding="utf-8",
                newline="\n",
            )
            self._prune_dataset_cache(cache_root, keep=fingerprint)

        assert manifest is not None
        artifacts_dir = self.runs.artifacts_dir(run.project_id, run.id)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        snapshot_archive = artifacts_dir / "dataset-snapshot.zip"
        shutil.copy2(cache_archive, snapshot_archive)
        digest = str(manifest["sha256"])
        class_names = [str(value) for value in manifest["class_names"]]
        run.dataset = DatasetSnapshot(
            sha256=digest,
            project_updated_at=project.updated_at,
            train_images=readiness.preview.train_images,
            validation_images=readiness.preview.validation_images,
            train_boxes=readiness.train_boxes,
            validation_boxes=readiness.validation_boxes,
            classes=readiness.preview.classes,
            class_names=class_names,
        )
        run.artifacts.append(
            TrainingArtifact(
                kind="dataset_snapshot",
                filename=snapshot_archive.name,
                size_bytes=snapshot_archive.stat().st_size,
            )
        )
        self.runs.save(run)
        return cache_dataset / "data.yaml", reused

    @staticmethod
    def _remove_cache_directory(cache_root: Path, candidate: Path) -> None:
        resolved_root = cache_root.resolve()
        resolved_candidate = candidate.resolve()
        if resolved_candidate.parent != resolved_root:
            raise TrainingExecutionError("Refused to remove a dataset cache outside its root")
        shutil.rmtree(resolved_candidate)

    def _prune_dataset_cache(self, cache_root: Path, keep: str) -> None:
        if not cache_root.is_dir():
            return
        directories = sorted(
            (
                path
                for path in cache_root.iterdir()
                if path.is_dir() and not path.name.startswith(".") and path.name != keep
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for stale in directories[2:]:
            self._remove_cache_directory(cache_root, stale)

    def _record_stage(
        self,
        project_id: UUID,
        run_id: UUID,
        stage: TrainingStage,
        message: str,
    ) -> None:
        run = self.get_run(project_id, run_id)
        if run.cancel_requested:
            return
        run.stage = stage
        run.stage_message = message
        if stage is TrainingStage.TRAINING:
            run.status = TrainingStatus.TRAINING
        elif stage is TrainingStage.VALIDATING:
            run.status = TrainingStatus.VALIDATING
        elif stage in {
            TrainingStage.LOADING_MODEL,
            TrainingStage.INITIALIZING_RUNTIME,
            TrainingStage.PREPARING_DATALOADER,
            TrainingStage.PREPARING_DATASET,
            TrainingStage.VALIDATING_DATASET,
        }:
            run.status = TrainingStatus.PREPARING
        self.runs.save(run)

    def _record_runtime(
        self,
        project_id: UUID,
        run_id: UUID,
        runtime: TrainingRuntimeDetails,
    ) -> None:
        run = self.get_run(project_id, run_id)
        run.runtime = runtime
        self.runs.save(run)

    def _merge_adapter_diagnostics(
        self,
        project_id: UUID,
        run_id: UUID,
        runtime: TrainingRuntimeDetails | None,
        timings: TrainingTimings | None,
    ) -> None:
        run = self.get_run(project_id, run_id)
        if runtime is not None:
            run.runtime = runtime
        if timings is not None:
            for field in (
                "model_load_seconds",
                "runtime_initialization_seconds",
                "dataloader_initialization_seconds",
                "startup_seconds",
                "validation_seconds",
            ):
                value = getattr(timings, field)
                if value is not None:
                    setattr(run.timings, field, value)
        self.runs.save(run)

    def _save_cancelled_checkpoints(
        self,
        project_id: UUID,
        run_id: UUID,
        result: object,
    ) -> None:
        best = getattr(result, "best_checkpoint", None)
        last = getattr(result, "last_checkpoint", None)
        if best is None and last is None:
            return
        run = self.get_run(project_id, run_id)
        existing = {artifact.kind for artifact in run.artifacts}
        records = self._register_checkpoints(run, best, last)
        run.artifacts.extend(record for record in records if record.kind not in existing)
        self.runs.save(run)

    def _register_checkpoints(
        self, run: TrainingRun, best_source: Path | None, last_source: Path | None
    ) -> list[TrainingArtifact]:
        artifacts_dir = self.runs.artifacts_dir(run.project_id, run.id)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        records: list[TrainingArtifact] = []
        for kind, source, filename in (
            ("best_checkpoint", best_source, "best.pt"),
            ("last_checkpoint", last_source, "last.pt"),
        ):
            if source is None or not source.is_file():
                continue
            destination = artifacts_dir / filename
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            records.append(TrainingArtifact(kind=kind, filename=filename, size_bytes=destination.stat().st_size))
        return records

    def _write_run_summary(self, run: TrainingRun) -> None:
        artifacts_dir = self.runs.artifacts_dir(run.project_id, run.id)
        destination = artifacts_dir / "training-result.json"
        payload = {
            "schema_version": 1,
            "run_id": str(run.id),
            "project_id": str(run.project_id),
            "configuration": run.config.model_dump(mode="json"),
            "resolved_device": run.resolved_device,
            "dataset": run.dataset.model_dump(mode="json") if run.dataset else None,
            "runtime": run.runtime.model_dump(mode="json") if run.runtime else None,
            "timings": run.timings.model_dump(mode="json"),
            "progress_history": [item.model_dump(mode="json") for item in run.progress_history],
            "validation_metrics": (run.validation_metrics.model_dump(mode="json") if run.validation_metrics else None),
            "per_class_metrics": [item.model_dump(mode="json") for item in run.per_class_metrics],
        }
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
        os.replace(temporary, destination)
        run.artifacts.append(
            TrainingArtifact(
                kind="configuration",
                filename=destination.name,
                size_bytes=destination.stat().st_size,
            )
        )
        self.runs.save(run)

    def _predict(
        self, project_id: UUID, run_id: UUID, image_path: Path, confidence: float
    ) -> tuple[DetectorPrediction, ...]:
        run = self.get_run(project_id, run_id)
        if run.status is not TrainingStatus.COMPLETED:
            raise TrainingPredictionError("Predictions require a completed training run")
        best, _ = self.artifact(project_id, run_id, "best_checkpoint")
        if run.resolved_device is None:
            raise TrainingPredictionError("Training run has no recorded device")
        try:
            return self.adapter.predict(
                checkpoint=best,
                image_path=image_path,
                image_size=run.config.image_size,
                device=run.resolved_device,
                confidence=confidence,
            )
        except TrainingDependencyUnavailableError:
            raise
        except Exception as error:
            LOGGER.exception("Trained-model prediction failed for run %s", run_id)
            raise TrainingPredictionError("Trained-model prediction failed") from error

    def _record_progress(
        self,
        project_id: UUID,
        run_id: UUID,
        point: TrainingProgressPoint,
    ) -> None:
        run = self.get_run(project_id, run_id)
        if run.cancel_requested or run.status not in {
            TrainingStatus.TRAINING,
            TrainingStatus.VALIDATING,
        }:
            return
        run.current_epoch = max(run.current_epoch, min(point.epoch, run.total_epochs))
        run.progress_history = [item for item in run.progress_history if item.epoch != point.epoch]
        run.progress_history.append(point)
        run.progress_history = sorted(run.progress_history, key=lambda item: item.epoch)[-300:]
        if point.metrics is not None:
            run.latest_metrics = point.metrics
        self.runs.save(run)

    def _mark_cancelled(self, project_id: UUID, run_id: UUID) -> None:
        run = self.get_run(project_id, run_id)
        run.status = TrainingStatus.CANCELLED
        run.stage = TrainingStage.CANCELLED
        run.stage_message = "Training stopped and GPU resources were released"
        run.cancel_requested = True
        run.completed_at = datetime.now(UTC)
        if run.cancel_requested_at is not None:
            run.timings.cancellation_seconds = max(
                0.0,
                (run.completed_at - run.cancel_requested_at).total_seconds(),
            )
        if run.started_at is not None:
            run.timings.total_seconds = max(
                0.0,
                (run.completed_at - run.started_at).total_seconds(),
            )
        run.failure_reason = None
        self.runs.save(run)

    def _mark_failed(self, project_id: UUID, run_id: UUID, reason: str) -> None:
        try:
            run = self.get_run(project_id, run_id)
        except Exception:
            LOGGER.exception("Could not persist failed state for training run %s", run_id)
            return
        run.status = TrainingStatus.FAILED
        run.stage = TrainingStage.FAILED
        run.stage_message = "Training stopped because the run failed"
        run.completed_at = datetime.now(UTC)
        if run.started_at is not None:
            run.timings.total_seconds = max(
                0.0,
                (run.completed_at - run.started_at).total_seconds(),
            )
        run.failure_reason = reason[:500]
        self.runs.save(run)

    def _recover_interrupted_runs(self) -> None:
        for run in self.runs.list_all():
            if run.status in ACTIVE_TRAINING_STATUSES:
                run.status = TrainingStatus.FAILED
                run.stage = TrainingStage.FAILED
                run.stage_message = "Training was interrupted by an application restart"
                run.completed_at = datetime.now(UTC)
                run.failure_reason = "Training was interrupted by an application restart."
                self.runs.save(run)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _failure_message(error: Exception) -> str:
        message = str(error).casefold()
        if "out of memory" in message and ("cuda" in message or "gpu" in message):
            return "Training ran out of GPU memory. Reduce the batch size or image size, then start a new CUDA run."
        if "no space left" in message or "disk full" in message:
            return "Training ran out of local storage space. Free space and start a new run."
        return "Training failed unexpectedly. Review the backend log for technical details."
