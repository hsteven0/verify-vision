from __future__ import annotations

import math
import time
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from threading import Event
from uuid import uuid4
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from app.domain.models import (
    Annotation,
    AnnotationSource,
    BoundingBox,
    ImageReviewState,
    Label,
    Project,
    ProjectImage,
    VerificationState,
)
from app.main import create_app
from app.persistence.repository import ProjectRepository
from app.training.adapter import (
    AdapterTrainingResult,
    AdapterValidationResult,
    TrainingDependencyUnavailableError,
    TrainingExecutionError,
    UltralyticsTrainingAdapter,
    cuda_training_supported,
    metrics_from_mapping,
    parse_validation_metrics,
)
from app.training.models import (
    ClassDetectionMetrics,
    DetectionMetrics,
    DetectorPrediction,
    ModelChoice,
    PredictionPreview,
    TrainingArtifact,
    TrainingAvailability,
    TrainingConfig,
    TrainingProgressPoint,
    TrainingRun,
    TrainingRuntimeDetails,
    TrainingStage,
    TrainingStatus,
    TrainingTimings,
)
from app.training.repository import TrainingRunRepository
from app.training.service import TrainingPredictionError, TrainingService


def png_bytes(color: tuple[int, int, int] = (30, 50, 70)) -> bytes:
    output = BytesIO()
    Image.new("RGB", (100, 100), color=color).save(output, format="PNG")
    return output.getvalue()


def verified_project(repository: ProjectRepository) -> Project:
    person = Label(name="Person")
    vehicle = Label(name="Vehicle")
    accepted_box = BoundingBox(x=10, y=20, width=30, height=40)
    project = Project(
        name="Training fixture",
        labels=[person, vehicle],
        images=[
            ProjectImage(
                filename="one.png",
                storage_name="one.png",
                media_type="image/png",
                width=100,
                height=100,
                review_state=ImageReviewState.COMPLETE,
                annotations=[
                    Annotation(
                        prediction_id=uuid4(),
                        label_id=person.id,
                        original_ai_label_id=person.id,
                        source=AnnotationSource.AI,
                        verification_state=VerificationState.ACCEPTED,
                        original_ai_box=accepted_box,
                        final_box=accepted_box,
                        provider="locateanything",
                        model="nvidia/LocateAnything-3B",
                        prompt="person",
                    ),
                    Annotation(
                        prediction_id=uuid4(),
                        label_id=vehicle.id,
                        original_ai_label_id=vehicle.id,
                        source=AnnotationSource.AI,
                        verification_state=VerificationState.ADJUSTED,
                        original_ai_box=BoundingBox(x=50, y=50, width=30, height=30),
                        final_box=BoundingBox(x=40, y=40, width=20, height=20),
                        provider="locateanything",
                        model="nvidia/LocateAnything-3B",
                        prompt="vehicle",
                    ),
                    Annotation(
                        prediction_id=uuid4(),
                        label_id=vehicle.id,
                        original_ai_label_id=vehicle.id,
                        source=AnnotationSource.AI,
                        verification_state=VerificationState.REJECTED,
                        original_ai_box=BoundingBox(x=1, y=1, width=10, height=10),
                        final_box=None,
                        provider="locateanything",
                        model="nvidia/LocateAnything-3B",
                        prompt="vehicle",
                    ),
                ],
            ),
            ProjectImage(
                filename="two.png",
                storage_name="two.png",
                media_type="image/png",
                width=100,
                height=100,
                review_state=ImageReviewState.COMPLETE,
                annotations=[
                    Annotation(
                        label_id=person.id,
                        source=AnnotationSource.HUMAN,
                        verification_state=VerificationState.HUMAN_ADDED,
                        final_box=BoundingBox(x=25, y=25, width=25, height=25),
                        review_prompt="person",
                    )
                ],
            ),
        ],
    )
    repository.save(project)
    repository.image_path(project.id, "one.png").write_bytes(png_bytes())
    repository.image_path(project.id, "two.png").write_bytes(png_bytes((60, 40, 20)))
    return project


class FakeTrainingAdapter:
    def __init__(self, mode: str = "success") -> None:
        self.mode = mode
        self.started = Event()
        self.snapshot_labels: list[str] = []
        self.predicted_paths: list[Path] = []

    def availability(self) -> TrainingAvailability:
        return TrainingAvailability(
            available=True,
            package_version="test",
            supported_models=[ModelChoice(checkpoint="yolo26n.pt", label="Nano", size="Test checkpoint")],
            device_options=["auto", "cuda:0"],
            detected_device="cuda:0",
            gpu_name="Synthetic NVIDIA GPU",
            gpu_memory_gb=8.0,
            torch_version="test",
            cuda_version="test",
        )

    def resolve_device(self, requested: str) -> str:
        if requested == "cpu":
            raise TrainingDependencyUnavailableError("CPU training is unsupported")
        return "cuda:0"

    def train(
        self,
        *,
        dataset_yaml: Path,
        config: TrainingConfig,
        output_dir: Path,
        device: str,
        class_names: list[str],
        on_progress: object,
        on_stage: object,
        on_runtime: object,
        is_cancelled: object,
    ) -> AdapterTrainingResult:
        del device
        on_stage(TrainingStage.TRAINING, "Training epoch 1")  # type: ignore[operator]
        on_runtime(  # type: ignore[operator]
            TrainingRuntimeDetails(
                gpu_name="Synthetic NVIDIA GPU",
                gpu_memory_gb=8.0,
                torch_version="test",
                cuda_version="test",
                amp_enabled=True,
                workers=0,
                batch_size=config.batch_size,
            )
        )
        self.started.set()
        snapshot = dataset_yaml.parent / "labels"
        self.snapshot_labels = [path.read_text(encoding="utf-8") for path in snapshot.rglob("*.txt")]
        if self.mode == "failure":
            raise TrainingExecutionError("Synthetic trainer failure")
        if self.mode == "oom":
            raise RuntimeError("CUDA out of memory while allocating tensor")
        if self.mode == "blocking":
            while not is_cancelled():  # type: ignore[operator]
                time.sleep(0.005)
            return AdapterTrainingResult(None, None, cancelled=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        best = output_dir / "adapter-best.pt"
        last = output_dir / "adapter-last.pt"
        best.write_bytes(b"best-checkpoint")
        last.write_bytes(b"last-checkpoint")
        on_progress(  # type: ignore[operator]
            TrainingProgressPoint(
                epoch=1,
                duration_seconds=0.1,
                training_loss=1.2,
                metrics=DetectionMetrics(precision=0.5, recall=0.4),
            )
        )
        on_progress(  # type: ignore[operator]
            TrainingProgressPoint(
                epoch=config.epochs,
                duration_seconds=0.1,
                training_loss=0.8,
                metrics=DetectionMetrics(precision=0.7, recall=0.6),
            )
        )
        if self.mode == "validation_failure":
            raise TrainingExecutionError("Synthetic validation failure")
        validation = AdapterValidationResult(
            metrics=DetectionMetrics(precision=0.8, recall=0.75, map50=0.7, map50_95=0.55),
            per_class=(
                ClassDetectionMetrics(
                    class_id=0,
                    class_name=class_names[0],
                    precision=0.9,
                    recall=0.8,
                    map50=0.75,
                    map50_95=0.6,
                ),
            ),
        )
        return AdapterTrainingResult(
            best,
            last,
            validation=validation,
            timings=TrainingTimings(
                model_load_seconds=0.2,
                runtime_initialization_seconds=0.1,
                dataloader_initialization_seconds=0.3,
                startup_seconds=0.6,
                validation_seconds=0.4,
            ),
        )

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
        del dataset_yaml, config, output_dir, device
        assert checkpoint.read_bytes() == b"best-checkpoint"
        if self.mode == "validation_failure":
            raise TrainingExecutionError("Synthetic validation failure")
        return AdapterValidationResult(
            metrics=DetectionMetrics(precision=0.8, recall=0.75, map50=0.7, map50_95=0.55),
            per_class=(
                ClassDetectionMetrics(
                    class_id=0,
                    class_name=class_names[0],
                    precision=0.9,
                    recall=0.8,
                    map50=0.75,
                    map50_95=0.6,
                ),
            ),
        )

    def predict(
        self,
        *,
        checkpoint: Path,
        image_path: Path,
        image_size: int,
        device: str,
        confidence: float,
    ) -> tuple[DetectorPrediction, ...]:
        del image_size, device
        assert checkpoint.read_bytes() == b"best-checkpoint"
        self.predicted_paths.append(image_path)
        return (
            DetectorPrediction(
                class_id=0,
                class_name="Person",
                confidence=max(confidence, 0.8),
                box=BoundingBox(x=10, y=12, width=30, height=35),
            ),
        )


def wait_for_terminal(service: TrainingService, project_id: object, run_id: object) -> TrainingRun:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        run = service.get_run(project_id, run_id)  # type: ignore[arg-type]
        if run.status not in {
            TrainingStatus.QUEUED,
            TrainingStatus.PREPARING,
            TrainingStatus.TRAINING,
            TrainingStatus.VALIDATING,
            TrainingStatus.CANCELLING,
        }:
            return run
        time.sleep(0.01)
    raise AssertionError("training run did not reach a terminal state")


def test_training_config_defaults_and_validation() -> None:
    config = TrainingConfig()
    assert config.checkpoint == "yolo26n.pt"
    assert config.epochs == 20
    assert config.train_ratio == 0.8
    assert config.device == "auto"

    with pytest.raises(ValidationError):
        TrainingConfig(epochs=0)
    with pytest.raises(ValidationError):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValidationError):
        TrainingConfig(image_size=641)
    with pytest.raises(ValidationError):
        TrainingConfig(checkpoint="arbitrary.pt")


def test_cuda_training_policy_and_device_override_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert cuda_training_supported("Windows", True)
    assert cuda_training_supported("Linux", True)
    assert not cuda_training_supported("Darwin", True)
    assert not cuda_training_supported("Windows", False)

    adapter = UltralyticsTrainingAdapter()
    available = TrainingAvailability(
        available=True,
        supported_models=adapter.model_choices,
        device_options=["auto", "cuda:0"],
        detected_device="cuda:0",
    )
    monkeypatch.setattr(adapter, "availability", lambda: available)
    monkeypatch.setattr(adapter, "_detected_device", lambda: "cuda:0")
    assert adapter.resolve_device("auto") == "cuda:0"
    assert adapter.resolve_device("cuda:0") == "cuda:0"
    with pytest.raises(TrainingDependencyUnavailableError, match="CPU training is not supported"):
        adapter.resolve_device("cpu")


def test_completed_run_reuses_verified_yolo_snapshot_and_persists(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "projects")
    project = verified_project(repository)
    adapter = FakeTrainingAdapter()
    service = TrainingService(repository, adapter=adapter)
    project_before = repository.get(project.id).model_dump_json()

    created = service.create_run(project.id, TrainingConfig(epochs=2, seed=19))
    completed = wait_for_terminal(service, project.id, created.id)

    assert completed.status is TrainingStatus.COMPLETED
    assert completed.current_epoch == 2
    assert completed.resolved_device == "cuda:0"
    assert completed.dataset is not None
    assert len(completed.dataset.sha256) == 64
    assert completed.dataset.train_images == 1
    assert completed.dataset.validation_images == 1
    assert completed.dataset.train_boxes + completed.dataset.validation_boxes == 3
    assert completed.validation_metrics == DetectionMetrics(precision=0.8, recall=0.75, map50=0.7, map50_95=0.55)
    assert completed.per_class_metrics[0].class_name == "Person"
    assert [point.epoch for point in completed.progress_history] == [1, 2]
    assert completed.progress_history[-1].training_loss == 0.8
    assert completed.runtime is not None
    assert completed.runtime.amp_enabled
    assert completed.timings.startup_seconds == 0.6
    assert {item.kind for item in completed.artifacts} == {
        "best_checkpoint",
        "last_checkpoint",
        "configuration",
        "dataset_snapshot",
    }
    assert repository.get(project.id).model_dump_json() == project_before

    rows = [row for content in adapter.snapshot_labels for row in content.splitlines()]
    assert len(rows) == 3
    assert "1 0.500000 0.500000 0.200000 0.200000" in rows
    assert not any("0.060000 0.060000 0.100000 0.100000" in row for row in rows)

    snapshot, _ = service.artifact(project.id, completed.id, "dataset_snapshot")
    with ZipFile(snapshot) as archive:
        assert "data.yaml" in archive.namelist()
    service.shutdown()

    persisted = TrainingRunRepository(repository).get(project.id, completed.id)
    assert persisted.status is TrainingStatus.COMPLETED
    assert persisted.validation_metrics == completed.validation_metrics


def test_dataset_cache_reuses_and_invalidates(
    tmp_path: Path,
) -> None:
    repository = ProjectRepository(tmp_path / "projects")
    project = verified_project(repository)
    service = TrainingService(repository, adapter=FakeTrainingAdapter())
    config = TrainingConfig(epochs=1, seed=41)

    first = wait_for_terminal(
        service,
        project.id,
        service.create_run(project.id, config).id,
    )
    second = wait_for_terminal(
        service,
        project.id,
        service.create_run(project.id, config).id,
    )

    assert first.status is TrainingStatus.COMPLETED
    assert second.status is TrainingStatus.COMPLETED
    assert not first.timings.dataset_reused
    assert second.timings.dataset_reused
    assert second.dataset is not None and first.dataset is not None
    assert second.dataset.sha256 == first.dataset.sha256

    revised = repository.get(project.id)
    revised.updated_at = datetime.now(UTC) + timedelta(seconds=1)
    repository.save(revised)
    third = wait_for_terminal(
        service,
        project.id,
        service.create_run(project.id, config).id,
    )

    assert third.status is TrainingStatus.COMPLETED
    assert not third.timings.dataset_reused
    service.shutdown()


def test_invalid_dataset_blocks_training_before_adapter_starts(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "projects")
    project = Project(name="Empty", labels=[Label(name="Object")])
    repository.save(project)
    adapter = FakeTrainingAdapter()
    service = TrainingService(repository, adapter=adapter)

    readiness = service.readiness(project.id, TrainingConfig())

    assert not readiness.ready
    assert "no eligible training annotations" in " ".join(readiness.blockers)
    with pytest.raises(Exception, match="no finalized images|no eligible"):
        service.create_run(project.id, TrainingConfig())
    assert not adapter.started.is_set()
    service.shutdown()


def test_failure_and_cancellation_preserve_project_data(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "projects")
    project = verified_project(repository)
    original = repository.get(project.id).model_dump_json()

    failing_service = TrainingService(repository, adapter=FakeTrainingAdapter("failure"))
    failed_id = failing_service.create_run(project.id, TrainingConfig()).id
    failed = wait_for_terminal(failing_service, project.id, failed_id)
    assert failed.status is TrainingStatus.FAILED
    assert failed.failure_reason == "Synthetic trainer failure"
    assert not any(item.kind == "best_checkpoint" for item in failed.artifacts)
    failing_service.shutdown()

    oom_service = TrainingService(repository, adapter=FakeTrainingAdapter("oom"))
    oom_id = oom_service.create_run(project.id, TrainingConfig()).id
    oom_failed = wait_for_terminal(oom_service, project.id, oom_id)
    assert oom_failed.status is TrainingStatus.FAILED
    assert oom_failed.failure_reason is not None
    assert "Reduce the batch size" in oom_failed.failure_reason
    assert "tensor" not in oom_failed.failure_reason
    oom_service.shutdown()

    validation_service = TrainingService(repository, adapter=FakeTrainingAdapter("validation_failure"))
    validation_failed_id = validation_service.create_run(project.id, TrainingConfig()).id
    validation_failed = wait_for_terminal(validation_service, project.id, validation_failed_id)
    assert validation_failed.status is TrainingStatus.FAILED
    assert validation_failed.failure_reason == "Synthetic validation failure"
    assert not any(item.kind in {"best_checkpoint", "last_checkpoint"} for item in validation_failed.artifacts)
    validation_service.shutdown()

    blocking = FakeTrainingAdapter("blocking")
    cancellation_service = TrainingService(repository, adapter=blocking)
    cancelled_id = cancellation_service.create_run(project.id, TrainingConfig()).id
    assert blocking.started.wait(2)
    assert cancellation_service.get_run(project.id, cancelled_id).status is TrainingStatus.TRAINING
    cancelling = cancellation_service.cancel(project.id, cancelled_id)
    assert cancelling.status is TrainingStatus.CANCELLING
    assert cancelling.stage is TrainingStage.CANCELLING
    cancelled = wait_for_terminal(cancellation_service, project.id, cancelled_id)
    assert cancelled.status is TrainingStatus.CANCELLED
    assert cancelled.cancel_requested
    assert cancelled.timings.cancellation_seconds is not None
    assert repository.get(project.id).model_dump_json() == original
    cancellation_service.shutdown()


def test_interrupted_run_is_marked_failed_on_service_restart(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "projects")
    project = verified_project(repository)
    run = TrainingRun(
        project_id=project.id,
        config=TrainingConfig(),
        total_epochs=20,
        status=TrainingStatus.TRAINING,
    )
    TrainingRunRepository(repository).save(run)

    service = TrainingService(repository, adapter=FakeTrainingAdapter())
    recovered = service.get_run(project.id, run.id)

    assert recovered.status is TrainingStatus.FAILED
    assert recovered.failure_reason == "Training was interrupted by an application restart."
    service.shutdown()


def test_artifact_resolution_cannot_escape_the_run_directory(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "projects")
    project = verified_project(repository)
    run = TrainingRun(
        project_id=project.id,
        config=TrainingConfig(),
        total_epochs=20,
        status=TrainingStatus.COMPLETED,
        artifacts=[
            TrainingArtifact(
                kind="best_checkpoint",
                filename="../run.json",
                size_bytes=1,
            )
        ],
    )
    TrainingRunRepository(repository).save(run)
    service = TrainingService(repository, adapter=FakeTrainingAdapter())

    with pytest.raises(TrainingPredictionError, match="missing"):
        service.artifact(project.id, run.id, "best_checkpoint")
    service.shutdown()


def test_metric_parsing_drops_nan_and_keeps_classes() -> None:
    missing = parse_validation_metrics(object(), ["A"])
    assert missing.metrics == DetectionMetrics()
    assert missing.per_class == ()

    current = metrics_from_mapping(
        {
            "metrics/precision(B)": 0.7,
            "metrics/recall(B)": float("nan"),
            "metrics/mAP50(B)": math.inf,
            "metrics/mAP50-95(B)": 0.4,
        }
    )
    assert current == DetectionMetrics(precision=0.7, map50_95=0.4)

    class BoxMetrics:
        mp = 0.8
        mr = 0.6
        map50 = 0.7
        map = 0.5
        ap_class_index = [1]

        @staticmethod
        def class_result(index: int) -> tuple[float, float, float, float]:
            assert index == 0
            return (0.75, 0.55, 0.65, 0.45)

    result = parse_validation_metrics(type("Raw", (), {"box": BoxMetrics()})(), ["A", "B"])
    assert result.metrics.map50_95 == 0.5
    assert result.per_class[0].class_name == "B"
    assert result.per_class[0].precision == 0.75


def test_training_api_lifecycle_artifacts_and_prediction_preview(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "projects")
    project = verified_project(repository)
    adapter = FakeTrainingAdapter()
    app = create_app(
        data_dir=repository.root,
        frontend_dist=tmp_path / "missing",
        training_adapter=adapter,
    )

    with TestClient(app) as client:
        availability = client.get(f"/api/projects/{project.id}/training/availability")
        assert availability.status_code == 200
        assert availability.json()["available"] is True

        ready = client.post(
            f"/api/projects/{project.id}/training/readiness",
            json=TrainingConfig(epochs=2).model_dump(mode="json"),
        )
        assert ready.status_code == 200
        assert ready.json()["ready"] is True

        started = client.post(
            f"/api/projects/{project.id}/training/runs",
            json=TrainingConfig(epochs=2).model_dump(mode="json"),
        )
        assert started.status_code == 202
        run_id = started.json()["id"]

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            response = client.get(f"/api/projects/{project.id}/training/runs/{run_id}")
            if response.json()["status"] == "completed":
                break
            time.sleep(0.01)
        run_payload = response.json()
        assert run_payload["status"] == "completed"
        assert run_payload["validation_metrics"] == {
            "precision": 0.8,
            "recall": 0.75,
            "map50": 0.7,
            "map50_95": 0.55,
        }
        assert "NaN" not in response.text and "Infinity" not in response.text

        best = client.get(f"/api/projects/{project.id}/training/runs/{run_id}/artifacts/best_checkpoint")
        assert best.status_code == 200
        assert best.content == b"best-checkpoint"
        assert "verifyvision-verified-dataset-best.pt" in best.headers["content-disposition"]

        image_id = project.images[0].id
        prediction = client.post(f"/api/projects/{project.id}/training/runs/{run_id}/predict/images/{image_id}")
        assert prediction.status_code == 200
        assert prediction.json()["source"] == "project_image"
        assert prediction.json()["detections"][0]["class_name"] == "Person"

        uploaded = client.post(
            f"/api/projects/{project.id}/training/runs/{run_id}/predict/upload",
            files={"file": ("held-out.png", png_bytes(), "image/png")},
        )
        assert uploaded.status_code == 200
        preview: PredictionPreview = PredictionPreview.model_validate(uploaded.json())
        image = client.get(f"/api/projects/{project.id}/training/runs/{run_id}/predictions/{preview.id}/image")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/png"

        missing = client.get(f"/api/projects/{project.id}/training/runs/{uuid4()}")
        assert missing.status_code == 404
