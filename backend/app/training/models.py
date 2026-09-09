from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, StringConstraints, field_validator

from app.domain.models import BoundingBox
from app.exporting.models import ExportPreview

SUPPORTED_CHECKPOINTS = ("yolo26n.pt", "yolo26s.pt", "yolo26m.pt")


class TrainingStatus(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATING = "validating"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE_TRAINING_STATUSES = {
    TrainingStatus.QUEUED,
    TrainingStatus.PREPARING,
    TrainingStatus.TRAINING,
    TrainingStatus.VALIDATING,
    TrainingStatus.CANCELLING,
}


class TrainingStage(StrEnum):
    QUEUED = "queued"
    VALIDATING_DATASET = "validating_dataset"
    PREPARING_DATASET = "preparing_dataset"
    LOADING_MODEL = "loading_model"
    INITIALIZING_RUNTIME = "initializing_runtime"
    PREPARING_DATALOADER = "preparing_dataloader"
    TRAINING = "training"
    VALIDATING = "validating"
    SAVING_CHECKPOINT = "saving_checkpoint"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ModelChoice(BaseModel):
    checkpoint: str
    label: str
    size: str


class TrainingAvailability(BaseModel):
    available: bool
    reason: str | None = None
    package_version: str | None = None
    supported_models: list[ModelChoice]
    device_options: list[str]
    detected_device: str
    gpu_name: str | None = None
    gpu_memory_gb: float | None = Field(default=None, gt=0)
    torch_version: str | None = None
    cuda_version: str | None = None


class TrainingConfig(BaseModel):
    checkpoint: str = "yolo26n.pt"
    epochs: int = Field(default=20, ge=1, le=300)
    image_size: int = Field(default=640, ge=128, le=2048)
    batch_size: int = Field(default=8, ge=1, le=128)
    train_ratio: float = Field(default=0.8, ge=0.5, le=0.95)
    seed: int = Field(default=1337, ge=0, le=2_147_483_647)
    device: Literal["auto", "cpu", "cuda:0"] = "auto"
    run_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)] = "verified-dataset"

    @field_validator("checkpoint")
    @classmethod
    def supported_checkpoint(cls, value: str) -> str:
        if value not in SUPPORTED_CHECKPOINTS:
            raise ValueError("Unsupported YOLO checkpoint")
        return value

    @field_validator("image_size")
    @classmethod
    def image_size_multiple(cls, value: int) -> int:
        if value % 32:
            raise ValueError("Image size must be a multiple of 32")
        return value


class TrainingReadiness(BaseModel):
    ready: bool
    preview: ExportPreview
    train_boxes: int = Field(ge=0)
    validation_boxes: int = Field(ge=0)
    blockers: list[str]
    advisories: list[str]


class DatasetSnapshot(BaseModel):
    sha256: str
    project_updated_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    train_images: int = Field(ge=0)
    validation_images: int = Field(ge=0)
    train_boxes: int = Field(ge=0)
    validation_boxes: int = Field(ge=0)
    classes: int = Field(ge=0)
    class_names: list[str]


class DetectionMetrics(BaseModel):
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    map50: float | None = Field(default=None, ge=0, le=1)
    map50_95: float | None = Field(default=None, ge=0, le=1)


class TrainingProgressPoint(BaseModel):
    epoch: int = Field(ge=1)
    duration_seconds: float | None = Field(default=None, ge=0)
    training_loss: float | None = Field(default=None, ge=0)
    metrics: DetectionMetrics | None = None


class TrainingRuntimeDetails(BaseModel):
    gpu_name: str | None = None
    gpu_memory_gb: float | None = Field(default=None, gt=0)
    torch_version: str | None = None
    cuda_version: str | None = None
    amp_enabled: bool | None = None
    workers: int | None = Field(default=None, ge=0)
    batch_size: int | None = Field(default=None, ge=1)
    cache_mode: str = "dataset snapshot"
    checkpoint_source: str | None = None


class TrainingTimings(BaseModel):
    request_to_worker_seconds: float | None = Field(default=None, ge=0)
    dataset_validation_seconds: float | None = Field(default=None, ge=0)
    dataset_preparation_seconds: float | None = Field(default=None, ge=0)
    dataset_reused: bool = False
    model_load_seconds: float | None = Field(default=None, ge=0)
    runtime_initialization_seconds: float | None = Field(default=None, ge=0)
    dataloader_initialization_seconds: float | None = Field(default=None, ge=0)
    startup_seconds: float | None = Field(default=None, ge=0)
    validation_seconds: float | None = Field(default=None, ge=0)
    artifact_save_seconds: float | None = Field(default=None, ge=0)
    cancellation_seconds: float | None = Field(default=None, ge=0)
    total_seconds: float | None = Field(default=None, ge=0)


class ClassDetectionMetrics(DetectionMetrics):
    class_id: int = Field(ge=0)
    class_name: str


class TrainingArtifact(BaseModel):
    kind: Literal["best_checkpoint", "last_checkpoint", "configuration", "dataset_snapshot"]
    filename: str
    size_bytes: int = Field(ge=0)


class TrainingRun(BaseModel):
    schema_version: Literal[1] = 1
    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    status: TrainingStatus = TrainingStatus.QUEUED
    stage: TrainingStage = TrainingStage.QUEUED
    stage_message: str = "Waiting for the local training worker"
    config: TrainingConfig
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    current_epoch: int = Field(default=0, ge=0)
    total_epochs: int = Field(ge=1)
    resolved_device: str | None = None
    dataset: DatasetSnapshot | None = None
    runtime: TrainingRuntimeDetails | None = None
    timings: TrainingTimings = Field(default_factory=TrainingTimings)
    progress_history: list[TrainingProgressPoint] = Field(default_factory=list)
    latest_metrics: DetectionMetrics | None = None
    validation_metrics: DetectionMetrics | None = None
    per_class_metrics: list[ClassDetectionMetrics] = Field(default_factory=list)
    validation_image_count: int = Field(default=0, ge=0)
    artifacts: list[TrainingArtifact] = Field(default_factory=list)
    failure_reason: str | None = None
    cancel_requested: bool = False


class DetectorPrediction(BaseModel):
    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0, le=1)
    box: BoundingBox


class PredictionPreview(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source: Literal["project_image", "uploaded_image"]
    filename: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    image_id: UUID | None = None
    detections: list[DetectorPrediction]
