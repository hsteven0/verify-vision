from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.models import Annotation, Project, ProjectImage

if TYPE_CHECKING:
    from app.persistence.repository import ProjectRepository


class ExportFormat(StrEnum):
    YOLO = "yolo"
    COCO = "coco"
    PASCAL_VOC = "pascal_voc"
    EVALUATION_CSV = "evaluation_csv"


class DatasetSplit(StrEnum):
    NONE = "none"
    TRAIN_VAL = "train_val"


class ExportOptions(BaseModel):
    split: DatasetSplit = DatasetSplit.NONE
    train_ratio: float = Field(default=0.8, gt=0, lt=1)
    seed: int = Field(default=1337, ge=0, le=2_147_483_647)


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ValidationFinding(BaseModel):
    severity: ValidationSeverity
    code: str
    message: str
    image_id: UUID | None = None
    annotation_id: UUID | None = None


class ExportPreview(BaseModel):
    total_images: int = Field(ge=0)
    exportable_images: int = Field(ge=0)
    train_images: int = Field(ge=0)
    validation_images: int = Field(ge=0)
    training_boxes: int = Field(ge=0)
    classes: int = Field(ge=0)
    accepted: int = Field(ge=0)
    adjusted: int = Field(ge=0)
    human_added: int = Field(ge=0)
    manual: int = Field(ge=0)
    excluded_rejected: int = Field(ge=0)
    excluded_unresolved: int = Field(ge=0)
    excluded_needs_review: int = Field(ge=0)
    blocking_error_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    findings: list[ValidationFinding]


@dataclass(frozen=True, slots=True)
class TrainingImage:
    image: ProjectImage
    annotations: tuple[Annotation, ...]
    portable_name: str
    split: Literal["all", "train", "val"] = "all"


@dataclass(frozen=True, slots=True)
class ExportContext:
    project: Project
    repository: ProjectRepository
    options: ExportOptions
    training_images: tuple[TrainingImage, ...]


@dataclass(frozen=True, slots=True)
class ExportArtifact:
    path: Path
    directory: Path
    filename: str
    media_type: str
    source_project_updated_at: datetime | None = None
    class_names: tuple[str, ...] = ()

    def cleanup(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)
