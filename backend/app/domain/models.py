from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class AnnotationSource(StrEnum):
    HUMAN = "human"
    AI = "ai"


class VerificationState(StrEnum):
    MANUAL = "manual"
    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    ADJUSTED = "adjusted"
    REJECTED = "rejected"
    HUMAN_ADDED = "human_added"


class ImageReviewState(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    NEEDS_REVIEW = "needs_review"


class BoundingBox(BaseModel):
    """Axis-aligned bounding box expressed in original image pixels."""

    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)

    @field_validator("x", "y", "width", "height")
    @classmethod
    def require_finite_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Bounding-box values must be finite")
        return value

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    def fits_within(self, image_width: int, image_height: int) -> bool:
        return self.right <= image_width and self.bottom <= image_height

    def intersection_over_union(self, other: BoundingBox) -> float:
        intersection_width = max(0.0, min(self.right, other.right) - max(self.x, other.x))
        intersection_height = max(0.0, min(self.bottom, other.bottom) - max(self.y, other.y))
        intersection = intersection_width * intersection_height
        if intersection == 0:
            return 0.0
        return intersection / (self.area + other.area - intersection)


class Label(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=80)
    color: str = "#32d6a0"

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Label name cannot be blank")
        return normalized

    @field_validator("color")
    @classmethod
    def require_hex_color(cls, value: str) -> str:
        if len(value) != 7 or value[0] != "#":
            raise ValueError("Color must use #RRGGBB format")
        try:
            int(value[1:], 16)
        except ValueError as error:
            raise ValueError("Color must use #RRGGBB format") from error
        return value.lower()


class Annotation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    prediction_id: UUID | None = None
    label_id: UUID
    original_ai_label_id: UUID | None = None
    source: AnnotationSource
    verification_state: VerificationState
    original_ai_box: BoundingBox | None = None
    final_box: BoundingBox | None = None
    provider: str | None = Field(default=None, max_length=120)
    model: str | None = Field(default=None, max_length=120)
    prompt: str | None = Field(default=None, max_length=500)
    confidence: float | None = Field(default=None, ge=0, le=1)
    review_prompt: str | None = Field(default=None, max_length=500)
    note: str | None = Field(default=None, max_length=1000)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def enforce_provenance_rules(self) -> Annotation:
        human_states = {VerificationState.MANUAL, VerificationState.HUMAN_ADDED}
        ai_states = {
            VerificationState.UNREVIEWED,
            VerificationState.ACCEPTED,
            VerificationState.ADJUSTED,
            VerificationState.REJECTED,
        }

        if self.source is AnnotationSource.HUMAN:
            if self.verification_state not in human_states:
                raise ValueError("Human annotations must be manual or human-added")
            if self.original_ai_box is not None:
                raise ValueError("Human annotations cannot contain an original AI box")
            if self.final_box is None:
                raise ValueError("Human annotations require a final box")
            ai_metadata = {
                "prediction ID": self.prediction_id,
                "original AI label": self.original_ai_label_id,
                "provider": self.provider,
                "model": self.model,
                "prompt": self.prompt,
                "confidence": self.confidence,
            }
            if any(value is not None for value in ai_metadata.values()):
                raise ValueError("Human annotations cannot contain AI provenance")
            if self.verification_state is VerificationState.MANUAL and self.review_prompt is not None:
                raise ValueError("Manual annotations cannot contain an AI review prompt")
        else:
            if self.verification_state not in ai_states:
                raise ValueError("AI annotations require an AI verification state")
            if self.original_ai_box is None:
                raise ValueError("AI annotations require the original AI box")
            if self.prediction_id is None:
                raise ValueError("AI annotations require a provider prediction ID")
            if self.original_ai_label_id is None:
                raise ValueError("AI annotations require the original AI label")
            if not self.provider or not self.model or not self.prompt:
                raise ValueError("AI annotations require provider, model, and prompt provenance")
            if self.review_prompt is not None:
                raise ValueError("AI annotations cannot contain a human review prompt")
            if self.verification_state is VerificationState.REJECTED:
                if self.final_box is not None:
                    raise ValueError("Rejected AI annotations cannot contain a final box")
            elif self.final_box is None:
                raise ValueError("Non-rejected AI annotations require a final box")
            if self.verification_state is VerificationState.ACCEPTED and (
                self.final_box != self.original_ai_box or self.label_id != self.original_ai_label_id
            ):
                raise ValueError("Accepted AI annotations must match the original prediction")
            if self.verification_state is VerificationState.ADJUSTED and (
                self.final_box == self.original_ai_box and self.label_id == self.original_ai_label_id
            ):
                raise ValueError("Adjusted AI annotations must differ from the original prediction")

        return self


class ProjectImage(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    filename: str = Field(min_length=1, max_length=255)
    storage_name: str = Field(min_length=1, max_length=255)
    media_type: str = Field(min_length=1, max_length=100)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    review_state: ImageReviewState = ImageReviewState.NOT_STARTED
    annotations: list[Annotation] = Field(default_factory=list)
    imported_at: datetime = Field(default_factory=utc_now)


class Project(BaseModel):
    schema_version: Literal[2] = 2
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=120)
    labels: list[Label] = Field(default_factory=list)
    images: list[ProjectImage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Project name cannot be blank")
        return normalized

    @model_validator(mode="after")
    def check_ids_and_references(self) -> Project:
        label_ids = [label.id for label in self.labels]
        if len(label_ids) != len(set(label_ids)):
            raise ValueError("Project contains duplicate label IDs")

        image_ids = [image.id for image in self.images]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("Project contains duplicate image IDs")

        annotation_ids = [annotation.id for image in self.images for annotation in image.annotations]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("Project contains duplicate annotation IDs")

        prediction_ids = [
            annotation.prediction_id
            for image in self.images
            for annotation in image.annotations
            if annotation.prediction_id is not None
        ]
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError("Project contains duplicate provider prediction IDs")

        valid_labels = set(label_ids)
        for image in self.images:
            for annotation in image.annotations:
                if annotation.label_id not in valid_labels:
                    raise ValueError(f"Annotation {annotation.id} references an unknown label")
                if annotation.original_ai_label_id is not None and annotation.original_ai_label_id not in valid_labels:
                    raise ValueError(f"Annotation {annotation.id} references an unknown original AI label")
                boxes = [annotation.original_ai_box, annotation.final_box]
                if any(box is not None and not box.fits_within(image.width, image.height) for box in boxes):
                    raise ValueError(f"Annotation {annotation.id} is outside the image bounds")

        return self


class ProjectSummary(BaseModel):
    id: UUID
    name: str
    image_count: int
    completed_image_count: int
    updated_at: datetime

    @classmethod
    def from_project(cls, project: Project) -> ProjectSummary:
        return cls(
            id=project.id,
            name=project.name,
            image_count=len(project.images),
            completed_image_count=sum(image.review_state is ImageReviewState.COMPLETE for image in project.images),
            updated_at=project.updated_at,
        )
