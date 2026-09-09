from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.models import Annotation, BoundingBox, ImageReviewState, Label
from app.domain.verification import VerificationDecision


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class LabelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    color: str = "#32d6a0"


class LabelUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class AnnotationLabelAssign(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    color: str = "#32d6a0"


class AnnotationCreate(BaseModel):
    label_id: UUID
    box: BoundingBox
    verification_state: Literal["manual", "human_added"] = "manual"
    review_prompt: str | None = Field(default=None, min_length=1, max_length=500)
    note: str | None = Field(default=None, max_length=1000)


class AnnotationUpdate(BaseModel):
    label_id: UUID | None = None
    box: BoundingBox | None = None
    note: str | None = Field(default=None, max_length=1000)


class ImageReviewUpdate(BaseModel):
    review_state: ImageReviewState


class ProposalCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    label_id: UUID | None = None


class ProjectHistoryImageState(BaseModel):
    id: UUID
    review_state: ImageReviewState
    annotations: list[Annotation]


class ProjectHistoryRestore(BaseModel):
    labels: list[Label]
    images: list[ProjectHistoryImageState]


class VerificationUpdate(BaseModel):
    decision: VerificationDecision


class HealthResponse(BaseModel):
    status: str
    api_version: str
    inference_provider: str
    inference_model: str
    inference_device: str
    inference_status: Literal["not_loaded", "loading", "ready", "unavailable", "error"]
    inference_available: bool
    inference_platform: str | None = None
    inference_python_version: str | None = None
    inference_torch_version: str | None = None
    inference_cuda_version: str | None = None
    inference_gpu: str | None = None
    inference_compute_capability: str | None = None
    inference_requested_dtype: str | None = None
    inference_selected_dtype: str | None = None
    inference_dtype_reason: str | None = None
    inference_detail: str | None = None


class PublicHealthResponse(BaseModel):
    status: str
    api_version: str


class CapabilityFeatures(BaseModel):
    inference: bool
    training: bool
    trained_model_prediction: bool
    manual_annotation: bool
    image_uploads: bool
    dataset_export: bool
    analytics: bool


class CapabilityInference(BaseModel):
    model: str
    available: bool
    device: str
    status: Literal["not_loaded", "loading", "ready", "unavailable", "error"]
    disclosure: str


class CapabilityLimits(BaseModel):
    max_upload_bytes: int
    prompt_max_characters: int


class CapabilitiesResponse(BaseModel):
    features: CapabilityFeatures
    inference: CapabilityInference
    limits: CapabilityLimits
