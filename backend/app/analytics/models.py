from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.evaluation import CountRate
from app.exporting.models import ValidationFinding


class AnalyticsFilters(BaseModel):
    label_id: UUID | None = None
    provider: str | None = Field(default=None, max_length=120)
    model: str | None = Field(default=None, max_length=120)
    prompt: str | None = Field(default=None, max_length=500)


class OutcomeBreakdown(BaseModel):
    total_ai_proposals: int = Field(ge=0)
    reviewed_ai_proposals: int = Field(ge=0)
    accepted: CountRate
    adjusted: CountRate
    rejected: CountRate
    unresolved: CountRate


class ModelTrustOverview(BaseModel):
    total_imported_images: int = Field(ge=0)
    reviewed_images: int = Field(ge=0)
    images_in_scope: int = Field(ge=0)
    images_needing_review: int = Field(ge=0)
    auto_label_trust: CountRate
    human_intervention: CountRate
    human_added_annotations: int = Field(ge=0)
    attributable_ai_misses: int = Field(ge=0)
    unattributed_human_added: int = Field(ge=0)


class IouBucket(BaseModel):
    key: str
    range_label: str
    interpretation: str
    count: int = Field(ge=0)
    rate: float = Field(ge=0, le=100)


class IouStatistics(BaseModel):
    count: int = Field(ge=0)
    mean: float | None = Field(default=None, ge=0, le=1)
    median: float | None = Field(default=None, ge=0, le=1)
    minimum: float | None = Field(default=None, ge=0, le=1)
    maximum: float | None = Field(default=None, ge=0, le=1)
    buckets: list[IouBucket]


class ClassAnalytics(BaseModel):
    label_id: UUID
    label_name: str
    total_ai_proposals: int = Field(ge=0)
    reviewed_ai_proposals: int = Field(ge=0)
    accepted: CountRate
    adjusted: CountRate
    rejected: CountRate
    unresolved: int = Field(ge=0)
    human_added: int = Field(ge=0)
    attributable_ai_misses: int = Field(ge=0)
    mean_adjusted_iou: float | None = Field(default=None, ge=0, le=1)
    final_verified_annotations: int = Field(ge=0)
    small_sample: bool


class PromptAnalytics(BaseModel):
    prompt: str
    inference_runs: int = Field(ge=0)
    total_ai_proposals: int = Field(ge=0)
    reviewed_ai_proposals: int = Field(ge=0)
    accepted: CountRate
    adjusted: CountRate
    rejected: CountRate
    unresolved: int = Field(ge=0)
    attributable_ai_misses: int = Field(ge=0)
    mean_adjusted_iou: float | None = Field(default=None, ge=0, le=1)
    small_sample: bool


class ConfidenceByOutcome(BaseModel):
    accepted_count: int = Field(ge=0)
    accepted_mean: float | None = Field(default=None, ge=0, le=1)
    adjusted_count: int = Field(ge=0)
    adjusted_mean: float | None = Field(default=None, ge=0, le=1)
    rejected_count: int = Field(ge=0)
    rejected_mean: float | None = Field(default=None, ge=0, le=1)


class ProviderModelAnalytics(BaseModel):
    provider: str
    model: str
    total_ai_proposals: int = Field(ge=0)
    reviewed_ai_proposals: int = Field(ge=0)
    accepted: CountRate
    adjusted: CountRate
    rejected: CountRate
    unresolved: int = Field(ge=0)
    attributable_ai_misses: int = Field(ge=0)
    mean_adjusted_iou: float | None = Field(default=None, ge=0, le=1)
    confidence: ConfidenceByOutcome | None = None


class ClassDistribution(BaseModel):
    label_id: UUID
    label_name: str
    count: int = Field(ge=0)
    rate: float = Field(ge=0, le=100)


class ValidationSummary(BaseModel):
    export_ready: bool
    blocking_error_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    findings: list[ValidationFinding]


class DatasetQualitySummary(BaseModel):
    total_imported_images: int = Field(ge=0)
    images_with_verified_annotations: int = Field(ge=0)
    images_without_final_annotations: int = Field(ge=0)
    exportable_images: int = Field(ge=0)
    total_final_training_boxes: int = Field(ge=0)
    total_classes: int = Field(ge=0)
    unresolved_annotations: int = Field(ge=0)
    review_needed_images: int = Field(ge=0)
    class_distribution: list[ClassDistribution]
    validation: ValidationSummary


class LabelFilterOption(BaseModel):
    id: UUID
    name: str


class ProviderModelFilterOption(BaseModel):
    provider: str
    model: str


class AnalyticsFilterOptions(BaseModel):
    labels: list[LabelFilterOption]
    provider_models: list[ProviderModelFilterOption]
    prompts: list[str]


class AnalyticsSummary(BaseModel):
    overview: ModelTrustOverview
    outcomes: OutcomeBreakdown
    adjusted_box_iou: IouStatistics
    classes: list[ClassAnalytics]
    prompts: list[PromptAnalytics]
    provider_models: list[ProviderModelAnalytics]
    dataset: DatasetQualitySummary
    filter_options: AnalyticsFilterOptions
    active_filters: AnalyticsFilters


class ExampleKind(StrEnum):
    REJECTED = "rejected"
    LOW_IOU = "low_iou"
    HUMAN_ADDED = "human_added"
    UNRESOLVED = "unresolved"
    NEEDS_REVIEW = "needs_review"
    INTERVENTION = "intervention"


class ProblemExample(BaseModel):
    image_id: UUID
    image_filename: str
    annotation_id: UUID | None
    label_id: UUID | None
    label_name: str | None
    kind: ExampleKind
    verification_state: str | None
    prompt: str | None
    provider: str | None
    model: str | None
    iou: float | None = Field(default=None, ge=0, le=1)
    human_added_is_attributed_ai_miss: bool = False


class AnalyticsExamples(BaseModel):
    kind: ExampleKind
    total: int = Field(ge=0)
    items: list[ProblemExample]
