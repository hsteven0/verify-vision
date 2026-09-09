from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.models import Annotation, AnnotationSource, Project, VerificationState
from app.domain.verification import verified_annotations


class CountRate(BaseModel):
    count: int = Field(ge=0)
    denominator: int = Field(ge=0)
    rate: float = Field(ge=0, le=100)


class ClassEvaluation(BaseModel):
    label_id: UUID
    label_name: str
    ai_proposals: int
    accepted: int
    adjusted: int
    rejected: int
    unresolved: int
    human_added: int


class EvaluationSummary(BaseModel):
    total_ai_proposals: int
    reviewed_ai_proposals: int
    accepted: CountRate
    adjusted: CountRate
    rejected: CountRate
    unresolved: CountRate
    human_added_count: int
    verified_annotation_count: int
    average_adjusted_iou: float | None
    by_class: list[ClassEvaluation]


def count_rate(count: int, denominator: int) -> CountRate:
    """Return a percentage without NaN or infinity."""

    return CountRate(
        count=count,
        denominator=denominator,
        rate=(count / denominator * 100) if denominator else 0,
    )


def ai_outcome_counts(annotations: Iterable[Annotation]) -> Counter[VerificationState]:
    return Counter(
        annotation.verification_state for annotation in annotations if annotation.source is AnnotationSource.AI
    )


def adjusted_annotation_iou(annotation: Annotation) -> float | None:
    if (
        annotation.source is AnnotationSource.AI
        and annotation.verification_state is VerificationState.ADJUSTED
        and annotation.original_ai_box is not None
        and annotation.final_box is not None
    ):
        return annotation.original_ai_box.intersection_over_union(annotation.final_box)
    return None


def calculate_evaluation(project: Project) -> EvaluationSummary:
    annotations = [annotation for image in project.images for annotation in image.annotations]
    ai_annotations = [annotation for annotation in annotations if annotation.source is AnnotationSource.AI]
    total = len(ai_annotations)
    counts = ai_outcome_counts(ai_annotations)
    accepted_count = counts[VerificationState.ACCEPTED]
    adjusted_count = counts[VerificationState.ADJUSTED]
    rejected_count = counts[VerificationState.REJECTED]
    unresolved_count = counts[VerificationState.UNREVIEWED]
    reviewed_count = accepted_count + adjusted_count + rejected_count
    human_added_count = sum(
        annotation.verification_state is VerificationState.HUMAN_ADDED for annotation in annotations
    )
    adjusted_ious = [iou for annotation in ai_annotations if (iou := adjusted_annotation_iou(annotation)) is not None]

    by_label: dict[UUID, dict[str, int]] = defaultdict(
        lambda: {
            "ai_proposals": 0,
            "accepted": 0,
            "adjusted": 0,
            "rejected": 0,
            "unresolved": 0,
            "human_added": 0,
        }
    )
    for annotation in ai_annotations:
        label_id = annotation.original_ai_label_id or annotation.label_id
        group = by_label[label_id]
        group["ai_proposals"] += 1
        state_key = (
            "unresolved"
            if annotation.verification_state is VerificationState.UNREVIEWED
            else annotation.verification_state.value
        )
        group[state_key] += 1
    for annotation in annotations:
        if annotation.verification_state is VerificationState.HUMAN_ADDED:
            by_label[annotation.label_id]["human_added"] += 1

    label_names = {label.id: label.name for label in project.labels}
    class_metrics = [
        ClassEvaluation(label_id=label_id, label_name=label_names[label_id], **values)
        for label_id, values in by_label.items()
        if label_id in label_names
    ]
    class_metrics.sort(key=lambda item: item.label_name.casefold())

    return EvaluationSummary(
        total_ai_proposals=total,
        reviewed_ai_proposals=reviewed_count,
        accepted=count_rate(accepted_count, reviewed_count),
        adjusted=count_rate(adjusted_count, reviewed_count),
        rejected=count_rate(rejected_count, reviewed_count),
        unresolved=count_rate(unresolved_count, total),
        human_added_count=human_added_count,
        verified_annotation_count=len(verified_annotations(annotations)),
        average_adjusted_iou=(sum(adjusted_ious) / len(adjusted_ious)) if adjusted_ious else None,
        by_class=class_metrics,
    )
