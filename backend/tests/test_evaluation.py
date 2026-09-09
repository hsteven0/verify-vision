from uuid import uuid4

import pytest

from app.domain.evaluation import calculate_evaluation
from app.domain.models import (
    Annotation,
    AnnotationSource,
    BoundingBox,
    Label,
    Project,
    ProjectImage,
    VerificationState,
)


def ai_annotation(
    label_id: object,
    state: VerificationState,
    final_box: BoundingBox | None,
) -> Annotation:
    original = BoundingBox(x=0, y=0, width=100, height=100)
    return Annotation(
        prediction_id=uuid4(),
        label_id=label_id,
        original_ai_label_id=label_id,
        source=AnnotationSource.AI,
        verification_state=state,
        original_ai_box=original,
        final_box=final_box,
        provider="mock",
        model="deterministic-layout-v1",
        prompt="vehicle",
        confidence=0.9,
    )


def test_evaluation_counts_rates_iou_and_class_breakdown() -> None:
    label = Label(name="Vehicle")
    original = BoundingBox(x=0, y=0, width=100, height=100)
    adjusted = BoundingBox(x=50, y=50, width=100, height=100)
    annotations = [
        ai_annotation(label.id, VerificationState.ACCEPTED, original),
        ai_annotation(label.id, VerificationState.ADJUSTED, adjusted),
        ai_annotation(label.id, VerificationState.REJECTED, None),
        ai_annotation(label.id, VerificationState.UNREVIEWED, original),
        Annotation(
            label_id=label.id,
            source=AnnotationSource.HUMAN,
            verification_state=VerificationState.HUMAN_ADDED,
            final_box=BoundingBox(x=200, y=200, width=20, height=20),
            review_prompt="vehicle",
        ),
    ]
    project = Project(
        name="Evaluation",
        labels=[label],
        images=[
            ProjectImage(
                filename="road.jpg",
                storage_name="road.jpg",
                media_type="image/jpeg",
                width=400,
                height=400,
                annotations=annotations,
            )
        ],
    )

    metrics = calculate_evaluation(project)

    assert metrics.total_ai_proposals == 4
    assert metrics.reviewed_ai_proposals == 3
    assert metrics.accepted.count == 1
    assert metrics.accepted.denominator == 3
    assert metrics.accepted.rate == pytest.approx(100 / 3)
    assert metrics.adjusted.rate == pytest.approx(100 / 3)
    assert metrics.rejected.rate == pytest.approx(100 / 3)
    assert metrics.unresolved.rate == 25
    assert metrics.human_added_count == 1
    assert metrics.verified_annotation_count == 3
    assert metrics.average_adjusted_iou == pytest.approx(2500 / 17500)
    assert metrics.by_class[0].human_added == 1
    assert metrics.by_class[0].ai_proposals == 4
