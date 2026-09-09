from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.domain.models import (
    Annotation,
    AnnotationSource,
    BoundingBox,
    Label,
    Project,
    ProjectImage,
    VerificationState,
)
from app.domain.verification import (
    InvalidVerificationTransition,
    VerificationDecision,
    verified_annotations,
    verify_ai_annotation,
)


def ai_annotation(
    *,
    state: VerificationState = VerificationState.UNREVIEWED,
    original: BoundingBox | None = None,
    final: BoundingBox | None = None,
) -> Annotation:
    label_id = uuid4()
    original_box = original or BoundingBox(x=10, y=10, width=40, height=40)
    return Annotation(
        prediction_id=uuid4(),
        label_id=label_id,
        original_ai_label_id=label_id,
        source=AnnotationSource.AI,
        verification_state=state,
        original_ai_box=original_box,
        final_box=original_box if final is None else final,
        provider="mock",
        model="deterministic-layout-v1",
        prompt="the bicycle",
        confidence=0.88,
    )


def test_bounding_box_iou_uses_pixel_geometry() -> None:
    first = BoundingBox(x=0, y=0, width=100, height=100)
    second = BoundingBox(x=50, y=50, width=100, height=100)

    assert first.intersection_over_union(second) == pytest.approx(2500 / 17500)
    assert first.intersection_over_union(BoundingBox(x=200, y=200, width=10, height=10)) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("x", -1), ("y", -1), ("width", 0), ("height", float("inf"))],
)
def test_bounding_box_rejects_invalid_geometry(field: str, value: float) -> None:
    values = {"x": 1, "y": 1, "width": 10, "height": 10, field: value}
    with pytest.raises(ValidationError):
        BoundingBox(**values)


def test_ai_annotation_preserves_original_and_final_boxes() -> None:
    label_id = uuid4()
    original = BoundingBox(x=10, y=10, width=40, height=40)
    corrected = BoundingBox(x=12, y=8, width=45, height=42)

    annotation = Annotation(
        prediction_id=uuid4(),
        label_id=label_id,
        original_ai_label_id=label_id,
        source=AnnotationSource.AI,
        verification_state=VerificationState.ADJUSTED,
        original_ai_box=original,
        final_box=corrected,
        provider="future-provider",
        model="future-model",
        prompt="the bicycle",
    )

    assert annotation.original_ai_box == original
    assert annotation.final_box == corrected


def test_ai_annotation_requires_original_prediction() -> None:
    with pytest.raises(ValidationError, match="original AI box"):
        Annotation(
            prediction_id=uuid4(),
            label_id=uuid4(),
            original_ai_label_id=uuid4(),
            source=AnnotationSource.AI,
            verification_state=VerificationState.UNREVIEWED,
            final_box=BoundingBox(x=1, y=1, width=5, height=5),
            provider="mock",
            model="mock-v1",
            prompt="object",
        )


def test_verification_transitions_preserve_prediction_and_original_box() -> None:
    proposal = ai_annotation()
    accepted = verify_ai_annotation(proposal, VerificationDecision.ACCEPTED)

    assert accepted.verification_state is VerificationState.ACCEPTED
    assert accepted.prediction_id == proposal.prediction_id
    assert accepted.original_ai_box == proposal.original_ai_box
    assert accepted.final_box == proposal.original_ai_box

    corrected_box = BoundingBox(x=12, y=8, width=45, height=42)
    changed = Annotation.model_validate({**proposal.model_dump(), "final_box": corrected_box})
    adjusted = verify_ai_annotation(changed, VerificationDecision.ADJUSTED)

    assert adjusted.original_ai_box == proposal.original_ai_box
    assert adjusted.final_box == corrected_box
    assert adjusted.verification_state is VerificationState.ADJUSTED


def test_invalid_verification_transition_is_rejected() -> None:
    proposal = ai_annotation()

    with pytest.raises(InvalidVerificationTransition, match="Move, resize"):
        verify_ai_annotation(proposal, VerificationDecision.ADJUSTED)

    accepted = verify_ai_annotation(proposal, VerificationDecision.ACCEPTED)
    with pytest.raises(InvalidVerificationTransition, match="Only unresolved"):
        verify_ai_annotation(accepted, VerificationDecision.REJECTED)


def test_rejected_predictions_are_not_verified_output() -> None:
    rejected = verify_ai_annotation(ai_annotation(), VerificationDecision.REJECTED)

    assert rejected.final_box is None
    assert rejected.original_ai_box is not None
    assert verified_annotations([rejected]) == []


def test_human_added_annotation_has_missed_object_provenance() -> None:
    annotation = Annotation(
        label_id=uuid4(),
        source=AnnotationSource.HUMAN,
        verification_state=VerificationState.HUMAN_ADDED,
        final_box=BoundingBox(x=4, y=5, width=20, height=25),
        review_prompt="all bicycles",
    )

    assert annotation.verification_state is VerificationState.HUMAN_ADDED
    assert annotation.prediction_id is None
    assert annotation.original_ai_box is None
    assert annotation.provider is None


def test_project_rejects_unknown_labels_and_out_of_bounds_boxes() -> None:
    label = Label(name="Vehicle")
    annotation = Annotation(
        label_id=label.id,
        source=AnnotationSource.HUMAN,
        verification_state=VerificationState.MANUAL,
        final_box=BoundingBox(x=90, y=90, width=20, height=20),
    )

    with pytest.raises(ValidationError, match="outside the image bounds"):
        Project(
            name="Road scenes",
            labels=[label],
            images=[
                ProjectImage(
                    filename="road.jpg",
                    storage_name="one.jpg",
                    width=100,
                    height=100,
                    media_type="image/jpeg",
                    annotations=[annotation],
                )
            ],
        )
