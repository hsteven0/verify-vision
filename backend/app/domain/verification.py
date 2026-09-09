from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from app.domain.models import Annotation, AnnotationSource, VerificationState


class VerificationDecision(StrEnum):
    ACCEPTED = "accepted"
    ADJUSTED = "adjusted"
    REJECTED = "rejected"


class InvalidVerificationTransition(ValueError):
    pass


def verify_ai_annotation(
    annotation: Annotation,
    decision: VerificationDecision,
) -> Annotation:
    if annotation.source is not AnnotationSource.AI:
        raise InvalidVerificationTransition("Only AI proposals can be verified")
    if annotation.verification_state is not VerificationState.UNREVIEWED:
        raise InvalidVerificationTransition("Only unresolved AI proposals can receive a decision")

    changed = (
        annotation.final_box != annotation.original_ai_box or annotation.label_id != annotation.original_ai_label_id
    )
    if decision is VerificationDecision.ACCEPTED and changed:
        raise InvalidVerificationTransition("This proposal was changed; confirm it as adjusted instead")
    if decision is VerificationDecision.ADJUSTED and not changed:
        raise InvalidVerificationTransition("Move, resize, or reclassify the proposal before confirming an adjustment")

    update: dict[str, object] = {
        "verification_state": VerificationState(decision.value),
        "updated_at": datetime.now(UTC),
    }
    if decision is VerificationDecision.REJECTED:
        update["final_box"] = None

    return Annotation.model_validate({**annotation.model_dump(), **update})


def verified_annotations(annotations: list[Annotation]) -> list[Annotation]:
    included_states = {
        VerificationState.MANUAL,
        VerificationState.HUMAN_ADDED,
        VerificationState.ACCEPTED,
        VerificationState.ADJUSTED,
    }
    return [
        annotation
        for annotation in annotations
        if annotation.verification_state in included_states and annotation.final_box is not None
    ]
