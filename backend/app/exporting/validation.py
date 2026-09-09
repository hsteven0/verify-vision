from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from uuid import UUID

from app.domain.models import (
    Annotation,
    AnnotationSource,
    BoundingBox,
    ImageReviewState,
    Project,
    ProjectImage,
    VerificationState,
)
from app.exporting.common import is_training_annotation, training_projection
from app.exporting.models import (
    DatasetSplit,
    ExportOptions,
    ExportPreview,
    ValidationFinding,
    ValidationSeverity,
)
from app.persistence.repository import ProjectRepository


def _duplicate_values(values: Iterable[UUID | None]) -> set[UUID]:
    counts = Counter(value for value in values if value is not None)
    return {value for value, count in counts.items() if count > 1}


class ProjectExportValidator:
    def __init__(self, repository: ProjectRepository) -> None:
        self.repository = repository

    def validate(self, project: Project, options: ExportOptions) -> ExportPreview:
        findings: list[ValidationFinding] = []
        label_ids = {label.id for label in project.labels}

        if not project.labels:
            self._add(findings, ValidationSeverity.ERROR, "no_labels", "Project has no classes.")

        self._report_duplicates(
            findings,
            "label",
            _duplicate_values(label.id for label in project.labels),
        )
        self._report_duplicates(
            findings,
            "image",
            _duplicate_values(image.id for image in project.images),
        )
        self._report_duplicates(
            findings,
            "annotation",
            _duplicate_values(annotation.id for image in project.images for annotation in image.annotations),
        )
        self._report_duplicates(
            findings,
            "prediction",
            _duplicate_values(annotation.prediction_id for image in project.images for annotation in image.annotations),
        )

        for image in project.images:
            if image.width <= 0 or image.height <= 0:
                self._add(
                    findings,
                    ValidationSeverity.ERROR,
                    "invalid_image_dimensions",
                    f'Image "{image.filename}" has invalid dimensions.',
                    image_id=image.id,
                )
            image_path = self.repository.image_path(project.id, image.storage_name)
            if not image_path.is_file():
                self._add(
                    findings,
                    ValidationSeverity.ERROR,
                    "missing_image_file",
                    f'Image file for "{image.filename}" is missing.',
                    image_id=image.id,
                )
            if image.review_state is ImageReviewState.NEEDS_REVIEW:
                self._add(
                    findings,
                    ValidationSeverity.WARNING,
                    "image_needs_review",
                    f'Image "{image.filename}" needs review and is excluded from training data.',
                    image_id=image.id,
                )

            unresolved = sum(
                annotation.verification_state is VerificationState.UNREVIEWED for annotation in image.annotations
            )
            if unresolved:
                self._add(
                    findings,
                    ValidationSeverity.WARNING,
                    "unresolved_predictions",
                    f'Image "{image.filename}" has {unresolved} unresolved AI '
                    f"proposal{'' if unresolved == 1 else 's'}; they are excluded.",
                    image_id=image.id,
                )

            for annotation in image.annotations:
                self._validate_annotation(findings, image, annotation, label_ids)

        projected = training_projection(project)
        if not projected:
            self._add(
                findings,
                ValidationSeverity.ERROR,
                "no_training_images",
                "Project has no finalized images or annotations eligible for training export.",
            )
        if options.split is DatasetSplit.TRAIN_VAL and len(projected) < 2:
            self._add(
                findings,
                ValidationSeverity.ERROR,
                "split_requires_two_images",
                "Train/validation split requires at least two exportable images.",
            )

        included_annotations = [annotation for item in projected for annotation in item.annotations]
        state_counts = Counter(annotation.verification_state for annotation in included_annotations)
        all_annotations = [annotation for image in project.images for annotation in image.annotations]
        excluded_needs_review = sum(
            is_training_annotation(annotation)
            for image in project.images
            if image.review_state is ImageReviewState.NEEDS_REVIEW
            for annotation in image.annotations
        )
        train_images = len(projected)
        validation_images = 0
        if options.split is DatasetSplit.TRAIN_VAL and len(projected) >= 2:
            train_images = max(1, min(len(projected) - 1, round(len(projected) * options.train_ratio)))
            validation_images = len(projected) - train_images

        error_count = sum(finding.severity is ValidationSeverity.ERROR for finding in findings)
        warning_count = sum(finding.severity is ValidationSeverity.WARNING for finding in findings)
        return ExportPreview(
            total_images=len(project.images),
            exportable_images=len(projected),
            train_images=train_images,
            validation_images=validation_images,
            training_boxes=len(included_annotations),
            classes=len(project.labels),
            accepted=state_counts[VerificationState.ACCEPTED],
            adjusted=state_counts[VerificationState.ADJUSTED],
            human_added=state_counts[VerificationState.HUMAN_ADDED],
            manual=state_counts[VerificationState.MANUAL],
            excluded_rejected=sum(
                annotation.verification_state is VerificationState.REJECTED for annotation in all_annotations
            ),
            excluded_unresolved=sum(
                annotation.verification_state is VerificationState.UNREVIEWED for annotation in all_annotations
            ),
            excluded_needs_review=excluded_needs_review,
            blocking_error_count=error_count,
            warning_count=warning_count,
            findings=findings,
        )

    def _validate_annotation(
        self,
        findings: list[ValidationFinding],
        image: ProjectImage,
        annotation: Annotation,
        label_ids: set[UUID],
    ) -> None:
        image_id = image.id
        filename = image.filename
        width = image.width
        height = image.height
        if annotation.label_id not in label_ids:
            self._add(
                findings,
                ValidationSeverity.ERROR,
                "unknown_label",
                f"Annotation {annotation.id} references an unknown final class.",
                image_id=image_id,
                annotation_id=annotation.id,
            )
        if annotation.original_ai_label_id is not None and annotation.original_ai_label_id not in label_ids:
            self._add(
                findings,
                ValidationSeverity.ERROR,
                "unknown_original_label",
                f"Annotation {annotation.id} references an unknown original AI class.",
                image_id=image_id,
                annotation_id=annotation.id,
            )

        requires_final = annotation.verification_state in {
            VerificationState.MANUAL,
            VerificationState.ACCEPTED,
            VerificationState.ADJUSTED,
            VerificationState.HUMAN_ADDED,
        }
        if requires_final and not isinstance(annotation.final_box, BoundingBox):
            self._add(
                findings,
                ValidationSeverity.ERROR,
                "missing_final_box",
                f'Annotation {annotation.id} on "{filename}" has no final verified box.',
                image_id=image_id,
                annotation_id=annotation.id,
            )
        if annotation.source is AnnotationSource.AI and not isinstance(annotation.original_ai_box, BoundingBox):
            self._add(
                findings,
                ValidationSeverity.ERROR,
                "missing_original_ai_box",
                f"AI annotation {annotation.id} has no original prediction box.",
                image_id=image_id,
                annotation_id=annotation.id,
            )

        for field_name, box in (
            ("original AI", annotation.original_ai_box),
            ("final", annotation.final_box),
        ):
            if box is None:
                continue
            if not isinstance(box, BoundingBox) or box.width <= 0 or box.height <= 0:
                self._add(
                    findings,
                    ValidationSeverity.ERROR,
                    "invalid_box",
                    f"Annotation {annotation.id} has an invalid {field_name} box.",
                    image_id=image_id,
                    annotation_id=annotation.id,
                )
            elif width <= 0 or height <= 0 or not box.fits_within(width, height):
                self._add(
                    findings,
                    ValidationSeverity.ERROR,
                    "box_outside_image",
                    f'Annotation {annotation.id} has a {field_name} box outside "{filename}".',
                    image_id=image_id,
                    annotation_id=annotation.id,
                )

    @staticmethod
    def _report_duplicates(findings: list[ValidationFinding], kind: str, duplicates: set[UUID]) -> None:
        for duplicate in sorted(duplicates, key=str):
            ProjectExportValidator._add(
                findings,
                ValidationSeverity.ERROR,
                f"duplicate_{kind}_id",
                f"Project contains duplicate {kind} ID {duplicate}.",
            )

    @staticmethod
    def _add(
        findings: list[ValidationFinding],
        severity: ValidationSeverity,
        code: str,
        message: str,
        *,
        image_id: UUID | None = None,
        annotation_id: UUID | None = None,
    ) -> None:
        findings.append(
            ValidationFinding(
                severity=severity,
                code=code,
                message=message,
                image_id=image_id,
                annotation_id=annotation_id,
            )
        )
