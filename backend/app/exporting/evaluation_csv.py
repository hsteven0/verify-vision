from __future__ import annotations

import csv
from pathlib import Path

from app.domain.models import AnnotationSource, VerificationState
from app.exporting.common import box_edges
from app.exporting.models import ExportContext, ExportFormat

CSV_COLUMNS = [
    "image_id",
    "image_name",
    "image_review_state",
    "annotation_id",
    "prediction_id",
    "source",
    "prompt",
    "review_prompt",
    "class",
    "provider",
    "model",
    "verification_status",
    "confidence",
    "original_x1",
    "original_y1",
    "original_x2",
    "original_y2",
    "verified_x1",
    "verified_y1",
    "verified_x2",
    "verified_y2",
    "iou",
    "note",
]


class EvaluationCsvExporter:
    format = ExportFormat.EVALUATION_CSV
    extension = ".csv"
    media_type = "text/csv; charset=utf-8"

    def write(self, context: ExportContext, destination: Path) -> None:
        label_names = {label.id: label.name for label in context.project.labels}
        with destination.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for image in context.project.images:
                for annotation in image.annotations:
                    original = box_edges(annotation.original_ai_box)
                    verified_box = (
                        annotation.final_box
                        if annotation.verification_state
                        in {
                            VerificationState.MANUAL,
                            VerificationState.ACCEPTED,
                            VerificationState.ADJUSTED,
                            VerificationState.HUMAN_ADDED,
                        }
                        else None
                    )
                    verified = box_edges(verified_box)
                    iou = None
                    if (
                        annotation.source is AnnotationSource.AI
                        and annotation.verification_state in {VerificationState.ACCEPTED, VerificationState.ADJUSTED}
                        and annotation.original_ai_box is not None
                        and verified_box is not None
                    ):
                        iou = annotation.original_ai_box.intersection_over_union(verified_box)
                    writer.writerow(
                        {
                            "image_id": str(image.id),
                            "image_name": image.filename,
                            "image_review_state": image.review_state.value,
                            "annotation_id": str(annotation.id),
                            "prediction_id": annotation.prediction_id or "",
                            "source": annotation.source.value,
                            "prompt": annotation.prompt or "",
                            "review_prompt": annotation.review_prompt or "",
                            "class": label_names.get(annotation.label_id, ""),
                            "provider": annotation.provider or "",
                            "model": annotation.model or "",
                            "verification_status": annotation.verification_state.value,
                            "confidence": self._number(annotation.confidence),
                            "original_x1": self._number(original[0]),
                            "original_y1": self._number(original[1]),
                            "original_x2": self._number(original[2]),
                            "original_y2": self._number(original[3]),
                            "verified_x1": self._number(verified[0]),
                            "verified_y1": self._number(verified[1]),
                            "verified_x2": self._number(verified[2]),
                            "verified_y2": self._number(verified[3]),
                            "iou": self._number(iou),
                            "note": annotation.note or "",
                        }
                    )

    @staticmethod
    def _number(value: float | None) -> str:
        if value is None:
            return ""
        return f"{value:.8g}"
