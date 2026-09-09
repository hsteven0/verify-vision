from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path

from app.domain.models import (
    Annotation,
    BoundingBox,
    ImageReviewState,
    Project,
    ProjectImage,
    VerificationState,
)
from app.exporting.models import DatasetSplit, ExportOptions, TrainingImage

TRAINING_STATES = {
    VerificationState.MANUAL,
    VerificationState.ACCEPTED,
    VerificationState.ADJUSTED,
    VerificationState.HUMAN_ADDED,
}


def is_training_annotation(annotation: Annotation) -> bool:
    return annotation.verification_state in TRAINING_STATES and isinstance(annotation.final_box, BoundingBox)


def training_annotations(image: ProjectImage) -> tuple[Annotation, ...]:
    if image.review_state is ImageReviewState.NEEDS_REVIEW:
        return ()
    return tuple(annotation for annotation in image.annotations if is_training_annotation(annotation))


def portable_image_name(image: ProjectImage) -> str:
    original = Path(image.filename)
    stem = safe_filename_component(original.stem, fallback="image")
    suffix = Path(image.storage_name).suffix.lower() or original.suffix.lower() or ".img"
    return f"{stem}-{str(image.id)[:8]}{suffix}"


def safe_filename_component(value: str, fallback: str = "export") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")
    return normalized[:80] or fallback


def training_projection(project: Project) -> tuple[TrainingImage, ...]:
    projected = []
    for image in project.images:
        annotations = training_annotations(image)
        if annotations or image.review_state is ImageReviewState.COMPLETE:
            projected.append(
                TrainingImage(
                    image=image,
                    annotations=annotations,
                    portable_name=portable_image_name(image),
                )
            )
    return tuple(projected)


def assign_dataset_splits(images: tuple[TrainingImage, ...], options: ExportOptions) -> tuple[TrainingImage, ...]:
    if options.split is DatasetSplit.NONE:
        return images
    if len(images) < 2:
        raise ValueError("A train/validation split requires at least two exportable images")

    ordered = sorted(
        images,
        key=lambda item: hashlib.sha256(f"{options.seed}:{item.image.id}".encode()).digest(),
    )
    train_count = max(1, min(len(ordered) - 1, round(len(ordered) * options.train_ratio)))
    train_ids = {item.image.id for item in ordered[:train_count]}
    return tuple(replace(item, split="train" if item.image.id in train_ids else "val") for item in images)


def yolo_coordinates(box: BoundingBox, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0 or not box.fits_within(image_width, image_height):
        raise ValueError("Bounding box must be valid and remain inside the image")
    values = (
        (box.x + box.width / 2) / image_width,
        (box.y + box.height / 2) / image_height,
        box.width / image_width,
        box.height / image_height,
    )
    if any(not math.isfinite(value) or value <= 0 or value > 1 for value in values):
        raise ValueError("YOLO coordinates must be finite values in the normalized 0..1 range")
    return values


def voc_coordinates(box: BoundingBox, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Convert zero-based box edges to one-based inclusive VOC coordinates."""

    values = (box.x, box.y, box.width, box.height)
    if image_width <= 0 or image_height <= 0 or any(not math.isfinite(value) for value in values):
        raise ValueError("Bounding box and image dimensions must be valid")
    left = max(0.0, min(float(image_width), box.x))
    top = max(0.0, min(float(image_height), box.y))
    right = max(0.0, min(float(image_width), box.right))
    bottom = max(0.0, min(float(image_height), box.bottom))
    if right <= left or bottom <= top:
        raise ValueError("Bounding box must overlap the image")
    return (
        min(image_width, math.floor(left) + 1),
        min(image_height, math.floor(top) + 1),
        max(1, math.ceil(right)),
        max(1, math.ceil(bottom)),
    )


def box_edges(box: BoundingBox | None) -> tuple[float | None, ...]:
    if box is None:
        return (None, None, None, None)
    return (box.x, box.y, box.right, box.bottom)


def yaml_quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)
