"""Create a stable project for analytics UI tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import UUID

from PIL import Image, ImageDraw

from app.domain.models import (
    Annotation,
    AnnotationSource,
    BoundingBox,
    ImageReviewState,
    Label,
    Project,
    ProjectImage,
    VerificationState,
)
from app.persistence.repository import ProjectRepository

PROJECT_ID = UUID("8c36c0cf-a512-4d4c-962a-91e2fe9b5e01")
PERSON_ID = UUID("8c36c0cf-a512-4d4c-962a-91e2fe9b5e02")
BICYCLE_ID = UUID("8c36c0cf-a512-4d4c-962a-91e2fe9b5e03")


def identifier(number: int) -> UUID:
    return UUID(f"8c36c0cf-a512-4d4c-962a-{number:012d}")


def ai_annotation(
    number: int,
    label_id: UUID,
    state: VerificationState,
    prompt: str,
    original: BoundingBox,
    *,
    final: BoundingBox | None = None,
    confidence: float = 0.8,
    provider: str = "locateanything",
    model: str = "nvidia/LocateAnything-3B",
) -> Annotation:
    verified = None if state is VerificationState.REJECTED else (final or original)
    return Annotation(
        id=identifier(100 + number),
        prediction_id=identifier(200 + number),
        label_id=label_id,
        original_ai_label_id=label_id,
        source=AnnotationSource.AI,
        verification_state=state,
        original_ai_box=original,
        final_box=verified,
        provider=provider,
        model=model,
        prompt=prompt,
        confidence=confidence,
    )


def build_project() -> Project:
    person = Label(id=PERSON_ID, name="Person", color="#32d6a0")
    bicycle = Label(id=BICYCLE_ID, name="Bicycle", color="#6ba8ff")
    return Project(
        id=PROJECT_ID,
        name="Analytics trust fixture",
        labels=[person, bicycle],
        images=[
            ProjectImage(
                id=identifier(10),
                filename="crosswalk.jpg",
                storage_name="crosswalk.jpg",
                media_type="image/jpeg",
                width=800,
                height=520,
                review_state=ImageReviewState.COMPLETE,
                annotations=[
                    ai_annotation(
                        1,
                        PERSON_ID,
                        VerificationState.ACCEPTED,
                        "person",
                        BoundingBox(x=105, y=95, width=120, height=325),
                        confidence=0.94,
                    ),
                    ai_annotation(
                        2,
                        PERSON_ID,
                        VerificationState.REJECTED,
                        "person",
                        BoundingBox(x=520, y=180, width=90, height=170),
                        confidence=0.27,
                    ),
                    ai_annotation(
                        3,
                        BICYCLE_ID,
                        VerificationState.ADJUSTED,
                        "bicycle",
                        BoundingBox(x=330, y=250, width=235, height=145),
                        final=BoundingBox(x=370, y=280, width=180, height=115),
                        confidence=0.76,
                    ),
                    ai_annotation(
                        4,
                        BICYCLE_ID,
                        VerificationState.REJECTED,
                        "bicycle",
                        BoundingBox(x=620, y=210, width=115, height=110),
                        confidence=0.21,
                    ),
                    Annotation(
                        id=identifier(301),
                        label_id=BICYCLE_ID,
                        source=AnnotationSource.HUMAN,
                        verification_state=VerificationState.HUMAN_ADDED,
                        final_box=BoundingBox(x=585, y=330, width=150, height=105),
                        review_prompt="bicycle",
                    ),
                ],
            ),
            ProjectImage(
                id=identifier(11),
                filename="helmet-review.jpg",
                storage_name="helmet-review.jpg",
                media_type="image/jpeg",
                width=800,
                height=520,
                review_state=ImageReviewState.COMPLETE,
                annotations=[
                    ai_annotation(
                        5,
                        PERSON_ID,
                        VerificationState.ACCEPTED,
                        "person",
                        BoundingBox(x=230, y=80, width=170, height=370),
                        confidence=0.91,
                    ),
                    ai_annotation(
                        6,
                        PERSON_ID,
                        VerificationState.ADJUSTED,
                        "person wearing a helmet",
                        BoundingBox(x=470, y=75, width=150, height=350),
                        final=BoundingBox(x=480, y=82, width=142, height=345),
                        confidence=0.72,
                    ),
                    ai_annotation(
                        7,
                        PERSON_ID,
                        VerificationState.REJECTED,
                        "person wearing a helmet",
                        BoundingBox(x=65, y=130, width=100, height=190),
                        confidence=0.34,
                    ),
                    Annotation(
                        id=identifier(302),
                        label_id=PERSON_ID,
                        source=AnnotationSource.HUMAN,
                        verification_state=VerificationState.HUMAN_ADDED,
                        final_box=BoundingBox(x=650, y=120, width=105, height=285),
                        review_prompt="person wearing a helmet",
                    ),
                    Annotation(
                        id=identifier(303),
                        label_id=PERSON_ID,
                        source=AnnotationSource.HUMAN,
                        verification_state=VerificationState.MANUAL,
                        final_box=BoundingBox(x=20, y=325, width=70, height=130),
                    ),
                ],
            ),
            ProjectImage(
                id=identifier(12),
                filename="open-review.jpg",
                storage_name="open-review.jpg",
                media_type="image/jpeg",
                width=800,
                height=520,
                review_state=ImageReviewState.NEEDS_REVIEW,
                annotations=[
                    ai_annotation(
                        8,
                        BICYCLE_ID,
                        VerificationState.UNREVIEWED,
                        "bicycle",
                        BoundingBox(x=260, y=255, width=255, height=150),
                        confidence=0.5,
                        provider="mock",
                        model="deterministic-layout-v1",
                    ),
                    Annotation(
                        id=identifier(304),
                        label_id=BICYCLE_ID,
                        source=AnnotationSource.HUMAN,
                        verification_state=VerificationState.HUMAN_ADDED,
                        final_box=BoundingBox(x=565, y=270, width=170, height=120),
                    ),
                ],
            ),
        ],
    )


def write_fixture_image(path: Path, index: int) -> None:
    image = Image.new("RGB", (800, 520), (18 + index * 4, 28 + index * 5, 25 + index * 3))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 340, 800, 520), fill=(45, 52, 49))
    for offset in range(-80, 880, 120):
        draw.polygon(
            ((offset, 520), (offset + 48, 520), (offset + 245, 340), (offset + 218, 340)),
            fill=(125, 132, 126),
        )
    draw.ellipse((55, 38, 170, 153), fill=(49, 78, 65), outline=(81, 140, 112), width=4)
    draw.rectangle((610, 70, 755, 220), fill=(33, 47, 42), outline=(74, 105, 91), width=3)
    image.save(path, format="JPEG", quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    repository = ProjectRepository(args.data_dir)
    destination = repository.project_dir(PROJECT_ID) / "project.json"
    if destination.exists():
        raise SystemExit(f"Fixture already exists at {destination}")
    project = repository.save(build_project())
    for index, image in enumerate(project.images):
        write_fixture_image(repository.image_path(project.id, image.storage_name), index)
    print(project.id)


if __name__ == "__main__":
    main()
