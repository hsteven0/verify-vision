from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.domain.models import (  # noqa: E402
    Annotation,
    AnnotationSource,
    BoundingBox,
    ImageReviewState,
    Label,
    Project,
    ProjectImage,
    VerificationState,
)
from app.persistence.repository import ProjectRepository  # noqa: E402
from app.training.models import TrainingConfig, TrainingStatus  # noqa: E402
from app.training.service import TrainingService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one small YOLO training check.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BACKEND_ROOT / "data" / "training-check",
        help="Local generated-project and training-artifact directory.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    parser.add_argument("--checkpoint", default="yolo26n.pt", choices=("yolo26n.pt",))
    return parser.parse_args()


def create_tiny_project(repository: ProjectRepository) -> Project:
    label = Label(name="Square")
    images: list[ProjectImage] = []
    project = Project(name="YOLO training check", labels=[label])
    for index in range(4):
        storage_name = f"square-{index}.png"
        offset = 12 + index * 3
        box = BoundingBox(x=offset, y=offset, width=32, height=32)
        images.append(
            ProjectImage(
                filename=storage_name,
                storage_name=storage_name,
                media_type="image/png",
                width=96,
                height=96,
                review_state=ImageReviewState.COMPLETE,
                annotations=[
                    Annotation(
                        label_id=label.id,
                        source=AnnotationSource.HUMAN,
                        verification_state=VerificationState.MANUAL,
                        final_box=box,
                    )
                ],
            )
        )
    project.images = images
    repository.save(project)
    for index, item in enumerate(project.images):
        image = Image.new("RGB", (96, 96), color=(22 + index * 5, 35, 42))
        box = item.annotations[0].final_box
        assert box is not None
        draw = ImageDraw.Draw(image)
        draw.rectangle((box.x, box.y, box.right, box.bottom), fill=(50, 210, 155))
        image.save(repository.image_path(project.id, item.storage_name), format="PNG")
    return project


def main() -> int:
    args = parse_args()
    repository = ProjectRepository(args.data_dir.resolve())
    project = create_tiny_project(repository)
    service = TrainingService(repository)
    availability = service.availability()
    if not availability.available:
        print(availability.reason)
        print('Install the optional extra with: python -m pip install -e "./backend[training]"')
        return 2

    run = service.create_run(
        project.id,
        TrainingConfig(
            checkpoint=args.checkpoint,
            epochs=1,
            image_size=320,
            batch_size=2,
            train_ratio=0.75,
            seed=1337,
            device=args.device,
            run_name="one-epoch-check",
        ),
    )
    while run.status in {
        TrainingStatus.QUEUED,
        TrainingStatus.PREPARING,
        TrainingStatus.TRAINING,
        TrainingStatus.VALIDATING,
    }:
        print(f"{run.status.value}: epoch {run.current_epoch}/{run.total_epochs}")
        time.sleep(1)
        run = service.get_run(project.id, run.id)

    if run.status is not TrainingStatus.COMPLETED:
        print(f"Training check failed: {run.failure_reason or run.status.value}")
        service.shutdown()
        return 1

    checkpoint, _ = service.artifact(project.id, run.id, "best_checkpoint")
    preview = service.predict_project_image(project.id, run.id, project.images[0].id, 0.25)
    print(f"completed: {checkpoint}")
    print(f"validation metrics: {run.validation_metrics}")
    print(f"sample predictions: {len(preview.detections)}")
    service.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
