from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

from app.exporting.models import ExportContext, ExportFormat, TrainingImage


class CocoExporter:
    format = ExportFormat.COCO
    extension = ".zip"
    media_type = "application/zip"

    def write(self, context: ExportContext, destination: Path) -> None:
        groups: dict[str, list[TrainingImage]] = {}
        for item in context.training_images:
            groups.setdefault(item.split, []).append(item)

        with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
            for split, items in groups.items():
                image_directory = PurePosixPath("images") if split == "all" else PurePosixPath("images") / split
                for item in items:
                    source = context.repository.image_path(context.project.id, item.image.storage_name)
                    archive.write(source, str(image_directory / item.portable_name))

                annotation_name = (
                    "annotations/instances.json" if split == "all" else f"annotations/instances_{split}.json"
                )
                archive.writestr(
                    annotation_name,
                    json.dumps(
                        self._payload(context, items, image_directory),
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                )

    @staticmethod
    def _payload(
        context: ExportContext,
        items: list[TrainingImage],
        image_directory: PurePosixPath,
    ) -> dict[str, list[dict[str, object]]]:
        category_ids = {label.id: index for index, label in enumerate(context.project.labels, start=1)}
        images: list[dict[str, object]] = []
        annotations: list[dict[str, object]] = []
        annotation_id = 1
        for image_id, item in enumerate(items, start=1):
            images.append(
                {
                    "id": image_id,
                    "file_name": str(image_directory / item.portable_name),
                    "width": item.image.width,
                    "height": item.image.height,
                }
            )
            for annotation in item.annotations:
                if annotation.final_box is None:
                    raise ValueError("Training annotation is missing its final box")
                box = annotation.final_box
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_ids[annotation.label_id],
                        "bbox": [box.x, box.y, box.width, box.height],
                        "area": box.area,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

        categories = [{"id": category_ids[label.id], "name": label.name} for label in context.project.labels]
        return {"images": images, "annotations": annotations, "categories": categories}
