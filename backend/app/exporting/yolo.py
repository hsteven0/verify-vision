from __future__ import annotations

from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

from app.exporting.common import yaml_quoted, yolo_coordinates
from app.exporting.models import ExportContext, ExportFormat, TrainingImage


class YoloExporter:
    format = ExportFormat.YOLO
    extension = ".zip"
    media_type = "application/zip"

    def write(self, context: ExportContext, destination: Path) -> None:
        class_ids = {label.id: index for index, label in enumerate(context.project.labels)}
        with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
            for item in context.training_images:
                image_directory, label_directory = self._directories(item)
                source = context.repository.image_path(context.project.id, item.image.storage_name)
                archive.write(source, str(image_directory / item.portable_name))
                rows = []
                for annotation in item.annotations:
                    if annotation.final_box is None:
                        raise ValueError("Training annotation is missing its final box")
                    coordinates = yolo_coordinates(annotation.final_box, item.image.width, item.image.height)
                    rows.append(
                        f"{class_ids[annotation.label_id]} " + " ".join(f"{value:.6f}" for value in coordinates)
                    )
                label_name = f"{Path(item.portable_name).stem}.txt"
                content = "\n".join(rows)
                if content:
                    content += "\n"
                archive.writestr(str(label_directory / label_name), content)
            archive.writestr("data.yaml", self._dataset_yaml(context))

    @staticmethod
    def _directories(item: TrainingImage) -> tuple[PurePosixPath, PurePosixPath]:
        if item.split == "all":
            return PurePosixPath("images"), PurePosixPath("labels")
        return (
            PurePosixPath("images") / item.split,
            PurePosixPath("labels") / item.split,
        )

    @staticmethod
    def _dataset_yaml(context: ExportContext) -> str:
        lines = ["path: ."]
        if any(item.split != "all" for item in context.training_images):
            lines.extend(["train: images/train", "val: images/val"])
        else:
            lines.append("train: images")
        lines.extend([f"nc: {len(context.project.labels)}", "names:"])
        lines.extend(f"  {index}: {yaml_quoted(label.name)}" for index, label in enumerate(context.project.labels))
        return "\n".join(lines) + "\n"
