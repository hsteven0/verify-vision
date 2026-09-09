from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import UUID
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZipFile

from app.exporting.common import voc_coordinates
from app.exporting.models import ExportContext, ExportFormat, TrainingImage


class PascalVocExporter:
    format = ExportFormat.PASCAL_VOC
    extension = ".zip"
    media_type = "application/zip"

    def write(self, context: ExportContext, destination: Path) -> None:
        label_names = {label.id: label.name for label in context.project.labels}
        with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
            for item in context.training_images:
                source = context.repository.image_path(context.project.id, item.image.storage_name)
                archive.write(source, f"JPEGImages/{item.portable_name}")
                annotation_name = f"Annotations/{Path(item.portable_name).stem}.xml"
                archive.writestr(annotation_name, self._xml(item, label_names))

            if any(item.split != "all" for item in context.training_images):
                for split in ("train", "val"):
                    names = [Path(item.portable_name).stem for item in context.training_images if item.split == split]
                    archive.writestr(f"ImageSets/Main/{split}.txt", "\n".join(names) + "\n")

    @classmethod
    def _xml(cls, item: TrainingImage, label_names: dict[UUID, str]) -> bytes:
        root = ElementTree.Element("annotation")
        cls._text(root, "folder", "JPEGImages")
        cls._text(root, "filename", item.portable_name)
        size = ElementTree.SubElement(root, "size")
        cls._text(size, "width", item.image.width)
        cls._text(size, "height", item.image.height)
        cls._text(size, "depth", 3)
        cls._text(root, "segmented", 0)

        for annotation in item.annotations:
            if annotation.final_box is None:
                raise ValueError("Training annotation is missing its final box")
            xmin, ymin, xmax, ymax = voc_coordinates(
                annotation.final_box,
                item.image.width,
                item.image.height,
            )
            object_node = ElementTree.SubElement(root, "object")
            cls._text(object_node, "name", label_names[annotation.label_id])
            cls._text(object_node, "pose", "Unspecified")
            cls._text(object_node, "truncated", 0)
            cls._text(object_node, "difficult", 0)
            box_node = ElementTree.SubElement(object_node, "bndbox")
            for name, value in zip(("xmin", "ymin", "xmax", "ymax"), (xmin, ymin, xmax, ymax), strict=True):
                cls._text(box_node, name, value)

        tree = ElementTree.ElementTree(root)
        ElementTree.indent(tree, space="  ")
        output = BytesIO()
        tree.write(output, encoding="utf-8", xml_declaration=True)
        return output.getvalue() + b"\n"

    @staticmethod
    def _text(parent: ElementTree.Element, name: str, value: object) -> None:
        ElementTree.SubElement(parent, name).text = str(value)
