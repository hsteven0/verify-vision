from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import UUID

from app.exporting.base import ProjectExporter
from app.exporting.coco import CocoExporter
from app.exporting.common import (
    assign_dataset_splits,
    safe_filename_component,
    training_projection,
)
from app.exporting.evaluation_csv import EvaluationCsvExporter
from app.exporting.models import (
    ExportArtifact,
    ExportContext,
    ExportFormat,
    ExportOptions,
    ExportPreview,
)
from app.exporting.pascal_voc import PascalVocExporter
from app.exporting.validation import ProjectExportValidator
from app.exporting.yolo import YoloExporter
from app.persistence.repository import ProjectRepository


class ExportValidationError(Exception):
    pass


class ExportService:
    def __init__(
        self,
        repository: ProjectRepository,
        exporters: tuple[ProjectExporter, ...] | None = None,
    ) -> None:
        self.repository = repository
        configured = exporters or (
            YoloExporter(),
            CocoExporter(),
            PascalVocExporter(),
            EvaluationCsvExporter(),
        )
        self.exporters = {exporter.format: exporter for exporter in configured}
        self.validator = ProjectExportValidator(repository)

    def preview(self, project_id: UUID, options: ExportOptions) -> ExportPreview:
        project = self.repository.get(project_id)
        return self.validator.validate(project, options)

    def export(self, project_id: UUID, export_format: ExportFormat, options: ExportOptions) -> ExportArtifact:
        project = self.repository.get(project_id)
        preview = self.validator.validate(project, options)
        if export_format in {ExportFormat.YOLO, ExportFormat.COCO, ExportFormat.PASCAL_VOC} and (
            preview.blocking_error_count
        ):
            first_error = next(finding for finding in preview.findings if finding.severity.value == "error")
            raise ExportValidationError(
                f"Training export blocked by {preview.blocking_error_count} validation "
                f"error{'' if preview.blocking_error_count == 1 else 's'}: "
                f"{first_error.message}"
            )

        training_images = training_projection(project)
        if export_format in {ExportFormat.YOLO, ExportFormat.COCO, ExportFormat.PASCAL_VOC}:
            training_images = assign_dataset_splits(training_images, options)
        exporter = self.exporters[export_format]
        project_name = safe_filename_component(project.name.casefold(), fallback="project")
        format_name = export_format.value.replace("_", "-")
        filename = f"verifyvision-{project_name}-{format_name}{exporter.extension}"
        directory = Path(tempfile.mkdtemp(prefix="verifyvision-export-"))
        destination = directory / filename
        context = ExportContext(
            project=project,
            repository=self.repository,
            options=options,
            training_images=training_images,
        )
        try:
            exporter.write(context, destination)
        except Exception:
            ExportArtifact(
                path=destination,
                directory=directory,
                filename=filename,
                media_type=exporter.media_type,
            ).cleanup()
            raise
        return ExportArtifact(
            path=destination,
            directory=directory,
            filename=filename,
            media_type=exporter.media_type,
            source_project_updated_at=project.updated_at,
            class_names=tuple(label.name for label in project.labels),
        )
