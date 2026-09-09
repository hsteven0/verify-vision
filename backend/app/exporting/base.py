from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.exporting.models import ExportContext, ExportFormat


class ProjectExporter(Protocol):
    format: ExportFormat
    extension: str
    media_type: str

    def write(self, context: ExportContext, destination: Path) -> None: ...
