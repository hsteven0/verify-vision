from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from uuid import UUID

from pydantic import ValidationError

from app.domain.models import Project, ProjectSummary


class ProjectNotFoundError(Exception):
    pass


class CorruptProjectError(Exception):
    pass


class ProjectRepository:
    """Stores one validated JSON file per project."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def list(self) -> list[ProjectSummary]:
        projects = [self._read(path) for path in self.root.glob("*/project.json")]
        projects.sort(key=lambda project: project.updated_at, reverse=True)
        return [ProjectSummary.from_project(project) for project in projects]

    def get(self, project_id: UUID) -> Project:
        path = self._project_file(project_id)
        if not path.is_file():
            raise ProjectNotFoundError(f"Project {project_id} was not found")
        return self._read(path)

    def save(self, project: Project) -> Project:
        with self._lock:
            # Recheck service changes before saving invalid state.
            project = Project.model_validate(project.model_dump())
            project_dir = self.project_dir(project.id)
            project_dir.mkdir(parents=True, exist_ok=True)
            destination = self._project_file(project.id)
            temporary = destination.with_suffix(".json.tmp")
            payload = project.model_dump_json(indent=2)
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as file:
                    file.write(payload)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return project

    def project_dir(self, project_id: UUID) -> Path:
        return self.root / str(project_id)

    def images_dir(self, project_id: UUID) -> Path:
        directory = self.project_dir(project_id) / "images"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def image_path(self, project_id: UUID, storage_name: str) -> Path:
        candidate = (self.images_dir(project_id) / storage_name).resolve()
        images_root = self.images_dir(project_id).resolve()
        if candidate.parent != images_root:
            raise ValueError("Invalid image storage path")
        return candidate

    def _project_file(self, project_id: UUID) -> Path:
        return self.project_dir(project_id) / "project.json"

    @staticmethod
    def _read(path: Path) -> Project:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            version = payload.get("schema_version", 1)
            if version == 1:
                payload["schema_version"] = 2
                for image in payload.get("images", []):
                    for annotation in image.get("annotations", []):
                        annotation.setdefault("prediction_id", None)
                        annotation.setdefault("original_ai_label_id", None)
                        annotation.setdefault("model", None)
                        annotation.setdefault("review_prompt", None)
            elif version != 2:
                raise ValueError(f"Unsupported project schema version: {version}")
            return Project.model_validate(payload)
        except (OSError, UnicodeError, ValueError, ValidationError) as error:
            raise CorruptProjectError(f"Could not load valid project data from {path}") from error
