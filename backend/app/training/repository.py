from __future__ import annotations

import os
from pathlib import Path
from threading import RLock
from uuid import UUID

from pydantic import ValidationError

from app.persistence.repository import ProjectRepository
from app.training.models import TrainingRun


class TrainingRunNotFoundError(Exception):
    pass


class CorruptTrainingRunError(Exception):
    pass


class TrainingRunRepository:
    """Stores run data and generated files under each project."""

    def __init__(self, projects: ProjectRepository) -> None:
        self.projects = projects
        self._lock = RLock()

    def save(self, run: TrainingRun) -> TrainingRun:
        with self._lock:
            run = TrainingRun.model_validate(run.model_dump())
            directory = self.run_dir(run.project_id, run.id)
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / "run.json"
            temporary = destination.with_suffix(".json.tmp")
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as file:
                    file.write(run.model_dump_json(indent=2))
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return run

    def get(self, project_id: UUID, run_id: UUID) -> TrainingRun:
        with self._lock:
            path = self.run_dir(project_id, run_id) / "run.json"
            if not path.is_file():
                raise TrainingRunNotFoundError(f"Training run {run_id} was not found")
            return self._read(path)

    def list(self, project_id: UUID) -> list[TrainingRun]:
        with self._lock:
            runs_root = self.projects.project_dir(project_id) / "training" / "runs"
            if not runs_root.is_dir():
                return []
            runs = [self._read(path) for path in runs_root.glob("*/run.json")]
            return sorted(runs, key=lambda run: run.created_at, reverse=True)

    def list_all(self) -> list[TrainingRun]:
        with self._lock:
            runs = [self._read(path) for path in self.projects.root.glob("*/training/runs/*/run.json")]
            return sorted(runs, key=lambda run: run.created_at, reverse=True)

    def run_dir(self, project_id: UUID, run_id: UUID) -> Path:
        return self.projects.project_dir(project_id) / "training" / "runs" / str(run_id)

    def snapshot_dir(self, project_id: UUID, run_id: UUID) -> Path:
        return self.run_dir(project_id, run_id) / "dataset"

    def artifacts_dir(self, project_id: UUID, run_id: UUID) -> Path:
        return self.run_dir(project_id, run_id) / "artifacts"

    def work_dir(self, project_id: UUID, run_id: UUID) -> Path:
        return self.run_dir(project_id, run_id) / "work"

    def predictions_dir(self, project_id: UUID, run_id: UUID) -> Path:
        return self.run_dir(project_id, run_id) / "predictions"

    @staticmethod
    def _read(path: Path) -> TrainingRun:
        try:
            return TrainingRun.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, ValidationError) as error:
            raise CorruptTrainingRunError(f"Could not load training run metadata from {path}") from error
