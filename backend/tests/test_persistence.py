from pathlib import Path

import pytest

from app.domain.models import (
    Annotation,
    AnnotationSource,
    BoundingBox,
    Label,
    Project,
    ProjectImage,
    VerificationState,
)
from app.persistence.repository import CorruptProjectError, ProjectRepository


def test_project_round_trip_uses_versioned_json(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path)
    project = Project(name="Wildlife", labels=[Label(name="Fox")])

    repository.save(project)
    loaded = repository.get(project.id)

    assert loaded == project
    assert loaded.schema_version == 2
    assert repository.list()[0].name == "Wildlife"
    assert not list(tmp_path.rglob("*.tmp"))


def test_corrupt_project_is_reported_instead_of_ignored(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path)
    project = Project(name="Damaged")
    repository.save(project)
    (repository.project_dir(project.id) / "project.json").write_text("{bad json", encoding="utf-8")

    with pytest.raises(CorruptProjectError):
        repository.get(project.id)


def test_ai_provenance_round_trip_and_v1_migration(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path)
    label = Label(name="Fox")
    annotation = Annotation(
        prediction_id=label.id,
        label_id=label.id,
        original_ai_label_id=label.id,
        source=AnnotationSource.AI,
        verification_state=VerificationState.ADJUSTED,
        original_ai_box=BoundingBox(x=10, y=10, width=50, height=50),
        final_box=BoundingBox(x=12, y=11, width=48, height=52),
        provider="mock",
        model="deterministic-layout-v1",
        prompt="fox",
        confidence=0.87,
    )
    project = Project(
        name="Provenance",
        labels=[label],
        images=[
            ProjectImage(
                filename="fox.jpg",
                storage_name="fox.jpg",
                media_type="image/jpeg",
                width=200,
                height=200,
                annotations=[annotation],
            )
        ],
    )
    repository.save(project)

    loaded = repository.get(project.id)
    restored = loaded.images[0].annotations[0]
    assert restored.prediction_id == annotation.prediction_id
    assert restored.original_ai_box == annotation.original_ai_box
    assert restored.final_box == annotation.final_box

    legacy = Project(name="Legacy", labels=[Label(name="Object")])
    repository.save(legacy)
    project_file = repository.project_dir(legacy.id) / "project.json"
    project_file.write_text(
        project_file.read_text(encoding="utf-8").replace('"schema_version": 2', '"schema_version": 1'),
        encoding="utf-8",
    )
    assert repository.get(legacy.id).schema_version == 2
