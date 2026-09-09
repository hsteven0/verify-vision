from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

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
from app.exporting.common import voc_coordinates, yolo_coordinates
from app.exporting.models import DatasetSplit, ExportFormat, ExportOptions
from app.exporting.service import ExportService, ExportValidationError
from app.exporting.validation import ProjectExportValidator
from app.main import create_app
from app.persistence.repository import ProjectRepository


@dataclass(frozen=True)
class ExportSample:
    repository: ProjectRepository
    project: Project
    vehicle: Label
    person: Label
    accepted: Annotation
    adjusted: Annotation
    rejected: Annotation
    unresolved: Annotation
    human_added: Annotation
    manual: Annotation


@pytest.fixture
def export_sample(tmp_path: Path) -> ExportSample:
    repository = ProjectRepository(tmp_path / "projects")
    vehicle = Label(name="Vehicle")
    person = Label(name="Person")
    accepted_box = BoundingBox(x=10, y=20, width=40, height=20)
    accepted = Annotation(
        prediction_id=uuid4(),
        label_id=person.id,
        original_ai_label_id=person.id,
        source=AnnotationSource.AI,
        verification_state=VerificationState.ACCEPTED,
        original_ai_box=accepted_box,
        final_box=accepted_box,
        provider="locateanything",
        model="nvidia/LocateAnything-3B",
        prompt="person, wearing a helmet",
        confidence=None,
    )
    adjusted = Annotation(
        prediction_id=uuid4(),
        label_id=vehicle.id,
        original_ai_label_id=person.id,
        source=AnnotationSource.AI,
        verification_state=VerificationState.ADJUSTED,
        original_ai_box=BoundingBox(x=15, y=15, width=30, height=30),
        final_box=BoundingBox(x=20, y=10, width=20, height=30),
        provider="locateanything",
        model="nvidia/LocateAnything-3B",
        prompt="rider",
        confidence=0.71,
        note="Changed class and geometry",
    )
    rejected = Annotation(
        prediction_id=uuid4(),
        label_id=vehicle.id,
        original_ai_label_id=vehicle.id,
        source=AnnotationSource.AI,
        verification_state=VerificationState.REJECTED,
        original_ai_box=BoundingBox(x=60, y=50, width=30, height=30),
        final_box=None,
        provider="mock",
        model="deterministic-layout-v1",
        prompt="parked car",
        confidence=0.8,
    )
    unresolved_box = BoundingBox(x=5, y=5, width=10, height=10)
    unresolved = Annotation(
        prediction_id=uuid4(),
        label_id=person.id,
        original_ai_label_id=person.id,
        source=AnnotationSource.AI,
        verification_state=VerificationState.UNREVIEWED,
        original_ai_box=unresolved_box,
        final_box=unresolved_box,
        provider="mock",
        model="deterministic-layout-v1",
        prompt="pedestrian",
        confidence=0.75,
    )
    human_added = Annotation(
        label_id=person.id,
        source=AnnotationSource.HUMAN,
        verification_state=VerificationState.HUMAN_ADDED,
        final_box=BoundingBox(x=50, y=20, width=50, height=40),
        review_prompt="all people",
        note="AI missed this person, near the curb",
    )
    manual = Annotation(
        label_id=vehicle.id,
        source=AnnotationSource.HUMAN,
        verification_state=VerificationState.MANUAL,
        final_box=BoundingBox(x=10, y=10, width=20, height=20),
    )
    review_excluded_box = BoundingBox(x=5, y=5, width=20, height=20)
    review_excluded = Annotation(
        prediction_id=uuid4(),
        label_id=vehicle.id,
        original_ai_label_id=vehicle.id,
        source=AnnotationSource.AI,
        verification_state=VerificationState.ACCEPTED,
        original_ai_box=review_excluded_box,
        final_box=review_excluded_box,
        provider="mock",
        model="deterministic-layout-v1",
        prompt="vehicle",
    )
    project = Project(
        name="Road review",
        labels=[vehicle, person],
        images=[
            ProjectImage(
                filename="street one.png",
                storage_name="street-one.png",
                media_type="image/png",
                width=100,
                height=100,
                review_state=ImageReviewState.IN_PROGRESS,
                annotations=[accepted, adjusted, rejected, unresolved],
            ),
            ProjectImage(
                filename="street two.png",
                storage_name="street-two.png",
                media_type="image/png",
                width=200,
                height=100,
                review_state=ImageReviewState.COMPLETE,
                annotations=[human_added, manual],
            ),
            ProjectImage(
                filename="check again.png",
                storage_name="check-again.png",
                media_type="image/png",
                width=100,
                height=100,
                review_state=ImageReviewState.NEEDS_REVIEW,
                annotations=[review_excluded],
            ),
        ],
    )
    repository.save(project)
    for index, image in enumerate(project.images, start=1):
        repository.image_path(project.id, image.storage_name).write_bytes(f"original-image-{index}".encode())
    return ExportSample(
        repository=repository,
        project=project,
        vehicle=vehicle,
        person=person,
        accepted=accepted,
        adjusted=adjusted,
        rejected=rejected,
        unresolved=unresolved,
        human_added=human_added,
        manual=manual,
    )


def test_yolo_coordinate_conversion_rejects_invalid_geometry() -> None:
    box = BoundingBox(x=10, y=20, width=40, height=20)
    assert yolo_coordinates(box, 100, 100) == pytest.approx((0.3, 0.3, 0.4, 0.2))

    with pytest.raises(ValueError, match="inside the image"):
        yolo_coordinates(box, 20, 20)

    zero_area = BoundingBox.model_construct(x=10, y=10, width=0, height=20)
    with pytest.raises(ValueError, match="normalized"):
        yolo_coordinates(zero_area, 100, 100)


def test_pascal_voc_coordinate_conversion_rounds_and_clamps() -> None:
    assert voc_coordinates(BoundingBox(x=10.25, y=20.75, width=40.5, height=20.1), 100, 100) == (
        11,
        21,
        51,
        41,
    )
    assert voc_coordinates(BoundingBox(x=0, y=0, width=100, height=50), 100, 50) == (1, 1, 100, 50)
    assert voc_coordinates(BoundingBox(x=4.7, y=8.2, width=0.2, height=0.3), 100, 100) == (5, 9, 5, 9)
    assert voc_coordinates(BoundingBox(x=99.7, y=20, width=1, height=1), 100, 100) == (100, 21, 100, 21)

    outside = BoundingBox.model_construct(x=101, y=20, width=1, height=1)
    with pytest.raises(ValueError, match="overlap"):
        voc_coordinates(outside, 100, 100)


def test_yolo_export_uses_final_boxes_and_deterministic_class_order(
    export_sample: ExportSample,
) -> None:
    artifact = ExportService(export_sample.repository).export(
        export_sample.project.id, ExportFormat.YOLO, ExportOptions()
    )
    try:
        with ZipFile(artifact.path) as archive:
            names = archive.namelist()
            assert "data.yaml" in names
            assert sum(name.startswith("images/") for name in names) == 2
            assert sum(name.startswith("labels/") for name in names) == 2
            yaml = archive.read("data.yaml").decode()
            assert '0: "Vehicle"' in yaml
            assert '1: "Person"' in yaml
            assert "val:" not in yaml

            first_labels = next(
                name
                for name in names
                if name.startswith("labels/") and str(export_sample.project.images[0].id)[:8] in name
            )
            rows = archive.read(first_labels).decode().splitlines()
            assert rows == [
                "1 0.300000 0.300000 0.400000 0.200000",
                "0 0.300000 0.250000 0.200000 0.300000",
            ]
            assert len(rows) == 2
    finally:
        artifact.cleanup()


def test_yolo_train_validation_split_is_deterministic(export_sample: ExportSample) -> None:
    options = ExportOptions(split=DatasetSplit.TRAIN_VAL, train_ratio=0.8, seed=42)
    service = ExportService(export_sample.repository)
    first = service.export(export_sample.project.id, ExportFormat.YOLO, options)
    second = service.export(export_sample.project.id, ExportFormat.YOLO, options)
    try:
        with ZipFile(first.path) as first_archive, ZipFile(second.path) as second_archive:
            assert first_archive.namelist() == second_archive.namelist()
            names = first_archive.namelist()
            assert sum(name.startswith("images/train/") for name in names) == 1
            assert sum(name.startswith("images/val/") for name in names) == 1
            yaml = first_archive.read("data.yaml").decode()
            assert "train: images/train" in yaml
            assert "val: images/val" in yaml
    finally:
        first.cleanup()
        second.cleanup()


def test_coco_export_uses_final_geometry(
    export_sample: ExportSample,
) -> None:
    artifact = ExportService(export_sample.repository).export(
        export_sample.project.id, ExportFormat.COCO, ExportOptions()
    )
    try:
        with ZipFile(artifact.path) as archive:
            payload = json.loads(archive.read("annotations/instances.json"))
            assert len(payload["images"]) == 2
            assert len(payload["annotations"]) == 4
            assert payload["categories"] == [
                {"id": 1, "name": "Vehicle"},
                {"id": 2, "name": "Person"},
            ]
            assert [item["id"] for item in payload["images"]] == [1, 2]
            assert [item["id"] for item in payload["annotations"]] == [1, 2, 3, 4]
            adjusted = payload["annotations"][1]
            assert adjusted["category_id"] == 1
            assert adjusted["bbox"] == [20.0, 10.0, 20.0, 30.0]
            assert adjusted["area"] == 600.0
            assert adjusted["iscrowd"] == 0
            assert all(item["id"] != str(export_sample.rejected.id) for item in payload["annotations"])
    finally:
        artifact.cleanup()


def test_coco_train_validation_split_writes_separate_annotation_files(
    export_sample: ExportSample,
) -> None:
    artifact = ExportService(export_sample.repository).export(
        export_sample.project.id,
        ExportFormat.COCO,
        ExportOptions(split=DatasetSplit.TRAIN_VAL, seed=9),
    )
    try:
        with ZipFile(artifact.path) as archive:
            names = archive.namelist()
            assert "annotations/instances_train.json" in names
            assert "annotations/instances_val.json" in names
            assert sum(name.startswith("images/train/") for name in names) == 1
            assert sum(name.startswith("images/val/") for name in names) == 1
            train = json.loads(archive.read("annotations/instances_train.json"))
            validation = json.loads(archive.read("annotations/instances_val.json"))
            assert len(train["images"]) == 1
            assert len(validation["images"]) == 1
    finally:
        artifact.cleanup()


def test_pascal_voc_export_uses_final_boxes_and_escapes_labels(
    export_sample: ExportSample,
) -> None:
    special_name = "Person & Rider <adult>"
    project = export_sample.project.model_copy(
        update={
            "labels": [
                export_sample.vehicle,
                export_sample.person.model_copy(update={"name": special_name}),
            ]
        }
    )
    export_sample.repository.save(project)
    artifact = ExportService(export_sample.repository).export(project.id, ExportFormat.PASCAL_VOC, ExportOptions())
    try:
        assert artifact.filename == "verifyvision-road-review-pascal-voc.zip"
        with ZipFile(artifact.path) as archive:
            names = archive.namelist()
            assert sum(name.startswith("JPEGImages/") for name in names) == 2
            assert sum(name.startswith("Annotations/") for name in names) == 2
            assert not any(name.startswith("ImageSets/") for name in names)

            first_xml = next(
                name for name in names if name.startswith("Annotations/") and str(project.images[0].id)[:8] in name
            )
            raw = archive.read(first_xml)
            assert b"Person &amp; Rider &lt;adult&gt;" in raw
            root = ElementTree.fromstring(raw)
            assert root.tag == "annotation"
            assert root.findtext("folder") == "JPEGImages"
            assert root.findtext("filename") == next(
                name.removeprefix("JPEGImages/")
                for name in names
                if name.startswith("JPEGImages/") and str(project.images[0].id)[:8] in name
            )
            assert root.findtext("size/width") == "100"
            assert root.findtext("size/height") == "100"
            assert root.findtext("size/depth") == "3"
            objects = root.findall("object")
            assert [item.findtext("name") for item in objects] == [special_name, "Vehicle"]
            assert [
                tuple(int(item.findtext(f"bndbox/{edge}") or 0) for edge in ("xmin", "ymin", "xmax", "ymax"))
                for item in objects
            ] == [(11, 21, 50, 40), (21, 11, 40, 40)]
    finally:
        artifact.cleanup()


def test_pascal_voc_split_is_deterministic_and_writes_image_sets(export_sample: ExportSample) -> None:
    service = ExportService(export_sample.repository)
    options = ExportOptions(split=DatasetSplit.TRAIN_VAL, seed=9)
    artifact = service.export(
        export_sample.project.id,
        ExportFormat.PASCAL_VOC,
        options,
    )
    repeated = service.export(export_sample.project.id, ExportFormat.PASCAL_VOC, options)
    try:
        with ZipFile(artifact.path) as archive, ZipFile(repeated.path) as repeated_archive:
            names = archive.namelist()
            assert names == repeated_archive.namelist()
            assert all(archive.read(name) == repeated_archive.read(name) for name in names)
            train = archive.read("ImageSets/Main/train.txt").decode().splitlines()
            validation = archive.read("ImageSets/Main/val.txt").decode().splitlines()
            exported_stems = {Path(name).stem for name in names if name.startswith("JPEGImages/")}
            assert len(train) == 1
            assert len(validation) == 1
            assert set(train + validation) == exported_stems
    finally:
        artifact.cleanup()
        repeated.cleanup()


def test_evaluation_csv_preserves_states_empty_values_iou_and_escaping(
    export_sample: ExportSample,
) -> None:
    artifact = ExportService(export_sample.repository).export(
        export_sample.project.id, ExportFormat.EVALUATION_CSV, ExportOptions()
    )
    try:
        raw = artifact.path.read_text(encoding="utf-8")
        assert '"person, wearing a helmet"' in raw
        rows = list(csv.DictReader(StringIO(raw)))
        assert len(rows) == 7
        by_id = {row["annotation_id"]: row for row in rows}
        assert by_id[str(export_sample.rejected.id)]["verified_x1"] == ""
        assert by_id[str(export_sample.unresolved.id)]["verified_x1"] == ""
        assert by_id[str(export_sample.human_added.id)]["original_x1"] == ""
        assert by_id[str(export_sample.human_added.id)]["verified_x1"] == "50"
        assert by_id[str(export_sample.adjusted.id)]["iou"] != ""
        assert by_id[str(export_sample.accepted.id)]["iou"] == "1"
    finally:
        artifact.cleanup()


def test_validation_reports_exclusions_and_missing_images(export_sample: ExportSample) -> None:
    validator = ProjectExportValidator(export_sample.repository)
    preview = validator.validate(export_sample.project, ExportOptions())

    assert preview.blocking_error_count == 0
    assert preview.training_boxes == 4
    assert preview.exportable_images == 2
    assert preview.accepted == 1
    assert preview.adjusted == 1
    assert preview.human_added == 1
    assert preview.manual == 1
    assert preview.excluded_rejected == 1
    assert preview.excluded_unresolved == 1
    assert preview.excluded_needs_review == 1
    assert {finding.code for finding in preview.findings} == {
        "unresolved_predictions",
        "image_needs_review",
    }

    export_sample.repository.image_path(export_sample.project.id, export_sample.project.images[0].storage_name).unlink()
    missing = validator.validate(export_sample.project, ExportOptions())
    assert "missing_image_file" in {finding.code for finding in missing.findings}
    assert missing.blocking_error_count == 1
    with pytest.raises(ExportValidationError, match="missing"):
        ExportService(export_sample.repository).export(export_sample.project.id, ExportFormat.YOLO, ExportOptions())


def test_validation_rejects_invalid_project_data(
    tmp_path: Path,
) -> None:
    repository = ProjectRepository(tmp_path / "projects")
    label = Label(name="Known")
    duplicate_id = uuid4()
    bad_box = BoundingBox.model_construct(x=90, y=90, width=20, height=20)
    missing_box = Annotation.model_construct(
        id=duplicate_id,
        label_id=uuid4(),
        source=AnnotationSource.HUMAN,
        verification_state=VerificationState.HUMAN_ADDED,
        final_box=None,
    )
    outside = Annotation.model_construct(
        id=duplicate_id,
        label_id=label.id,
        source=AnnotationSource.HUMAN,
        verification_state=VerificationState.MANUAL,
        final_box=bad_box,
    )
    zero_area = Annotation.model_construct(
        id=uuid4(),
        label_id=label.id,
        source=AnnotationSource.HUMAN,
        verification_state=VerificationState.MANUAL,
        final_box=BoundingBox.model_construct(x=1, y=1, width=0, height=2),
    )
    ai_missing_final = Annotation.model_construct(
        id=uuid4(),
        prediction_id=uuid4(),
        label_id=label.id,
        original_ai_label_id=uuid4(),
        source=AnnotationSource.AI,
        verification_state=VerificationState.ADJUSTED,
        original_ai_box=BoundingBox(x=1, y=1, width=2, height=2),
        final_box=None,
        provider="mock",
        model="mock-v1",
        prompt="object",
    )
    image_id = uuid4()
    image = ProjectImage.model_construct(
        id=image_id,
        filename="bad.png",
        storage_name="bad.png",
        media_type="image/png",
        width=0,
        height=100,
        review_state=ImageReviewState.COMPLETE,
        annotations=[missing_box, outside, zero_area, ai_missing_final],
    )
    project = Project.model_construct(
        id=uuid4(),
        name="Invalid",
        labels=[label, label],
        images=[image, image],
    )
    repository.image_path(project.id, image.storage_name).write_bytes(b"image")

    preview = ProjectExportValidator(repository).validate(project, ExportOptions())
    codes = {finding.code for finding in preview.findings}

    assert {
        "duplicate_label_id",
        "duplicate_image_id",
        "duplicate_annotation_id",
        "unknown_label",
        "unknown_original_label",
        "missing_final_box",
        "invalid_image_dimensions",
        "invalid_box",
        "box_outside_image",
    }.issubset(codes)
    assert preview.blocking_error_count >= 6


def test_split_requires_two_exportable_images(
    export_sample: ExportSample,
) -> None:
    project = export_sample.project.model_copy(update={"images": [export_sample.project.images[0]]})
    preview = ProjectExportValidator(export_sample.repository).validate(
        project, ExportOptions(split=DatasetSplit.TRAIN_VAL)
    )

    assert "split_requires_two_images" in {finding.code for finding in preview.findings}


def test_export_preview_and_download_api(export_sample: ExportSample) -> None:
    app = create_app(
        data_dir=export_sample.repository.root,
        frontend_dist=export_sample.repository.root / "missing",
    )
    client = TestClient(app)

    preview = client.get(f"/api/projects/{export_sample.project.id}/exports/preview")
    assert preview.status_code == 200
    assert preview.json()["training_boxes"] == 4

    response = client.post(
        f"/api/projects/{export_sample.project.id}/exports/yolo",
        json={"split": "none", "train_ratio": 0.8, "seed": 1337},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "verifyvision-road-review-yolo.zip" in response.headers["content-disposition"]
    with ZipFile(BytesIO(response.content)) as archive:
        assert "data.yaml" in archive.namelist()

    voc_response = client.post(
        f"/api/projects/{export_sample.project.id}/exports/pascal_voc",
        json={"split": "none", "train_ratio": 0.8, "seed": 1337},
    )
    assert voc_response.status_code == 200
    assert "verifyvision-road-review-pascal-voc.zip" in voc_response.headers["content-disposition"]
    with ZipFile(BytesIO(voc_response.content)) as archive:
        assert sum(name.startswith("Annotations/") for name in archive.namelist()) == 2

    removed_response = client.post(
        f"/api/projects/{export_sample.project.id}/exports/verifyvision_json",
        json={"split": "none", "train_ratio": 0.8, "seed": 1337},
    )
    assert removed_response.status_code == 422
