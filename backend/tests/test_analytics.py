from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.analytics.models import AnalyticsFilters, ExampleKind
from app.analytics.service import AnalyticsService
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
from app.exporting.models import ExportFormat, ExportOptions
from app.exporting.service import ExportService
from app.main import create_app
from app.persistence.repository import ProjectRepository


def fixed_uuid(number: int) -> UUID:
    return UUID(int=number)


def ai_annotation(
    number: int,
    label_id: UUID,
    state: VerificationState,
    *,
    prompt: str,
    provider: str = "nvidia",
    model: str = "nvidia/LocateAnything-3B",
    confidence: float = 0.8,
    final_box: BoundingBox | None = None,
) -> Annotation:
    original = BoundingBox(x=20, y=20, width=100, height=100)
    if state is VerificationState.REJECTED:
        verified = None
    elif final_box is not None:
        verified = final_box
    else:
        verified = original
    return Annotation(
        id=fixed_uuid(100 + number),
        prediction_id=fixed_uuid(200 + number),
        label_id=label_id,
        original_ai_label_id=label_id,
        source=AnnotationSource.AI,
        verification_state=state,
        original_ai_box=original,
        final_box=verified,
        provider=provider,
        model=model,
        prompt=prompt,
        confidence=confidence,
    )


def analytics_project() -> Project:
    person = Label(id=fixed_uuid(1), name="Person", color="#32d6a0")
    bicycle = Label(id=fixed_uuid(2), name="Bicycle", color="#6ba8ff")
    first = ProjectImage(
        id=fixed_uuid(10),
        filename="city-one.jpg",
        storage_name="city-one.jpg",
        media_type="image/jpeg",
        width=400,
        height=300,
        review_state=ImageReviewState.COMPLETE,
        annotations=[
            ai_annotation(1, person.id, VerificationState.ACCEPTED, prompt="person"),
            ai_annotation(
                2,
                person.id,
                VerificationState.REJECTED,
                prompt="person",
                confidence=0.25,
            ),
            ai_annotation(
                3,
                bicycle.id,
                VerificationState.ADJUSTED,
                prompt="bicycle",
                final_box=BoundingBox(x=30, y=30, width=100, height=100),
            ),
            ai_annotation(
                4,
                bicycle.id,
                VerificationState.REJECTED,
                prompt="bicycle",
                confidence=0.2,
            ),
            Annotation(
                id=fixed_uuid(301),
                label_id=bicycle.id,
                source=AnnotationSource.HUMAN,
                verification_state=VerificationState.HUMAN_ADDED,
                final_box=BoundingBox(x=220, y=100, width=40, height=50),
                review_prompt="bicycle",
            ),
        ],
    )
    second = ProjectImage(
        id=fixed_uuid(11),
        filename="city-two.jpg",
        storage_name="city-two.jpg",
        media_type="image/jpeg",
        width=400,
        height=300,
        review_state=ImageReviewState.COMPLETE,
        annotations=[
            ai_annotation(5, person.id, VerificationState.ACCEPTED, prompt="person"),
            ai_annotation(
                6,
                person.id,
                VerificationState.ADJUSTED,
                prompt="person wearing a helmet",
                final_box=BoundingBox(x=22, y=22, width=100, height=100),
            ),
            ai_annotation(
                7,
                person.id,
                VerificationState.REJECTED,
                prompt="person wearing a helmet",
                confidence=0.35,
            ),
            Annotation(
                id=fixed_uuid(302),
                label_id=person.id,
                source=AnnotationSource.HUMAN,
                verification_state=VerificationState.HUMAN_ADDED,
                final_box=BoundingBox(x=250, y=30, width=50, height=100),
                review_prompt="person wearing a helmet",
            ),
            Annotation(
                id=fixed_uuid(303),
                label_id=person.id,
                source=AnnotationSource.HUMAN,
                verification_state=VerificationState.MANUAL,
                final_box=BoundingBox(x=5, y=150, width=25, height=45),
            ),
        ],
    )
    third = ProjectImage(
        id=fixed_uuid(12),
        filename="city-three.jpg",
        storage_name="city-three.jpg",
        media_type="image/jpeg",
        width=400,
        height=300,
        review_state=ImageReviewState.NEEDS_REVIEW,
        annotations=[
            ai_annotation(
                8,
                bicycle.id,
                VerificationState.UNREVIEWED,
                prompt="bicycle",
                provider="mock",
                model="deterministic-layout-v1",
                confidence=0.5,
            ),
            Annotation(
                id=fixed_uuid(304),
                label_id=bicycle.id,
                source=AnnotationSource.HUMAN,
                verification_state=VerificationState.HUMAN_ADDED,
                final_box=BoundingBox(x=150, y=150, width=30, height=30),
            ),
        ],
    )
    return Project(
        id=fixed_uuid(50),
        name="Deterministic trust fixture",
        labels=[person, bicycle],
        images=[first, second, third],
    )


@pytest.fixture
def analytics_repository(tmp_path: Path) -> tuple[ProjectRepository, Project]:
    repository = ProjectRepository(tmp_path / "projects")
    project = repository.save(analytics_project())
    for image in project.images:
        repository.image_path(project.id, image.storage_name).write_bytes(b"fixture")
    return repository, project


def test_overview_uses_reviewed_proposals_as_rate_denominator(
    analytics_repository: tuple[ProjectRepository, Project],
) -> None:
    repository, project = analytics_repository
    summary = AnalyticsService(repository).summarize(project.id)

    assert summary.outcomes.total_ai_proposals == 8
    assert summary.outcomes.reviewed_ai_proposals == 7
    assert summary.outcomes.accepted.count == 2
    assert summary.outcomes.accepted.denominator == 7
    assert summary.outcomes.accepted.rate == pytest.approx(2 / 7 * 100)
    assert summary.outcomes.adjusted.count == 2
    assert summary.outcomes.rejected.count == 3
    assert summary.outcomes.unresolved.count == 1
    assert summary.outcomes.unresolved.denominator == 8
    assert summary.overview.human_added_annotations == 3
    assert summary.overview.attributable_ai_misses == 2
    assert summary.overview.unattributed_human_added == 1
    assert summary.overview.human_intervention.count == 8
    assert summary.overview.human_intervention.denominator == 10
    assert summary.overview.human_intervention.rate == 80
    assert summary.overview.auto_label_trust.rate == pytest.approx(2 / 7 * 100)


def test_iou_statistics_and_verifyvision_interpretation_buckets(
    analytics_repository: tuple[ProjectRepository, Project],
) -> None:
    repository, project = analytics_repository
    statistics = AnalyticsService(repository).summarize(project.id).adjusted_box_iou
    expected = [8100 / 11900, 9604 / 10396]

    assert statistics.count == 2
    assert statistics.mean == pytest.approx(sum(expected) / 2)
    assert statistics.median == pytest.approx(sum(expected) / 2)
    assert statistics.minimum == pytest.approx(expected[0])
    assert statistics.maximum == pytest.approx(expected[1])
    assert {bucket.key: bucket.count for bucket in statistics.buckets} == {
        "major": 0,
        "significant": 1,
        "moderate": 0,
        "minor": 1,
    }


def test_class_prompt_and_provider_grouping(
    analytics_repository: tuple[ProjectRepository, Project],
) -> None:
    repository, project = analytics_repository
    summary = AnalyticsService(repository).summarize(project.id)
    by_class = {item.label_name: item for item in summary.classes}
    by_prompt = {item.prompt: item for item in summary.prompts}
    by_provider = {(item.provider, item.model): item for item in summary.provider_models}

    assert by_class["Person"].total_ai_proposals == 5
    assert by_class["Person"].rejected.count == 2
    assert by_class["Person"].human_added == 1
    assert by_class["Person"].final_verified_annotations == 5
    assert by_class["Bicycle"].total_ai_proposals == 3
    assert by_class["Bicycle"].unresolved == 1
    assert by_class["Bicycle"].human_added == 2
    assert by_class["Bicycle"].small_sample is True

    assert by_prompt["person"].inference_runs == 2
    assert by_prompt["person"].total_ai_proposals == 3
    assert by_prompt["person wearing a helmet"].accepted.count == 0
    assert by_prompt["person wearing a helmet"].attributable_ai_misses == 1
    assert by_prompt["bicycle"].inference_runs == 2

    nvidia = by_provider[("nvidia", "nvidia/LocateAnything-3B")]
    assert nvidia.total_ai_proposals == 7
    assert nvidia.attributable_ai_misses == 2
    assert nvidia.confidence is not None
    assert nvidia.confidence.rejected_mean == pytest.approx((0.25 + 0.2 + 0.35) / 3)
    assert by_provider[("mock", "deterministic-layout-v1")].unresolved == 1


def test_filters_are_exact_and_keep_provider_models_separate(
    analytics_repository: tuple[ProjectRepository, Project],
) -> None:
    repository, project = analytics_repository
    service = AnalyticsService(repository)
    prompt_summary = service.summarize(project.id, AnalyticsFilters(prompt="person wearing a helmet"))
    mock_summary = service.summarize(
        project.id,
        AnalyticsFilters(provider="mock", model="deterministic-layout-v1"),
    )

    assert prompt_summary.outcomes.total_ai_proposals == 2
    assert prompt_summary.outcomes.accepted.count == 0
    assert prompt_summary.overview.human_added_annotations == 1
    assert mock_summary.outcomes.total_ai_proposals == 1
    assert mock_summary.outcomes.unresolved.count == 1
    assert mock_summary.overview.human_added_annotations == 0


def test_dataset_summary_uses_export_eligibility_and_validation(
    analytics_repository: tuple[ProjectRepository, Project],
) -> None:
    repository, project = analytics_repository
    dataset = AnalyticsService(repository).summarize(project.id).dataset

    assert dataset.total_imported_images == 3
    assert dataset.images_with_verified_annotations == 3
    assert dataset.exportable_images == 2
    assert dataset.total_final_training_boxes == 7
    assert dataset.unresolved_annotations == 1
    assert dataset.review_needed_images == 1
    assert dataset.validation.export_ready is True
    assert dataset.validation.warning_count == 2
    assert {item.label_name: item.count for item in dataset.class_distribution} == {
        "Person": 5,
        "Bicycle": 2,
    }


def test_problem_example_queries_return_relevant_images(
    analytics_repository: tuple[ProjectRepository, Project],
) -> None:
    repository, project = analytics_repository
    service = AnalyticsService(repository)

    rejected = service.examples(project.id, ExampleKind.REJECTED)
    low_iou = service.examples(project.id, ExampleKind.LOW_IOU)
    human_added = service.examples(project.id, ExampleKind.HUMAN_ADDED)
    bicycle = service.examples(
        project.id,
        ExampleKind.INTERVENTION,
        AnalyticsFilters(label_id=fixed_uuid(2)),
    )

    assert rejected.total == 3
    assert low_iou.total == 1
    assert low_iou.items[0].label_name == "Bicycle"
    assert human_added.total == 3
    assert sum(item.human_added_is_attributed_ai_miss for item in human_added.items) == 2
    assert bicycle.total == 4
    assert {item.image_filename for item in bicycle.items} == {
        "city-one.jpg",
        "city-three.jpg",
    }


def test_empty_project_response_has_no_nan_or_infinity(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "projects")
    project = repository.save(Project(name="Empty"))
    summary = AnalyticsService(repository).summarize(project.id)
    encoded = json.dumps(summary.model_dump(mode="json"), allow_nan=False)

    assert summary.outcomes.total_ai_proposals == 0
    assert summary.outcomes.accepted.rate == 0
    assert summary.adjusted_box_iou.mean is None
    assert summary.adjusted_box_iou.median is None
    assert '"mean": null' in encoded


def test_analytics_api_schema_for_populated_and_empty_projects(
    tmp_path: Path,
    analytics_repository: tuple[ProjectRepository, Project],
) -> None:
    repository, project = analytics_repository
    app = create_app(
        data_dir=repository.root,
        frontend_dist=tmp_path / "missing-dist",
    )
    client = TestClient(app)

    populated = client.get(f"/api/projects/{project.id}/analytics")
    examples = client.get(
        f"/api/projects/{project.id}/analytics/examples",
        params={"kind": "rejected", "limit": 2},
    )
    empty = client.post("/api/projects", json={"name": "API empty"}).json()
    empty_response = client.get(f"/api/projects/{empty['id']}/analytics")

    assert populated.status_code == 200
    assert populated.json()["outcomes"]["reviewed_ai_proposals"] == 7
    assert populated.json()["outcomes"]["accepted"]["denominator"] == 7
    assert examples.status_code == 200
    assert examples.json()["total"] == 3
    assert len(examples.json()["items"]) == 2
    assert empty_response.status_code == 200
    assert empty_response.json()["adjusted_box_iou"]["mean"] is None
    json.dumps(populated.json(), allow_nan=False)
    json.dumps(empty_response.json(), allow_nan=False)


def test_dashboard_totals_match_evaluation_csv(
    analytics_repository: tuple[ProjectRepository, Project],
) -> None:
    repository, project = analytics_repository
    analytics = AnalyticsService(repository).summarize(project.id)
    export_service = ExportService(repository)
    csv_artifact = export_service.export(project.id, ExportFormat.EVALUATION_CSV, ExportOptions())
    try:
        with csv_artifact.path.open(encoding="utf-8", newline="") as source:
            csv_counts = Counter(row["verification_status"] for row in csv.DictReader(source))

        assert csv_counts["accepted"] == analytics.outcomes.accepted.count
        assert csv_counts["adjusted"] == analytics.outcomes.adjusted.count
        assert csv_counts["rejected"] == analytics.outcomes.rejected.count
        assert csv_counts["human_added"] == analytics.overview.human_added_annotations
    finally:
        csv_artifact.cleanup()
