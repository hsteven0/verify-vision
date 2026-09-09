from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.domain.models import BoundingBox
from app.exporting.service import ExportValidationError
from app.inference.provider import (
    LocalizationPrediction,
    LocalizationRequest,
    ProviderResponseError,
    ProviderUnavailableError,
)
from app.main import create_app
from app.persistence.repository import CorruptProjectError, ProjectNotFoundError
from app.services.projects import (
    InferenceFailedError,
    InferenceInputError,
    InferenceUnavailableError,
    InvalidImageError,
    ProjectConflictError,
    ResourceNotFoundError,
)
from app.training.adapter import TrainingDependencyUnavailableError
from app.training.repository import CorruptTrainingRunError, TrainingRunNotFoundError
from app.training.service import TrainingConflictError, TrainingPredictionError, TrainingValidationError
from tests.fakes import MockVisionLocalizationProvider


def png_bytes(width: int = 320, height: int = 200) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), color=(40, 55, 70)).save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (ProjectNotFoundError("Project not found"), 404, "Project not found"),
        (ResourceNotFoundError("Annotation not found"), 404, "Annotation not found"),
        (ProjectConflictError("Project already exists"), 409, "Project already exists"),
        (InvalidImageError("Image is invalid"), 422, "Image is invalid"),
        (InferenceInputError("Prompt is invalid"), 422, "Prompt is invalid"),
        (InferenceUnavailableError("CUDA is unavailable"), 503, "CUDA is unavailable"),
        (InferenceFailedError("Inference failed"), 502, "Inference failed"),
        (ExportValidationError("Dataset is not ready"), 409, "Dataset is not ready"),
        (TrainingRunNotFoundError("Run not found"), 404, "Run not found"),
        (TrainingConflictError("Training is active"), 409, "Training is active"),
        (TrainingValidationError("Training data is invalid"), 409, "Training data is invalid"),
        (TrainingDependencyUnavailableError("CUDA is unavailable"), 503, "CUDA is unavailable"),
        (TrainingPredictionError("Checkpoint is missing"), 422, "Checkpoint is missing"),
        (CorruptProjectError("raw detail"), 500, "Project data is invalid."),
        (CorruptTrainingRunError("raw detail"), 500, "Training run data is invalid."),
    ],
)
def test_service_errors_keep_http_contract(
    tmp_path: Path,
    error: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    app = create_app(data_dir=tmp_path, frontend_dist=tmp_path / "missing")

    @app.get("/test-service-error")
    def fail() -> None:
        raise error

    response = TestClient(app).get("/test-service-error")

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}


def test_unexpected_errors_stay_unhandled(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path, frontend_dist=tmp_path / "missing")

    @app.get("/test-unexpected-error")
    def fail() -> None:
        raise RuntimeError("unexpected")

    with pytest.raises(RuntimeError, match="unexpected"):
        TestClient(app).get("/test-unexpected-error")


def test_project_names_are_unique_case_insensitively(tmp_path: Path) -> None:
    client = TestClient(create_app(data_dir=tmp_path, frontend_dist=tmp_path / "missing"))

    created = client.post("/api/projects", json={"name": "  Wildlife   Dataset  "})
    duplicate = client.post("/api/projects", json={"name": "wildlife dataset"})

    assert created.status_code == 201
    assert created.json()["name"] == "Wildlife Dataset"
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == 'A project named "wildlife dataset" already exists'
    assert [item["name"] for item in client.get("/api/projects").json()] == ["Wildlife Dataset"]


def test_manual_annotation_vertical_slice_persists_across_app_restart(tmp_path: Path) -> None:
    client = TestClient(create_app(data_dir=tmp_path, frontend_dist=tmp_path / "missing"))

    project_response = client.post("/api/projects", json={"name": "Street audit"})
    assert project_response.status_code == 201
    project = project_response.json()
    project_id = project["id"]
    label_id = project["labels"][0]["id"]

    upload_response = client.post(
        f"/api/projects/{project_id}/images",
        files=[("files", ("street.png", png_bytes(), "image/png"))],
    )
    assert upload_response.status_code == 200
    image_id = upload_response.json()["images"][0]["id"]

    annotation_response = client.post(
        f"/api/projects/{project_id}/images/{image_id}/annotations",
        json={
            "label_id": label_id,
            "box": {"x": 12, "y": 20, "width": 90, "height": 70},
        },
    )
    assert annotation_response.status_code == 201
    annotation = annotation_response.json()["images"][0]["annotations"][0]
    assert annotation["source"] == "human"
    assert annotation["verification_state"] == "manual"

    restarted = TestClient(create_app(data_dir=tmp_path, frontend_dist=tmp_path / "missing"))
    persisted = restarted.get(f"/api/projects/{project_id}")
    assert persisted.status_code == 200
    assert persisted.json()["images"][0]["annotations"][0]["final_box"]["width"] == 90

    content = restarted.get(f"/api/projects/{project_id}/images/{image_id}/content")
    assert content.status_code == 200
    assert content.headers["content-type"] == "image/png"


def test_upload_rejects_non_image_data_without_mutating_project(tmp_path: Path) -> None:
    client = TestClient(create_app(data_dir=tmp_path, frontend_dist=tmp_path / "missing"))
    project = client.post("/api/projects", json={"name": "Clean project"}).json()

    response = client.post(
        f"/api/projects/{project['id']}/images",
        files=[("files", ("not-image.txt", b"hello", "text/plain"))],
    )

    assert response.status_code == 422
    stored = client.get(f"/api/projects/{project['id']}").json()
    assert stored["images"] == []


def test_clear_annotations_can_be_undone(tmp_path: Path) -> None:
    client = TestClient(create_app(data_dir=tmp_path, frontend_dist=tmp_path / "missing"))
    project = client.post("/api/projects", json={"name": "Clear safely"}).json()
    uploaded = client.post(
        f"/api/projects/{project['id']}/images",
        files=[("files", ("scene.png", png_bytes(), "image/png"))],
    ).json()
    image = uploaded["images"][0]
    populated = client.post(
        f"/api/projects/{project['id']}/images/{image['id']}/annotations",
        json={
            "label_id": project["labels"][0]["id"],
            "box": {"x": 10, "y": 10, "width": 40, "height": 50},
        },
    ).json()
    snapshot = {
        "labels": populated["labels"],
        "images": [
            {
                "id": item["id"],
                "review_state": item["review_state"],
                "annotations": item["annotations"],
            }
            for item in populated["images"]
        ],
    }

    cleared = client.delete(f"/api/projects/{project['id']}/images/{image['id']}/annotations")
    assert cleared.status_code == 200
    assert len(cleared.json()["images"]) == 1
    assert cleared.json()["images"][0]["annotations"] == []
    assert cleared.json()["images"][0]["review_state"] == "not_started"
    restored = client.put(f"/api/projects/{project['id']}/history-state", json=snapshot)
    assert restored.status_code == 200
    assert len(restored.json()["images"][0]["annotations"]) == 1


def test_delete_image_removes_metadata_and_stored_content(tmp_path: Path) -> None:
    client = TestClient(create_app(data_dir=tmp_path, frontend_dist=tmp_path / "missing"))
    project = client.post("/api/projects", json={"name": "Delete image"}).json()
    uploaded = client.post(
        f"/api/projects/{project['id']}/images",
        files=[
            ("files", ("first.png", png_bytes(), "image/png")),
            ("files", ("second.png", png_bytes(), "image/png")),
        ],
    ).json()
    removed = uploaded["images"][0]

    response = client.delete(f"/api/projects/{project['id']}/images/{removed['id']}")

    assert response.status_code == 200
    assert [item["filename"] for item in response.json()["images"]] == ["second.png"]
    assert client.get(f"/api/projects/{project['id']}/images/{removed['id']}/content").status_code == 404


def test_delete_class_only_when_it_is_unused(tmp_path: Path) -> None:
    client = TestClient(create_app(data_dir=tmp_path, frontend_dist=tmp_path / "missing"))
    project = client.post("/api/projects", json={"name": "Classes"}).json()
    unused = client.post(
        f"/api/projects/{project['id']}/labels",
        json={"name": "Unused", "color": "#6ba8ff"},
    ).json()["labels"][1]
    deleted = client.delete(f"/api/projects/{project['id']}/labels/{unused['id']}")
    assert deleted.status_code == 200
    assert [label["name"] for label in deleted.json()["labels"]] == ["Unlabeled"]

    uploaded = client.post(
        f"/api/projects/{project['id']}/images",
        files=[("files", ("scene.png", png_bytes(), "image/png"))],
    ).json()
    image = uploaded["images"][0]
    client.post(
        f"/api/projects/{project['id']}/images/{image['id']}/annotations",
        json={
            "label_id": project["labels"][0]["id"],
            "box": {"x": 10, "y": 10, "width": 40, "height": 50},
        },
    )
    blocked = client.delete(f"/api/projects/{project['id']}/labels/{project['labels'][0]['id']}")
    assert blocked.status_code == 409


def test_health_reports_injected_localization_provider(tmp_path: Path) -> None:
    class StubLocateAnythingProvider:
        name = "locateanything"
        model_name = "nvidia/LocateAnything-3B"

        async def localize(self, request: object) -> list[object]:
            return []

    client = TestClient(
        create_app(
            data_dir=tmp_path,
            frontend_dist=tmp_path / "missing",
            localization_provider=StubLocateAnythingProvider(),
        )
    )

    health = client.get("/api/health").json()
    assert health == {
        "status": "ok",
        "api_version": "1",
        "inference_provider": "locateanything",
        "inference_model": "nvidia/LocateAnything-3B",
        "inference_device": "not_applicable",
        "inference_status": "ready",
        "inference_available": True,
        "inference_platform": None,
        "inference_python_version": None,
        "inference_torch_version": None,
        "inference_cuda_version": None,
        "inference_gpu": None,
        "inference_compute_capability": None,
        "inference_requested_dtype": None,
        "inference_selected_dtype": None,
        "inference_dtype_reason": None,
        "inference_detail": None,
    }


def test_capabilities_require_cuda_inference(
    tmp_path: Path,
) -> None:
    class RuntimeDiagnostics:
        available = True
        platform = "win32"
        python_version = "3.12.10"
        torch_version = "2.12.1+cu126"
        cuda_version = "12.6"
        gpu_name = "NVIDIA GeForce RTX Test"
        detail = None

    class AvailableLocateAnythingProvider:
        name = "locateanything"
        model_name = "nvidia/LocateAnything-3B"
        device = "cuda:0"
        runtime_status = "not_loaded"
        runtime_diagnostics = RuntimeDiagnostics()

        async def localize(self, request: object) -> list[object]:
            return []

    client = TestClient(
        create_app(
            data_dir=tmp_path,
            frontend_dist=tmp_path / "missing",
            localization_provider=AvailableLocateAnythingProvider(),
        )
    )

    capabilities = client.get("/api/capabilities").json()
    health = client.get("/api/health").json()
    assert capabilities["features"]["inference"] is True
    assert capabilities["inference"] == {
        "model": "nvidia/LocateAnything-3B",
        "available": True,
        "device": "cuda:0",
        "status": "not_loaded",
        "disclosure": "Live local inference through NVIDIA LocateAnything-3B on NVIDIA CUDA.",
    }
    assert health["inference_gpu"] == "NVIDIA GeForce RTX Test"
    assert health["inference_available"] is True


@pytest.mark.parametrize(
    ("provider_error", "expected_status"),
    [
        (ProviderUnavailableError("model runtime unavailable"), 503),
        (ProviderResponseError("malformed model response"), 502),
    ],
)
def test_inference_provider_errors_are_safe_api_responses(
    tmp_path: Path, provider_error: Exception, expected_status: int
) -> None:
    class FailingProvider:
        name = "locateanything"
        model_name = "nvidia/LocateAnything-3B"

        async def localize(self, request: object) -> list[object]:
            raise provider_error

    client = TestClient(
        create_app(
            data_dir=tmp_path,
            frontend_dist=tmp_path / "missing",
            localization_provider=FailingProvider(),
        )
    )
    project = client.post("/api/projects", json={"name": "Errors"}).json()
    uploaded = client.post(
        f"/api/projects/{project['id']}/images",
        files=[("files", ("scene.png", png_bytes(), "image/png"))],
    ).json()

    response = client.post(
        f"/api/projects/{project['id']}/images/{uploaded['images'][0]['id']}/proposals",
        json={"prompt": "person", "label_id": project["labels"][0]["id"]},
    )

    assert response.status_code == expected_status
    assert response.json()["detail"] == str(provider_error)
    assert "Traceback" not in response.text


def test_predictions_keep_prompt_and_provenance(tmp_path: Path) -> None:
    class NormalizedProvider:
        name = "locateanything"
        model_name = "nvidia/LocateAnything-3B"
        device = "cuda"
        runtime_status = "ready"
        requests: list[LocalizationRequest] = []

        async def localize(self, request: LocalizationRequest) -> list[LocalizationPrediction]:
            self.requests.append(request)
            return [
                LocalizationPrediction(
                    prediction_id=uuid4(),
                    box=BoundingBox(x=32, y=20, width=64, height=80),
                    confidence=None,
                    label_hint="Person",
                )
            ]

    provider = NormalizedProvider()
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            frontend_dist=tmp_path / "missing",
            localization_provider=provider,
        )
    )
    project = client.post("/api/projects", json={"name": "Provenance"}).json()
    uploaded = client.post(
        f"/api/projects/{project['id']}/images",
        files=[("files", ("scene.png", png_bytes(), "image/png"))],
    ).json()
    prompt = "person  wearing a helmet"

    generated = client.post(
        f"/api/projects/{project['id']}/images/{uploaded['images'][0]['id']}/proposals",
        json={"prompt": prompt, "label_id": project["labels"][0]["id"]},
    )

    assert generated.status_code == 200
    assert "service;dur=" in generated.headers["Server-Timing"]
    assert float(generated.headers["X-VerifyVision-Inference-Ms"]) >= 0
    assert "X-VerifyVision-Model-Reused" not in generated.headers
    annotation = generated.json()["images"][0]["annotations"][0]
    assert annotation["provider"] == "locateanything"
    assert annotation["model"] == "nvidia/LocateAnything-3B"
    assert annotation["prompt"] == prompt
    assert annotation["confidence"] is None
    assert annotation["original_ai_box"] == annotation["final_box"]
    assert len(provider.requests) == 1
    inference_request = provider.requests[0]
    assert inference_request.prompt == prompt
    assert inference_request.image_key == uploaded["images"][0]["storage_name"]
    assert inference_request.image_path.name == inference_request.image_key
    assert inference_request.image_path.is_file()
    assert (inference_request.image_width, inference_request.image_height) == (320, 200)


@pytest.mark.parametrize(
    ("prompt", "expected_label"),
    [
        ("people", "People"),
        ("guitar", "Guitar"),
        ("traffic light", "Traffic Light"),
        ("red car", "Red Car"),
        ("person wearing helmet", "Person Wearing Helmet"),
    ],
)
def test_prompt_creates_human_friendly_proposal_class(tmp_path: Path, prompt: str, expected_label: str) -> None:
    class RecordingProvider:
        name = "locateanything"
        model_name = "nvidia/LocateAnything-3B"
        requests: list[LocalizationRequest] = []

        async def localize(self, request: LocalizationRequest) -> list[LocalizationPrediction]:
            self.requests.append(request)
            return [
                LocalizationPrediction(
                    prediction_id=uuid4(),
                    box=BoundingBox(x=10, y=12, width=40, height=50),
                    confidence=0.8,
                    label_hint=request.label_name,
                )
            ]

    provider = RecordingProvider()
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            frontend_dist=tmp_path / "missing",
            localization_provider=provider,
        )
    )
    project = client.post("/api/projects", json={"name": "Prompt labels"}).json()
    uploaded = client.post(
        f"/api/projects/{project['id']}/images",
        files=[("files", ("scene.png", png_bytes(), "image/png"))],
    ).json()
    image_id = uploaded["images"][0]["id"]

    response = client.post(
        f"/api/projects/{project['id']}/images/{image_id}/proposals",
        json={"prompt": prompt},
    )

    assert response.status_code == 200
    payload = response.json()
    created_label = next(label for label in payload["labels"] if label["name"] == expected_label)
    annotation = payload["images"][0]["annotations"][0]
    assert annotation["label_id"] == created_label["id"]
    assert annotation["original_ai_label_id"] == created_label["id"]
    assert annotation["prompt"] == prompt
    assert provider.requests[0].prompt == prompt
    assert provider.requests[0].label_name == expected_label


def test_prompt_classes_reuse_and_allow_override(tmp_path: Path) -> None:
    class RecordingProvider:
        name = "locateanything"
        model_name = "nvidia/LocateAnything-3B"

        async def localize(self, request: LocalizationRequest) -> list[LocalizationPrediction]:
            return [
                LocalizationPrediction(
                    prediction_id=uuid4(),
                    box=BoundingBox(x=10, y=12, width=40, height=50),
                    confidence=None,
                    label_hint=request.label_name,
                )
            ]

    client = TestClient(
        create_app(
            data_dir=tmp_path,
            frontend_dist=tmp_path / "missing",
            localization_provider=RecordingProvider(),
        )
    )
    project = client.post("/api/projects", json={"name": "Reuse"}).json()
    uploaded = client.post(
        f"/api/projects/{project['id']}/images",
        files=[("files", ("scene.png", png_bytes(), "image/png"))],
    ).json()
    image_id = uploaded["images"][0]["id"]
    endpoint = f"/api/projects/{project['id']}/images/{image_id}/proposals"

    for prompt in ("people", "People", "PEOPLE"):
        response = client.post(endpoint, json={"prompt": prompt})
        assert response.status_code == 200

    generated = response.json()
    people_labels = [label for label in generated["labels"] if label["name"].casefold() == "people"]
    assert len(people_labels) == 1
    annotations = generated["images"][0]["annotations"]
    assert {annotation["prompt"] for annotation in annotations} == {
        "people",
        "People",
        "PEOPLE",
    }
    assert {annotation["label_id"] for annotation in annotations} == {people_labels[0]["id"]}

    first = annotations[0]
    override = client.put(
        f"/api/projects/{project['id']}/annotations/{first['id']}/label",
        json={"name": "Crowd", "color": "#ffbd59"},
    )
    assert override.status_code == 200
    overridden = next(
        annotation for annotation in override.json()["images"][0]["annotations"] if annotation["id"] == first["id"]
    )
    assert overridden["label_id"] != overridden["original_ai_label_id"]
    assert overridden["prompt"] == "people"


def test_history_restore_preserves_ai_provenance(tmp_path: Path) -> None:
    class LocateAnythingBoundary(MockVisionLocalizationProvider):
        name = "locateanything"
        model_name = "nvidia/LocateAnything-3B-test-double"

    client = TestClient(
        create_app(
            data_dir=tmp_path,
            frontend_dist=tmp_path / "missing",
            localization_provider=LocateAnythingBoundary(),
        )
    )
    project = client.post("/api/projects", json={"name": "History"}).json()
    uploaded = client.post(
        f"/api/projects/{project['id']}/images",
        files=[("files", ("scene.png", png_bytes(), "image/png"))],
    ).json()
    image_id = uploaded["images"][0]["id"]
    generated = client.post(
        f"/api/projects/{project['id']}/images/{image_id}/proposals",
        json={"prompt": "guitar"},
    ).json()
    annotation = generated["images"][0]["annotations"][0]
    snapshot = {
        "labels": generated["labels"],
        "images": [
            {
                "id": image["id"],
                "review_state": image["review_state"],
                "annotations": image["annotations"],
            }
            for image in generated["images"]
        ],
    }

    changed = client.post(
        f"/api/projects/{project['id']}/annotations/{annotation['id']}/verify",
        json={"decision": "rejected"},
    )
    assert changed.json()["images"][0]["annotations"][0]["verification_state"] == "rejected"

    restored = client.put(f"/api/projects/{project['id']}/history-state", json=snapshot)
    assert restored.status_code == 200
    restored_annotation = restored.json()["images"][0]["annotations"][0]
    for field in (
        "prediction_id",
        "original_ai_label_id",
        "original_ai_box",
        "provider",
        "model",
        "prompt",
        "confidence",
    ):
        assert restored_annotation[field] == annotation[field]
    assert restored_annotation["verification_state"] == "unreviewed"


def test_zero_detections_succeed_and_cache(tmp_path: Path) -> None:
    class EmptyProvider:
        name = "locateanything"
        model_name = "nvidia/LocateAnything-3B"

        def __init__(self) -> None:
            self.call_count = 0

        async def localize(self, request: object) -> list[LocalizationPrediction]:
            self.call_count += 1
            return []

    provider = EmptyProvider()
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            frontend_dist=tmp_path / "missing",
            localization_provider=provider,
        )
    )
    project = client.post("/api/projects", json={"name": "Empty result"}).json()
    uploaded = client.post(
        f"/api/projects/{project['id']}/images",
        files=[("files", ("scene.png", png_bytes(), "image/png"))],
    ).json()
    endpoint = f"/api/projects/{project['id']}/images/{uploaded['images'][0]['id']}/proposals"
    request = {"prompt": "traffic light", "label_id": project["labels"][0]["id"]}

    first = client.post(endpoint, json=request)
    second = client.post(endpoint, json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["images"][0]["annotations"] == []
    assert provider.call_count == 1


def test_verification_flow_persists_metrics(
    tmp_path: Path,
) -> None:
    class DeterministicLocateAnythingBoundary(MockVisionLocalizationProvider):
        name = "locateanything"
        model_name = "nvidia/LocateAnything-3B-test-double"

    client = TestClient(
        create_app(
            data_dir=tmp_path,
            frontend_dist=tmp_path / "missing",
            localization_provider=DeterministicLocateAnythingBoundary(),
        )
    )
    project = client.post("/api/projects", json={"name": "Verification"}).json()
    project_id = project["id"]
    label_id = project["labels"][0]["id"]
    uploaded = client.post(
        f"/api/projects/{project_id}/images",
        files=[("files", ("scene.png", png_bytes(600, 400), "image/png"))],
    ).json()
    image_id = uploaded["images"][0]["id"]

    generated = client.post(
        f"/api/projects/{project_id}/images/{image_id}/proposals",
        json={"prompt": "all objects", "label_id": label_id},
    )
    assert generated.status_code == 200
    proposals = generated.json()["images"][0]["annotations"]
    assert len(proposals) == 3
    assert all(item["verification_state"] == "unreviewed" for item in proposals)
    assert len({item["prediction_id"] for item in proposals}) == 3

    cached = client.post(
        f"/api/projects/{project_id}/images/{image_id}/proposals",
        json={"prompt": "all objects", "label_id": label_id},
    )
    assert len(cached.json()["images"][0]["annotations"]) == 3

    accepted = client.post(
        f"/api/projects/{project_id}/annotations/{proposals[0]['id']}/verify",
        json={"decision": "accepted"},
    )
    assert accepted.status_code == 200

    original_second = proposals[1]["original_ai_box"]
    corrected = {**original_second, "x": original_second["x"] + 5}
    edited = client.patch(
        f"/api/projects/{project_id}/annotations/{proposals[1]['id']}",
        json={"box": corrected},
    )
    assert edited.status_code == 200
    adjusted = client.post(
        f"/api/projects/{project_id}/annotations/{proposals[1]['id']}/verify",
        json={"decision": "adjusted"},
    )
    adjusted_annotation = adjusted.json()["images"][0]["annotations"][1]
    assert adjusted_annotation["original_ai_box"] == original_second
    assert adjusted_annotation["final_box"] == corrected

    rejected = client.post(
        f"/api/projects/{project_id}/annotations/{proposals[2]['id']}/verify",
        json={"decision": "rejected"},
    )
    rejected_annotation = rejected.json()["images"][0]["annotations"][2]
    assert rejected_annotation["final_box"] is None
    assert rejected_annotation["original_ai_box"] is not None

    human_added = client.post(
        f"/api/projects/{project_id}/images/{image_id}/annotations",
        json={
            "label_id": label_id,
            "box": {"x": 20, "y": 20, "width": 30, "height": 40},
            "verification_state": "human_added",
            "review_prompt": "all objects",
        },
    )
    assert human_added.status_code == 201

    completed = client.patch(
        f"/api/projects/{project_id}/images/{image_id}",
        json={"review_state": "complete"},
    )
    assert completed.status_code == 200

    metrics = client.get(f"/api/projects/{project_id}/evaluation").json()
    assert metrics["total_ai_proposals"] == 3
    assert metrics["accepted"]["count"] == 1
    assert metrics["accepted"]["rate"] == pytest.approx(100 / 3)
    assert metrics["adjusted"]["count"] == 1
    assert metrics["rejected"]["count"] == 1
    assert metrics["human_added_count"] == 1
    assert metrics["average_adjusted_iou"] is not None

    restarted = TestClient(create_app(data_dir=tmp_path, frontend_dist=tmp_path / "missing"))
    persisted = restarted.get(f"/api/projects/{project_id}").json()
    states = [item["verification_state"] for item in persisted["images"][0]["annotations"]]
    assert states == ["accepted", "adjusted", "rejected", "human_added"]
