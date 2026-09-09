from __future__ import annotations

from time import perf_counter
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, FastAPI, File, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from app.analytics.models import (
    AnalyticsExamples,
    AnalyticsFilters,
    AnalyticsSummary,
    ExampleKind,
)
from app.analytics.service import AnalyticsService
from app.api.schemas import (
    AnnotationCreate,
    AnnotationLabelAssign,
    AnnotationUpdate,
    CapabilitiesResponse,
    CapabilityFeatures,
    CapabilityInference,
    CapabilityLimits,
    HealthResponse,
    ImageReviewUpdate,
    LabelCreate,
    LabelUpdate,
    ProjectCreate,
    ProjectHistoryRestore,
    ProposalCreate,
    VerificationUpdate,
)
from app.dependencies.models import DependencyUpdateStatus
from app.domain.evaluation import EvaluationSummary, calculate_evaluation
from app.domain.models import Project, ProjectSummary
from app.exporting.common import safe_filename_component
from app.exporting.models import DatasetSplit, ExportFormat, ExportOptions, ExportPreview
from app.exporting.service import ExportService, ExportValidationError
from app.inference.provider import describe_provider
from app.persistence.repository import CorruptProjectError, ProjectNotFoundError
from app.runtime import ApplicationRuntime, ServiceBundle
from app.services.projects import (
    MAX_IMAGE_BYTES,
    ImageUpload,
    InferenceFailedError,
    InferenceInputError,
    InferenceUnavailableError,
    InvalidImageError,
    ProjectConflictError,
    ProjectService,
    ResourceNotFoundError,
)
from app.training.adapter import TrainingDependencyUnavailableError
from app.training.models import (
    PredictionPreview,
    TrainingAvailability,
    TrainingConfig,
    TrainingReadiness,
    TrainingRun,
)
from app.training.repository import CorruptTrainingRunError, TrainingRunNotFoundError
from app.training.service import (
    TrainingConflictError,
    TrainingPredictionError,
    TrainingService,
    TrainingValidationError,
)

router = APIRouter(prefix="/api")

SERVICE_ERRORS = (
    ProjectNotFoundError,
    ResourceNotFoundError,
    ProjectConflictError,
    InvalidImageError,
    InferenceUnavailableError,
    InferenceInputError,
    InferenceFailedError,
    ExportValidationError,
    TrainingRunNotFoundError,
    TrainingConflictError,
    TrainingValidationError,
    TrainingDependencyUnavailableError,
    TrainingPredictionError,
    CorruptTrainingRunError,
    CorruptProjectError,
)


def services_from(request: Request) -> ServiceBundle:
    runtime: ApplicationRuntime = request.app.state.runtime
    return runtime.services()


def service_from(request: Request) -> ProjectService:
    return services_from(request).projects


def export_service_from(request: Request) -> ExportService:
    return services_from(request).exports


def analytics_service_from(request: Request) -> AnalyticsService:
    return services_from(request).analytics


def training_service_from(request: Request) -> TrainingService:
    return services_from(request).training


async def handle_service_error(_request: Request, error: Exception) -> JSONResponse:
    if isinstance(error, (ProjectNotFoundError, ResourceNotFoundError, TrainingRunNotFoundError)):
        code = status.HTTP_404_NOT_FOUND
    elif isinstance(
        error,
        (ProjectConflictError, ExportValidationError, TrainingConflictError, TrainingValidationError),
    ):
        code = status.HTTP_409_CONFLICT
    elif isinstance(error, (InvalidImageError, InferenceInputError, TrainingPredictionError)):
        code = status.HTTP_422_UNPROCESSABLE_CONTENT
    elif isinstance(error, (InferenceUnavailableError, TrainingDependencyUnavailableError)):
        code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(error, InferenceFailedError):
        code = status.HTTP_502_BAD_GATEWAY
    elif isinstance(error, CorruptTrainingRunError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Training run data is invalid."},
        )
    elif isinstance(error, CorruptProjectError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Project data is invalid."},
        )
    else:
        raise error
    return JSONResponse(status_code=code, content={"detail": str(error)})


def register_service_error_handlers(app: FastAPI) -> None:
    for error_type in SERVICE_ERRORS:
        app.add_exception_handler(error_type, handle_service_error)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    runtime: ApplicationRuntime = request.app.state.runtime
    provider_info = describe_provider(runtime.provider)
    return HealthResponse(
        status="ok",
        api_version="1",
        inference_provider=provider_info.name,
        inference_model=provider_info.model_name,
        inference_device=provider_info.device,
        inference_status=provider_info.runtime_status,
        inference_available=provider_info.available,
        inference_platform=provider_info.platform,
        inference_python_version=provider_info.python_version,
        inference_torch_version=provider_info.torch_version,
        inference_cuda_version=provider_info.cuda_version,
        inference_gpu=provider_info.gpu_name,
        inference_compute_capability=provider_info.compute_capability,
        inference_requested_dtype=provider_info.requested_dtype,
        inference_selected_dtype=provider_info.selected_dtype,
        inference_dtype_reason=provider_info.dtype_reason,
        inference_detail=provider_info.detail,
    )


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities(request: Request) -> CapabilitiesResponse:
    runtime: ApplicationRuntime = request.app.state.runtime
    provider_info = describe_provider(runtime.provider)
    return CapabilitiesResponse(
        features=CapabilityFeatures(
            inference=provider_info.name == "locateanything" and provider_info.available,
            training=True,
            trained_model_prediction=True,
            manual_annotation=True,
            image_uploads=True,
            dataset_export=True,
            analytics=True,
        ),
        inference=CapabilityInference(
            model=provider_info.model_name,
            available=provider_info.available,
            device=provider_info.device,
            status=provider_info.runtime_status,
            disclosure=(
                "Live local inference through NVIDIA LocateAnything-3B on NVIDIA CUDA."
                if provider_info.available
                else provider_info.detail or "The required local NVIDIA CUDA runtime is unavailable."
            ),
        ),
        limits=CapabilityLimits(
            max_upload_bytes=MAX_IMAGE_BYTES,
            prompt_max_characters=500,
        ),
    )


@router.get("/dependencies/updates", response_model=DependencyUpdateStatus)
def dependency_updates(request: Request) -> DependencyUpdateStatus:
    runtime: ApplicationRuntime = request.app.state.runtime
    return runtime.dependencies.status()


@router.post("/dependencies/updates/check", response_model=DependencyUpdateStatus)
async def check_dependency_updates(request: Request) -> DependencyUpdateStatus:
    runtime: ApplicationRuntime = request.app.state.runtime
    return await run_in_threadpool(runtime.dependencies.check_now)


@router.get("/projects", response_model=list[ProjectSummary])
def list_projects(request: Request) -> list[ProjectSummary]:
    return service_from(request).list_projects()


@router.post("/projects", response_model=Project, status_code=status.HTTP_201_CREATED)
def create_project(data: ProjectCreate, request: Request) -> Project:
    return service_from(request).create_project(data.name)


@router.get("/projects/{project_id}", response_model=Project)
def get_project(project_id: UUID, request: Request) -> Project:
    return service_from(request).get_project(project_id)


@router.get("/projects/{project_id}/evaluation", response_model=EvaluationSummary)
def project_evaluation(project_id: UUID, request: Request) -> EvaluationSummary:
    return calculate_evaluation(service_from(request).get_project(project_id))


@router.get("/projects/{project_id}/analytics", response_model=AnalyticsSummary)
def project_analytics(
    project_id: UUID,
    request: Request,
    label_id: UUID | None = None,
    provider: Annotated[str | None, Query(max_length=120)] = None,
    model: Annotated[str | None, Query(max_length=120)] = None,
    prompt: Annotated[str | None, Query(max_length=500)] = None,
) -> AnalyticsSummary:
    return analytics_service_from(request).summarize(
        project_id,
        AnalyticsFilters(
            label_id=label_id,
            provider=provider,
            model=model,
            prompt=prompt,
        ),
    )


@router.get("/projects/{project_id}/analytics/examples", response_model=AnalyticsExamples)
def project_analytics_examples(
    project_id: UUID,
    request: Request,
    kind: ExampleKind,
    label_id: UUID | None = None,
    provider: Annotated[str | None, Query(max_length=120)] = None,
    model: Annotated[str | None, Query(max_length=120)] = None,
    prompt: Annotated[str | None, Query(max_length=500)] = None,
    iou_max: Annotated[float, Query(ge=0, le=1)] = 0.75,
    limit: Annotated[int, Query(ge=1, le=50)] = 25,
) -> AnalyticsExamples:
    return analytics_service_from(request).examples(
        project_id,
        kind,
        AnalyticsFilters(
            label_id=label_id,
            provider=provider,
            model=model,
            prompt=prompt,
        ),
        iou_max=iou_max,
        limit=limit,
    )


@router.get("/projects/{project_id}/exports/preview", response_model=ExportPreview)
def export_preview(
    project_id: UUID,
    request: Request,
    split: DatasetSplit = DatasetSplit.NONE,
    train_ratio: Annotated[float, Query(gt=0, lt=1)] = 0.8,
    seed: Annotated[int, Query(ge=0, le=2_147_483_647)] = 1337,
) -> ExportPreview:
    return export_service_from(request).preview(
        project_id,
        ExportOptions(split=split, train_ratio=train_ratio, seed=seed),
    )


@router.post("/projects/{project_id}/exports/{export_format}", response_class=FileResponse)
def download_export(
    project_id: UUID,
    export_format: ExportFormat,
    data: ExportOptions,
    request: Request,
) -> FileResponse:
    artifact = export_service_from(request).export(project_id, export_format, data)
    return FileResponse(
        artifact.path,
        media_type=artifact.media_type,
        filename=artifact.filename,
        background=BackgroundTask(artifact.cleanup),
    )


@router.get("/projects/{project_id}/training/availability", response_model=TrainingAvailability)
def training_availability(project_id: UUID, request: Request) -> TrainingAvailability:
    service_from(request).get_project(project_id)
    return training_service_from(request).availability()


@router.post("/projects/{project_id}/training/readiness", response_model=TrainingReadiness)
def training_readiness(project_id: UUID, data: TrainingConfig, request: Request) -> TrainingReadiness:
    return training_service_from(request).readiness(project_id, data)


@router.get("/projects/{project_id}/training/runs", response_model=list[TrainingRun])
def list_training_runs(project_id: UUID, request: Request) -> list[TrainingRun]:
    return training_service_from(request).list_runs(project_id)


@router.post(
    "/projects/{project_id}/training/runs",
    response_model=TrainingRun,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_training_run(project_id: UUID, data: TrainingConfig, request: Request) -> TrainingRun:
    return training_service_from(request).create_run(project_id, data)


@router.get("/projects/{project_id}/training/runs/{run_id}", response_model=TrainingRun)
def get_training_run(project_id: UUID, run_id: UUID, request: Request) -> TrainingRun:
    return training_service_from(request).get_run(project_id, run_id)


@router.post("/projects/{project_id}/training/runs/{run_id}/cancel", response_model=TrainingRun)
def cancel_training_run(project_id: UUID, run_id: UUID, request: Request) -> TrainingRun:
    return training_service_from(request).cancel(project_id, run_id)


@router.get(
    "/projects/{project_id}/training/runs/{run_id}/artifacts/{artifact_kind}",
    response_class=FileResponse,
)
def download_training_artifact(project_id: UUID, run_id: UUID, artifact_kind: str, request: Request) -> FileResponse:
    path, stored_name = training_service_from(request).artifact(project_id, run_id, artifact_kind)
    run = training_service_from(request).get_run(project_id, run_id)
    run_name = safe_filename_component(run.config.run_name, fallback="training-run")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"verifyvision-{run_name}-{stored_name}",
    )


@router.post(
    "/projects/{project_id}/training/runs/{run_id}/predict/images/{image_id}",
    response_model=PredictionPreview,
)
async def predict_project_image(
    project_id: UUID,
    run_id: UUID,
    image_id: UUID,
    request: Request,
    confidence: Annotated[float, Query(ge=0.01, le=1)] = 0.25,
) -> PredictionPreview:
    return await run_in_threadpool(
        training_service_from(request).predict_project_image,
        project_id,
        run_id,
        image_id,
        confidence,
    )


@router.post(
    "/projects/{project_id}/training/runs/{run_id}/predict/upload",
    response_model=PredictionPreview,
)
async def predict_uploaded_image(
    project_id: UUID,
    run_id: UUID,
    request: Request,
    file: Annotated[UploadFile, File(description="A held-out JPEG, PNG, or WebP image")],
    confidence: Annotated[float, Query(ge=0.01, le=1)] = 0.25,
) -> PredictionPreview:
    content = await file.read(MAX_IMAGE_BYTES + 1)
    return await run_in_threadpool(
        training_service_from(request).predict_uploaded_image,
        project_id,
        run_id,
        file.filename or "preview.png",
        content,
        confidence,
    )


@router.get(
    "/projects/{project_id}/training/runs/{run_id}/predictions/{preview_id}/image",
    response_class=FileResponse,
)
def training_prediction_image(project_id: UUID, run_id: UUID, preview_id: UUID, request: Request) -> FileResponse:
    path = training_service_from(request).prediction_image(project_id, run_id, preview_id)
    return FileResponse(path, media_type="image/png", filename="prediction-preview.png")


@router.post("/projects/{project_id}/labels", response_model=Project)
def add_label(project_id: UUID, data: LabelCreate, request: Request) -> Project:
    return service_from(request).add_label(project_id, data)


@router.patch("/projects/{project_id}/labels/{label_id}", response_model=Project)
def rename_label(project_id: UUID, label_id: UUID, data: LabelUpdate, request: Request) -> Project:
    return service_from(request).rename_label(project_id, label_id, data)


@router.delete("/projects/{project_id}/labels/{label_id}", response_model=Project)
def delete_label(project_id: UUID, label_id: UUID, request: Request) -> Project:
    return service_from(request).delete_label(project_id, label_id)


@router.post("/projects/{project_id}/images", response_model=Project)
async def add_images(
    project_id: UUID,
    request: Request,
    files: Annotated[list[UploadFile], File(description="JPEG, PNG, or WebP images")],
) -> Project:
    project_service = service_from(request)
    uploads = [
        ImageUpload(
            filename=file.filename or "",
            content=await file.read(project_service.max_image_bytes + 1),
        )
        for file in files
    ]
    return project_service.add_images(project_id, uploads)


@router.post("/projects/{project_id}/images/{image_id}/proposals", response_model=Project)
async def generate_proposals(
    project_id: UUID,
    image_id: UUID,
    data: ProposalCreate,
    request: Request,
    response: Response,
) -> Project:
    started = perf_counter()
    project = await service_from(request).generate_ai_proposals(project_id, image_id, data.prompt)
    elapsed_ms = (perf_counter() - started) * 1000
    timing = getattr(request.app.state.runtime.provider, "last_timing", None)
    metrics = [f"service;dur={elapsed_ms:.2f}"]
    if (
        timing is not None
        and timing.completed_at >= started
        and timing.image_key
        == next(
            (image.storage_name for image in project.images if image.id == image_id),
            None,
        )
        and timing.prompt == data.prompt
    ):
        metrics.extend(
            (
                f"model-load;dur={timing.model_load_ms:.2f}",
                f"image-decode;dur={timing.image_decode_ms:.2f}",
                f"input-preparation;dur={timing.input_preparation_ms:.2f}",
                f"gpu-inference;dur={timing.gpu_inference_ms:.2f}",
                f"output-parse;dur={timing.output_parse_ms:.2f}",
                f"coordinate-normalization;dur={timing.coordinate_normalization_ms:.2f}",
                f"prediction-build;dur={timing.prediction_build_ms:.2f}",
            )
        )
        response.headers["X-VerifyVision-Model-Reused"] = str(not timing.model_loaded_this_request).lower()
    response.headers["Server-Timing"] = ", ".join(metrics)
    response.headers["X-VerifyVision-Inference-Ms"] = f"{elapsed_ms:.2f}"
    return project


@router.get("/projects/{project_id}/images/{image_id}/content", response_class=FileResponse)
def image_content(project_id: UUID, image_id: UUID, request: Request) -> FileResponse:
    path, image = service_from(request).image_file(project_id, image_id)
    return FileResponse(path, media_type=image.media_type, filename=image.filename)


@router.patch("/projects/{project_id}/images/{image_id}", response_model=Project)
def update_image_review(
    project_id: UUID,
    image_id: UUID,
    data: ImageReviewUpdate,
    request: Request,
) -> Project:
    return service_from(request).set_image_review_state(project_id, image_id, data.review_state)


@router.delete("/projects/{project_id}/images/{image_id}/annotations", response_model=Project)
def clear_image_annotations(project_id: UUID, image_id: UUID, request: Request) -> Project:
    return service_from(request).clear_image_annotations(project_id, image_id)


@router.delete("/projects/{project_id}/images/{image_id}", response_model=Project)
def delete_image(project_id: UUID, image_id: UUID, request: Request) -> Project:
    return service_from(request).delete_image(project_id, image_id)


@router.post(
    "/projects/{project_id}/images/{image_id}/annotations",
    response_model=Project,
    status_code=status.HTTP_201_CREATED,
)
def add_annotation(
    project_id: UUID,
    image_id: UUID,
    data: AnnotationCreate,
    request: Request,
) -> Project:
    return service_from(request).add_manual_annotation(project_id, image_id, data)


@router.patch("/projects/{project_id}/annotations/{annotation_id}", response_model=Project)
def update_annotation(
    project_id: UUID,
    annotation_id: UUID,
    data: AnnotationUpdate,
    request: Request,
) -> Project:
    return service_from(request).update_annotation(project_id, annotation_id, data)


@router.put("/projects/{project_id}/annotations/{annotation_id}/label", response_model=Project)
def assign_annotation_label(
    project_id: UUID,
    annotation_id: UUID,
    data: AnnotationLabelAssign,
    request: Request,
) -> Project:
    return service_from(request).assign_annotation_label(project_id, annotation_id, data)


@router.post("/projects/{project_id}/annotations/{annotation_id}/verify", response_model=Project)
def verify_annotation(
    project_id: UUID,
    annotation_id: UUID,
    data: VerificationUpdate,
    request: Request,
) -> Project:
    return service_from(request).verify_annotation(project_id, annotation_id, data.decision)


@router.delete("/projects/{project_id}/annotations/{annotation_id}", response_model=Project)
def delete_annotation(project_id: UUID, annotation_id: UUID, request: Request) -> Project:
    return service_from(request).delete_annotation(project_id, annotation_id)


@router.put("/projects/{project_id}/history-state", response_model=Project)
def restore_history_state(
    project_id: UUID,
    data: ProjectHistoryRestore,
    request: Request,
) -> Project:
    return service_from(request).restore_history_state(project_id, data)
