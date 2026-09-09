from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import capabilities, register_service_error_handlers, router
from app.api.schemas import CapabilitiesResponse, PublicHealthResponse
from app.config import AppSettings
from app.inference.factory import create_localization_provider, validate_runtime_provider
from app.inference.provider import VisionLocalizationProvider, describe_provider
from app.runtime import ApplicationRuntime
from app.training.adapter import DetectorTrainingAdapter
from app.version import APP_VERSION

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BACKEND_ROOT.parent
logger = logging.getLogger("uvicorn.error")


def public_health(request: Request) -> PublicHealthResponse:
    return PublicHealthResponse(status="ok", api_version="1")


def create_app(
    data_dir: Path | None = None,
    frontend_dist: Path | None = None,
    localization_provider: VisionLocalizationProvider | None = None,
    training_adapter: DetectorTrainingAdapter | None = None,
    *,
    settings: AppSettings | None = None,
    environment: Mapping[str, str] | None = None,
) -> FastAPI:
    resolved_settings = settings or AppSettings.from_environment(
        environment,
        default_data_dir=BACKEND_ROOT / "data" / "projects",
    )
    if data_dir is not None:
        resolved_settings = replace(resolved_settings, data_dir=data_dir)
    app = FastAPI(
        title="VerifyVision API",
        version=APP_VERSION,
        description="Human verification, evaluation, and annotation API.",
    )

    if localization_provider is not None:
        configured_provider = localization_provider
    else:
        configured_provider = create_localization_provider(environment)
    validate_runtime_provider(configured_provider)
    app.state.runtime = ApplicationRuntime(
        resolved_settings,
        configured_provider,
        training_adapter,
    )

    def log_runtime_configuration() -> None:
        provider_info = describe_provider(configured_provider)
        logger.info("VerifyVision runtime: local only")
        logger.info("Vision provider: %s", provider_info.name)
        logger.info("Model: %s", provider_info.model_name)
        model_revision = getattr(configured_provider, "model_revision", None)
        if model_revision:
            logger.info("Model revision: %s", model_revision)
        if provider_info.platform:
            logger.info("Platform: %s", provider_info.platform)
        if provider_info.python_version:
            logger.info("Python: %s", provider_info.python_version)
        if provider_info.torch_version:
            logger.info("PyTorch: %s", provider_info.torch_version)
        if provider_info.cuda_version:
            logger.info("CUDA runtime: %s", provider_info.cuda_version)
        logger.info("Device: %s", provider_info.device)
        if provider_info.gpu_name:
            logger.info("GPU: %s", provider_info.gpu_name)
        if provider_info.compute_capability:
            logger.info("CUDA capability: %s", provider_info.compute_capability)
        if provider_info.selected_dtype:
            logger.info(
                "Inference dtype: requested=%s selected=%s reason=%s",
                provider_info.requested_dtype,
                provider_info.selected_dtype,
                provider_info.dtype_reason,
            )
        logger.info("Inference available: %s", provider_info.available)
        if provider_info.detail:
            logger.warning("Inference runtime: %s", provider_info.detail)
        if provider_info.runtime_status == "not_loaded":
            logger.info("LocateAnything will load on first inference request")
        if resolved_settings.dependency_update_check_enabled:
            started = app.state.runtime.dependencies.check_if_due_in_background()
            logger.info(
                "Dependency update check: %s",
                "started in background" if started else "cached result is still current",
            )

    app.router.add_event_handler("startup", log_runtime_configuration)
    app.router.add_event_handler("shutdown", app.state.runtime.shutdown)
    register_service_error_handlers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
        expose_headers=["Content-Disposition"],
    )
    app.include_router(router)
    app.add_api_route(
        "/health",
        public_health,
        response_model=PublicHealthResponse,
        methods=["GET"],
    )
    app.add_api_route(
        "/capabilities",
        capabilities,
        response_model=CapabilitiesResponse,
        methods=["GET"],
    )

    resolved_frontend = frontend_dist or REPOSITORY_ROOT / "frontend" / "dist"
    if resolved_frontend.is_dir():
        app.mount("/", StaticFiles(directory=resolved_frontend, html=True), name="frontend")

    return app


app = create_app()
