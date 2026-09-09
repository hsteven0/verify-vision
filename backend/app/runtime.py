from __future__ import annotations

from dataclasses import dataclass

from app.analytics.service import AnalyticsService
from app.config import AppSettings
from app.dependencies.service import DependencyUpdateService
from app.exporting.service import ExportService
from app.inference.provider import VisionLocalizationProvider
from app.persistence.repository import ProjectRepository
from app.services.projects import ProjectService
from app.training.adapter import DetectorTrainingAdapter
from app.training.service import TrainingService


@dataclass(slots=True)
class ServiceBundle:
    projects: ProjectService
    exports: ExportService
    analytics: AnalyticsService
    training: TrainingService

    def shutdown(self) -> None:
        self.training.shutdown()


class ApplicationRuntime:
    """Creates the shared local services."""

    def __init__(
        self,
        settings: AppSettings,
        provider: VisionLocalizationProvider,
        training_adapter: DetectorTrainingAdapter | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        repository = ProjectRepository(settings.data_dir)
        projects = ProjectService(repository, provider, max_image_bytes=25 * 1024 * 1024)
        exports = ExportService(repository)
        self._services = ServiceBundle(
            projects=projects,
            exports=exports,
            analytics=AnalyticsService(repository),
            training=TrainingService(repository, export_service=exports, adapter=training_adapter),
        )
        self.dependencies = DependencyUpdateService(
            settings.dependency_update_cache,
            interval_hours=settings.dependency_update_interval_hours,
        )

    def services(self) -> ServiceBundle:
        return self._services

    def shutdown(self) -> None:
        self._services.shutdown()
