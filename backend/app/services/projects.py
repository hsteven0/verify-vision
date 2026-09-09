from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from app.api.schemas import (
    AnnotationCreate,
    AnnotationLabelAssign,
    AnnotationUpdate,
    LabelCreate,
    LabelUpdate,
    ProjectHistoryRestore,
)
from app.domain.models import (
    Annotation,
    AnnotationSource,
    ImageReviewState,
    Label,
    Project,
    ProjectImage,
    ProjectSummary,
    VerificationState,
)
from app.domain.verification import (
    InvalidVerificationTransition,
    VerificationDecision,
    verify_ai_annotation,
)
from app.inference.provider import (
    LocalizationRequest,
    ProviderInferenceError,
    ProviderInputError,
    ProviderResponseError,
    ProviderUnavailableError,
    VisionLocalizationProvider,
)
from app.persistence.repository import ProjectRepository

MAX_IMAGE_BYTES = 25 * 1024 * 1024
SUPPORTED_IMAGE_FORMATS = {
    "JPEG": (".jpg", "image/jpeg"),
    "PNG": (".png", "image/png"),
    "WEBP": (".webp", "image/webp"),
}
DEFAULT_LABEL_COLOR = "#32d6a0"
LABEL_COLORS = ("#32d6a0", "#6ba8ff", "#ffbd59", "#ee7dba", "#a98bff", "#ff796d")


class ProjectConflictError(Exception):
    pass


class InvalidImageError(Exception):
    pass


class ResourceNotFoundError(Exception):
    pass


class InferenceUnavailableError(Exception):
    pass


class InferenceInputError(Exception):
    pass


class InferenceFailedError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ImageUpload:
    filename: str
    content: bytes


@dataclass(frozen=True, slots=True)
class PreparedImage:
    filename: str
    storage_name: str
    media_type: str
    width: int
    height: int
    content: bytes


class ProjectService:
    def __init__(
        self,
        repository: ProjectRepository,
        localization_provider: VisionLocalizationProvider | None = None,
        *,
        max_image_bytes: int = MAX_IMAGE_BYTES,
    ) -> None:
        self.repository = repository
        self.localization_provider = localization_provider
        self.max_image_bytes = max_image_bytes
        self._proposal_lock = asyncio.Lock()
        self._empty_inference_cache: set[tuple[str, str, UUID, str, str]] = set()

    def list_projects(self) -> list[ProjectSummary]:
        return self.repository.list()

    def create_project(self, name: str) -> Project:
        normalized_name = " ".join(name.split())
        if not normalized_name:
            raise ProjectConflictError("Enter a project name")
        projects = self.repository.list()
        if any(project.name.casefold() == normalized_name.casefold() for project in projects):
            raise ProjectConflictError(f'A project named "{normalized_name}" already exists')
        project = Project(
            name=normalized_name,
            labels=[Label(name="Unlabeled", color=DEFAULT_LABEL_COLOR)],
        )
        return self.repository.save(project)

    def get_project(self, project_id: UUID) -> Project:
        return self.repository.get(project_id)

    def add_label(self, project_id: UUID, data: LabelCreate) -> Project:
        project = self.repository.get(project_id)
        normalized_name = " ".join(data.name.split())
        if any(label.name.casefold() == normalized_name.casefold() for label in project.labels):
            raise ProjectConflictError(f'A label named "{normalized_name}" already exists')
        project.labels.append(Label(name=normalized_name, color=data.color))
        return self._save_updated(project)

    def rename_label(self, project_id: UUID, label_id: UUID, data: LabelUpdate) -> Project:
        project = self.repository.get(project_id)
        label = self._require_label(project, label_id)
        normalized_name = " ".join(data.name.split())
        if any(
            candidate.id != label_id and candidate.name.casefold() == normalized_name.casefold()
            for candidate in project.labels
        ):
            raise ProjectConflictError(f'A label named "{normalized_name}" already exists')
        index = project.labels.index(label)
        project.labels[index] = Label(
            id=label.id,
            name=normalized_name,
            color=label.color,
        )
        return self._save_updated(project)

    def delete_label(self, project_id: UUID, label_id: UUID) -> Project:
        project = self.repository.get(project_id)
        label = self._require_label(project, label_id)
        if any(
            annotation.label_id == label_id or annotation.original_ai_label_id == label_id
            for image in project.images
            for annotation in image.annotations
        ):
            raise ProjectConflictError(f'Class "{label.name}" is still used by an annotation and cannot be deleted')
        project.labels.remove(label)
        return self._save_updated(project)

    def assign_annotation_label(
        self,
        project_id: UUID,
        annotation_id: UUID,
        data: AnnotationLabelAssign,
    ) -> Project:
        project = self.repository.get(project_id)
        image, annotation, index = self._find_annotation(project, annotation_id)
        if annotation.verification_state is VerificationState.REJECTED:
            raise ProjectConflictError("Rejected AI proposals cannot be edited")

        normalized_name = " ".join(data.name.split())
        label = next(
            (candidate for candidate in project.labels if candidate.name.casefold() == normalized_name.casefold()),
            None,
        )
        if label is None:
            label = Label(name=normalized_name, color=data.color)
            project.labels.append(label)

        update = annotation.model_dump()
        update["label_id"] = label.id
        if (
            annotation.source is AnnotationSource.AI
            and annotation.verification_state is VerificationState.ACCEPTED
            and label.id != annotation.original_ai_label_id
        ):
            update["verification_state"] = VerificationState.ADJUSTED
        update["updated_at"] = datetime.now(UTC)
        image.annotations[index] = Annotation.model_validate(update)
        return self._save_updated(project)

    def add_images(self, project_id: UUID, uploads: list[ImageUpload]) -> Project:
        if not uploads:
            raise InvalidImageError("Choose at least one image to import")

        project = self.repository.get(project_id)
        prepared = [self._prepare_image(upload) for upload in uploads]
        written_paths: list[Path] = []
        try:
            for image in prepared:
                destination = self.repository.image_path(project.id, image.storage_name)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_bytes(image.content)
                os.replace(temporary, destination)
                written_paths.append(destination)

            project.images.extend(
                ProjectImage(
                    filename=image.filename,
                    storage_name=image.storage_name,
                    media_type=image.media_type,
                    width=image.width,
                    height=image.height,
                )
                for image in prepared
            )
            return self._save_updated(project)
        except Exception:
            for path in written_paths:
                path.unlink(missing_ok=True)
            raise

    def image_file(self, project_id: UUID, image_id: UUID) -> tuple[Path, ProjectImage]:
        project = self.repository.get(project_id)
        image = self._find_image(project, image_id)
        path = self.repository.image_path(project_id, image.storage_name)
        if not path.is_file():
            raise ResourceNotFoundError(f"Image file for {image.filename} is missing")
        return path, image

    def clear_image_annotations(self, project_id: UUID, image_id: UUID) -> Project:
        project = self.repository.get(project_id)
        image = self._find_image(project, image_id)
        image.annotations = []
        image.review_state = ImageReviewState.NOT_STARTED
        return self._save_updated(project)

    def delete_image(self, project_id: UUID, image_id: UUID) -> Project:
        project = self.repository.get(project_id)
        image = self._find_image(project, image_id)
        source = self.repository.image_path(project_id, image.storage_name)
        pending = source.with_name(f".{source.name}.{uuid4().hex}.delete")
        moved = False
        try:
            if source.is_file():
                os.replace(source, pending)
                moved = True
            project.images.remove(image)
            saved = self._save_updated(project)
        except Exception:
            if moved and pending.is_file():
                os.replace(pending, source)
            raise
        pending.unlink(missing_ok=True)
        return saved

    def add_manual_annotation(self, project_id: UUID, image_id: UUID, data: AnnotationCreate) -> Project:
        project = self.repository.get(project_id)
        image = self._find_image(project, image_id)
        self._require_label(project, data.label_id)
        if not data.box.fits_within(image.width, image.height):
            raise ProjectConflictError("Bounding box must stay within the image")

        verification_state = VerificationState(data.verification_state)
        if verification_state is VerificationState.MANUAL and data.review_prompt is not None:
            raise ProjectConflictError("Only human-added annotations can store a review prompt")
        image.annotations.append(
            Annotation(
                label_id=data.label_id,
                source=AnnotationSource.HUMAN,
                verification_state=verification_state,
                final_box=data.box,
                review_prompt=data.review_prompt,
                note=data.note,
            )
        )
        if image.review_state is ImageReviewState.NOT_STARTED:
            image.review_state = ImageReviewState.IN_PROGRESS
        return self._save_updated(project)

    async def generate_ai_proposals(
        self,
        project_id: UUID,
        image_id: UUID,
        prompt: str,
    ) -> Project:
        # Keep the cache check, inference, and save together to block duplicate model work.
        async with self._proposal_lock:
            return await self._generate_ai_proposals(project_id, image_id, prompt)

    async def _generate_ai_proposals(
        self,
        project_id: UUID,
        image_id: UUID,
        prompt: str,
    ) -> Project:
        if self.localization_provider is None:
            raise InferenceUnavailableError("No vision localization provider is configured")

        project = self.repository.get(project_id)
        image = self._find_image(project, image_id)
        if not prompt.strip():
            raise ProjectConflictError("Enter a localization prompt")
        preserved_prompt = prompt
        display_label = self._display_label_from_prompt(prompt)
        display_label_key = display_label.casefold()
        labels_by_id = {label.id: label for label in project.labels}
        inference_key = (
            self.localization_provider.name,
            self.localization_provider.model_name,
            image_id,
            display_label_key,
            preserved_prompt,
        )

        cached = any(
            annotation.source is AnnotationSource.AI
            and annotation.provider == self.localization_provider.name
            and annotation.model == self.localization_provider.model_name
            and annotation.prompt == preserved_prompt
            and annotation.original_ai_label_id in labels_by_id
            and labels_by_id[annotation.original_ai_label_id].name.casefold() == display_label_key
            for annotation in image.annotations
        )
        if cached or inference_key in self._empty_inference_cache:
            return project

        image_path = self.repository.image_path(project_id, image.storage_name)
        try:
            predictions = await self.localization_provider.localize(
                LocalizationRequest(
                    image_key=image.storage_name,
                    image_path=image_path,
                    image_width=image.width,
                    image_height=image.height,
                    prompt=preserved_prompt,
                    label_name=display_label,
                )
            )
        except ProviderUnavailableError as error:
            raise InferenceUnavailableError(str(error)) from error
        except ProviderInputError as error:
            raise InferenceInputError(str(error)) from error
        except (ProviderInferenceError, ProviderResponseError) as error:
            raise InferenceFailedError(str(error)) from error

        # Reload after inference so another process's edits are not overwritten.
        project = self.repository.get(project_id)
        image = self._find_image(project, image_id)
        if not predictions:
            self._empty_inference_cache.add(inference_key)
            return self._save_updated(project)

        label = next(
            (candidate for candidate in project.labels if candidate.name.casefold() == display_label_key),
            None,
        )
        if label is None:
            label = Label(
                name=display_label,
                color=LABEL_COLORS[len(project.labels) % len(LABEL_COLORS)],
            )
            project.labels.append(label)

        existing_predictions = {
            annotation.prediction_id: (image, index, annotation)
            for index, annotation in enumerate(image.annotations)
            if annotation.prediction_id is not None
        }
        for prediction in predictions:
            existing = existing_predictions.get(prediction.prediction_id)
            if existing is not None:
                existing_image, existing_index, existing_annotation = existing
                existing_label = next(
                    (candidate for candidate in project.labels if candidate.id == existing_annotation.label_id),
                    None,
                )
                if (
                    existing_image.id == image.id
                    and existing_annotation.source is AnnotationSource.AI
                    and existing_annotation.verification_state is VerificationState.UNREVIEWED
                    and existing_annotation.prompt == preserved_prompt
                    and existing_annotation.label_id == existing_annotation.original_ai_label_id
                    and existing_label is not None
                    and existing_label.name.casefold() == "object"
                ):
                    migrated = existing_annotation.model_copy(
                        update={
                            "label_id": label.id,
                            "original_ai_label_id": label.id,
                            "updated_at": datetime.now(UTC),
                        }
                    )
                    existing_image.annotations[existing_index] = Annotation.model_validate(migrated.model_dump())
                continue
            if not prediction.box.fits_within(image.width, image.height):
                raise ProjectConflictError("The localization provider returned an invalid box")
            image.annotations.append(
                Annotation(
                    prediction_id=prediction.prediction_id,
                    label_id=label.id,
                    original_ai_label_id=label.id,
                    source=AnnotationSource.AI,
                    verification_state=VerificationState.UNREVIEWED,
                    original_ai_box=prediction.box,
                    final_box=prediction.box,
                    provider=self.localization_provider.name,
                    model=self.localization_provider.model_name,
                    prompt=preserved_prompt,
                    confidence=prediction.confidence,
                )
            )
            existing_predictions[prediction.prediction_id] = (
                image,
                len(image.annotations) - 1,
                image.annotations[-1],
            )

        if image.review_state in {ImageReviewState.NOT_STARTED, ImageReviewState.COMPLETE}:
            image.review_state = ImageReviewState.IN_PROGRESS
        return self._save_updated(project)

    def restore_history_state(
        self,
        project_id: UUID,
        data: ProjectHistoryRestore,
    ) -> Project:
        project = self.repository.get(project_id)
        current_image_ids = [image.id for image in project.images]
        restored_image_ids = [image.id for image in data.images]
        if len(restored_image_ids) != len(set(restored_image_ids)):
            raise ProjectConflictError("Undo history contains duplicate image IDs")
        if set(current_image_ids) != set(restored_image_ids):
            raise ProjectConflictError("Undo history no longer matches this project's imported images")

        state_by_image = {image.id: image for image in data.images}
        payload = project.model_dump()
        payload["labels"] = [label.model_dump() for label in data.labels]
        for image_payload in payload["images"]:
            state = state_by_image[image_payload["id"]]
            image_payload["review_state"] = state.review_state
            image_payload["annotations"] = [annotation.model_dump() for annotation in state.annotations]
        restored = Project.model_validate(payload)
        return self._save_updated(restored)

    def verify_annotation(
        self,
        project_id: UUID,
        annotation_id: UUID,
        decision: VerificationDecision,
    ) -> Project:
        project = self.repository.get(project_id)
        image, annotation, index = self._find_annotation(project, annotation_id)
        try:
            image.annotations[index] = verify_ai_annotation(annotation, decision)
        except InvalidVerificationTransition as error:
            raise ProjectConflictError(str(error)) from error
        return self._save_updated(project)

    def update_annotation(
        self,
        project_id: UUID,
        annotation_id: UUID,
        data: AnnotationUpdate,
    ) -> Project:
        project = self.repository.get(project_id)
        image, annotation, index = self._find_annotation(project, annotation_id)
        if annotation.verification_state is VerificationState.REJECTED:
            raise ProjectConflictError("Rejected AI proposals cannot be edited")
        update: dict[str, object] = annotation.model_dump()
        if data.label_id is not None:
            self._require_label(project, data.label_id)
            update["label_id"] = data.label_id
        if data.box is not None:
            if not data.box.fits_within(image.width, image.height):
                raise ProjectConflictError("Bounding box must stay within the image")
            update["final_box"] = data.box
        if "note" in data.model_fields_set:
            update["note"] = data.note
        if (
            annotation.source is AnnotationSource.AI
            and annotation.verification_state is VerificationState.ACCEPTED
            and (
                update["final_box"] != annotation.original_ai_box
                or update["label_id"] != annotation.original_ai_label_id
            )
        ):
            update["verification_state"] = VerificationState.ADJUSTED
        update["updated_at"] = datetime.now(UTC)
        image.annotations[index] = Annotation.model_validate(update)
        return self._save_updated(project)

    def delete_annotation(self, project_id: UUID, annotation_id: UUID) -> Project:
        project = self.repository.get(project_id)
        image, annotation, _ = self._find_annotation(project, annotation_id)
        if annotation.source is AnnotationSource.AI:
            raise ProjectConflictError("AI proposals must be rejected, not deleted")
        image.annotations.remove(annotation)
        return self._save_updated(project)

    def set_image_review_state(self, project_id: UUID, image_id: UUID, state: ImageReviewState) -> Project:
        project = self.repository.get(project_id)
        image = self._find_image(project, image_id)
        if state is ImageReviewState.COMPLETE and any(
            annotation.verification_state is VerificationState.UNREVIEWED for annotation in image.annotations
        ):
            raise ProjectConflictError("Resolve every AI proposal before marking the image reviewed")
        image.review_state = state
        return self._save_updated(project)

    def _prepare_image(self, upload: ImageUpload) -> PreparedImage:
        if not upload.filename.strip():
            raise InvalidImageError("Every uploaded image must have a filename")
        if not upload.content:
            raise InvalidImageError(f"{upload.filename} is empty")
        if len(upload.content) > self.max_image_bytes:
            limit_mb = self.max_image_bytes // (1024 * 1024)
            raise InvalidImageError(f"{upload.filename} exceeds the {limit_mb} MB upload limit")

        try:
            with Image.open(BytesIO(upload.content)) as opened:
                image_format = opened.format
                if image_format not in SUPPORTED_IMAGE_FORMATS:
                    raise InvalidImageError(f"{upload.filename} is not a supported JPEG, PNG, or WebP image")
                opened.load()
                normalized = ImageOps.exif_transpose(opened)
                extension, media_type = SUPPORTED_IMAGE_FORMATS[image_format]
                if image_format == "JPEG" and normalized.mode not in {"RGB", "L"}:
                    normalized = normalized.convert("RGB")
                output = BytesIO()
                normalized.save(output, format=image_format)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise InvalidImageError(f"{upload.filename} could not be read as an image") from error

        return PreparedImage(
            filename=Path(upload.filename).name,
            storage_name=f"{uuid4()}{extension}",
            media_type=media_type,
            width=normalized.width,
            height=normalized.height,
            content=output.getvalue(),
        )

    @staticmethod
    def _find_image(project: Project, image_id: UUID) -> ProjectImage:
        try:
            return next(image for image in project.images if image.id == image_id)
        except StopIteration as error:
            raise ResourceNotFoundError(f"Image {image_id} was not found") from error

    @staticmethod
    def _find_annotation(project: Project, annotation_id: UUID) -> tuple[ProjectImage, Annotation, int]:
        for image in project.images:
            for index, annotation in enumerate(image.annotations):
                if annotation.id == annotation_id:
                    return image, annotation, index
        raise ResourceNotFoundError(f"Annotation {annotation_id} was not found")

    @staticmethod
    def _require_label(project: Project, label_id: UUID) -> Label:
        try:
            return next(label for label in project.labels if label.id == label_id)
        except StopIteration as error:
            raise ResourceNotFoundError(f"Label {label_id} was not found") from error

    @staticmethod
    def _display_label_from_prompt(prompt: str) -> str:
        words = " ".join(prompt.split()).split(" ")
        return " ".join(
            word.capitalize() if word.islower() or word.isupper() else word[0].upper() + word[1:] for word in words
        )

    def _save_updated(self, project: Project) -> Project:
        project.updated_at = datetime.now(UTC)
        return self.repository.save(project)
