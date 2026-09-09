from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from statistics import fmean, median
from uuid import UUID

from app.analytics.models import (
    AnalyticsExamples,
    AnalyticsFilterOptions,
    AnalyticsFilters,
    AnalyticsSummary,
    ClassAnalytics,
    ClassDistribution,
    ConfidenceByOutcome,
    DatasetQualitySummary,
    ExampleKind,
    IouBucket,
    IouStatistics,
    LabelFilterOption,
    ModelTrustOverview,
    OutcomeBreakdown,
    ProblemExample,
    PromptAnalytics,
    ProviderModelAnalytics,
    ProviderModelFilterOption,
    ValidationSummary,
)
from app.domain.evaluation import (
    adjusted_annotation_iou,
    ai_outcome_counts,
    calculate_evaluation,
    count_rate,
)
from app.domain.models import (
    Annotation,
    AnnotationSource,
    ImageReviewState,
    Project,
    ProjectImage,
    VerificationState,
)
from app.domain.verification import verified_annotations
from app.exporting.common import is_training_annotation, training_projection
from app.exporting.models import DatasetSplit, ExportOptions, ExportPreview
from app.exporting.validation import ProjectExportValidator
from app.persistence.repository import ProjectRepository

SMALL_SAMPLE_THRESHOLD = 10


class AnalyticsService:
    """Builds project analytics from saved annotation data."""

    def __init__(self, repository: ProjectRepository) -> None:
        self.repository = repository
        self.validator = ProjectExportValidator(repository)

    def summarize(self, project_id: UUID, filters: AnalyticsFilters | None = None) -> AnalyticsSummary:
        project = self.repository.get(project_id)
        active = filters or AnalyticsFilters()
        selected_ai, selected_human = self._selected_annotations(project, active)
        evaluation = calculate_evaluation(project)

        if active == AnalyticsFilters():
            outcomes = OutcomeBreakdown(
                total_ai_proposals=evaluation.total_ai_proposals,
                reviewed_ai_proposals=evaluation.reviewed_ai_proposals,
                accepted=evaluation.accepted,
                adjusted=evaluation.adjusted,
                rejected=evaluation.rejected,
                unresolved=evaluation.unresolved,
            )
        else:
            outcomes = self._outcomes(selected_ai)

        attributed_human = [
            (image, annotation) for image, annotation in selected_human if self._is_attributable_miss(image, annotation)
        ]
        images_in_scope = {image.id for image, _ in [*selected_ai, *selected_human]}
        intervention_count = outcomes.adjusted.count + outcomes.rejected.count + len(selected_human)
        intervention_denominator = outcomes.reviewed_ai_proposals + len(selected_human)

        preview = self.validator.validate(project, ExportOptions(split=DatasetSplit.NONE))
        return AnalyticsSummary(
            overview=ModelTrustOverview(
                total_imported_images=len(project.images),
                reviewed_images=sum(image.review_state is ImageReviewState.COMPLETE for image in project.images),
                images_in_scope=len(images_in_scope),
                images_needing_review=sum(
                    image.review_state is ImageReviewState.NEEDS_REVIEW for image in project.images
                ),
                auto_label_trust=count_rate(outcomes.accepted.count, outcomes.reviewed_ai_proposals),
                human_intervention=count_rate(intervention_count, intervention_denominator),
                human_added_annotations=len(selected_human),
                attributable_ai_misses=len(attributed_human),
                unattributed_human_added=len(selected_human) - len(attributed_human),
            ),
            outcomes=outcomes,
            adjusted_box_iou=self._iou_statistics(selected_ai),
            classes=self._class_metrics(project, selected_ai, selected_human, active),
            prompts=self._prompt_metrics(selected_ai, selected_human),
            provider_models=self._provider_metrics(selected_ai, selected_human),
            dataset=self._dataset_summary(project, preview),
            filter_options=self._filter_options(project),
            active_filters=active,
        )

    def examples(
        self,
        project_id: UUID,
        kind: ExampleKind,
        filters: AnalyticsFilters | None = None,
        *,
        iou_max: float = 0.75,
        limit: int = 25,
    ) -> AnalyticsExamples:
        project = self.repository.get(project_id)
        active = filters or AnalyticsFilters()
        label_names = {label.id: label.name for label in project.labels}
        items: list[ProblemExample] = []

        if kind is ExampleKind.NEEDS_REVIEW:
            for image in project.images:
                if image.review_state is not ImageReviewState.NEEDS_REVIEW:
                    continue
                if active != AnalyticsFilters() and not any(
                    self._annotation_matches(image, annotation, active) for annotation in image.annotations
                ):
                    continue
                items.append(
                    ProblemExample(
                        image_id=image.id,
                        image_filename=image.filename,
                        annotation_id=None,
                        label_id=None,
                        label_name=None,
                        kind=kind,
                        verification_state=None,
                        prompt=None,
                        provider=None,
                        model=None,
                    )
                )
        else:
            for image in project.images:
                for annotation in image.annotations:
                    if not self._annotation_matches(image, annotation, active):
                        continue
                    iou = adjusted_annotation_iou(annotation)
                    if not self._matches_example_kind(annotation, kind, iou, iou_max):
                        continue
                    is_human_added = annotation.verification_state is VerificationState.HUMAN_ADDED
                    provider = annotation.provider
                    model = annotation.model
                    if is_human_added:
                        attribution = self._provider_attribution(image, annotation)
                        provider = attribution[0] if attribution else None
                        model = attribution[1] if attribution else None
                    label_id = (
                        annotation.original_ai_label_id
                        if annotation.source is AnnotationSource.AI
                        else annotation.label_id
                    )
                    items.append(
                        ProblemExample(
                            image_id=image.id,
                            image_filename=image.filename,
                            annotation_id=annotation.id,
                            label_id=label_id,
                            label_name=label_names.get(label_id),
                            kind=kind,
                            verification_state=annotation.verification_state.value,
                            prompt=annotation.prompt or annotation.review_prompt,
                            provider=provider,
                            model=model,
                            iou=iou,
                            human_added_is_attributed_ai_miss=(
                                is_human_added and self._is_attributable_miss(image, annotation)
                            ),
                        )
                    )

        items.sort(
            key=lambda item: (
                item.iou if item.iou is not None else 2,
                item.image_filename.casefold(),
                str(item.annotation_id or ""),
            )
        )
        return AnalyticsExamples(kind=kind, total=len(items), items=items[:limit])

    @staticmethod
    def _outcomes(ai_records: list[tuple[ProjectImage, Annotation]]) -> OutcomeBreakdown:
        annotations = [annotation for _, annotation in ai_records]
        counts = ai_outcome_counts(annotations)
        total = len(annotations)
        reviewed = sum(
            counts[state]
            for state in (
                VerificationState.ACCEPTED,
                VerificationState.ADJUSTED,
                VerificationState.REJECTED,
            )
        )
        return OutcomeBreakdown(
            total_ai_proposals=total,
            reviewed_ai_proposals=reviewed,
            accepted=count_rate(counts[VerificationState.ACCEPTED], reviewed),
            adjusted=count_rate(counts[VerificationState.ADJUSTED], reviewed),
            rejected=count_rate(counts[VerificationState.REJECTED], reviewed),
            unresolved=count_rate(counts[VerificationState.UNREVIEWED], total),
        )

    def _selected_annotations(
        self, project: Project, filters: AnalyticsFilters
    ) -> tuple[list[tuple[ProjectImage, Annotation]], list[tuple[ProjectImage, Annotation]]]:
        ai_records: list[tuple[ProjectImage, Annotation]] = []
        human_added: list[tuple[ProjectImage, Annotation]] = []
        for image in project.images:
            for annotation in image.annotations:
                if not self._annotation_matches(image, annotation, filters):
                    continue
                if annotation.source is AnnotationSource.AI:
                    ai_records.append((image, annotation))
                elif annotation.verification_state is VerificationState.HUMAN_ADDED:
                    human_added.append((image, annotation))
        return ai_records, human_added

    def _annotation_matches(self, image: ProjectImage, annotation: Annotation, filters: AnalyticsFilters) -> bool:
        if annotation.source is AnnotationSource.AI:
            label_id = annotation.original_ai_label_id or annotation.label_id
            return (
                (filters.label_id is None or label_id == filters.label_id)
                and (filters.provider is None or annotation.provider == filters.provider)
                and (filters.model is None or annotation.model == filters.model)
                and (filters.prompt is None or annotation.prompt == filters.prompt)
            )

        if annotation.verification_state is not VerificationState.HUMAN_ADDED:
            return False
        if filters.label_id is not None and annotation.label_id != filters.label_id:
            return False
        if filters.prompt is not None and annotation.review_prompt != filters.prompt:
            return False
        if filters.provider is not None or filters.model is not None:
            attribution = self._provider_attribution(image, annotation)
            if attribution is None:
                return False
            provider, model = attribution
            if filters.provider is not None and provider != filters.provider:
                return False
            if filters.model is not None and model != filters.model:
                return False
        return True

    @staticmethod
    def _is_attributable_miss(image: ProjectImage, annotation: Annotation) -> bool:
        return bool(
            annotation.verification_state is VerificationState.HUMAN_ADDED
            and annotation.review_prompt
            and any(
                candidate.source is AnnotationSource.AI and candidate.prompt == annotation.review_prompt
                for candidate in image.annotations
            )
        )

    @staticmethod
    def _provider_attribution(image: ProjectImage, annotation: Annotation) -> tuple[str, str] | None:
        if not annotation.review_prompt:
            return None
        candidates = {
            (candidate.provider, candidate.model)
            for candidate in image.annotations
            if candidate.source is AnnotationSource.AI
            and candidate.prompt == annotation.review_prompt
            and candidate.provider is not None
            and candidate.model is not None
        }
        if len(candidates) != 1:
            return None
        provider, model = next(iter(candidates))
        return provider, model

    @staticmethod
    def _iou_statistics(
        records: list[tuple[ProjectImage, Annotation]],
    ) -> IouStatistics:
        values = sorted(iou for _, annotation in records if (iou := adjusted_annotation_iou(annotation)) is not None)
        definitions = (
            ("major", "< 0.50", "Major correction", lambda value: value < 0.5),
            (
                "significant",
                "0.50–0.74",
                "Significant adjustment",
                lambda value: 0.5 <= value < 0.75,
            ),
            (
                "moderate",
                "0.75–0.89",
                "Moderate adjustment",
                lambda value: 0.75 <= value < 0.9,
            ),
            ("minor", "0.90–1.00", "Minor adjustment", lambda value: value >= 0.9),
        )
        buckets = []
        for key, range_label, interpretation, predicate in definitions:
            count = sum(predicate(value) for value in values)
            buckets.append(
                IouBucket(
                    key=key,
                    range_label=range_label,
                    interpretation=interpretation,
                    count=count,
                    rate=(count / len(values) * 100) if values else 0,
                )
            )
        return IouStatistics(
            count=len(values),
            mean=fmean(values) if values else None,
            median=median(values) if values else None,
            minimum=values[0] if values else None,
            maximum=values[-1] if values else None,
            buckets=buckets,
        )

    def _class_metrics(
        self,
        project: Project,
        ai_records: list[tuple[ProjectImage, Annotation]],
        human_records: list[tuple[ProjectImage, Annotation]],
        filters: AnalyticsFilters,
    ) -> list[ClassAnalytics]:
        by_label_ai: dict[UUID, list[tuple[ProjectImage, Annotation]]] = defaultdict(list)
        by_label_human: Counter[UUID] = Counter()
        by_label_attributed: Counter[UUID] = Counter()
        for record in ai_records:
            annotation = record[1]
            by_label_ai[annotation.original_ai_label_id or annotation.label_id].append(record)
        for image, annotation in human_records:
            by_label_human[annotation.label_id] += 1
            if self._is_attributable_miss(image, annotation):
                by_label_attributed[annotation.label_id] += 1

        final_counts: Counter[UUID] = Counter(
            annotation.label_id for image in project.images for annotation in verified_annotations(image.annotations)
        )
        results = []
        for label in project.labels:
            if filters.label_id is not None and filters.label_id != label.id:
                continue
            records = by_label_ai[label.id]
            human_count = by_label_human[label.id]
            model_cohort_filter = any((filters.provider, filters.model, filters.prompt))
            if model_cohort_filter and not records and not human_count:
                continue
            if not model_cohort_filter and not records and not human_count and not final_counts[label.id]:
                continue
            outcomes = self._outcomes(records)
            ious = [iou for _, annotation in records if (iou := adjusted_annotation_iou(annotation)) is not None]
            results.append(
                ClassAnalytics(
                    label_id=label.id,
                    label_name=label.name,
                    total_ai_proposals=outcomes.total_ai_proposals,
                    reviewed_ai_proposals=outcomes.reviewed_ai_proposals,
                    accepted=outcomes.accepted,
                    adjusted=outcomes.adjusted,
                    rejected=outcomes.rejected,
                    unresolved=outcomes.unresolved.count,
                    human_added=human_count,
                    attributable_ai_misses=by_label_attributed[label.id],
                    mean_adjusted_iou=fmean(ious) if ious else None,
                    final_verified_annotations=final_counts[label.id],
                    small_sample=outcomes.reviewed_ai_proposals < SMALL_SAMPLE_THRESHOLD,
                )
            )
        return sorted(
            results,
            key=lambda item: (
                -item.rejected.rate,
                -item.adjusted.rate,
                item.label_name.casefold(),
            ),
        )

    def _prompt_metrics(
        self,
        ai_records: list[tuple[ProjectImage, Annotation]],
        human_records: list[tuple[ProjectImage, Annotation]],
    ) -> list[PromptAnalytics]:
        by_prompt: dict[str, list[tuple[ProjectImage, Annotation]]] = defaultdict(list)
        for record in ai_records:
            if record[1].prompt:
                by_prompt[record[1].prompt].append(record)
        attributable = Counter(
            annotation.review_prompt
            for image, annotation in human_records
            if self._is_attributable_miss(image, annotation) and annotation.review_prompt
        )

        results = []
        for prompt, records in by_prompt.items():
            outcomes = self._outcomes(records)
            runs = {
                (
                    image.id,
                    annotation.provider,
                    annotation.model,
                    annotation.prompt,
                    annotation.original_ai_label_id,
                )
                for image, annotation in records
            }
            ious = [iou for _, annotation in records if (iou := adjusted_annotation_iou(annotation)) is not None]
            results.append(
                PromptAnalytics(
                    prompt=prompt,
                    inference_runs=len(runs),
                    total_ai_proposals=outcomes.total_ai_proposals,
                    reviewed_ai_proposals=outcomes.reviewed_ai_proposals,
                    accepted=outcomes.accepted,
                    adjusted=outcomes.adjusted,
                    rejected=outcomes.rejected,
                    unresolved=outcomes.unresolved.count,
                    attributable_ai_misses=attributable[prompt],
                    mean_adjusted_iou=fmean(ious) if ious else None,
                    small_sample=outcomes.reviewed_ai_proposals < SMALL_SAMPLE_THRESHOLD,
                )
            )
        return sorted(results, key=lambda item: (-item.rejected.rate, item.prompt.casefold()))

    def _provider_metrics(
        self,
        ai_records: list[tuple[ProjectImage, Annotation]],
        human_records: list[tuple[ProjectImage, Annotation]],
    ) -> list[ProviderModelAnalytics]:
        grouped: dict[tuple[str, str], list[tuple[ProjectImage, Annotation]]] = defaultdict(list)
        for record in ai_records:
            annotation = record[1]
            provider_key = annotation.provider or "unknown"
            model_key = annotation.model or "unknown"
            grouped[(provider_key, model_key)].append(record)
        attributed = Counter(
            attribution
            for image, annotation in human_records
            if (attribution := self._provider_attribution(image, annotation)) is not None
        )
        results = []
        for (provider, model), records in grouped.items():
            outcomes = self._outcomes(records)
            ious = [iou for _, annotation in records if (iou := adjusted_annotation_iou(annotation)) is not None]
            confidence = self._confidence_summary(annotation for _, annotation in records)
            results.append(
                ProviderModelAnalytics(
                    provider=provider,
                    model=model,
                    total_ai_proposals=outcomes.total_ai_proposals,
                    reviewed_ai_proposals=outcomes.reviewed_ai_proposals,
                    accepted=outcomes.accepted,
                    adjusted=outcomes.adjusted,
                    rejected=outcomes.rejected,
                    unresolved=outcomes.unresolved.count,
                    attributable_ai_misses=attributed[(provider, model)],
                    mean_adjusted_iou=fmean(ious) if ious else None,
                    confidence=confidence,
                )
            )
        return sorted(results, key=lambda item: (item.provider.casefold(), item.model.casefold()))

    @staticmethod
    def _confidence_summary(annotations: Iterable[Annotation]) -> ConfidenceByOutcome | None:
        values: dict[VerificationState, list[float]] = defaultdict(list)
        for annotation in annotations:
            if annotation.confidence is not None:
                values[annotation.verification_state].append(annotation.confidence)
        reviewed_values = [
            *values[VerificationState.ACCEPTED],
            *values[VerificationState.ADJUSTED],
            *values[VerificationState.REJECTED],
        ]
        if not reviewed_values:
            return None

        def mean(state: VerificationState) -> float | None:
            return fmean(values[state]) if values[state] else None

        return ConfidenceByOutcome(
            accepted_count=len(values[VerificationState.ACCEPTED]),
            accepted_mean=mean(VerificationState.ACCEPTED),
            adjusted_count=len(values[VerificationState.ADJUSTED]),
            adjusted_mean=mean(VerificationState.ADJUSTED),
            rejected_count=len(values[VerificationState.REJECTED]),
            rejected_mean=mean(VerificationState.REJECTED),
        )

    @staticmethod
    def _dataset_summary(project: Project, preview: ExportPreview) -> DatasetQualitySummary:
        # The validator preview is the source of truth for export eligibility.
        projected = training_projection(project)
        distribution: Counter[UUID] = Counter(
            annotation.label_id for item in projected for annotation in item.annotations
        )
        total_boxes = sum(distribution.values())
        class_distribution = [
            ClassDistribution(
                label_id=label.id,
                label_name=label.name,
                count=distribution[label.id],
                rate=(distribution[label.id] / total_boxes * 100) if total_boxes else 0,
            )
            for label in project.labels
            if distribution[label.id]
        ]
        class_distribution.sort(key=lambda item: (-item.count, item.label_name.casefold()))
        images_with_final = sum(
            any(is_training_annotation(annotation) for annotation in image.annotations) for image in project.images
        )
        return DatasetQualitySummary(
            total_imported_images=len(project.images),
            images_with_verified_annotations=images_with_final,
            images_without_final_annotations=len(project.images) - images_with_final,
            exportable_images=preview.exportable_images,
            total_final_training_boxes=preview.training_boxes,
            total_classes=len(project.labels),
            unresolved_annotations=preview.excluded_unresolved,
            review_needed_images=sum(image.review_state is ImageReviewState.NEEDS_REVIEW for image in project.images),
            class_distribution=class_distribution,
            validation=ValidationSummary(
                export_ready=preview.blocking_error_count == 0,
                blocking_error_count=preview.blocking_error_count,
                warning_count=preview.warning_count,
                findings=preview.findings,
            ),
        )

    @staticmethod
    def _filter_options(project: Project) -> AnalyticsFilterOptions:
        provider_models = {
            (annotation.provider or "unknown", annotation.model or "unknown")
            for image in project.images
            for annotation in image.annotations
            if annotation.source is AnnotationSource.AI
        }
        prompts = {
            annotation.prompt
            for image in project.images
            for annotation in image.annotations
            if annotation.source is AnnotationSource.AI and annotation.prompt
        }
        return AnalyticsFilterOptions(
            labels=[LabelFilterOption(id=label.id, name=label.name) for label in project.labels],
            provider_models=[
                ProviderModelFilterOption(provider=provider, model=model)
                for provider, model in sorted(
                    provider_models, key=lambda item: (item[0].casefold(), item[1].casefold())
                )
            ],
            prompts=sorted(prompts, key=str.casefold),
        )

    @staticmethod
    def _matches_example_kind(
        annotation: Annotation,
        kind: ExampleKind,
        iou: float | None,
        iou_max: float,
    ) -> bool:
        state = annotation.verification_state
        if kind is ExampleKind.REJECTED:
            return state is VerificationState.REJECTED
        if kind is ExampleKind.LOW_IOU:
            return state is VerificationState.ADJUSTED and iou is not None and iou < iou_max
        if kind is ExampleKind.HUMAN_ADDED:
            return state is VerificationState.HUMAN_ADDED
        if kind is ExampleKind.UNRESOLVED:
            return state is VerificationState.UNREVIEWED
        if kind is ExampleKind.INTERVENTION:
            return state in {
                VerificationState.ADJUSTED,
                VerificationState.REJECTED,
                VerificationState.HUMAN_ADDED,
            }
        return False
