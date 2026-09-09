export type UUID = string;

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Label {
  id: UUID;
  name: string;
  color: string;
}

export type AnnotationSource = "human" | "ai";
export type VerificationState = "manual" | "unreviewed" | "accepted" | "adjusted" | "rejected" | "human_added";
export type ImageReviewState = "not_started" | "in_progress" | "complete" | "needs_review";

export interface Annotation {
  id: UUID;
  prediction_id: UUID | null;
  label_id: UUID;
  original_ai_label_id: UUID | null;
  source: AnnotationSource;
  verification_state: VerificationState;
  original_ai_box: BoundingBox | null;
  final_box: BoundingBox | null;
  provider: string | null;
  model: string | null;
  prompt: string | null;
  confidence: number | null;
  review_prompt: string | null;
  note: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProjectImage {
  id: UUID;
  filename: string;
  storage_name: string;
  media_type: string;
  width: number;
  height: number;
  review_state: ImageReviewState;
  annotations: Annotation[];
  imported_at: string;
}

export interface Project {
  schema_version: 2;
  id: UUID;
  name: string;
  labels: Label[];
  images: ProjectImage[];
  created_at: string;
  updated_at: string;
}

export interface ProjectHistoryImageState {
  id: UUID;
  review_state: ImageReviewState;
  annotations: Annotation[];
}

export interface ProjectHistorySnapshot {
  labels: Label[];
  images: ProjectHistoryImageState[];
}

export interface ProjectSummary {
  id: UUID;
  name: string;
  image_count: number;
  completed_image_count: number;
  updated_at: string;
}

export type VerificationDecision = "accepted" | "adjusted" | "rejected";
export type HumanAnnotationState = "manual" | "human_added";

export interface CountRate {
  count: number;
  denominator: number;
  rate: number;
}

export interface ClassEvaluation {
  label_id: UUID;
  label_name: string;
  ai_proposals: number;
  accepted: number;
  adjusted: number;
  rejected: number;
  unresolved: number;
  human_added: number;
}

export interface EvaluationSummary {
  total_ai_proposals: number;
  reviewed_ai_proposals: number;
  accepted: CountRate;
  adjusted: CountRate;
  rejected: CountRate;
  unresolved: CountRate;
  human_added_count: number;
  verified_annotation_count: number;
  average_adjusted_iou: number | null;
  by_class: ClassEvaluation[];
}

export type ProviderRuntimeStatus = "not_loaded" | "loading" | "ready" | "unavailable" | "error";

export interface HealthResponse {
  status: string;
  api_version: string;
  inference_provider: string;
  inference_model: string;
  inference_device: string;
  inference_status: ProviderRuntimeStatus;
  inference_available: boolean;
  inference_platform: string | null;
  inference_python_version: string | null;
  inference_torch_version: string | null;
  inference_cuda_version: string | null;
  inference_gpu: string | null;
  inference_compute_capability: string | null;
  inference_requested_dtype: string | null;
  inference_selected_dtype: string | null;
  inference_dtype_reason: string | null;
  inference_detail: string | null;
}

export interface ApplicationCapabilities {
  features: {
    inference: boolean;
    training: boolean;
    trained_model_prediction: boolean;
    manual_annotation: boolean;
    image_uploads: boolean;
    dataset_export: boolean;
    analytics: boolean;
  };
  inference: {
    model: string;
    available: boolean;
    device: string;
    status: ProviderRuntimeStatus;
    disclosure: string;
  };
  limits: {
    max_upload_bytes: number;
    prompt_max_characters: number;
  };
}

export type DependencyUpdateState = "not_checked" | "checking" | "up_to_date" | "updates_available" | "unavailable";

export interface DependencyUpdate {
  package: string;
  ecosystem: "python" | "frontend";
  risk: "critical_runtime" | "application";
  current_version: string;
  latest_version: string;
  release_url: string;
}

export interface DependencyUpdateStatus {
  state: DependencyUpdateState;
  last_checked_at: string | null;
  next_check_at: string | null;
  packages_checked: number;
  updates: DependencyUpdate[];
  detail: string | null;
}

export type ExportFormat = "yolo" | "coco" | "pascal_voc" | "evaluation_csv";
export type DatasetSplit = "none" | "train_val";

export interface ExportOptions {
  split: DatasetSplit;
  train_ratio: number;
  seed: number;
}

export interface ValidationFinding {
  severity: "error" | "warning";
  code: string;
  message: string;
  image_id: UUID | null;
  annotation_id: UUID | null;
}

export interface ExportPreview {
  total_images: number;
  exportable_images: number;
  train_images: number;
  validation_images: number;
  training_boxes: number;
  classes: number;
  accepted: number;
  adjusted: number;
  human_added: number;
  manual: number;
  excluded_rejected: number;
  excluded_unresolved: number;
  excluded_needs_review: number;
  blocking_error_count: number;
  warning_count: number;
  findings: ValidationFinding[];
}

export interface AnalyticsFilters {
  label_id: UUID | null;
  provider: string | null;
  model: string | null;
  prompt: string | null;
}

export interface OutcomeBreakdown {
  total_ai_proposals: number;
  reviewed_ai_proposals: number;
  accepted: CountRate;
  adjusted: CountRate;
  rejected: CountRate;
  unresolved: CountRate;
}

export interface ModelTrustOverview {
  total_imported_images: number;
  reviewed_images: number;
  images_in_scope: number;
  images_needing_review: number;
  auto_label_trust: CountRate;
  human_intervention: CountRate;
  human_added_annotations: number;
  attributable_ai_misses: number;
  unattributed_human_added: number;
}

export interface IouBucket {
  key: string;
  range_label: string;
  interpretation: string;
  count: number;
  rate: number;
}

export interface IouStatistics {
  count: number;
  mean: number | null;
  median: number | null;
  minimum: number | null;
  maximum: number | null;
  buckets: IouBucket[];
}

export interface ClassAnalytics {
  label_id: UUID;
  label_name: string;
  total_ai_proposals: number;
  reviewed_ai_proposals: number;
  accepted: CountRate;
  adjusted: CountRate;
  rejected: CountRate;
  unresolved: number;
  human_added: number;
  attributable_ai_misses: number;
  mean_adjusted_iou: number | null;
  final_verified_annotations: number;
  small_sample: boolean;
}

export interface PromptAnalytics {
  prompt: string;
  inference_runs: number;
  total_ai_proposals: number;
  reviewed_ai_proposals: number;
  accepted: CountRate;
  adjusted: CountRate;
  rejected: CountRate;
  unresolved: number;
  attributable_ai_misses: number;
  mean_adjusted_iou: number | null;
  small_sample: boolean;
}

export interface ConfidenceByOutcome {
  accepted_count: number;
  accepted_mean: number | null;
  adjusted_count: number;
  adjusted_mean: number | null;
  rejected_count: number;
  rejected_mean: number | null;
}

export interface ProviderModelAnalytics {
  provider: string;
  model: string;
  total_ai_proposals: number;
  reviewed_ai_proposals: number;
  accepted: CountRate;
  adjusted: CountRate;
  rejected: CountRate;
  unresolved: number;
  attributable_ai_misses: number;
  mean_adjusted_iou: number | null;
  confidence: ConfidenceByOutcome | null;
}

export interface ClassDistribution {
  label_id: UUID;
  label_name: string;
  count: number;
  rate: number;
}

export interface ValidationSummary {
  export_ready: boolean;
  blocking_error_count: number;
  warning_count: number;
  findings: ValidationFinding[];
}

export interface DatasetQualitySummary {
  total_imported_images: number;
  images_with_verified_annotations: number;
  images_without_final_annotations: number;
  exportable_images: number;
  total_final_training_boxes: number;
  total_classes: number;
  unresolved_annotations: number;
  review_needed_images: number;
  class_distribution: ClassDistribution[];
  validation: ValidationSummary;
}

export interface AnalyticsFilterOptions {
  labels: Array<{ id: UUID; name: string }>;
  provider_models: Array<{ provider: string; model: string }>;
  prompts: string[];
}

export interface AnalyticsSummary {
  overview: ModelTrustOverview;
  outcomes: OutcomeBreakdown;
  adjusted_box_iou: IouStatistics;
  classes: ClassAnalytics[];
  prompts: PromptAnalytics[];
  provider_models: ProviderModelAnalytics[];
  dataset: DatasetQualitySummary;
  filter_options: AnalyticsFilterOptions;
  active_filters: AnalyticsFilters;
}

export type ExampleKind = "rejected" | "low_iou" | "human_added" | "unresolved" | "needs_review" | "intervention";

export interface ProblemExample {
  image_id: UUID;
  image_filename: string;
  annotation_id: UUID | null;
  label_id: UUID | null;
  label_name: string | null;
  kind: ExampleKind;
  verification_state: string | null;
  prompt: string | null;
  provider: string | null;
  model: string | null;
  iou: number | null;
  human_added_is_attributed_ai_miss: boolean;
}

export interface AnalyticsExamples {
  kind: ExampleKind;
  total: number;
  items: ProblemExample[];
}

export type TrainingStatus =
  "queued" | "preparing" | "training" | "validating" | "cancelling" | "completed" | "failed" | "cancelled";

export type TrainingStage =
  | "queued"
  | "validating_dataset"
  | "preparing_dataset"
  | "loading_model"
  | "initializing_runtime"
  | "preparing_dataloader"
  | "training"
  | "validating"
  | "saving_checkpoint"
  | "cancelling"
  | "completed"
  | "cancelled"
  | "failed";

export interface ModelChoice {
  checkpoint: string;
  label: string;
  size: string;
}

export interface TrainingAvailability {
  available: boolean;
  reason: string | null;
  package_version: string | null;
  supported_models: ModelChoice[];
  device_options: string[];
  detected_device: string;
  gpu_name: string | null;
  gpu_memory_gb: number | null;
  torch_version: string | null;
  cuda_version: string | null;
}

export interface TrainingConfig {
  checkpoint: string;
  epochs: number;
  image_size: number;
  batch_size: number;
  train_ratio: number;
  seed: number;
  device: "auto" | "cpu" | "cuda:0";
  run_name: string;
}

export interface TrainingReadiness {
  ready: boolean;
  preview: ExportPreview;
  train_boxes: number;
  validation_boxes: number;
  blockers: string[];
  advisories: string[];
}

export interface DatasetSnapshot {
  sha256: string;
  project_updated_at: string;
  created_at: string;
  train_images: number;
  validation_images: number;
  train_boxes: number;
  validation_boxes: number;
  classes: number;
  class_names: string[];
}

export interface DetectionMetrics {
  precision: number | null;
  recall: number | null;
  map50: number | null;
  map50_95: number | null;
}

export interface TrainingProgressPoint {
  epoch: number;
  duration_seconds: number | null;
  training_loss: number | null;
  metrics: DetectionMetrics | null;
}

export interface TrainingRuntimeDetails {
  gpu_name: string | null;
  gpu_memory_gb: number | null;
  torch_version: string | null;
  cuda_version: string | null;
  amp_enabled: boolean | null;
  workers: number | null;
  batch_size: number | null;
  cache_mode: string;
  checkpoint_source: string | null;
}

export interface TrainingTimings {
  request_to_worker_seconds: number | null;
  dataset_validation_seconds: number | null;
  dataset_preparation_seconds: number | null;
  dataset_reused: boolean;
  model_load_seconds: number | null;
  runtime_initialization_seconds: number | null;
  dataloader_initialization_seconds: number | null;
  startup_seconds: number | null;
  validation_seconds: number | null;
  artifact_save_seconds: number | null;
  cancellation_seconds: number | null;
  total_seconds: number | null;
}

export interface ClassDetectionMetrics extends DetectionMetrics {
  class_id: number;
  class_name: string;
}

export type TrainingArtifactKind = "best_checkpoint" | "last_checkpoint" | "configuration" | "dataset_snapshot";

export interface TrainingArtifact {
  kind: TrainingArtifactKind;
  filename: string;
  size_bytes: number;
}

export interface TrainingRun {
  schema_version: 1;
  id: UUID;
  project_id: UUID;
  status: TrainingStatus;
  stage: TrainingStage;
  stage_message: string;
  config: TrainingConfig;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  cancel_requested_at: string | null;
  current_epoch: number;
  total_epochs: number;
  resolved_device: string | null;
  dataset: DatasetSnapshot | null;
  runtime: TrainingRuntimeDetails | null;
  timings: TrainingTimings;
  progress_history: TrainingProgressPoint[];
  latest_metrics: DetectionMetrics | null;
  validation_metrics: DetectionMetrics | null;
  per_class_metrics: ClassDetectionMetrics[];
  validation_image_count: number;
  artifacts: TrainingArtifact[];
  failure_reason: string | null;
  cancel_requested: boolean;
}

export interface DetectorPrediction {
  class_id: number;
  class_name: string;
  confidence: number;
  box: BoundingBox;
}

export interface PredictionPreview {
  id: UUID;
  source: "project_image" | "uploaded_image";
  filename: string;
  width: number;
  height: number;
  image_id: UUID | null;
  detections: DetectorPrediction[];
}
