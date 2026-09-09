import type {
  ApplicationCapabilities,
  AnalyticsExamples,
  AnalyticsFilters,
  AnalyticsSummary,
  BoundingBox,
  DependencyUpdateStatus,
  EvaluationSummary,
  ExportFormat,
  ExportOptions,
  ExportPreview,
  HumanAnnotationState,
  HealthResponse,
  ImageReviewState,
  Project,
  ProjectHistorySnapshot,
  ProjectSummary,
  UUID,
  VerificationDecision,
  ExampleKind,
  PredictionPreview,
  TrainingArtifactKind,
  TrainingAvailability,
  TrainingConfig,
  TrainingReadiness,
  TrainingRun,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
  }
}

export interface InferenceRequestTiming {
  initiated_at_ms: number;
  response_received_at_ms: number;
  completed_at_ms: number;
  request_to_headers_ms: number;
  response_parse_ms: number;
  server_timing: string | null;
  model_reused: boolean | null;
}

let latestInferenceTiming: InferenceRequestTiming | null = null;

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
  return (await response.json()) as T;
}

async function requestWithStartupRetry<T>(path: string): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 12_000);
    try {
      return await request<T>(path, { signal: controller.signal });
    } catch (error) {
      lastError = error;
      const retryable = !(error instanceof ApiError) || [502, 503, 504].includes(error.status);
      if (!retryable || attempt === 3) throw error;
      await new Promise((resolve) => window.setTimeout(resolve, 700 * 2 ** attempt));
    } finally {
      window.clearTimeout(timeout);
    }
  }
  throw lastError;
}

const jsonHeaders = { "Content-Type": "application/json" };

function analyticsQuery(filters: AnalyticsFilters): URLSearchParams {
  const query = new URLSearchParams();
  if (filters.label_id) query.set("label_id", filters.label_id);
  if (filters.provider) query.set("provider", filters.provider);
  if (filters.model) query.set("model", filters.model);
  if (filters.prompt) query.set("prompt", filters.prompt);
  return query;
}

async function errorMessage(response: Response): Promise<string> {
  let message = `Request failed (${response.status})`;
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) message = payload.detail;
  } catch {
    // The status code remains useful when the response is not JSON.
  }
  return message;
}

async function downloadResponse(response: Response, fallbackFilename: string): Promise<void> {
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);

  const disposition = response.headers.get("Content-Disposition") ?? "";
  const filename = disposition.match(/filename="?([^";]+)"?/i)?.[1] ?? fallbackFilename;
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

async function requestInferenceProject(projectId: UUID, imageId: UUID, prompt: string): Promise<Project> {
  const initiatedAt = performance.now();
  const response = await fetch(`/api/projects/${projectId}/images/${imageId}/proposals`, {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ prompt }),
  });
  const responseReceivedAt = performance.now();
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
  const parseStartedAt = performance.now();
  const project = (await response.json()) as Project;
  const completedAt = performance.now();
  const reusedHeader = response.headers.get("X-VerifyVision-Model-Reused");
  latestInferenceTiming = {
    initiated_at_ms: initiatedAt,
    response_received_at_ms: responseReceivedAt,
    completed_at_ms: completedAt,
    request_to_headers_ms: responseReceivedAt - initiatedAt,
    response_parse_ms: completedAt - parseStartedAt,
    server_timing: response.headers.get("Server-Timing"),
    model_reused: reusedHeader === null ? null : reusedHeader === "true",
  };
  return project;
}

export const api = {
  getCapabilities: () => requestWithStartupRetry<ApplicationCapabilities>("/api/capabilities"),

  getHealth: () => request<HealthResponse>("/api/health"),

  getDependencyUpdates: () => request<DependencyUpdateStatus>("/api/dependencies/updates"),

  checkDependencyUpdates: () => request<DependencyUpdateStatus>("/api/dependencies/updates/check", { method: "POST" }),

  listProjects: () => request<ProjectSummary[]>("/api/projects"),

  getProject: (projectId: UUID) => request<Project>(`/api/projects/${projectId}`),

  getEvaluation: (projectId: UUID) => request<EvaluationSummary>(`/api/projects/${projectId}/evaluation`),

  getAnalytics: (projectId: UUID, filters: AnalyticsFilters) => {
    const query = analyticsQuery(filters);
    const suffix = query.size ? `?${query}` : "";
    return request<AnalyticsSummary>(`/api/projects/${projectId}/analytics${suffix}`);
  },

  getAnalyticsExamples: (projectId: UUID, kind: ExampleKind, filters: AnalyticsFilters, limit = 25) => {
    const query = analyticsQuery(filters);
    query.set("kind", kind);
    query.set("limit", String(limit));
    return request<AnalyticsExamples>(`/api/projects/${projectId}/analytics/examples?${query}`);
  },

  getExportPreview: (projectId: UUID, options: ExportOptions) => {
    const query = new URLSearchParams({
      split: options.split,
      train_ratio: String(options.train_ratio),
      seed: String(options.seed),
    });
    return request<ExportPreview>(`/api/projects/${projectId}/exports/preview?${query}`);
  },

  downloadExport: async (projectId: UUID, format: ExportFormat, options: ExportOptions) => {
    const response = await fetch(`/api/projects/${projectId}/exports/${format}`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(options),
    });
    await downloadResponse(response, `verifyvision-${format}`);
  },

  createProject: (name: string) =>
    request<Project>("/api/projects", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ name }),
    }),

  addLabel: (projectId: UUID, name: string, color: string) =>
    request<Project>(`/api/projects/${projectId}/labels`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ name, color }),
    }),

  renameLabel: (projectId: UUID, labelId: UUID, name: string) =>
    request<Project>(`/api/projects/${projectId}/labels/${labelId}`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify({ name }),
    }),

  deleteLabel: (projectId: UUID, labelId: UUID) =>
    request<Project>(`/api/projects/${projectId}/labels/${labelId}`, {
      method: "DELETE",
    }),

  uploadImages: (projectId: UUID, files: File[]) => {
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    return request<Project>(`/api/projects/${projectId}/images`, { method: "POST", body });
  },

  createAnnotation: (
    projectId: UUID,
    imageId: UUID,
    labelId: UUID,
    box: BoundingBox,
    verificationState: HumanAnnotationState = "manual",
    reviewPrompt?: string,
  ) =>
    request<Project>(`/api/projects/${projectId}/images/${imageId}/annotations`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({
        label_id: labelId,
        box,
        verification_state: verificationState,
        review_prompt: verificationState === "human_added" ? reviewPrompt : undefined,
      }),
    }),

  generateProposals: requestInferenceProject,

  getLastInferenceTiming: () => latestInferenceTiming,

  verifyAnnotation: (projectId: UUID, annotationId: UUID, decision: VerificationDecision) =>
    request<Project>(`/api/projects/${projectId}/annotations/${annotationId}/verify`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ decision }),
    }),

  updateAnnotation: (
    projectId: UUID,
    annotationId: UUID,
    update: { box?: BoundingBox; label_id?: UUID; note?: string | null },
  ) =>
    request<Project>(`/api/projects/${projectId}/annotations/${annotationId}`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify(update),
    }),

  assignAnnotationLabel: (projectId: UUID, annotationId: UUID, name: string, color: string) =>
    request<Project>(`/api/projects/${projectId}/annotations/${annotationId}/label`, {
      method: "PUT",
      headers: jsonHeaders,
      body: JSON.stringify({ name, color }),
    }),

  restoreHistoryState: (projectId: UUID, snapshot: ProjectHistorySnapshot) =>
    request<Project>(`/api/projects/${projectId}/history-state`, {
      method: "PUT",
      headers: jsonHeaders,
      body: JSON.stringify(snapshot),
    }),

  deleteAnnotation: (projectId: UUID, annotationId: UUID) =>
    request<Project>(`/api/projects/${projectId}/annotations/${annotationId}`, {
      method: "DELETE",
    }),

  clearImageAnnotations: (projectId: UUID, imageId: UUID) =>
    request<Project>(`/api/projects/${projectId}/images/${imageId}/annotations`, {
      method: "DELETE",
    }),

  deleteImage: (projectId: UUID, imageId: UUID) =>
    request<Project>(`/api/projects/${projectId}/images/${imageId}`, {
      method: "DELETE",
    }),

  updateImageReview: (projectId: UUID, imageId: UUID, reviewState: ImageReviewState) =>
    request<Project>(`/api/projects/${projectId}/images/${imageId}`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify({ review_state: reviewState }),
    }),

  imageUrl: (projectId: UUID, imageId: UUID) => `/api/projects/${projectId}/images/${imageId}/content`,

  getTrainingAvailability: (projectId: UUID) =>
    request<TrainingAvailability>(`/api/projects/${projectId}/training/availability`),

  getTrainingReadiness: (projectId: UUID, config: TrainingConfig) =>
    request<TrainingReadiness>(`/api/projects/${projectId}/training/readiness`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(config),
    }),

  listTrainingRuns: (projectId: UUID) => request<TrainingRun[]>(`/api/projects/${projectId}/training/runs`),

  createTrainingRun: (projectId: UUID, config: TrainingConfig) =>
    request<TrainingRun>(`/api/projects/${projectId}/training/runs`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(config),
    }),

  getTrainingRun: (projectId: UUID, runId: UUID) =>
    request<TrainingRun>(`/api/projects/${projectId}/training/runs/${runId}`),

  cancelTrainingRun: (projectId: UUID, runId: UUID) =>
    request<TrainingRun>(`/api/projects/${projectId}/training/runs/${runId}/cancel`, {
      method: "POST",
    }),

  downloadTrainingArtifact: async (projectId: UUID, runId: UUID, kind: TrainingArtifactKind) => {
    const response = await fetch(`/api/projects/${projectId}/training/runs/${runId}/artifacts/${kind}`);
    await downloadResponse(response, kind);
  },

  predictProjectImage: (projectId: UUID, runId: UUID, imageId: UUID, confidence = 0.25) =>
    request<PredictionPreview>(
      `/api/projects/${projectId}/training/runs/${runId}/predict/images/${imageId}?confidence=${confidence}`,
      { method: "POST" },
    ),

  predictUploadedImage: (projectId: UUID, runId: UUID, file: File, confidence = 0.25) => {
    const body = new FormData();
    body.append("file", file);
    return request<PredictionPreview>(
      `/api/projects/${projectId}/training/runs/${runId}/predict/upload?confidence=${confidence}`,
      { method: "POST", body },
    );
  },

  trainingPredictionImageUrl: (projectId: UUID, runId: UUID, previewId: UUID) =>
    `/api/projects/${projectId}/training/runs/${runId}/predictions/${previewId}/image`,
};
