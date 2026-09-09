import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import type {
  EvaluationSummary,
  Project,
  TrainingAvailability,
  TrainingReadiness,
  TrainingRun,
  TrainingStatus,
} from "../types";
import { TrainingWorkspace } from "./TrainingWorkspace";

vi.mock("../api", () => ({
  api: {
    getTrainingAvailability: vi.fn(),
    getTrainingReadiness: vi.fn(),
    listTrainingRuns: vi.fn(),
    getTrainingRun: vi.fn(),
    createTrainingRun: vi.fn(),
    cancelTrainingRun: vi.fn(),
    downloadTrainingArtifact: vi.fn(),
  },
}));

const project: Project = {
  schema_version: 2,
  id: "00000000-0000-0000-0000-000000000001",
  name: "Training UI fixture",
  labels: [{ id: "00000000-0000-0000-0000-000000000020", name: "Person", color: "#32d6a0" }],
  images: [
    {
      id: "00000000-0000-0000-0000-000000000010",
      filename: "street.png",
      storage_name: "street.png",
      media_type: "image/png",
      width: 100,
      height: 100,
      review_state: "complete",
      annotations: [],
      imported_at: "2026-08-12T00:00:00Z",
    },
  ],
  created_at: "2026-08-12T00:00:00Z",
  updated_at: "2026-08-12T00:00:00Z",
};

const availability: TrainingAvailability = {
  available: true,
  reason: null,
  package_version: "8.4.119",
  supported_models: [{ checkpoint: "yolo26n.pt", label: "Nano", size: "Smallest checkpoint" }],
  device_options: ["auto", "cuda:0"],
  detected_device: "cuda:0",
  gpu_name: "NVIDIA GeForce RTX 2060 SUPER",
  gpu_memory_gb: 8,
  torch_version: "2.12.1+cu126",
  cuda_version: "12.6",
};

const readiness: TrainingReadiness = {
  ready: true,
  preview: {
    total_images: 10,
    exportable_images: 10,
    train_images: 8,
    validation_images: 2,
    training_boxes: 20,
    classes: 1,
    accepted: 12,
    adjusted: 4,
    human_added: 3,
    manual: 1,
    excluded_rejected: 2,
    excluded_unresolved: 0,
    excluded_needs_review: 0,
    blocking_error_count: 0,
    warning_count: 1,
    findings: [],
  },
  train_boxes: 16,
  validation_boxes: 4,
  blockers: [],
  advisories: ["Small validation split"],
};

const evaluation: EvaluationSummary = {
  total_ai_proposals: 20,
  reviewed_ai_proposals: 20,
  accepted: { count: 12, denominator: 20, rate: 60 },
  adjusted: { count: 4, denominator: 20, rate: 20 },
  rejected: { count: 4, denominator: 20, rate: 20 },
  unresolved: { count: 0, denominator: 20, rate: 0 },
  human_added_count: 3,
  verified_annotation_count: 19,
  average_adjusted_iou: 0.8,
  by_class: [],
};

const baseRun: TrainingRun = {
  schema_version: 1,
  id: "00000000-0000-0000-0000-000000000100",
  project_id: project.id,
  status: "completed",
  stage: "completed",
  stage_message: "Training and held-out validation completed",
  config: {
    checkpoint: "yolo26n.pt",
    epochs: 2,
    image_size: 640,
    batch_size: 8,
    train_ratio: 0.8,
    seed: 1337,
    device: "auto",
    run_name: "portfolio-detector",
  },
  created_at: "2026-08-12T00:00:00Z",
  started_at: "2026-08-12T00:00:01Z",
  completed_at: "2026-08-12T00:01:00Z",
  cancel_requested_at: null,
  current_epoch: 2,
  total_epochs: 2,
  resolved_device: "cuda:0",
  dataset: {
    sha256: "a".repeat(64),
    project_updated_at: "2026-08-12T00:00:00Z",
    created_at: "2026-08-12T00:00:01Z",
    train_images: 8,
    validation_images: 2,
    train_boxes: 16,
    validation_boxes: 4,
    classes: 1,
    class_names: ["Person"],
  },
  runtime: {
    gpu_name: "NVIDIA GeForce RTX 2060 SUPER",
    gpu_memory_gb: 8,
    torch_version: "2.12.1+cu126",
    cuda_version: "12.6",
    amp_enabled: true,
    workers: 0,
    batch_size: 8,
    cache_mode: "reusable verified snapshot",
    checkpoint_source: "B:/Projects/VerifyVision/yolo26n.pt",
  },
  timings: {
    request_to_worker_seconds: 0.01,
    dataset_validation_seconds: 0.1,
    dataset_preparation_seconds: 0.2,
    dataset_reused: true,
    model_load_seconds: 0.4,
    runtime_initialization_seconds: 0.3,
    dataloader_initialization_seconds: 1.2,
    startup_seconds: 2.1,
    validation_seconds: 3.4,
    artifact_save_seconds: 0.5,
    cancellation_seconds: null,
    total_seconds: 59,
  },
  progress_history: [
    {
      epoch: 1,
      duration_seconds: 20,
      training_loss: 1.4,
      metrics: { precision: 0.6, recall: 0.55, map50: 0.58, map50_95: 0.4 },
    },
    {
      epoch: 2,
      duration_seconds: 18,
      training_loss: 0.9,
      metrics: { precision: 0.8, recall: 0.7, map50: 0.74, map50_95: 0.52 },
    },
  ],
  latest_metrics: { precision: 0.8, recall: 0.7, map50: 0.74, map50_95: 0.52 },
  validation_metrics: { precision: 0.8, recall: 0.7, map50: 0.74, map50_95: 0.52 },
  per_class_metrics: [
    {
      class_id: 0,
      class_name: "Person",
      precision: 0.8,
      recall: 0.7,
      map50: 0.74,
      map50_95: 0.52,
    },
  ],
  validation_image_count: 2,
  artifacts: [
    { kind: "best_checkpoint", filename: "best.pt", size_bytes: 2048 },
    { kind: "configuration", filename: "training-result.json", size_bytes: 1024 },
  ],
  failure_reason: null,
  cancel_requested: false,
};

function runWith(status: TrainingStatus, overrides: Partial<TrainingRun> = {}): TrainingRun {
  const terminal = ["completed", "cancelled", "failed"].includes(status);
  return {
    ...baseRun,
    status,
    stage: status === "preparing" ? "preparing_dataloader" : status,
    stage_message: status === "preparing" ? "Scanning labels and preparing the data loader" : humanStage(status),
    current_epoch: status === "training" || status === "validating" || status === "cancelling" ? 1 : 0,
    completed_at: terminal ? baseRun.completed_at : null,
    cancel_requested: status === "cancelling" || status === "cancelled",
    cancel_requested_at: status === "cancelling" || status === "cancelled" ? "2026-08-12T00:00:45Z" : null,
    validation_metrics: status === "completed" ? baseRun.validation_metrics : null,
    progress_history:
      status === "training" || status === "validating" || status === "cancelling"
        ? [baseRun.progress_history[0]]
        : status === "completed"
          ? baseRun.progress_history
          : [],
    failure_reason: status === "failed" ? "Synthetic CUDA failure" : null,
    ...overrides,
  };
}

function humanStage(status: TrainingStatus) {
  return status === "cancelling" ? "Stopping after the current safe training operation" : status;
}

describe("TrainingWorkspace", () => {
  beforeEach(() => {
    vi.mocked(api.getTrainingAvailability).mockResolvedValue(availability);
    vi.mocked(api.getTrainingReadiness).mockResolvedValue(readiness);
    vi.mocked(api.listTrainingRuns).mockResolvedValue([baseRun]);
    vi.mocked(api.createTrainingRun).mockResolvedValue(runWith("queued"));
    vi.mocked(api.cancelTrainingRun).mockResolvedValue(runWith("cancelling"));
    vi.mocked(api.getTrainingRun).mockResolvedValue(baseRun);
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("shows readiness, config, and completed metrics", async () => {
    const onOpenTest = vi.fn();
    render(<TrainingWorkspace project={project} evaluation={evaluation} onOpenTest={onOpenTest} />);

    expect(await screen.findByText("Train a YOLO model")).toBeInTheDocument();
    expect(screen.getByText("10 images")).toBeInTheDocument();
    expect(screen.getByText("RTX 2060 SUPER")).toBeInTheDocument();
    expect(screen.getByText("Dataset ready")).toBeInTheDocument();
    expect(screen.getAllByText("74.0%")).toHaveLength(3);
    expect(screen.getByRole("img", { name: /Training loss and validation mAP50/ })).toBeInTheDocument();

    const advanced = screen.getByText("Advanced settings").closest("details");
    expect(advanced).not.toHaveAttribute("open");
    fireEvent.click(screen.getByText("Advanced settings"));
    expect(advanced).toHaveAttribute("open");

    fireEvent.click(screen.getByRole("button", { name: "Test model" }));
    expect(onOpenTest).toHaveBeenCalledOnce();
  });

  it("starts one run with the selected configuration", async () => {
    vi.mocked(api.listTrainingRuns).mockResolvedValue([]);
    render(<TrainingWorkspace project={project} evaluation={evaluation} />);

    fireEvent.change(await screen.findByLabelText("Epochs"), { target: { value: "12" } });
    fireEvent.click(screen.getByRole("button", { name: "Start training" }));

    await waitFor(() => expect(api.createTrainingRun).toHaveBeenCalledOnce());
    expect(api.createTrainingRun).toHaveBeenCalledWith(
      project.id,
      expect.objectContaining({ epochs: 12, device: "auto" }),
    );
  });

  it("uses a real empty state before the first run", async () => {
    vi.mocked(api.listTrainingRuns).mockResolvedValue([]);
    render(<TrainingWorkspace project={project} evaluation={evaluation} />);

    expect(await screen.findByText("No training run yet")).toBeInTheDocument();
    expect(screen.queryByText("0.0%")).not.toBeInTheDocument();
  });

  it("communicates preparation without a fake percentage", async () => {
    vi.mocked(api.listTrainingRuns).mockResolvedValue([runWith("preparing")]);
    render(<TrainingWorkspace project={project} evaluation={evaluation} />);

    expect(await screen.findByText("Scanning labels and preparing the data loader")).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Training preparation steps" })).toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it.each([
    ["training", "Training"],
    ["validating", "Validating"],
  ] as const)("shows the %s state with measured progress", async (status, label) => {
    vi.mocked(api.listTrainingRuns).mockResolvedValue([runWith(status)]);
    render(<TrainingWorkspace project={project} evaluation={evaluation} />);

    expect((await screen.findAllByText(label)).length).toBeGreaterThan(0);
    expect(screen.getByRole("progressbar", { name: "Training epoch progress" })).toHaveAttribute("aria-valuenow", "1");
    expect(screen.getByText("Loss 1.400")).toBeInTheDocument();
  });

  it("shows cancelling and disables repeated cancellation", async () => {
    vi.mocked(api.listTrainingRuns).mockResolvedValue([runWith("cancelling")]);
    render(<TrainingWorkspace project={project} evaluation={evaluation} />);

    expect(await screen.findByText("Cancelling safely")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancelling…" })).toBeDisabled();
  });

  it("requests cancellation from an active run", async () => {
    const active = runWith("training");
    vi.mocked(api.listTrainingRuns).mockResolvedValue([active]);
    render(<TrainingWorkspace project={project} evaluation={evaluation} />);

    fireEvent.click(await screen.findByRole("button", { name: "Cancel training" }));
    await waitFor(() => expect(api.cancelTrainingRun).toHaveBeenCalledWith(project.id, active.id));
  });

  it("shows cancelled and failed terminal states clearly", async () => {
    vi.mocked(api.listTrainingRuns).mockResolvedValue([
      runWith("cancelled", { timings: { ...baseRun.timings, cancellation_seconds: 1.7 } }),
      runWith("failed", { id: "00000000-0000-0000-0000-000000000101" }),
    ]);
    const { rerender } = render(<TrainingWorkspace project={project} evaluation={evaluation} />);

    expect(await screen.findByText(/Cancelled after 1.7s/)).toBeInTheDocument();
    vi.mocked(api.listTrainingRuns).mockResolvedValue([runWith("failed")]);
    rerender(
      <TrainingWorkspace
        project={{ ...project, id: "00000000-0000-0000-0000-000000000002" }}
        evaluation={evaluation}
      />,
    );
    expect(await screen.findByText("Synthetic CUDA failure")).toBeInTheDocument();
  });

  it("blocks training when CUDA is unavailable", async () => {
    vi.mocked(api.getTrainingAvailability).mockResolvedValue({
      ...availability,
      available: false,
      reason: "Local training requires an NVIDIA CUDA GPU visible to PyTorch.",
      device_options: [],
      detected_device: "unavailable",
      gpu_name: null,
      gpu_memory_gb: null,
    });
    vi.mocked(api.listTrainingRuns).mockResolvedValue([]);
    render(<TrainingWorkspace project={project} evaluation={evaluation} />);

    expect(await screen.findByText("CUDA required")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start training" })).toBeDisabled();
    expect(screen.getByText(/NVIDIA CUDA GPU visible to PyTorch/)).toBeInTheDocument();
  });
});
