import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import type { PredictionPreview, Project, TrainingRun } from "../types";
import { TestWorkspace } from "./TestWorkspace";

vi.mock("../api", () => ({
  api: {
    listTrainingRuns: vi.fn(),
    predictProjectImage: vi.fn(),
    predictUploadedImage: vi.fn(),
    imageUrl: vi.fn(() => "/project-image.png"),
    trainingPredictionImageUrl: vi.fn(() => "/uploaded-image.png"),
  },
}));

const imageId = "00000000-0000-0000-0000-000000000010";
const project: Project = {
  schema_version: 2,
  id: "00000000-0000-0000-0000-000000000001",
  name: "Test fixture",
  labels: [{ id: "00000000-0000-0000-0000-000000000020", name: "Person", color: "#32d6a0" }],
  images: [
    {
      id: imageId,
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

const completedRun: TrainingRun = {
  schema_version: 1,
  id: "00000000-0000-0000-0000-000000000100",
  project_id: project.id,
  status: "completed",
  stage: "completed",
  stage_message: "Training completed",
  config: {
    checkpoint: "yolo26n.pt",
    epochs: 2,
    image_size: 640,
    batch_size: 8,
    train_ratio: 0.8,
    seed: 1337,
    device: "auto",
    run_name: "best-detector",
  },
  created_at: "2026-08-12T00:00:00Z",
  started_at: "2026-08-12T00:00:01Z",
  completed_at: "2026-08-12T00:01:00Z",
  cancel_requested_at: null,
  current_epoch: 2,
  total_epochs: 2,
  resolved_device: "cuda:0",
  dataset: null,
  runtime: null,
  timings: {
    request_to_worker_seconds: 0.01,
    dataset_validation_seconds: 0.1,
    dataset_preparation_seconds: 0.1,
    dataset_reused: false,
    model_load_seconds: 0.2,
    runtime_initialization_seconds: 0.2,
    dataloader_initialization_seconds: 0.3,
    startup_seconds: 0.8,
    validation_seconds: 1,
    artifact_save_seconds: 0.2,
    cancellation_seconds: null,
    total_seconds: 59,
  },
  progress_history: [],
  latest_metrics: null,
  validation_metrics: { precision: 0.8, recall: 0.7, map50: 0.74, map50_95: 0.52 },
  per_class_metrics: [],
  validation_image_count: 4,
  artifacts: [],
  failure_reason: null,
  cancel_requested: false,
};

const preview: PredictionPreview = {
  id: "00000000-0000-0000-0000-000000000200",
  source: "project_image",
  filename: "street.png",
  width: 100,
  height: 100,
  image_id: imageId,
  detections: [
    {
      class_id: 0,
      class_name: "Person",
      confidence: 0.88,
      box: { x: 10, y: 10, width: 30, height: 40 },
    },
  ],
};

describe("TestWorkspace", () => {
  beforeEach(() => {
    vi.mocked(api.listTrainingRuns).mockResolvedValue([completedRun]);
    vi.mocked(api.predictProjectImage).mockResolvedValue(preview);
    vi.mocked(api.predictUploadedImage).mockResolvedValue({
      ...preview,
      source: "uploaded_image",
      filename: "held-out.jpg",
      image_id: null,
    });
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("separates single-image testing from saved validation metrics", async () => {
    render(<TestWorkspace project={project} />);

    expect(await screen.findByText("Test a trained model")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Validation metrics" })).toBeInTheDocument();
    expect(screen.getByText(/validation images from this training run/i)).toBeInTheDocument();
    expect(screen.getByText("80.0%")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Run test" }));

    expect(await screen.findByText("1 trained-model detection")).toBeInTheDocument();
    expect(api.predictProjectImage).toHaveBeenCalledWith(project.id, completedRun.id, imageId, 0.25);
    expect(project.images[0].annotations).toEqual([]);
  });

  it("supports a held-out upload through the same primary action", async () => {
    const view = render(<TestWorkspace project={project} />);
    await screen.findByText("Test a trained model");
    fireEvent.click(screen.getByRole("button", { name: "Held-out upload" }));

    const file = new File(["image"], "held-out.jpg", { type: "image/jpeg" });
    const input = view.container.querySelector('input[type="file"]');
    expect(input).not.toBeNull();
    fireEvent.change(input!, { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Run test" }));

    expect(await screen.findByText("1 trained-model detection")).toBeInTheDocument();
    expect(screen.getAllByText("Held-out upload").length).toBeGreaterThan(0);
    expect(api.predictUploadedImage).toHaveBeenCalledWith(project.id, completedRun.id, file, 0.25);
  });

  it("keeps confidence settings in the shared disclosure", async () => {
    render(<TestWorkspace project={project} />);
    await screen.findByText("Test a trained model");

    const summary = screen.getByText("Advanced");
    const disclosure = summary.closest("details");
    expect(disclosure).toHaveClass("secondary-advanced");
    expect(disclosure).not.toHaveAttribute("open");

    fireEvent.click(summary);
    const slider = screen.getByRole("slider", { name: "Confidence threshold" });
    expect(slider).toBeVisible();
    fireEvent.change(slider, { target: { value: "0.4" } });
    expect(screen.getByText("40%")).toBeVisible();
  });

  it("shows an empty state without a trained model", async () => {
    vi.mocked(api.listTrainingRuns).mockResolvedValue([]);
    render(<TestWorkspace project={project} />);

    expect(await screen.findByText("No trained model available")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run test" })).not.toBeInTheDocument();
  });

  it("skips training queries when prediction is unavailable", () => {
    render(<TestWorkspace project={project} enabled={false} />);

    expect(screen.getByText("No trained model available")).toBeInTheDocument();
    expect(screen.getByText(/complete a local training run/i)).toBeInTheDocument();
    expect(api.listTrainingRuns).not.toHaveBeenCalled();
  });
});
