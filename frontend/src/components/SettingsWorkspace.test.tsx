import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import type { ApplicationCapabilities, HealthResponse, Project } from "../types";
import { SettingsWorkspace } from "./SettingsWorkspace";

const project: Project = {
  schema_version: 2,
  id: "project-1",
  name: "Local dataset",
  labels: [],
  images: [],
  created_at: "2026-08-21T00:00:00Z",
  updated_at: "2026-08-21T00:00:00Z",
};

const capabilities: ApplicationCapabilities = {
  features: {
    inference: true,
    training: true,
    trained_model_prediction: true,
    manual_annotation: true,
    image_uploads: true,
    dataset_export: true,
    analytics: true,
  },
  inference: {
    model: "nvidia/LocateAnything-3B",
    available: true,
    device: "cuda:0",
    status: "ready",
    disclosure: "Live local inference through NVIDIA LocateAnything-3B on NVIDIA CUDA.",
  },
  limits: { max_upload_bytes: 25 * 1024 * 1024, prompt_max_characters: 500 },
};

const health: HealthResponse = {
  status: "ok",
  api_version: "1",
  inference_provider: "locateanything",
  inference_model: "nvidia/LocateAnything-3B",
  inference_device: "cuda:0",
  inference_status: "ready",
  inference_available: true,
  inference_platform: "win32",
  inference_python_version: "3.12.10",
  inference_torch_version: "2.12.1+cu126",
  inference_cuda_version: "12.6",
  inference_gpu: "NVIDIA GeForce RTX 2060 SUPER",
  inference_compute_capability: "7.5",
  inference_requested_dtype: "auto",
  inference_selected_dtype: "float16",
  inference_dtype_reason: "pre-Ampere CUDA device",
  inference_detail: null,
};

describe("SettingsWorkspace", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("shows local runtime context and a cached dependency status", async () => {
    vi.spyOn(api, "getDependencyUpdates").mockResolvedValue({
      state: "up_to_date",
      last_checked_at: "2026-08-21T12:00:00Z",
      next_check_at: "2026-08-22T12:00:00Z",
      packages_checked: 18,
      updates: [],
      detail: null,
    });

    render(
      <SettingsWorkspace
        project={project}
        capabilities={capabilities}
        health={health}
        themePreference="system"
        onThemeChange={vi.fn()}
      />,
    );

    expect(await screen.findByText("Up to date")).toBeInTheDocument();
    expect(screen.getByText("NVIDIA GeForce RTX 2060 SUPER")).toBeInTheDocument();
    expect(screen.getByText("v1.0.0")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "GitHub repository" })).toHaveAttribute(
      "href",
      "https://github.com/hsteven0/verify-vision",
    );
    expect(screen.queryByText(/demo/i)).not.toBeInTheDocument();
  });

  it("reviews updates without offering an automatic install action", async () => {
    vi.spyOn(api, "getDependencyUpdates").mockResolvedValue({
      state: "not_checked",
      last_checked_at: null,
      next_check_at: null,
      packages_checked: 0,
      updates: [],
      detail: null,
    });
    vi.spyOn(api, "checkDependencyUpdates").mockResolvedValue({
      state: "updates_available",
      last_checked_at: "2026-08-21T12:00:00Z",
      next_check_at: "2026-08-22T12:00:00Z",
      packages_checked: 18,
      updates: [
        {
          package: "torch",
          ecosystem: "python",
          risk: "critical_runtime",
          current_version: "2.12.1+cu126",
          latest_version: "2.13.0",
          release_url: "https://pypi.org/project/torch/",
        },
      ],
      detail: "Nothing is installed automatically.",
    });

    render(
      <SettingsWorkspace
        project={project}
        capabilities={capabilities}
        health={health}
        themePreference="dark"
        onThemeChange={vi.fn()}
      />,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Check for updates" }));

    await waitFor(() => expect(screen.getByText("1 update available")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Review updates"));
    expect(screen.getByText("torch")).toBeInTheDocument();
    expect(screen.getByText("ML runtime")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /install|update now/i })).not.toBeInTheDocument();
  });

  it("opens a configured repository in a new tab", async () => {
    vi.spyOn(api, "getDependencyUpdates").mockResolvedValue({
      state: "not_checked",
      last_checked_at: null,
      next_check_at: null,
      packages_checked: 0,
      updates: [],
      detail: null,
    });

    render(
      <SettingsWorkspace
        project={project}
        capabilities={capabilities}
        health={health}
        themePreference="system"
        onThemeChange={vi.fn()}
        repositoryUrl="https://github.com/example/verifyvision"
      />,
    );

    const link = await screen.findByRole("link", { name: "GitHub repository" });
    expect(link).toHaveAttribute("href", "https://github.com/example/verifyvision");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noreferrer");
  });
});
