import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "./api";
import type { ApplicationCapabilities, DependencyUpdateStatus, Project } from "./types";

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
    status: "not_loaded",
    disclosure: "Live local inference through NVIDIA LocateAnything-3B on NVIDIA CUDA.",
  },
  limits: { max_upload_bytes: 25 * 1024 * 1024, prompt_max_characters: 500 },
};

const project = {
  schema_version: 2,
  id: "11000000-0000-4000-8000-000000000001",
  name: "City Objects",
  labels: [],
  images: [],
  created_at: "2026-01-15T18:00:00Z",
  updated_at: "2026-01-15T18:08:00Z",
} satisfies Project;

describe("local-only API client", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("loads capabilities without provider headers", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify(capabilities), { status: 200 }));

    await expect(api.getCapabilities()).resolves.toEqual(capabilities);

    expect(fetchMock.mock.calls[0][0]).toBe("/api/capabilities");
    expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get("X-VerifyVision-Demo-Session")).toBe(null);
  });

  it("checks dependency updates only through an explicit action", async () => {
    const result: DependencyUpdateStatus = {
      state: "up_to_date",
      last_checked_at: "2026-08-21T12:00:00Z",
      next_check_at: "2026-08-22T12:00:00Z",
      packages_checked: 12,
      updates: [],
      detail: null,
    };
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify(result), { status: 200 }));

    await expect(api.checkDependencyUpdates()).resolves.toEqual(result);

    expect(fetchMock.mock.calls[0][0]).toBe("/api/dependencies/updates/check");
    expect(fetchMock.mock.calls[0][1]?.method).toBe("POST");
  });

  it("sends the exact prompt without requiring a preselected class", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(project), {
        status: 200,
        headers: {
          "Server-Timing": "service;dur=3012.40, gpu-inference;dur=2988.20",
          "X-VerifyVision-Model-Reused": "true",
        },
      }),
    );

    await api.generateProposals("project-1", "image-1", "people  near a car");

    expect(fetchMock.mock.calls[0][0]).toBe("/api/projects/project-1/images/image-1/proposals");
    expect(fetchMock.mock.calls[0][1]?.body).toBe(JSON.stringify({ prompt: "people  near a car" }));
    expect(api.getLastInferenceTiming()).toMatchObject({ model_reused: true });
  });
});
