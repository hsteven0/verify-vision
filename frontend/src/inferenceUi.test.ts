import { describe, expect, it } from "vitest";

import { activeInferenceLabel, inferenceActionLabel, isCurrentProposal } from "./inferenceUi";
import type { HealthResponse } from "./types";

function health(status: HealthResponse["inference_status"], device = "auto"): HealthResponse {
  return {
    status: "ok",
    api_version: "1",
    inference_provider: "locateanything",
    inference_model: "nvidia/LocateAnything-3B",
    inference_device: device,
    inference_status: status,
    inference_available: status !== "error" && status !== "unavailable",
    inference_platform: "win32",
    inference_python_version: "3.12.10",
    inference_torch_version: "2.12.1+cu126",
    inference_cuda_version: "12.6",
    inference_gpu: "NVIDIA GeForce RTX Test",
    inference_compute_capability: "7.5",
    inference_requested_dtype: "auto",
    inference_selected_dtype: "float16",
    inference_dtype_reason: "pre-Ampere CUDA device",
    inference_detail: null,
  };
}

describe("inference UI", () => {
  it("rejects saved predictions from another provider", () => {
    expect(
      isCurrentProposal(
        { source: "ai", provider: "locateanything", model: "nvidia/LocateAnything-3B" },
        "locateanything",
        "nvidia/LocateAnything-3B",
      ),
    ).toBe(true);
    expect(
      isCurrentProposal(
        { source: "ai", provider: "retired-provider", model: "old-model" },
        "locateanything",
        "nvidia/LocateAnything-3B",
      ),
    ).toBe(false);
  });

  it("distinguishes first model load from active inference", () => {
    expect(inferenceActionLabel(true, health("not_loaded"))).toBe("Loading LocateAnything-3B…");
    expect(inferenceActionLabel(true, health("ready", "cuda:0"))).toBe("Finding objects…");
    expect(inferenceActionLabel(false, health("ready"))).toBe("Find objects");
  });

  it("reports local CUDA and unsupported runtime states honestly", () => {
    expect(activeInferenceLabel(null)).toBe("LocateAnything-3B · checking runtime");
    expect(activeInferenceLabel(health("ready", "cuda:0"))).toBe("LocateAnything-3B · Local CUDA");
    expect(activeInferenceLabel(health("error"))).toBe("LocateAnything-3B · runtime unavailable");
  });
});
