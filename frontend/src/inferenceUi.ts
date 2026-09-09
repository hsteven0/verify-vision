import type { Annotation, HealthResponse } from "./types";

export function isCurrentProposal(
  annotation: Pick<Annotation, "source" | "provider" | "model">,
  provider: string | null | undefined,
  model: string | null | undefined,
): boolean {
  return (
    annotation.source === "ai" &&
    (!provider || annotation.provider === provider) &&
    (!model || annotation.model === model)
  );
}

export function inferenceActionLabel(inferenceBusy: boolean, health: HealthResponse | null): string {
  if (!inferenceBusy) return "Find objects";
  if (health?.inference_status === "not_loaded" || health?.inference_status === "loading") {
    return "Loading LocateAnything-3B…";
  }
  return "Finding objects…";
}

export function activeInferenceLabel(health: HealthResponse | null): string {
  if (!health) return "LocateAnything-3B · checking runtime";
  if (health.inference_status === "loading") return "LocateAnything-3B · loading model";
  if (!health.inference_available || health.inference_status === "error") {
    return "LocateAnything-3B · runtime unavailable";
  }
  if (health.inference_device.startsWith("cuda")) return "LocateAnything-3B · Local CUDA";
  return "LocateAnything-3B · Local inference";
}
