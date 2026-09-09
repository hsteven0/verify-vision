import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import type { AnalyticsSummary, Project } from "../types";
import { AnalyticsDashboard } from "./AnalyticsDashboard";

vi.mock("../api", () => ({
  api: {
    getAnalytics: vi.fn(),
    getAnalyticsExamples: vi.fn(),
  },
}));

const countRate = (count: number, denominator: number) => ({
  count,
  denominator,
  rate: denominator ? (count / denominator) * 100 : 0,
});

const project: Project = {
  schema_version: 2,
  id: "00000000-0000-0000-0000-000000000050",
  name: "Trust fixture",
  labels: [{ id: "00000000-0000-0000-0000-000000000001", name: "Person", color: "#32d6a0" }],
  images: [],
  created_at: "2026-08-12T00:00:00Z",
  updated_at: "2026-08-12T00:00:00Z",
};

function populatedSummary(): AnalyticsSummary {
  return {
    overview: {
      total_imported_images: 3,
      reviewed_images: 2,
      images_in_scope: 2,
      images_needing_review: 1,
      auto_label_trust: countRate(2, 4),
      human_intervention: countRate(3, 5),
      human_added_annotations: 1,
      attributable_ai_misses: 1,
      unattributed_human_added: 0,
    },
    outcomes: {
      total_ai_proposals: 5,
      reviewed_ai_proposals: 4,
      accepted: countRate(2, 4),
      adjusted: countRate(1, 4),
      rejected: countRate(1, 4),
      unresolved: countRate(1, 5),
    },
    adjusted_box_iou: {
      count: 1,
      mean: 0.68,
      median: 0.68,
      minimum: 0.68,
      maximum: 0.68,
      buckets: [
        {
          key: "major",
          range_label: "< 0.50",
          interpretation: "Major correction",
          count: 0,
          rate: 0,
        },
        {
          key: "significant",
          range_label: "0.50–0.74",
          interpretation: "Significant adjustment",
          count: 1,
          rate: 100,
        },
        {
          key: "moderate",
          range_label: "0.75–0.89",
          interpretation: "Moderate adjustment",
          count: 0,
          rate: 0,
        },
        {
          key: "minor",
          range_label: "0.90–1.00",
          interpretation: "Minor adjustment",
          count: 0,
          rate: 0,
        },
      ],
    },
    classes: [
      {
        label_id: project.labels[0].id,
        label_name: "Person",
        total_ai_proposals: 5,
        reviewed_ai_proposals: 4,
        accepted: countRate(2, 4),
        adjusted: countRate(1, 4),
        rejected: countRate(1, 4),
        unresolved: 1,
        human_added: 1,
        attributable_ai_misses: 1,
        mean_adjusted_iou: 0.68,
        final_verified_annotations: 4,
        small_sample: true,
      },
    ],
    prompts: [
      {
        prompt: "person wearing a helmet",
        inference_runs: 2,
        total_ai_proposals: 5,
        reviewed_ai_proposals: 4,
        accepted: countRate(2, 4),
        adjusted: countRate(1, 4),
        rejected: countRate(1, 4),
        unresolved: 1,
        attributable_ai_misses: 1,
        mean_adjusted_iou: 0.68,
        small_sample: true,
      },
    ],
    provider_models: [
      {
        provider: "nvidia",
        model: "nvidia/LocateAnything-3B",
        total_ai_proposals: 5,
        reviewed_ai_proposals: 4,
        accepted: countRate(2, 4),
        adjusted: countRate(1, 4),
        rejected: countRate(1, 4),
        unresolved: 1,
        attributable_ai_misses: 1,
        mean_adjusted_iou: 0.68,
        confidence: null,
      },
    ],
    dataset: {
      total_imported_images: 3,
      images_with_verified_annotations: 2,
      images_without_final_annotations: 1,
      exportable_images: 2,
      total_final_training_boxes: 4,
      total_classes: 1,
      unresolved_annotations: 1,
      review_needed_images: 1,
      class_distribution: [{ label_id: project.labels[0].id, label_name: "Person", count: 4, rate: 100 }],
      validation: {
        export_ready: true,
        blocking_error_count: 0,
        warning_count: 2,
        findings: [],
      },
    },
    filter_options: {
      labels: [{ id: project.labels[0].id, name: "Person" }],
      provider_models: [{ provider: "nvidia", model: "nvidia/LocateAnything-3B" }],
      prompts: ["person wearing a helmet"],
    },
    active_filters: { label_id: null, provider: null, model: null, prompt: null },
  };
}

describe("AnalyticsDashboard", () => {
  afterEach(cleanup);

  beforeEach(() => {
    vi.mocked(api.getAnalytics).mockResolvedValue(populatedSummary());
    vi.mocked(api.getAnalyticsExamples).mockResolvedValue({
      kind: "rejected",
      total: 1,
      items: [
        {
          image_id: "00000000-0000-0000-0000-000000000010",
          image_filename: "problem.jpg",
          annotation_id: "00000000-0000-0000-0000-000000000101",
          label_id: project.labels[0].id,
          label_name: "Person",
          kind: "rejected",
          verification_state: "rejected",
          prompt: "person wearing a helmet",
          provider: "nvidia",
          model: "nvidia/LocateAnything-3B",
          iou: null,
          human_added_is_attributed_ai_miss: false,
        },
      ],
    });
  });

  it("renders the outcome-first dashboard and dataset readiness", async () => {
    render(<AnalyticsDashboard project={project} onInspectImage={vi.fn()} onOpenExport={vi.fn()} />);

    expect(await screen.findByText("How accurate were the proposals?")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Verification outcomes" })).toBeInTheDocument();
    expect(screen.getAllByText("50.0%").length).toBeGreaterThan(0);
    expect(screen.getByText("Class results")).toBeInTheDocument();
    expect(screen.getAllByText("person wearing a helmet", { exact: false })).toHaveLength(2);
    expect(screen.getByText("Ready with exclusions")).toBeInTheDocument();
    expect(screen.getByText("Unresolved work will be excluded")).toBeInTheDocument();
    expect(screen.getByText("Box placement")).toBeInTheDocument();
    expect(screen.getByText("Major corrections")).toBeInTheDocument();
    expect(screen.getByText("LocateAnything-3B")).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: /Provider/i })).not.toBeInTheDocument();
  });

  it("shows the active model without a provider selector", async () => {
    render(
      <AnalyticsDashboard
        project={project}
        activeSource={{ label: "Model", value: "LocateAnything-3B" }}
        onInspectImage={vi.fn()}
        onOpenExport={vi.fn()}
      />,
    );

    expect(await screen.findByText("LocateAnything-3B")).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: /Provider/i })).not.toBeInTheDocument();
  });

  it("shows provider filtering only when distinct cohorts exist", async () => {
    const mixed = populatedSummary();
    mixed.filter_options.provider_models.push({
      provider: "another-provider",
      model: "another-model",
    });
    vi.mocked(api.getAnalytics).mockResolvedValue(mixed);
    render(<AnalyticsDashboard project={project} onInspectImage={vi.fn()} onOpenExport={vi.fn()} />);

    expect(await screen.findByRole("combobox", { name: /Provider cohort/i })).toBeVisible();
  });

  it("hides duplicate provider cohorts", async () => {
    const duplicated = populatedSummary();
    duplicated.filter_options.provider_models.push({
      provider: "NVIDIA",
      model: "NVIDIA/LOCATEANYTHING-3B",
    });
    vi.mocked(api.getAnalytics).mockResolvedValue(duplicated);
    render(<AnalyticsDashboard project={project} onInspectImage={vi.fn()} onOpenExport={vi.fn()} />);

    await screen.findByText("How accurate were the proposals?");
    expect(screen.queryByRole("combobox", { name: /Provider cohort/i })).not.toBeInTheDocument();
  });

  it("opens a rejected example in the annotation workspace", async () => {
    const onInspectImage = vi.fn();
    render(<AnalyticsDashboard project={project} onInspectImage={onInspectImage} onOpenExport={vi.fn()} />);
    await screen.findByText("How accurate were the proposals?");
    fireEvent.click(screen.getByRole("button", { name: /Rejected predictions/i }));
    fireEvent.click(await screen.findByRole("button", { name: /problem.jpg/i }));

    expect(onInspectImage).toHaveBeenCalledWith(
      "00000000-0000-0000-0000-000000000010",
      "00000000-0000-0000-0000-000000000101",
    );
  });

  it("explains the no-proposals state without invalid rates", async () => {
    const empty = populatedSummary();
    empty.outcomes = {
      total_ai_proposals: 0,
      reviewed_ai_proposals: 0,
      accepted: countRate(0, 0),
      adjusted: countRate(0, 0),
      rejected: countRate(0, 0),
      unresolved: countRate(0, 0),
    };
    vi.mocked(api.getAnalytics).mockResolvedValue(empty);
    render(<AnalyticsDashboard project={project} onInspectImage={vi.fn()} onOpenExport={vi.fn()} />);

    expect(await screen.findByText("No reviewed proposals yet")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText("NaN")).not.toBeInTheDocument());
  });
});
