import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import type { ExportPreview, Project } from "../types";
import { ExportWorkspace } from "./ExportWorkspace";

vi.mock("../api", () => ({
  api: {
    getExportPreview: vi.fn(),
    downloadExport: vi.fn(),
  },
}));

const project: Project = {
  schema_version: 2,
  id: "00000000-0000-0000-0000-000000000001",
  name: "Export fixture",
  labels: [],
  images: [],
  created_at: "2026-08-12T00:00:00Z",
  updated_at: "2026-08-12T00:00:00Z",
};

const readyPreview: ExportPreview = {
  total_images: 12,
  exportable_images: 10,
  train_images: 10,
  validation_images: 0,
  training_boxes: 84,
  classes: 4,
  accepted: 60,
  adjusted: 18,
  human_added: 4,
  manual: 2,
  excluded_rejected: 8,
  excluded_unresolved: 1,
  excluded_needs_review: 0,
  blocking_error_count: 0,
  warning_count: 0,
  findings: [],
};

describe("ExportWorkspace", () => {
  beforeEach(() => {
    vi.mocked(api.getExportPreview).mockResolvedValue(readyPreview);
    vi.mocked(api.downloadExport).mockResolvedValue(undefined);
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("presents readiness and four plain-language export choices", async () => {
    render(<ExportWorkspace project={project} />);

    expect(await screen.findByText("Ready with exclusions")).toBeInTheDocument();
    expect(screen.getByText("84")).toBeInTheDocument();
    expect(screen.getByText("Export the verified dataset for YOLO training.")).toBeInTheDocument();
    expect(screen.getByText("Export annotations in standard COCO JSON format.")).toBeInTheDocument();
    expect(screen.getByText(/one Pascal VOC XML annotation file per image/i)).toBeInTheDocument();
    expect(screen.getByText(/annotation metrics for Excel/i)).toBeInTheDocument();
    expect(screen.queryByText("VerifyVision JSON")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Export Pascal VOC" }));
    expect(api.downloadExport).toHaveBeenCalledWith(project.id, "pascal_voc", {
      split: "none",
      train_ratio: 0.8,
      seed: 1337,
    });
  });

  it("keeps advanced split settings collapsed until requested", async () => {
    render(<ExportWorkspace project={project} />);
    await screen.findByText("Ready with exclusions");

    const summary = screen.getByText("Configure dataset split");
    expect(summary.closest("details")).toHaveClass("secondary-advanced");
    expect(summary.closest("details")).not.toHaveAttribute("open");
    fireEvent.click(summary);
    expect(summary.closest("details")).toHaveAttribute("open");
    expect(screen.getByRole("button", { name: "Train / validation" })).toBeVisible();
  });

  it("blocks dataset exports but keeps evaluation CSV available", async () => {
    vi.mocked(api.getExportPreview).mockResolvedValue({
      ...readyPreview,
      blocking_error_count: 1,
      warning_count: 1,
      findings: [
        {
          severity: "error",
          code: "unresolved_annotation",
          message: "1 annotation still requires review.",
          image_id: null,
          annotation_id: null,
        },
      ],
    });
    render(<ExportWorkspace project={project} />);

    expect(await screen.findByText("Resolve export issues")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Resolve validation issues" })).toHaveLength(3);
    expect(screen.getAllByRole("button", { name: "Resolve validation issues" })[0]).toBeDisabled();
    expect(screen.getByRole("button", { name: "Export CSV" })).toBeEnabled();
  });
});
