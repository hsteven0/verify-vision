import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Project, ProjectImage, ProjectSummary } from "../types";
import { ImageList } from "./AnnotateLibraryPane";
import { imageLibraryWindow } from "./imageLibraryWindow";
import { disambiguateProjectNames } from "./projectOptionLabels";

function makeImage(index: number): ProjectImage {
  return {
    id: `image-${index}`,
    filename: `street_${String(index).padStart(3, "0")}.jpg`,
    storage_name: `street_${index}.jpg`,
    media_type: "image/jpeg",
    width: 1920,
    height: 1080,
    review_state: index === 1 ? "complete" : "not_started",
    annotations: [],
    imported_at: "2026-08-18T00:00:00Z",
  };
}

function makeProject(imageCount = 12): Project {
  return {
    schema_version: 2,
    id: "project-1",
    name: "Street review",
    labels: [],
    images: Array.from({ length: imageCount }, (_, index) => makeImage(index + 1)),
    created_at: "2026-08-18T00:00:00Z",
    updated_at: "2026-08-18T00:00:00Z",
  };
}

const summary: ProjectSummary = {
  id: "project-1",
  name: "Street review",
  image_count: 12,
  completed_image_count: 1,
  updated_at: "2026-08-18T00:00:00Z",
};

function renderPane(imageCount = 12) {
  const onSelectImage = vi.fn();
  const result = render(
    <ImageList
      project={makeProject(imageCount)}
      projects={[{ ...summary, image_count: imageCount }]}
      selectedImageId="image-1"
      busy={false}
      inferenceBusy={false}
      onOpenProject={vi.fn()}
      onCreateProject={vi.fn(() => Promise.resolve())}
      onSelectImage={onSelectImage}
      onImport={vi.fn()}
    />,
  );
  return { ...result, onSelectImage };
}

describe("ImageList", () => {
  afterEach(cleanup);

  it("keeps project and images visible without global navigation", () => {
    const { onSelectImage } = renderPane();

    expect(screen.getByLabelText("Current project")).toBeVisible();
    expect(screen.getByLabelText("Project images")).toBeVisible();
    expect(screen.queryByLabelText("Project sections")).not.toBeInTheDocument();
    expect(screen.getByTitle("street_001.jpg")).toHaveAttribute("aria-current", "true");
    expect(screen.getByTitle("Complete")).toBeVisible();
    fireEvent.click(screen.getByTitle("street_002.jpg"));
    expect(onSelectImage).toHaveBeenCalledWith("image-2");

    fireEvent.change(screen.getByPlaceholderText("Search images…"), {
      target: { value: "012" },
    });
    expect(screen.getByTitle("street_012.jpg")).toBeVisible();
    expect(screen.queryByTitle("street_001.jpg")).not.toBeInTheDocument();
  });

  it("creates projects in a focused dialog", async () => {
    const onCreateProject = vi.fn(() => Promise.resolve());
    render(
      <ImageList
        project={makeProject()}
        projects={[summary]}
        selectedImageId="image-1"
        busy={false}
        inferenceBusy={false}
        onOpenProject={vi.fn()}
        onCreateProject={onCreateProject}
        onSelectImage={vi.fn()}
        onImport={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Create another project" }));
    expect(screen.getByRole("dialog", { name: "New project" })).toBeVisible();
    expect(screen.getByRole("textbox", { name: "Name" })).toHaveFocus();
    const form = screen.getByRole("form", { name: "Create project" });
    fireEvent.change(screen.getByRole("textbox", { name: "Name" }), { target: { value: "New dataset" } });
    fireEvent.submit(form);
    expect(onCreateProject).toHaveBeenCalledWith("New dataset");
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "New project" })).not.toBeInTheDocument());
  });

  it("cancels project creation and restores focus", () => {
    renderPane();
    const createButton = screen.getByRole("button", { name: "Create another project" });
    createButton.focus();
    fireEvent.click(createButton);
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(createButton).toHaveFocus();

    fireEvent.click(createButton);
    const dialog = screen.getByRole("dialog", { name: "New project" });
    fireEvent(dialog, new Event("cancel", { cancelable: true }));
    expect(dialog).not.toBeInTheDocument();
    expect(createButton).toHaveFocus();

    fireEvent.click(createButton);
    const backdrop = screen.getByRole("dialog", { name: "New project" });
    fireEvent.mouseDown(backdrop);
    expect(backdrop).not.toBeInTheDocument();
    expect(createButton).toHaveFocus();
  });

  it("keeps the full project name available when the selector truncates", () => {
    const longName = "VerifyVision Showcase for a very long downtown street review project";
    const longProject = { ...makeProject(), name: longName };
    render(
      <ImageList
        project={longProject}
        projects={[{ ...summary, name: longName }]}
        selectedImageId="image-1"
        busy={false}
        inferenceBusy={false}
        onOpenProject={vi.fn()}
        onCreateProject={vi.fn(() => Promise.resolve())}
        onSelectImage={vi.fn()}
        onImport={vi.fn()}
      />,
    );

    expect(screen.getByRole("combobox", { name: "Project" })).toHaveAttribute("title", longName);
  });

  it("disambiguates historical duplicate names without exposing IDs", () => {
    const labels = disambiguateProjectNames([
      summary,
      {
        ...summary,
        id: "project-2",
        image_count: 1,
        updated_at: "2026-08-17T00:00:00Z",
      },
    ]);

    expect(labels.get("project-1")).toMatch(/^Street review · Aug 18 · 12 images$/);
    expect(labels.get("project-2")).toMatch(/^Street review · Aug 17 · 1 image$/);
    expect([...labels.values()].join(" ")).not.toContain("project-");
  });

  it.each([10, 100, 500])("keeps a %i-image library usable", (imageCount) => {
    renderPane(imageCount);
    expect(screen.getByLabelText("Current project")).toBeVisible();
    expect(screen.getByLabelText("Project images")).toBeVisible();
    expect(screen.getByTitle("street_001.jpg")).toHaveAttribute("aria-current", "true");
    if (imageCount >= 100) {
      expect(document.querySelectorAll(".sidebar-image-item").length).toBeLessThan(25);
    }
  });

  it("bounds rendering work for 500-image projects", () => {
    expect(imageLibraryWindow(500, 7200, 720)).toEqual({ start: 119, end: 142 });
  });
});
