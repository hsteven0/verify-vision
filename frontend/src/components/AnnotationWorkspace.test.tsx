import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ComponentProps } from "react";

import type { Annotation, Project, ProjectImage } from "../types";
import { AnnotationWorkspace } from "./AnnotationWorkspace";

const proposal: Annotation = {
  id: "annotation-1",
  prediction_id: "prediction-1",
  label_id: "people",
  original_ai_label_id: "people",
  source: "ai",
  verification_state: "unreviewed",
  original_ai_box: { x: 10, y: 20, width: 30, height: 40 },
  final_box: { x: 10, y: 20, width: 30, height: 40 },
  provider: "locateanything",
  model: "nvidia/LocateAnything-3B",
  prompt: "people",
  confidence: null,
  review_prompt: null,
  note: null,
  created_at: "2026-08-13T00:00:00Z",
  updated_at: "2026-08-13T00:00:00Z",
};

const image: ProjectImage = {
  id: "image-1",
  filename: "people.jpg",
  storage_name: "people.jpg",
  media_type: "image/jpeg",
  width: 100,
  height: 100,
  review_state: "in_progress",
  annotations: [proposal],
  imported_at: "2026-08-13T00:00:00Z",
};

const project: Project = {
  schema_version: 2,
  id: "project-1",
  name: "Prompt labels",
  labels: [{ id: "people", name: "People", color: "#32d6a0" }],
  images: [image],
  created_at: "2026-08-13T00:00:00Z",
  updated_at: "2026-08-13T00:00:00Z",
};

function workspaceProps(
  overrides: Partial<ComponentProps<typeof AnnotationWorkspace>> = {},
): ComponentProps<typeof AnnotationWorkspace> {
  return {
    project,
    image,
    selectedImageId: image.id,
    selectedImageIndex: 0,
    selectedLabelId: "people",
    selectedAnnotationId: proposal.id,
    selectedAnnotation: proposal,
    capabilities: null,
    health: null,
    prompt: "people",
    busy: false,
    inferenceBusy: false,
    inferenceNotice: null,
    canUndo: false,
    canRedo: true,
    undoDescription: null,
    redoDescription: "Change class",
    onSelectImage: vi.fn(),
    onMoveImage: vi.fn(),
    onSelectLabel: vi.fn(),
    onSelectAnnotation: vi.fn(),
    onPromptChange: vi.fn(),
    onGenerate: vi.fn(),
    onDismissInferenceNotice: vi.fn(),
    onCreateBox: vi.fn(),
    onUpdateBox: vi.fn(),
    onUndo: vi.fn(),
    onRedo: vi.fn(),
    onMarkReviewed: vi.fn(),
    onImport: vi.fn(),
    onDropFiles: vi.fn(),
    onAddLabel: vi.fn(),
    onRenameLabel: vi.fn(),
    onDeleteLabel: vi.fn(),
    onAssignAnnotationClass: vi.fn(),
    onChangeAnnotationLabel: vi.fn(),
    onDeleteAnnotation: vi.fn(),
    onClearAnnotations: vi.fn(),
    onDeleteImage: vi.fn(),
    onVerify: vi.fn(),
    ...overrides,
  };
}

describe("AnnotationWorkspace", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("separates class from status and shows history state", () => {
    render(<AnnotationWorkspace {...workspaceProps()} />);

    expect(screen.getAllByText("People").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Needs review").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Undo" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Redo" })).toBeEnabled();
    expect(screen.getByRole("combobox", { name: "Class" })).toHaveValue("people");
    expect(screen.queryByText("Project class manager")).not.toBeInTheDocument();
    expect(screen.getByText("Prompt")).toBeVisible();
    expect(screen.getByText("Original")).toBeVisible();
    expect(screen.getByText("Final")).toBeVisible();
    expect(screen.getByText("people", { selector: "dd" })).toBeVisible();
    expect(screen.queryByText("Details")).not.toBeInTheDocument();
    expect(screen.queryByText("Source")).not.toBeInTheDocument();
    expect(screen.queryByText("nvidia/LocateAnything-3B")).not.toBeInTheDocument();
  });

  it("keeps verification in the review bar and moves to the next object", () => {
    const onVerify = vi.fn();
    const onSelectAnnotation = vi.fn();
    const second = { ...proposal, id: "annotation-2", prediction_id: "prediction-2" };
    const human: Annotation = {
      ...proposal,
      id: "annotation-human",
      prediction_id: null,
      original_ai_label_id: null,
      source: "human",
      verification_state: "human_added",
      original_ai_box: null,
      provider: null,
      model: null,
      prompt: null,
    };
    render(
      <AnnotationWorkspace
        {...workspaceProps({
          image: { ...image, annotations: [proposal, human, second] },
          onVerify,
          onSelectAnnotation,
        })}
      />,
    );

    const review = screen.getByRole("complementary", { name: "AI proposal review" });
    expect(review).toBeVisible();
    expect(review).toHaveTextContent("1 / 2");
    expect(screen.getByText("AI proposals")).toBeVisible();
    expect(screen.getByRole("button", { name: "Accept" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Adjust" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Reject" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Accept" }));
    fireEvent.click(screen.getByRole("button", { name: "Next proposal" }));

    expect(onVerify).toHaveBeenCalledWith("accepted");
    expect(onSelectAnnotation).toHaveBeenCalledWith("annotation-2");
  });

  it("leaves the persistent image library to the application sidebar", () => {
    render(<AnnotationWorkspace {...workspaceProps()} />);

    expect(screen.queryByRole("dialog", { name: "Project images" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Images 1/ })).not.toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: "Class" }), {
      target: { value: "__manage_classes__" },
    });
    expect(screen.getByRole("dialog", { name: "Manage classes" })).toBeVisible();
  });

  it("closes the class dialog and restores focus", () => {
    render(<AnnotationWorkspace {...workspaceProps()} />);
    const classSelect = screen.getByRole("combobox", { name: "Class" });
    classSelect.focus();

    fireEvent.change(classSelect, { target: { value: "__create_class__" } });
    const dialog = screen.getByRole("dialog", { name: "Create new class" });
    expect(screen.getByRole("textbox", { name: "Class name" })).toHaveFocus();

    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(dialog).not.toBeInTheDocument();
    expect(classSelect).toHaveFocus();
  });

  it("closes the class dialog from its backdrop", () => {
    render(<AnnotationWorkspace {...workspaceProps()} />);
    fireEvent.change(screen.getByRole("combobox", { name: "Class" }), {
      target: { value: "__manage_classes__" },
    });
    const dialog = screen.getByRole("dialog", { name: "Manage classes" });

    fireEvent.mouseDown(dialog);

    expect(dialog).not.toBeInTheDocument();
  });

  it("shows drawing class only in draw mode", () => {
    render(<AnnotationWorkspace {...workspaceProps()} />);

    expect(screen.queryByRole("button", { name: "Add missed" })).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "Drawing class" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Draw" }));
    expect(screen.getByRole("combobox", { name: "Drawing class" })).toBeVisible();
  });

  it("supports toolbar and keyboard zoom outside inputs", () => {
    render(<AnnotationWorkspace {...workspaceProps()} />);

    expect(screen.getByRole("status", { name: "Current zoom" })).toHaveTextContent("100%");
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(screen.getByRole("status", { name: "Current zoom" })).toHaveTextContent("125%");

    fireEvent.keyDown(window, { key: "-", ctrlKey: true });
    expect(screen.getByRole("status", { name: "Current zoom" })).toHaveTextContent("100%");

    const prompt = screen.getByRole("textbox", { name: "Find objects in this image" });
    prompt.focus();
    fireEvent.keyDown(prompt, { key: "+", ctrlKey: true });
    expect(screen.getByRole("status", { name: "Current zoom" })).toHaveTextContent("100%");
  });

  it("lets the user reduce or hide canvas labels", () => {
    const view = render(<AnnotationWorkspace {...workspaceProps()} />);
    const mode = screen.getByRole("combobox", { name: "Canvas label display" });

    expect(mode).toHaveValue("hover");
    expect(view.container.querySelectorAll(".box-label")).toHaveLength(1);

    fireEvent.change(mode, { target: { value: "classes" } });
    expect(mode).toHaveValue("classes");
    expect(view.container.querySelectorAll(".box-label")).toHaveLength(1);

    fireEvent.change(mode, { target: { value: "hidden" } });
    expect(mode).toHaveValue("hidden");
    expect(view.container.querySelectorAll(".box-label")).toHaveLength(0);
  });

  it("requires confirmation before clear and delete image actions", () => {
    const onClearAnnotations = vi.fn();
    const onDeleteImage = vi.fn();
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<AnnotationWorkspace {...workspaceProps({ onClearAnnotations, onDeleteImage })} />);

    fireEvent.click(screen.getByLabelText("Image actions"));
    fireEvent.click(screen.getByRole("button", { name: "Clear annotations" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete image" }));

    expect(confirm).toHaveBeenCalledTimes(2);
    expect(onClearAnnotations).toHaveBeenCalledOnce();
    expect(onDeleteImage).toHaveBeenCalledOnce();
  });

  it("keeps sequential previous and next image navigation", () => {
    const onMoveImage = vi.fn();
    render(
      <AnnotationWorkspace
        {...workspaceProps({
          selectedImageIndex: 1,
          project: { ...project, images: [image, { ...image, id: "image-2" }, image] },
          onMoveImage,
        })}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Previous image" }));
    fireEvent.click(screen.getByRole("button", { name: "Next image" }));
    expect(onMoveImage).toHaveBeenNthCalledWith(1, -1);
    expect(onMoveImage).toHaveBeenNthCalledWith(2, 1);
  });

  it("supports image arrows without taking them from forms or dialogs", () => {
    const onMoveImage = vi.fn();
    render(<AnnotationWorkspace {...workspaceProps({ onMoveImage })} />);

    fireEvent.keyDown(window, { key: "ArrowLeft" });
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(onMoveImage).toHaveBeenNthCalledWith(1, -1);
    expect(onMoveImage).toHaveBeenNthCalledWith(2, 1);

    fireEvent.keyDown(screen.getByRole("textbox", { name: "Find objects in this image" }), { key: "ArrowRight" });
    const slider = document.createElement("input");
    slider.type = "range";
    document.body.append(slider);
    fireEvent.keyDown(slider, { key: "ArrowRight" });
    slider.remove();
    fireEvent.change(screen.getByRole("combobox", { name: "Class" }), {
      target: { value: "__manage_classes__" },
    });
    fireEvent.keyDown(screen.getByRole("dialog", { name: "Manage classes" }), { key: "ArrowLeft" });
    expect(onMoveImage).toHaveBeenCalledTimes(2);
  });
});
