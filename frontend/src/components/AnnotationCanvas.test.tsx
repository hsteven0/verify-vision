import { act, fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { Annotation, ProjectImage } from "../types";
import { AnnotationCanvas } from "./AnnotationCanvas";
import {
  annotationTargetsAtPoint,
  anchoredScrollDelta,
  canvasPointFromClient,
  movedBox,
  normalizedBox,
  resizedBox,
  zoomAnchorFromClient,
} from "./annotationGeometry";

const image: ProjectImage = {
  id: "image-1",
  filename: "large.jpg",
  storage_name: "large.jpg",
  media_type: "image/jpeg",
  width: 2000,
  height: 1000,
  review_state: "in_progress",
  annotations: [],
  imported_at: "2026-08-18T00:00:00Z",
};

const annotation: Annotation = {
  id: "annotation-1",
  prediction_id: "prediction-1",
  label_id: "people",
  original_ai_label_id: "people",
  source: "ai",
  verification_state: "adjusted",
  original_ai_box: { x: 180, y: 120, width: 320, height: 240 },
  final_box: { x: 200, y: 140, width: 300, height: 220 },
  provider: "locateanything",
  model: "nvidia/LocateAnything-3B",
  prompt: "people",
  confidence: null,
  review_prompt: null,
  note: null,
  created_at: "2026-08-18T00:00:00Z",
  updated_at: "2026-08-18T00:00:00Z",
};

describe("AnnotationCanvas zoom geometry", () => {
  it("shows state detail for a hovered annotation", () => {
    const annotatedImage: ProjectImage = {
      ...image,
      annotations: [annotation],
    };
    const view = render(
      <AnnotationCanvas
        image={annotatedImage}
        imageUrl="/large.jpg"
        labels={[{ id: "people", name: "People", color: "#5c7cff" }]}
        selectedLabelId={null}
        selectedAnnotationId={null}
        interactionMode="select"
        zoom={1}
        onZoomChange={vi.fn()}
        onSelectAnnotation={vi.fn()}
        onCreate={vi.fn()}
        onUpdate={vi.fn()}
      />,
    );

    expect(view.container.querySelector(".box-label.has-detail")).toBeNull();
    const overlay = view.container.querySelector(".annotation-overlay") as SVGSVGElement;
    vi.spyOn(overlay, "getBoundingClientRect").mockReturnValue({
      left: 0,
      top: 0,
      width: 2000,
      height: 1000,
    } as DOMRect);
    fireEvent.pointerMove(overlay, { clientX: 250, clientY: 200 });
    expect(view.container.querySelector(".box-label.has-detail")).not.toBeNull();
    expect(view.container.querySelector(".box-label.has-detail")?.textContent).toContain("Adjusted");
  });

  it("uses class color and a dash pattern for selection, then raises hover", () => {
    const overlapping = {
      ...annotation,
      id: "annotation-2",
      prediction_id: "prediction-2",
      label_id: "car",
      original_ai_label_id: "car",
      final_box: { x: 260, y: 180, width: 100, height: 80 },
    };
    const view = render(
      <AnnotationCanvas
        image={{ ...image, annotations: [annotation, overlapping] }}
        imageUrl="/large.jpg"
        labels={[
          { id: "people", name: "People", color: "#5c7cff" },
          { id: "car", name: "Car", color: "#f3b44c" },
        ]}
        selectedLabelId={null}
        selectedAnnotationId="annotation-1"
        interactionMode="select"
        zoom={1}
        onZoomChange={vi.fn()}
        onSelectAnnotation={vi.fn()}
        onCreate={vi.fn()}
        onUpdate={vi.fn()}
      />,
    );

    const selected = view.container.querySelector('[data-annotation-id="annotation-1"]');
    const selectedBox = selected?.querySelector(".annotation-box");
    expect(selected).toHaveClass("is-selected");
    expect(selectedBox).toHaveAttribute("stroke", "#5c7cff");
    expect(selectedBox).toHaveAttribute("stroke-dasharray");
    expect(view.container.querySelectorAll(".resize-handle")).toHaveLength(4);

    const overlay = view.container.querySelector(".annotation-overlay") as SVGSVGElement;
    vi.spyOn(overlay, "getBoundingClientRect").mockReturnValue({
      left: 0,
      top: 0,
      width: 2000,
      height: 1000,
    } as DOMRect);
    fireEvent.pointerMove(overlay, { clientX: 300, clientY: 210 });
    const layers = view.container.querySelectorAll(".annotation-layer");
    expect(layers[layers.length - 1]).toHaveAttribute("data-annotation-id", "annotation-2");
    expect(layers[layers.length - 1]).toHaveClass("is-hovered");
    expect(view.container.querySelector(".annotation-overlay")?.lastElementChild).toHaveClass("resize-sw");

    fireEvent.pointerMove(overlay, { clientX: 220, clientY: 160 });
    const selectedAndHovered = view.container.querySelector('[data-annotation-id="annotation-1"]');
    expect(selectedAndHovered).toHaveClass("is-selected", "is-hovered");
  });

  it("selects the smallest overlapping box and cycles on a repeated click", () => {
    const small = {
      ...annotation,
      id: "annotation-2",
      prediction_id: "prediction-2",
      final_box: { x: 260, y: 180, width: 100, height: 80 },
    };
    const onSelectAnnotation = vi.fn();
    const props = {
      image: { ...image, annotations: [annotation, small] },
      imageUrl: "/large.jpg",
      labels: [{ id: "people", name: "People", color: "#5c7cff" }],
      selectedLabelId: null,
      interactionMode: "select" as const,
      zoom: 1,
      onZoomChange: vi.fn(),
      onSelectAnnotation,
      onCreate: vi.fn(),
      onUpdate: vi.fn(),
    };
    const view = render(<AnnotationCanvas {...props} selectedAnnotationId={null} />);
    let overlay = view.container.querySelector(".annotation-overlay") as SVGSVGElement;
    Object.defineProperty(overlay, "setPointerCapture", { value: vi.fn() });
    vi.spyOn(overlay, "getBoundingClientRect").mockReturnValue({
      left: 0,
      top: 0,
      width: 2000,
      height: 1000,
    } as DOMRect);

    fireEvent.pointerDown(overlay, { button: 0, pointerId: 1, clientX: 300, clientY: 210 });
    fireEvent.pointerUp(overlay, { pointerId: 1, clientX: 300, clientY: 210 });
    expect(onSelectAnnotation).toHaveBeenLastCalledWith("annotation-2");

    view.rerender(<AnnotationCanvas {...props} selectedAnnotationId="annotation-2" />);
    overlay = view.container.querySelector(".annotation-overlay") as SVGSVGElement;
    fireEvent.pointerDown(overlay, { button: 0, pointerId: 2, clientX: 300, clientY: 210 });
    fireEvent.pointerUp(overlay, { pointerId: 2, clientX: 300, clientY: 210 });
    expect(onSelectAnnotation).toHaveBeenLastCalledWith("annotation-1");
  });

  it("keeps source coordinates stable at every zoom", () => {
    const fitted = canvasPointFromClient({ left: 10, top: 20, width: 1000, height: 500 }, image, 260, 145);
    const zoomed = canvasPointFromClient({ left: 10, top: 20, width: 2000, height: 1000 }, image, 510, 270);

    expect(fitted).toEqual({ x: 500, y: 250 });
    expect(zoomed).toEqual(fitted);
    expect(normalizedBox(fitted, { x: 900, y: 650 })).toEqual({
      x: 500,
      y: 250,
      width: 400,
      height: 400,
    });
  });

  it("keeps move and resize operations in source-image coordinates", () => {
    const original = { x: 200, y: 150, width: 300, height: 200 };

    expect(
      movedBox(
        {
          type: "move",
          annotationId: "annotation-1",
          start: { x: 250, y: 200 },
          current: { x: 400, y: 300 },
          original,
        },
        image,
      ),
    ).toEqual({ x: 350, y: 250, width: 300, height: 200 });

    expect(
      resizedBox(
        {
          type: "resize",
          annotationId: "annotation-1",
          handle: "se",
          start: { x: 500, y: 350 },
          current: { x: 750, y: 500 },
          original,
        },
        image,
      ),
    ).toEqual({ x: 200, y: 150, width: 550, height: 350 });
  });

  it("handles Ctrl+wheel zoom locally without changing annotation data", () => {
    const onZoomChange = vi.fn();
    const view = render(
      <AnnotationCanvas
        image={image}
        imageUrl="/large.jpg"
        labels={[]}
        selectedLabelId={null}
        selectedAnnotationId={null}
        interactionMode="select"
        zoom={1}
        onZoomChange={onZoomChange}
        onSelectAnnotation={vi.fn()}
        onCreate={vi.fn()}
        onUpdate={vi.fn()}
      />,
    );
    const viewport = view.container.querySelector(".canvas-viewport");
    expect(viewport).not.toBeNull();

    const canvasWheel = new WheelEvent("wheel", {
      bubbles: true,
      cancelable: true,
      ctrlKey: true,
      deltaY: -100,
      clientX: 250,
      clientY: 150,
    });
    act(() => expect(viewport!.dispatchEvent(canvasWheel)).toBe(false));
    expect(onZoomChange).toHaveBeenCalledWith(1.25);
    expect(image.width).toBe(2000);
    expect(image.annotations).toEqual([]);
  });

  it("does not intercept ordinary wheel input inside the canvas", () => {
    const onZoomChange = vi.fn();
    const view = render(
      <AnnotationCanvas
        image={image}
        imageUrl="/large.jpg"
        labels={[]}
        selectedLabelId={null}
        selectedAnnotationId={null}
        interactionMode="select"
        zoom={1}
        onZoomChange={onZoomChange}
        onSelectAnnotation={vi.fn()}
        onCreate={vi.fn()}
        onUpdate={vi.fn()}
      />,
    );
    const viewport = view.container.querySelector(".canvas-viewport");
    const ordinaryWheel = new WheelEvent("wheel", { bubbles: true, cancelable: true, deltaY: 100 });
    expect(viewport!.dispatchEvent(ordinaryWheel)).toBe(true);
    expect(onZoomChange).not.toHaveBeenCalled();
  });

  it("preserves the source point under the cursor when the stage grows", () => {
    const anchor = zoomAnchorFromClient({ left: 100, top: 50, width: 800, height: 400 }, 300, 250);
    expect(anchor).toMatchObject({ xRatio: 0.25, yRatio: 0.5 });
    expect(anchoredScrollDelta({ left: 100, top: 50, width: 1200, height: 600 }, anchor)).toEqual({
      x: 100,
      y: 100,
    });
  });

  it("targets box borders first and otherwise prefers the smallest containing box", () => {
    const targets = [
      { id: "large", box: { x: 0, y: 0, width: 100, height: 100 } },
      { id: "small", box: { x: 20, y: 20, width: 20, height: 20 } },
      { id: "partial", box: { x: 25, y: 25, width: 50, height: 50 } },
    ];

    expect(annotationTargetsAtPoint(targets, { x: 30, y: 30 }, 3)).toEqual(["small", "partial", "large"]);
    expect(annotationTargetsAtPoint(targets, { x: 1, y: 60 }, 3)).toEqual(["large"]);
    expect(annotationTargetsAtPoint(targets, { x: 24, y: 30 }, 3)).toEqual(["partial", "small", "large"]);
  });
});
