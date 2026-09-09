import { describe, expect, it } from "vitest";

import { layoutCanvasLabels, type CanvasLabelCandidate } from "./canvasLabels";

const image = { width: 1000, height: 600 };

function candidate(id: string, x: number, priority: CanvasLabelCandidate["priority"] = "normal") {
  return {
    id,
    box: { x, y: 100, width: 140, height: 100 },
    classText: "People",
    stateText: "Accepted",
    priority,
  };
}

describe("canvas label layout", () => {
  it("keeps labels inside image bounds", () => {
    const labels = layoutCanvasLabels(
      [
        {
          ...candidate("edge", 960, "selected"),
          box: { x: 960, y: 2, width: 38, height: 40 },
        },
      ],
      image,
      1,
      "hover",
    );

    expect(labels).toHaveLength(1);
    expect(labels[0].x).toBeGreaterThanOrEqual(0);
    expect(labels[0].y).toBeGreaterThanOrEqual(0);
    expect(labels[0].x + labels[0].width).toBeLessThanOrEqual(image.width);
    expect(labels[0].y + labels[0].height).toBeLessThanOrEqual(image.height);
  });

  it("drops normal labels that collide", () => {
    const crowded = Array.from({ length: 8 }, (_, index) => candidate(`normal-${index}`, 100));
    const labels = layoutCanvasLabels([candidate("selected", 100, "selected"), ...crowded], image, 1, "classes");

    expect(labels.some((label) => label.id === "selected")).toBe(true);
    expect(labels.filter((label) => label.id.startsWith("normal")).length).toBeLessThan(crowded.length);
  });

  it("keeps selected and hovered detail ahead of normal labels", () => {
    const labels = layoutCanvasLabels(
      [candidate("normal", 100), candidate("hovered", 102, "hovered"), candidate("selected", 104, "selected")],
      image,
      1,
      "hover",
    );

    expect(labels[0]).toMatchObject({ id: "selected", detail: true });
    expect(labels[1]).toMatchObject({ id: "hovered", detail: true });
  });

  it("honors the three display modes", () => {
    const candidates = [candidate("normal", 100), candidate("selected", 300, "selected")];

    expect(layoutCanvasLabels(candidates, image, 1, "hover").map((label) => label.id)).toEqual(["selected"]);
    expect(layoutCanvasLabels(candidates, image, 1, "classes").every((label) => !label.detail)).toBe(true);
    expect(layoutCanvasLabels(candidates, image, 1, "classes").map((label) => label.id)).toEqual([
      "selected",
      "normal",
    ]);
    expect(layoutCanvasLabels(candidates, image, 1, "hidden")).toEqual([]);
  });

  it("shows only focused labels on dense images", () => {
    const candidates = Array.from({ length: 12 }, (_, index) =>
      candidate(`box-${index}`, index * 60, index === 4 ? "selected" : index === 7 ? "hovered" : "normal"),
    );

    expect(layoutCanvasLabels(candidates, image, 1, "hover").map((label) => label.id)).toEqual(["box-4", "box-7"]);
  });
});
