import type { BoundingBox, UUID } from "../types";

export type CanvasLabelMode = "hover" | "classes" | "hidden";

export interface CanvasLabelCandidate {
  id: UUID;
  box: BoundingBox;
  classText: string;
  stateText: string;
  priority: "selected" | "hovered" | "normal";
}

export interface CanvasLabelLayout {
  id: UUID;
  x: number;
  y: number;
  width: number;
  height: number;
  detail: boolean;
}

interface ImageSize {
  width: number;
  height: number;
}

function overlap(first: CanvasLabelLayout, second: CanvasLabelLayout, gap: number): boolean {
  return !(
    first.x + first.width + gap <= second.x ||
    second.x + second.width + gap <= first.x ||
    first.y + first.height + gap <= second.y ||
    second.y + second.height + gap <= first.y
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function positions(
  candidate: CanvasLabelCandidate,
  width: number,
  height: number,
  image: ImageSize,
  unit: number,
): Array<{ x: number; y: number }> {
  const { box } = candidate;
  const gap = 3 * unit;
  const left = clamp(box.x + gap, 0, Math.max(0, image.width - width));
  const right = clamp(box.x + box.width - width - gap, 0, Math.max(0, image.width - width));
  const insideTop = clamp(box.y + gap, 0, Math.max(0, image.height - height));
  const insideBottom = clamp(box.y + box.height - height - gap, 0, Math.max(0, image.height - height));
  const above = clamp(box.y - height - gap, 0, Math.max(0, image.height - height));
  const below = clamp(box.y + box.height + gap, 0, Math.max(0, image.height - height));

  const fitsInside = box.width >= width + gap * 2 && box.height >= height + gap * 2;
  const preferred = fitsInside
    ? [
        { x: left, y: insideTop },
        { x: right, y: insideBottom },
        { x: left, y: above },
        { x: left, y: below },
      ]
    : [
        { x: left, y: above },
        { x: left, y: below },
        { x: left, y: insideTop },
        { x: right, y: insideBottom },
      ];

  return preferred.filter(
    (position, index, items) =>
      items.findIndex((item) => Math.abs(item.x - position.x) < 0.1 && Math.abs(item.y - position.y) < 0.1) === index,
  );
}

export function layoutCanvasLabels(
  candidates: CanvasLabelCandidate[],
  image: ImageSize,
  unit: number,
  mode: CanvasLabelMode,
): CanvasLabelLayout[] {
  if (mode === "hidden") return [];

  const rank = { selected: 0, hovered: 1, normal: 2 } as const;
  const visible = [...candidates]
    .filter((candidate) => mode === "classes" || candidate.priority !== "normal")
    .sort((first, second) => rank[first.priority] - rank[second.priority]);
  const placed: CanvasLabelLayout[] = [];

  for (const candidate of visible) {
    const detail = mode === "hover";
    const width = Math.min(
      image.width,
      Math.max(
        76 * unit,
        (candidate.classText.length * 9.5 + (detail ? candidate.stateText.length * 2.5 : 0) + 28) * unit,
      ),
    );
    const height = (detail ? 43 : 29) * unit;
    const choices = positions(candidate, width, height, image, unit);
    const open = choices.find((position) => {
      const layout = { id: candidate.id, ...position, width, height, detail };
      return placed.every((current) => !overlap(layout, current, 3 * unit));
    });

    if (!open && candidate.priority === "normal") continue;
    const position = open ?? choices[0];
    if (!position) continue;
    placed.push({ id: candidate.id, ...position, width, height, detail });
  }

  return placed;
}
