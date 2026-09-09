import type { BoundingBox, ProjectImage, UUID } from "../types";

export type Point = { x: number; y: number };
export type ResizeHandle = "nw" | "ne" | "se" | "sw";

export interface ZoomAnchor {
  clientX: number;
  clientY: number;
  xRatio: number;
  yRatio: number;
}

export interface AnnotationTarget {
  id: UUID;
  box: BoundingBox;
}

export interface MoveGesture {
  type: "move";
  annotationId: UUID;
  start: Point;
  original: BoundingBox;
  current: Point;
}

export interface ResizeGesture {
  type: "resize";
  annotationId: UUID;
  handle: ResizeHandle;
  start: Point;
  original: BoundingBox;
  current: Point;
}

export const clamp = (value: number, minimum: number, maximum: number) => Math.min(maximum, Math.max(minimum, value));

export function normalizedBox(start: Point, end: Point): BoundingBox {
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  };
}

export function movedBox(gesture: MoveGesture, image: ProjectImage): BoundingBox {
  const dx = gesture.current.x - gesture.start.x;
  const dy = gesture.current.y - gesture.start.y;
  return {
    ...gesture.original,
    x: clamp(gesture.original.x + dx, 0, image.width - gesture.original.width),
    y: clamp(gesture.original.y + dy, 0, image.height - gesture.original.height),
  };
}

export function resizedBox(gesture: ResizeGesture, image: ProjectImage): BoundingBox {
  const minimum = Math.max(4, Math.min(image.width, image.height) * 0.005);
  const dx = gesture.current.x - gesture.start.x;
  const dy = gesture.current.y - gesture.start.y;
  let left = gesture.original.x;
  let top = gesture.original.y;
  let right = gesture.original.x + gesture.original.width;
  let bottom = gesture.original.y + gesture.original.height;

  if (gesture.handle.includes("w")) left = clamp(left + dx, 0, right - minimum);
  if (gesture.handle.includes("e")) right = clamp(right + dx, left + minimum, image.width);
  if (gesture.handle.includes("n")) top = clamp(top + dy, 0, bottom - minimum);
  if (gesture.handle.includes("s")) bottom = clamp(bottom + dy, top + minimum, image.height);

  return { x: left, y: top, width: right - left, height: bottom - top };
}

export function canvasPointFromClient(
  bounds: Pick<DOMRect, "left" | "top" | "width" | "height">,
  image: Pick<ProjectImage, "width" | "height">,
  clientX: number,
  clientY: number,
): Point {
  return {
    x: clamp(((clientX - bounds.left) / bounds.width) * image.width, 0, image.width),
    y: clamp(((clientY - bounds.top) / bounds.height) * image.height, 0, image.height),
  };
}

export function zoomAnchorFromClient(
  bounds: Pick<DOMRect, "left" | "top" | "width" | "height">,
  clientX: number,
  clientY: number,
): ZoomAnchor {
  return {
    clientX,
    clientY,
    xRatio: clamp((clientX - bounds.left) / bounds.width, 0, 1),
    yRatio: clamp((clientY - bounds.top) / bounds.height, 0, 1),
  };
}

export function anchoredScrollDelta(
  bounds: Pick<DOMRect, "left" | "top" | "width" | "height">,
  anchor: ZoomAnchor,
): Point {
  return {
    x: bounds.left + anchor.xRatio * bounds.width - anchor.clientX,
    y: bounds.top + anchor.yRatio * bounds.height - anchor.clientY,
  };
}

export function annotationTargetsAtPoint(targets: AnnotationTarget[], point: Point, borderTolerance: number): UUID[] {
  return targets
    .flatMap((target) => {
      const { box } = target;
      const right = box.x + box.width;
      const bottom = box.y + box.height;
      const inside = point.x >= box.x && point.x <= right && point.y >= box.y && point.y <= bottom;
      const outsideX = Math.max(box.x - point.x, 0, point.x - right);
      const outsideY = Math.max(box.y - point.y, 0, point.y - bottom);
      const outsideDistance = Math.hypot(outsideX, outsideY);
      if (!inside && outsideDistance > borderTolerance) return [];
      const borderDistance = inside
        ? Math.min(point.x - box.x, right - point.x, point.y - box.y, bottom - point.y)
        : outsideDistance;
      return [{ ...target, borderDistance, onBorder: borderDistance <= borderTolerance }];
    })
    .sort((first, second) => {
      if (first.onBorder !== second.onBorder) return first.onBorder ? -1 : 1;
      if (first.onBorder && first.borderDistance !== second.borderDistance) {
        return first.borderDistance - second.borderDistance;
      }
      return first.box.width * first.box.height - second.box.width * second.box.height;
    })
    .map((target) => target.id);
}
