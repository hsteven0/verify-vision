import type { Annotation, BoundingBox } from "./types";

export function imageStateLabel(state: string): string {
  return state.replaceAll("_", " ");
}

export function titleCase(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function annotationStateLabel(annotation: Annotation): string {
  if (annotation.verification_state === "unreviewed") return "AI proposal · needs review";
  if (annotation.verification_state === "human_added") return "Human added · possible miss";
  if (annotation.verification_state === "manual") return "Manual annotation";
  return `AI proposal · ${annotation.verification_state}`;
}

export function boxesEqual(first: BoundingBox | null, second: BoundingBox | null): boolean {
  if (!first || !second) return first === second;
  return first.x === second.x && first.y === second.y && first.width === second.width && first.height === second.height;
}

export function proposalWasChanged(annotation: Annotation): boolean {
  return (
    annotation.source === "ai" &&
    (!boxesEqual(annotation.original_ai_box, annotation.final_box) ||
      annotation.original_ai_label_id !== annotation.label_id)
  );
}
