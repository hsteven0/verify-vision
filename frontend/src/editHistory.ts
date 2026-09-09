import type { Project, ProjectHistorySnapshot } from "./types";

export interface EditHistoryEntry {
  description: string;
  before: ProjectHistorySnapshot;
  after: ProjectHistorySnapshot;
}

export interface EditHistory {
  entries: EditHistoryEntry[];
  index: number;
  limit: number;
}

export function createEditHistory(limit = 50): EditHistory {
  return { entries: [], index: 0, limit };
}

export function snapshotProject(project: Project): ProjectHistorySnapshot {
  return {
    labels: project.labels.map((label) => ({ ...label })),
    images: project.images.map((image) => ({
      id: image.id,
      review_state: image.review_state,
      annotations: image.annotations.map((annotation) => ({
        ...annotation,
        original_ai_box: annotation.original_ai_box ? { ...annotation.original_ai_box } : null,
        final_box: annotation.final_box ? { ...annotation.final_box } : null,
      })),
    })),
  };
}

export function recordEdit(
  history: EditHistory,
  description: string,
  before: ProjectHistorySnapshot,
  after: ProjectHistorySnapshot,
): EditHistory {
  if (JSON.stringify(before) === JSON.stringify(after)) return history;
  const entries = [...history.entries.slice(0, history.index), { description, before, after }];
  const bounded = entries.slice(Math.max(0, entries.length - history.limit));
  return { ...history, entries: bounded, index: bounded.length };
}

export function undoTarget(
  history: EditHistory,
): { snapshot: ProjectHistorySnapshot; description: string; history: EditHistory } | null {
  if (history.index === 0) return null;
  const entry = history.entries[history.index - 1];
  return {
    snapshot: entry.before,
    description: entry.description,
    history: { ...history, index: history.index - 1 },
  };
}

export function redoTarget(
  history: EditHistory,
): { snapshot: ProjectHistorySnapshot; description: string; history: EditHistory } | null {
  if (history.index >= history.entries.length) return null;
  const entry = history.entries[history.index];
  return {
    snapshot: entry.after,
    description: entry.description,
    history: { ...history, index: history.index + 1 },
  };
}
