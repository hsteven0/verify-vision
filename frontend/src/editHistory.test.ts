import { describe, expect, it } from "vitest";

import { createEditHistory, recordEdit, redoTarget, snapshotProject, undoTarget } from "./editHistory";
import type { Annotation, Project, VerificationState } from "./types";

const annotation: Annotation = {
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

function projectWith(overrides: Partial<Annotation> = {}): Project {
  return {
    schema_version: 2,
    id: "project-1",
    name: "History",
    labels: [
      { id: "people", name: "People", color: "#32d6a0" },
      { id: "crowd", name: "Crowd", color: "#6ba8ff" },
    ],
    images: [
      {
        id: "image-1",
        filename: "people.jpg",
        storage_name: "people.jpg",
        media_type: "image/jpeg",
        width: 100,
        height: 100,
        review_state: "in_progress",
        annotations: [{ ...annotation, ...overrides }],
        imported_at: "2026-08-13T00:00:00Z",
      },
    ],
    created_at: "2026-08-13T00:00:00Z",
    updated_at: "2026-08-13T00:00:00Z",
  };
}

describe("annotation edit history", () => {
  it.each([
    ["Move box", { final_box: { x: 20, y: 25, width: 30, height: 40 } }],
    ["Resize box", { final_box: { x: 10, y: 20, width: 42, height: 48 } }],
    ["Change class", { label_id: "crowd" }],
    ["Accept proposal", { verification_state: "accepted" as VerificationState }],
    ["Confirm adjustment", { verification_state: "adjusted" as VerificationState }],
    ["Reject proposal", { verification_state: "rejected" as VerificationState }],
  ])("undoes and redoes %s", (description, change) => {
    const before = snapshotProject(projectWith());
    const after = snapshotProject(projectWith(change));
    const history = recordEdit(createEditHistory(), description, before, after);

    const undone = undoTarget(history);
    expect(undone?.snapshot).toEqual(before);
    const redone = redoTarget(undone!.history);
    expect(redone?.snapshot).toEqual(after);
    expect(redone?.snapshot.images[0].annotations[0].prediction_id).toBe("prediction-1");
    expect(redone?.snapshot.images[0].annotations[0].prompt).toBe("people");
  });

  it("handles manual creation, human-added creation, and deletion", () => {
    const empty = projectWith();
    empty.images[0].annotations = [];
    const manual = projectWith({ source: "human", verification_state: "manual" });
    const humanAdded = projectWith({
      source: "human",
      verification_state: "human_added",
      review_prompt: "people",
    });
    let history = createEditHistory();
    history = recordEdit(history, "Create box", snapshotProject(empty), snapshotProject(manual));
    history = recordEdit(history, "Add missing object", snapshotProject(manual), snapshotProject(humanAdded));
    history = recordEdit(history, "Delete box", snapshotProject(humanAdded), snapshotProject(empty));

    expect(undoTarget(history)?.snapshot.images[0].annotations[0].verification_state).toBe("human_added");
  });

  it("clears redo entries when a new action follows undo", () => {
    const before = snapshotProject(projectWith());
    const moved = snapshotProject(projectWith({ final_box: { x: 20, y: 20, width: 30, height: 40 } }));
    const resized = snapshotProject(projectWith({ final_box: { x: 10, y: 20, width: 45, height: 45 } }));
    let history = recordEdit(createEditHistory(), "Move box", before, moved);
    history = recordEdit(history, "Resize box", moved, resized);
    history = undoTarget(history)!.history;
    history = recordEdit(history, "Change class", moved, snapshotProject(projectWith({ label_id: "crowd" })));

    expect(history.entries.map((entry) => entry.description)).toEqual(["Move box", "Change class"]);
    expect(redoTarget(history)).toBeNull();
  });

  it("keeps history bounded", () => {
    let history = createEditHistory(3);
    for (let index = 0; index < 6; index += 1) {
      history = recordEdit(
        history,
        `Edit ${index}`,
        snapshotProject(projectWith({ note: `${index}` })),
        snapshotProject(projectWith({ note: `${index + 1}` })),
      );
    }
    expect(history.entries).toHaveLength(3);
    expect(history.index).toBe(3);
    expect(history.entries[0].description).toBe("Edit 3");
  });
});
