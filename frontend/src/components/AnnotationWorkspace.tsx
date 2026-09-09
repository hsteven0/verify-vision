import { useEffect, useMemo, useRef, useState, type DragEvent, type FormEvent, type ReactNode } from "react";

import { api } from "../api";
import { imageStateLabel, proposalWasChanged, titleCase } from "../annotationUi";
import { inferenceActionLabel } from "../inferenceUi";
import type {
  Annotation,
  ApplicationCapabilities,
  BoundingBox,
  HealthResponse,
  Project,
  ProjectImage,
  UUID,
  VerificationDecision,
} from "../types";
import { AnnotationCanvas } from "./AnnotationCanvas";
import type { CanvasLabelMode } from "./canvasLabels";
import { Icon } from "./Icon";

const LABEL_COLORS = ["#32d6a0", "#6ba8ff", "#ffbd59", "#ee7dba", "#a98bff", "#ff796d"];
const CREATE_CLASS_VALUE = "__create_class__";
const MANAGE_CLASSES_VALUE = "__manage_classes__";
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 4;
const ZOOM_STEP = 0.25;

function blocksWorkspaceShortcut(target: EventTarget | null): boolean {
  return (
    target instanceof Element &&
    Boolean(target.closest("button, input, textarea, select, form, dialog, details[open], [contenteditable]"))
  );
}

interface AnnotationWorkspaceProps {
  project: Project;
  image: ProjectImage | null;
  selectedImageId: UUID | null;
  selectedImageIndex: number;
  selectedLabelId: UUID | null;
  selectedAnnotationId: UUID | null;
  selectedAnnotation: Annotation | null;
  capabilities: ApplicationCapabilities | null;
  health: HealthResponse | null;
  prompt: string;
  busy: boolean;
  inferenceBusy: boolean;
  inferenceNotice: string | null;
  canUndo: boolean;
  canRedo: boolean;
  undoDescription: string | null;
  redoDescription: string | null;
  onSelectImage: (id: UUID) => void;
  onMoveImage: (direction: -1 | 1) => void;
  onSelectLabel: (id: UUID) => void;
  onSelectAnnotation: (id: UUID | null) => void;
  onPromptChange: (prompt: string) => void;
  onGenerate: () => Promise<void>;
  onDismissInferenceNotice: () => void;
  onCreateBox: (box: BoundingBox) => Promise<void>;
  onUpdateBox: (annotationId: UUID, box: BoundingBox, action: "Move box" | "Resize box") => Promise<void>;
  onUndo: () => Promise<void>;
  onRedo: () => Promise<void>;
  onMarkReviewed: () => Promise<void>;
  onImport: () => void;
  onDropFiles: (files: File[]) => Promise<void>;
  onAddLabel: (name: string, color: string) => Promise<UUID | null>;
  onRenameLabel: (labelId: UUID, name: string) => Promise<boolean>;
  onDeleteLabel: (labelId: UUID) => Promise<boolean>;
  onAssignAnnotationClass: (annotationId: UUID, name: string, color: string) => Promise<boolean>;
  onChangeAnnotationLabel: (annotationId: UUID, labelId: UUID) => Promise<void>;
  onDeleteAnnotation: (annotationId: UUID) => Promise<void>;
  onClearAnnotations: () => Promise<void>;
  onDeleteImage: () => Promise<void>;
  onVerify: (decision: VerificationDecision) => Promise<void>;
}

type ClassDialogIntent = "drawing" | "annotation" | null;

export function AnnotationWorkspace(props: AnnotationWorkspaceProps) {
  const { onMoveImage } = props;
  const [interactionMode, setInteractionMode] = useState<"select" | "draw">("select");
  const [zoom, setZoom] = useState(1);
  const [labelMode, setLabelMode] = useState<CanvasLabelMode>("hover");
  const [classDialogIntent, setClassDialogIntent] = useState<ClassDialogIntent>(null);
  const [classManagerOpen, setClassManagerOpen] = useState(false);
  const unresolvedOnImage =
    props.image?.annotations.filter((annotation) => annotation.verification_state === "unreviewed").length ?? 0;

  useEffect(() => {
    setInteractionMode("select");
    setZoom(1);
    setClassDialogIntent(null);
  }, [props.selectedImageId]);

  useEffect(() => {
    const handleWorkspaceShortcut = (event: KeyboardEvent) => {
      if (blocksWorkspaceShortcut(event.target)) return;
      const modifier = event.ctrlKey || event.metaKey;
      if (modifier && (event.key === "+" || event.key === "=")) {
        event.preventDefault();
        setZoom((current) => Math.min(MAX_ZOOM, current + ZOOM_STEP));
      } else if (modifier && event.key === "-") {
        event.preventDefault();
        setZoom((current) => Math.max(MIN_ZOOM, current - ZOOM_STEP));
      } else if (modifier && event.key === "0") {
        event.preventDefault();
        setZoom(1);
      } else if (!modifier && event.key === "ArrowLeft") {
        event.preventDefault();
        onMoveImage(-1);
      } else if (!modifier && event.key === "ArrowRight") {
        event.preventDefault();
        onMoveImage(1);
      }
    };
    window.addEventListener("keydown", handleWorkspaceShortcut);
    return () => window.removeEventListener("keydown", handleWorkspaceShortcut);
  }, [onMoveImage]);

  const clearAnnotations = () => {
    if (!props.image || props.image.annotations.length === 0) return;
    if (
      window.confirm(
        `Clear all annotations from “${props.image.filename}”?\n\nThe imported image will remain, and you can undo this action.`,
      )
    ) {
      void props.onClearAnnotations();
    }
  };

  const deleteImage = () => {
    if (!props.image) return;
    if (
      window.confirm(
        `Delete image?\n\nThis will remove “${props.image.filename}” and its annotations from the project.`,
      )
    ) {
      void props.onDeleteImage();
    }
  };

  const proposals = props.image?.annotations.filter((item) => item.source === "ai") ?? [];
  const proposalIndex = proposals.findIndex((item) => item.id === props.selectedAnnotationId);
  const selectNextProposal = () => {
    if (proposals.length === 0) return;
    const start = proposalIndex < 0 ? -1 : proposalIndex;
    const nextUnresolved = Array.from(
      { length: proposals.length },
      (_, offset) => proposals[(start + offset + 1) % proposals.length],
    ).find((item) => item.verification_state === "unreviewed");
    props.onSelectAnnotation((nextUnresolved ?? proposals[(start + 1) % proposals.length]).id);
  };

  return (
    <main className="annotation-workspace">
      <section
        className={`annotation-center ${props.project.images.length === 0 ? "is-empty" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
        }}
        onDrop={(event: DragEvent) => {
          event.preventDefault();
          void props.onDropFiles(Array.from(event.dataTransfer.files));
        }}
      >
        {props.image ? (
          <>
            <FindBar {...props} />

            {props.inferenceNotice && (
              <div className="inference-notice compact-notice" role="status">
                <span aria-hidden="true">●</span>
                <span>{props.inferenceNotice}</span>
                <button onClick={props.onDismissInferenceNotice} aria-label="Dismiss result">
                  <Icon name="close" />
                </button>
              </div>
            )}

            <div className="canvas-tools" aria-label="Canvas tools">
              <div className="tool-segment" aria-label="Interaction mode">
                <button
                  className={interactionMode === "select" ? "active" : ""}
                  onClick={() => setInteractionMode("select")}
                  aria-pressed={interactionMode === "select"}
                >
                  <Icon name="pointer" />
                  Select
                </button>
                <button
                  className={interactionMode === "draw" ? "active" : ""}
                  onClick={() => setInteractionMode("draw")}
                  aria-pressed={interactionMode === "draw"}
                >
                  <Icon name="box" />
                  Draw
                </button>
              </div>

              {interactionMode === "draw" && (
                <ClassSelect
                  id="drawing-class"
                  label="Drawing class"
                  value={props.selectedLabelId ?? ""}
                  project={props.project}
                  disabled={props.busy}
                  onClassChange={props.onSelectLabel}
                  onCreate={() => setClassDialogIntent("drawing")}
                  onManage={() => setClassManagerOpen(true)}
                />
              )}

              <div className="history-controls" aria-label="Edit history">
                <button
                  disabled={props.busy || props.inferenceBusy || !props.canUndo}
                  onClick={() => void props.onUndo()}
                  title={props.undoDescription ? `Undo ${props.undoDescription} (Ctrl+Z)` : "Undo (Ctrl+Z)"}
                  aria-label="Undo"
                >
                  <Icon name="undo" />
                  <span>Undo</span>
                </button>
                <button
                  disabled={props.busy || props.inferenceBusy || !props.canRedo}
                  onClick={() => void props.onRedo()}
                  title={
                    props.redoDescription ? `Redo ${props.redoDescription} (Ctrl+Y)` : "Redo (Ctrl+Y or Ctrl+Shift+Z)"
                  }
                  aria-label="Redo"
                >
                  <Icon name="redo" />
                  <span>Redo</span>
                </button>
              </div>

              <ZoomControls zoom={zoom} onZoomChange={setZoom} />
              <LabelModeControl value={labelMode} onChange={setLabelMode} />
            </div>

            <AnnotationCanvas
              image={props.image}
              imageUrl={api.imageUrl(props.project.id, props.image.id)}
              labels={props.project.labels}
              selectedLabelId={props.selectedLabelId}
              selectedAnnotationId={props.selectedAnnotationId}
              interactionMode={interactionMode}
              zoom={zoom}
              labelMode={labelMode}
              disabled={props.busy}
              onZoomChange={setZoom}
              onSelectAnnotation={props.onSelectAnnotation}
              onCreate={props.onCreateBox}
              onUpdate={props.onUpdateBox}
            />

            <div className="image-pager-bar">
              <div className="image-pager">
                <button
                  onClick={() => props.onMoveImage(-1)}
                  disabled={props.selectedImageIndex <= 0}
                  title="Previous image (P)"
                  aria-label="Previous image"
                >
                  <Icon name="previous" />
                  Previous image
                </button>
                <strong>
                  {props.selectedImageIndex + 1} <span>/ {props.project.images.length}</span>
                </strong>
                <button
                  onClick={() => props.onMoveImage(1)}
                  disabled={props.selectedImageIndex >= props.project.images.length - 1}
                  title="Next image (N)"
                  aria-label="Next image"
                >
                  Next image
                  <Icon name="next" />
                </button>
              </div>
              <div className="image-context" title={props.image.filename}>
                <strong>{props.image.filename}</strong>
                <span className={`status-dot state-${props.image.review_state}`} />
                <span>{titleCase(imageStateLabel(props.image.review_state))}</span>
                {unresolvedOnImage > 0 && <span>· {unresolvedOnImage} to review</span>}
                {props.busy && <span>· Saving…</span>}
              </div>
              <button
                className={`mark-reviewed-button ${props.image.review_state === "complete" ? "complete" : ""}`}
                disabled={props.busy || unresolvedOnImage > 0}
                onClick={() => void props.onMarkReviewed()}
                title={unresolvedOnImage > 0 ? "Resolve all AI proposals first" : undefined}
              >
                <Icon name="check" />
                {props.image.review_state === "complete" ? "Reviewed" : "Mark reviewed"}
              </button>
              <ImageActions
                key={props.image.id}
                canClear={props.image.annotations.length > 0}
                busy={props.busy || props.inferenceBusy}
                onClear={clearAnnotations}
                onDelete={deleteImage}
              />
            </div>
          </>
        ) : (
          <EmptyCanvas
            onImport={props.onImport}
            maxUploadBytes={props.capabilities?.limits.max_upload_bytes ?? 25 * 1024 * 1024}
          />
        )}
      </section>

      <ObjectPanel
        project={props.project}
        annotation={props.selectedAnnotation}
        busy={props.busy}
        onChangeAnnotationLabel={props.onChangeAnnotationLabel}
        onCreateClass={() => setClassDialogIntent("annotation")}
        onManageClasses={() => setClassManagerOpen(true)}
        onDeleteAnnotation={props.onDeleteAnnotation}
      />

      <ReviewBar
        annotation={props.selectedAnnotation}
        current={proposalIndex >= 0 ? proposalIndex + 1 : 0}
        total={proposals.length}
        busy={props.busy}
        onVerify={props.onVerify}
        onNext={selectNextProposal}
      />

      {classDialogIntent && (
        <CreateClassDialog
          project={props.project}
          busy={props.busy}
          intent={classDialogIntent}
          onClose={() => setClassDialogIntent(null)}
          onCreate={async (name, color) => {
            if (classDialogIntent === "annotation" && props.selectedAnnotation) {
              const saved = await props.onAssignAnnotationClass(props.selectedAnnotation.id, name, color);
              if (saved) setClassDialogIntent(null);
              return;
            }
            const labelId = await props.onAddLabel(name, color);
            if (labelId) {
              props.onSelectLabel(labelId);
              setClassDialogIntent(null);
            }
          }}
        />
      )}

      {classManagerOpen && (
        <ClassManagerDialog
          project={props.project}
          busy={props.busy}
          onClose={() => setClassManagerOpen(false)}
          onAddLabel={props.onAddLabel}
          onRenameLabel={props.onRenameLabel}
          onDeleteLabel={props.onDeleteLabel}
        />
      )}
    </main>
  );
}

function ZoomControls({ zoom, onZoomChange }: { zoom: number; onZoomChange: (zoom: number) => void }) {
  return (
    <div className="zoom-controls" aria-label="Image zoom controls">
      <button
        type="button"
        onClick={() => onZoomChange(Math.max(MIN_ZOOM, zoom - ZOOM_STEP))}
        disabled={zoom <= MIN_ZOOM}
        aria-label="Zoom out"
        title="Zoom out (Ctrl+-)"
      >
        <Icon name="minus" />
      </button>
      <output aria-label="Current zoom">{Math.round(zoom * 100)}%</output>
      <button
        type="button"
        onClick={() => onZoomChange(Math.min(MAX_ZOOM, zoom + ZOOM_STEP))}
        disabled={zoom >= MAX_ZOOM}
        aria-label="Zoom in"
        title="Zoom in (Ctrl++)"
      >
        <Icon name="plus" />
      </button>
      <button
        type="button"
        className="zoom-fit"
        onClick={() => onZoomChange(1)}
        disabled={zoom === 1}
        aria-label="Fit image to view"
        title="Fit to view (Ctrl+0)"
      >
        <Icon name="fit" />
        Fit
      </button>
    </div>
  );
}

function LabelModeControl({ value, onChange }: { value: CanvasLabelMode; onChange: (mode: CanvasLabelMode) => void }) {
  return (
    <label className="label-mode-control">
      <Icon name="classes" />
      <span>Labels</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value as CanvasLabelMode)}
        aria-label="Canvas label display"
      >
        <option value="hover">Hover</option>
        <option value="classes">Classes</option>
        <option value="hidden">Hidden</option>
      </select>
    </label>
  );
}

function FindBar(props: AnnotationWorkspaceProps) {
  return (
    <div className="find-bar">
      <form
        className="find-form"
        onSubmit={(event) => {
          event.preventDefault();
          void props.onGenerate();
        }}
      >
        <Icon name="search" />
        <label className="visually-hidden" htmlFor="localization-prompt">
          Find objects in this image
        </label>
        <input
          id="localization-prompt"
          value={props.prompt}
          onChange={(event) => props.onPromptChange(event.target.value)}
          placeholder="Find objects in this image…"
          maxLength={props.capabilities?.limits.prompt_max_characters ?? 500}
        />
        <kbd>Enter</kbd>
        <button
          className="primary-button find-button"
          type="submit"
          disabled={
            props.busy || props.inferenceBusy || !props.prompt.trim() || !props.capabilities?.features.inference
          }
        >
          {inferenceActionLabel(props.inferenceBusy, props.health)}
        </button>
      </form>
    </div>
  );
}

function ClassSelect({
  id,
  label,
  value,
  project,
  disabled,
  onClassChange,
  onCreate,
  onManage,
}: {
  id: string;
  label: string;
  value: string;
  project: Project;
  disabled: boolean;
  onClassChange: (id: UUID) => void;
  onCreate: () => void;
  onManage: () => void;
}) {
  return (
    <label className="class-select-control" htmlFor={id}>
      <span>{label}</span>
      <select
        id={id}
        aria-label={label}
        value={value}
        disabled={disabled}
        onChange={(event) => {
          if (event.target.value === CREATE_CLASS_VALUE) onCreate();
          else if (event.target.value === MANAGE_CLASSES_VALUE) onManage();
          else onClassChange(event.target.value);
        }}
      >
        {project.labels.length === 0 && <option value="">Choose a class</option>}
        {project.labels.map((item) => (
          <option key={item.id} value={item.id}>
            {item.name}
          </option>
        ))}
        <option disabled>──────────</option>
        <option value={CREATE_CLASS_VALUE}>+ Create new class</option>
        <option value={MANAGE_CLASSES_VALUE}>Manage classes…</option>
      </select>
    </label>
  );
}

function ObjectPanel({
  project,
  annotation,
  busy,
  onChangeAnnotationLabel,
  onCreateClass,
  onManageClasses,
  onDeleteAnnotation,
}: {
  project: Project;
  annotation: Annotation | null;
  busy: boolean;
  onChangeAnnotationLabel: (annotationId: UUID, labelId: UUID) => Promise<void>;
  onCreateClass: () => void;
  onManageClasses: () => void;
  onDeleteAnnotation: (annotationId: UUID) => Promise<void>;
}) {
  const label = project.labels.find((item) => item.id === annotation?.label_id);
  const status = annotation ? verificationStateLabel(annotation) : null;

  return (
    <section className="object-panel" aria-label="Selected object">
      {annotation ? (
        <div className="object-panel-content">
          <div className="selected-object-summary">
            <span className="selected-class-color" style={{ background: label?.color }} />
            <div>
              <h2>{label?.name || "Unlabeled"}</h2>
              <p>
                {annotation.source === "ai"
                  ? "AI proposal"
                  : annotation.verification_state === "human_added"
                    ? "Human added"
                    : "Human annotation"}
              </p>
            </div>
            <span className={`verification-badge state-${annotation.verification_state}`}>{status}</span>
          </div>
          <div className="object-class-field">
            <ClassSelect
              id="annotation-class"
              label="Class"
              value={annotation.label_id}
              project={project}
              disabled={busy || annotation.verification_state === "rejected"}
              onClassChange={(labelId) => void onChangeAnnotationLabel(annotation.id, labelId)}
              onCreate={onCreateClass}
              onManage={onManageClasses}
            />
          </div>
          <dl className="annotation-details">
            <DetailTerm term="Prompt" value={annotation.prompt || "—"} />
            <DetailTerm
              term="Original"
              value={annotation.original_ai_box ? formatBox(annotation.original_ai_box) : "—"}
            />
            <DetailTerm term="Final" value={annotation.final_box ? formatBox(annotation.final_box) : "—"} />
          </dl>
          {annotation.source === "human" && (
            <button
              className="delete-annotation-button"
              disabled={busy}
              onClick={() => {
                if (window.confirm("Delete this annotation? You can undo this action."))
                  void onDeleteAnnotation(annotation.id);
              }}
            >
              <Icon name="trash" />
              Delete annotation
            </button>
          )}
        </div>
      ) : (
        <div className="object-panel-empty">
          <Icon name="pointer" />
          <div>
            <h2>No annotation selected</h2>
            <p>Select a box to review or edit it.</p>
          </div>
        </div>
      )}
    </section>
  );
}

function ReviewBar({
  annotation,
  current,
  total,
  busy,
  onVerify,
  onNext,
}: {
  annotation: Annotation | null;
  current: number;
  total: number;
  busy: boolean;
  onVerify: (decision: VerificationDecision) => Promise<void>;
  onNext: () => void;
}) {
  const unresolved = annotation?.source === "ai" && annotation.verification_state === "unreviewed";
  const changed = annotation ? proposalWasChanged(annotation) : false;
  return (
    <aside className="review-bar" aria-label="AI proposal review">
      <div className="review-count">
        <span className="review-heading">AI proposals</span>
        <span>
          <strong>{current || "–"}</strong> / {total}
        </span>
      </div>
      <button
        className="review-accept"
        disabled={busy || !unresolved || changed}
        onClick={() => void onVerify("accepted")}
        title="Accept unchanged (A)"
      >
        <Icon name="check" />
        <span>Accept</span>
      </button>
      <button
        className="review-adjust"
        disabled={busy || !unresolved || !changed}
        onClick={() => void onVerify("adjusted")}
        title="Confirm adjustment"
      >
        <Icon name="box" />
        <span>Adjust</span>
      </button>
      <button
        className="review-reject"
        disabled={busy || !unresolved}
        onClick={() => void onVerify("rejected")}
        title="Reject proposal (R)"
      >
        <Icon name="close" />
        <span>Reject</span>
      </button>
      <button
        className="review-next"
        disabled={busy || total === 0}
        onClick={onNext}
        title="Next proposal"
        aria-label="Next proposal"
      >
        <Icon name="next" />
        <span>Next proposal</span>
      </button>
    </aside>
  );
}

function DetailTerm({ term, value }: { term: string; value: string }) {
  return (
    <div>
      <dt>{term}</dt>
      <dd title={value}>{value}</dd>
    </div>
  );
}

function ImageActions({
  canClear,
  busy,
  onClear,
  onDelete,
}: {
  canClear: boolean;
  busy: boolean;
  onClear: () => void;
  onDelete: () => void;
}) {
  return (
    <details className="image-actions-menu">
      <summary aria-label="Image actions" title="Image actions">
        <Icon name="more" />
      </summary>
      <div>
        <button disabled={busy || !canClear} onClick={onClear}>
          Clear annotations
        </button>
        <button className="destructive" disabled={busy} onClick={onDelete}>
          Delete image
        </button>
      </div>
    </details>
  );
}

function CreateClassDialog({
  project,
  busy,
  intent,
  onClose,
  onCreate,
}: {
  project: Project;
  busy: boolean;
  intent: Exclude<ClassDialogIntent, null>;
  onClose: () => void;
  onCreate: (name: string, color: string) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const color = LABEL_COLORS[project.labels.length % LABEL_COLORS.length];
  return (
    <Modal title="Create new class" onClose={onClose}>
      <form
        className="class-create-form"
        onSubmit={(event: FormEvent) => {
          event.preventDefault();
          if (name.trim()) void onCreate(name.trim(), color);
        }}
      >
        <p>
          {intent === "annotation"
            ? "Create and assign a class to the selected object."
            : "Create a reusable class for manual drawing."}
        </p>
        <label htmlFor="new-class-name">Class name</label>
        <div className="class-name-input">
          <span style={{ background: color }} />
          <input
            id="new-class-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            maxLength={80}
            autoFocus
            placeholder="e.g. Traffic Light"
          />
        </div>
        <div className="modal-actions">
          <button type="button" className="quiet-button" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="primary-button" disabled={busy || !name.trim()}>
            Create class
          </button>
        </div>
      </form>
    </Modal>
  );
}

function ClassManagerDialog({
  project,
  busy,
  onClose,
  onAddLabel,
  onRenameLabel,
  onDeleteLabel,
}: {
  project: Project;
  busy: boolean;
  onClose: () => void;
  onAddLabel: (name: string, color: string) => Promise<UUID | null>;
  onRenameLabel: (labelId: UUID, name: string) => Promise<boolean>;
  onDeleteLabel: (labelId: UUID) => Promise<boolean>;
}) {
  const [name, setName] = useState("");
  const [editingId, setEditingId] = useState<UUID | null>(null);
  const [renamed, setRenamed] = useState("");
  const usedLabels = useMemo(
    () =>
      new Set(
        project.images.flatMap((image) =>
          image.annotations.flatMap(
            (annotation) => [annotation.label_id, annotation.original_ai_label_id].filter(Boolean) as UUID[],
          ),
        ),
      ),
    [project.images],
  );
  const nextColor = LABEL_COLORS[project.labels.length % LABEL_COLORS.length];
  return (
    <Modal title="Manage classes" onClose={onClose} wide>
      <div className="class-manager">
        <p>Rename classes or remove classes that are not referenced by annotation history.</p>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            if (!name.trim()) return;
            void onAddLabel(name.trim(), nextColor).then((id) => {
              if (id) setName("");
            });
          }}
        >
          <span style={{ background: nextColor }} />
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="New class name"
            maxLength={80}
          />
          <button className="primary-button" disabled={busy || !name.trim()}>
            Add class
          </button>
        </form>
        <div className="class-manager-rows">
          {project.labels.map((item) =>
            editingId === item.id ? (
              <form
                key={item.id}
                onSubmit={(event) => {
                  event.preventDefault();
                  if (!renamed.trim()) return;
                  void onRenameLabel(item.id, renamed.trim()).then((saved) => {
                    if (saved) setEditingId(null);
                  });
                }}
              >
                <span style={{ background: item.color }} />
                <input
                  value={renamed}
                  onChange={(event) => setRenamed(event.target.value)}
                  autoFocus
                  aria-label={`Rename ${item.name}`}
                />
                <button disabled={busy || !renamed.trim()}>Save</button>
                <button type="button" onClick={() => setEditingId(null)}>
                  Cancel
                </button>
              </form>
            ) : (
              <div key={item.id}>
                <span style={{ background: item.color }} />
                <strong>{item.name}</strong>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => {
                    setEditingId(item.id);
                    setRenamed(item.name);
                  }}
                >
                  Rename
                </button>
                <button
                  type="button"
                  className="class-delete-button"
                  disabled={busy || usedLabels.has(item.id)}
                  title={
                    usedLabels.has(item.id) ? "This class is referenced by annotation history" : "Delete unused class"
                  }
                  onClick={() => {
                    if (window.confirm(`Delete unused class “${item.name}”?`)) void onDeleteLabel(item.id);
                  }}
                >
                  Delete
                </button>
              </div>
            ),
          )}
        </div>
      </div>
    </Modal>
  );
}

function Modal({
  title,
  children,
  onClose,
  wide = false,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const previousFocusRef = useRef(document.activeElement instanceof HTMLElement ? document.activeElement : null);

  useEffect(() => {
    const dialog = dialogRef.current;
    const previousFocus = previousFocusRef.current;
    dialog?.showModal();
    dialog?.querySelector<HTMLElement>("input:not([disabled]), textarea:not([disabled])")?.focus();
    return () => {
      if (dialog?.open) dialog.close();
      previousFocus?.focus();
    };
  }, []);

  const close = () => {
    dialogRef.current?.close();
    previousFocusRef.current?.focus();
    onClose();
  };

  return (
    <dialog
      ref={dialogRef}
      className="modal-layer"
      aria-label={title}
      onCancel={(event) => {
        event.preventDefault();
        close();
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          close();
        }
      }}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) close();
      }}
    >
      <section className={`modal-panel ${wide ? "wide" : ""}`}>
        <header>
          <h2>{title}</h2>
          <button onClick={close} aria-label={`Close ${title}`}>
            <Icon name="close" />
          </button>
        </header>
        {children}
      </section>
    </dialog>
  );
}

function EmptyCanvas({ onImport, maxUploadBytes }: { onImport: () => void; maxUploadBytes: number }) {
  return (
    <div className="empty-canvas">
      <div className="empty-frame">
        <Icon name="images" />
      </div>
      <h1>Bring in your first images</h1>
      <p>Drop JPEG, PNG, or WebP files here, or choose images from your computer.</p>
      <button className="primary-button" onClick={onImport}>
        <Icon name="import" />
        Import Images
      </button>
      <small>Up to {Math.floor(maxUploadBytes / (1024 * 1024))} MB per image</small>
    </div>
  );
}

function verificationStateLabel(annotation: Annotation): string {
  if (annotation.verification_state === "unreviewed") return "Needs review";
  if (annotation.verification_state === "human_added") return "Human added";
  if (annotation.verification_state === "manual") return "Manual";
  return titleCase(annotation.verification_state);
}

function formatBox(box: BoundingBox): string {
  return `${Math.round(box.x)}, ${Math.round(box.y)} · ${Math.round(box.width)} × ${Math.round(box.height)}`;
}
