import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";

import { api } from "./api";
import { proposalWasChanged } from "./annotationUi";
import { AnalyticsDashboard } from "./components/AnalyticsDashboard";
import { ImageList } from "./components/AnnotateLibraryPane";
import { AnnotationWorkspace } from "./components/AnnotationWorkspace";
import type { AppView } from "./components/AppNavigation";
import { ExportWorkspace } from "./components/ExportWorkspace";
import { GlobalNavigationDrawer } from "./components/GlobalNavigationDrawer";
import { Icon } from "./components/Icon";
import { SettingsWorkspace } from "./components/SettingsWorkspace";
import { TrainingWorkspace } from "./components/TrainingWorkspace";
import { TestWorkspace } from "./components/TestWorkspace";
import {
  createEditHistory,
  recordEdit,
  redoTarget,
  snapshotProject,
  undoTarget,
  type EditHistory,
} from "./editHistory";
import { isCurrentProposal } from "./inferenceUi";
import { useTheme } from "./theme";
import type {
  ApplicationCapabilities,
  BoundingBox,
  EvaluationSummary,
  HealthResponse,
  Project,
  ProjectSummary,
  UUID,
  VerificationDecision,
} from "./types";

const ACTIVE_PROJECT_KEY = "verifyvision.activeProject";

function App() {
  const { preference: themePreference, resolvedTheme, setPreference: setThemePreference } = useTheme();
  const [capabilities, setCapabilities] = useState<ApplicationCapabilities | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
  const [selectedImageId, setSelectedImageId] = useState<UUID | null>(null);
  const [selectedLabelId, setSelectedLabelId] = useState<UUID | null>(null);
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<UUID | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [inferenceBusy, setInferenceBusy] = useState(false);
  const [inferenceNotice, setInferenceNotice] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [navigationOpen, setNavigationOpen] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [activeView, setActiveView] = useState<AppView>("workspace");
  const [connectionAttempt, setConnectionAttempt] = useState(0);
  const [history, setHistory] = useState<EditHistory>(() => createEditHistory());
  const fileInput = useRef<HTMLInputElement>(null);
  const historyRef = useRef(history);
  const historyProjectId = useRef<UUID | null>(null);

  const replaceHistory = useCallback((next: EditHistory) => {
    historyRef.current = next;
    setHistory(next);
  }, []);

  const clearHistory = useCallback(() => replaceHistory(createEditHistory()), [replaceHistory]);

  const selectedImage = useMemo(
    () => project?.images.find((image) => image.id === selectedImageId) ?? null,
    [project, selectedImageId],
  );
  const selectedAnnotation = useMemo(
    () => selectedImage?.annotations.find((annotation) => annotation.id === selectedAnnotationId) ?? null,
    [selectedImage, selectedAnnotationId],
  );
  const selectedImageIndex =
    project && selectedImage ? project.images.findIndex((image) => image.id === selectedImage.id) : -1;

  useEffect(() => {
    const initialize = async () => {
      setLoading(true);
      setError(null);
      try {
        const nextCapabilities = await api.getCapabilities();
        setCapabilities(nextCapabilities);
        const [summaries, providerHealth] = await Promise.all([api.listProjects(), api.getHealth()]);
        setProjects(summaries);
        setHealth(providerHealth);
        if (summaries.length > 0) {
          const remembered = localStorage.getItem(ACTIVE_PROJECT_KEY);
          const active = summaries.find((item) => item.id === remembered) ?? summaries[0];
          const [loadedProject, loadedEvaluation] = await Promise.all([
            api.getProject(active.id),
            api.getEvaluation(active.id),
          ]);
          setProject(loadedProject);
          setEvaluation(loadedEvaluation);
        }
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Could not load VerifyVision");
      } finally {
        setLoading(false);
      }
    };
    void initialize();
  }, [connectionAttempt]);

  useEffect(() => {
    if (!project) return;
    if (historyProjectId.current !== project.id) {
      historyProjectId.current = project.id;
      clearHistory();
    }
    localStorage.setItem(ACTIVE_PROJECT_KEY, project.id);
    if (!project.images.some((image) => image.id === selectedImageId)) {
      setSelectedImageId(project.images[0]?.id ?? null);
      setSelectedAnnotationId(null);
    }
    if (!project.labels.some((label) => label.id === selectedLabelId)) {
      setSelectedLabelId(project.labels[0]?.id ?? null);
    }
    if (
      selectedAnnotationId &&
      !project.images.some((image) => image.annotations.some((annotation) => annotation.id === selectedAnnotationId))
    ) {
      setSelectedAnnotationId(null);
    }
  }, [clearHistory, project, selectedAnnotationId, selectedImageId, selectedLabelId]);

  const latestImagePrompt = useMemo(
    () =>
      [...(selectedImage?.annotations ?? [])].reverse().find((annotation) => annotation.source === "ai")?.prompt ?? "",
    [selectedImage],
  );

  useEffect(() => {
    setPrompt(latestImagePrompt);
  }, [selectedImageId, latestImagePrompt]);

  useEffect(() => setInferenceNotice(null), [selectedImageId]);

  useEffect(() => {
    if (!inferenceBusy) return;
    let active = true;
    const refreshInferenceHealth = async () => {
      try {
        const nextHealth = await api.getHealth();
        if (active) setHealth(nextHealth);
      } catch {
        // The active inference request owns the user-facing error state.
      }
    };
    void refreshInferenceHealth();
    const interval = window.setInterval(() => void refreshInferenceHealth(), 2000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [inferenceBusy]);

  const refreshProjectList = useCallback(async () => setProjects(await api.listProjects()), []);

  const applyMutation = useCallback(
    async (operation: () => Promise<Project>, historyDescription?: string): Promise<Project | null> => {
      setBusy(true);
      setError(null);
      try {
        const before = historyDescription && project ? snapshotProject(project) : null;
        const updated = await operation();
        if (before && historyDescription) {
          replaceHistory(recordEdit(historyRef.current, historyDescription, before, snapshotProject(updated)));
        }
        setProject(updated);
        try {
          await refreshProjectList();
        } catch {
          setError("The change was saved, but the project list could not be refreshed.");
        }
        try {
          setEvaluation(await api.getEvaluation(updated.id));
        } catch {
          setError("The change was saved, but evaluation metrics could not be refreshed.");
        }
        return updated;
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "The change could not be saved");
        return null;
      } finally {
        setBusy(false);
      }
    },
    [project, refreshProjectList, replaceHistory],
  );

  const restoreHistory = useCallback(
    async (direction: "undo" | "redo") => {
      if (!project || busy || inferenceBusy) return;
      const target = direction === "undo" ? undoTarget(historyRef.current) : redoTarget(historyRef.current);
      if (!target) return;
      setBusy(true);
      setError(null);
      try {
        const restored = await api.restoreHistoryState(project.id, target.snapshot);
        setProject(restored);
        replaceHistory(target.history);
        const results = await Promise.allSettled([refreshProjectList(), api.getEvaluation(restored.id)]);
        if (results[1].status === "fulfilled") setEvaluation(results[1].value);
        if (results.some((result) => result.status === "rejected")) {
          setError(`${direction === "undo" ? "Undo" : "Redo"} worked, but the view did not fully refresh.`);
        }
      } catch (caught) {
        setError(
          caught instanceof Error ? caught.message : `${direction === "undo" ? "Undo" : "Redo"} could not be completed`,
        );
      } finally {
        setBusy(false);
      }
    },
    [busy, inferenceBusy, project, refreshProjectList, replaceHistory],
  );

  const createProject = async (name: string) => {
    const created = await applyMutation(() => api.createProject(name));
    if (!created) return;
    setSelectedImageId(null);
    setSelectedAnnotationId(null);
    setActiveView("workspace");
  };

  const openProject = async (projectId: string) => {
    setLoading(true);
    setError(null);
    try {
      const [loadedProject, loadedEvaluation] = await Promise.all([
        api.getProject(projectId),
        api.getEvaluation(projectId),
      ]);
      setProject(loadedProject);
      setEvaluation(loadedEvaluation);
      setSelectedImageId(null);
      setSelectedAnnotationId(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The project could not be opened");
    } finally {
      setLoading(false);
    }
  };

  const importFiles = async (files: File[]) => {
    if (!project || files.length === 0) return;
    const before = project.images.length;
    const updated = await applyMutation(() => api.uploadImages(project.id, files));
    if (!updated) return;
    clearHistory();
    setSelectedImageId(updated.images[before]?.id ?? updated.images[0]?.id ?? null);
  };

  const createBox = async (box: BoundingBox) => {
    if (!project || !selectedImage || !selectedLabelId) return;
    const beforeIds = new Set(selectedImage.annotations.map((annotation) => annotation.id));
    const updated = await applyMutation(
      () =>
        api.createAnnotation(
          project.id,
          selectedImage.id,
          selectedLabelId,
          box,
          "human_added",
          prompt.trim() || undefined,
        ),
      "Draw box",
    );
    if (!updated) return;
    const updatedImage = updated.images.find((image) => image.id === selectedImage.id);
    const created = updatedImage?.annotations.find((annotation) => !beforeIds.has(annotation.id));
    setSelectedAnnotationId(created?.id ?? null);
  };

  const updateBox = async (annotationId: UUID, box: BoundingBox, action: "Move box" | "Resize box") => {
    if (!project) return;
    await applyMutation(() => api.updateAnnotation(project.id, annotationId, { box }), action);
  };

  const moveImage = useCallback(
    (direction: -1 | 1) => {
      if (!project || selectedImageIndex < 0) return;
      const next = project.images[selectedImageIndex + direction];
      if (next) {
        setSelectedImageId(next.id);
        setSelectedAnnotationId(null);
      }
    },
    [project, selectedImageIndex],
  );

  const generateProposals = async () => {
    if (!project || !selectedImage || !prompt.trim() || inferenceBusy) {
      return;
    }
    const requestedPrompt = prompt;
    const beforeIds = new Set(selectedImage.annotations.map((annotation) => annotation.id));
    setInferenceBusy(true);
    setInferenceNotice(null);
    setError(null);
    try {
      const updated = await api.generateProposals(project.id, selectedImage.id, requestedPrompt);
      setProject(updated);
      clearHistory();
      const updatedImage = updated.images.find((image) => image.id === selectedImage.id);
      const newProposals =
        updatedImage?.annotations.filter(
          (annotation) => !beforeIds.has(annotation.id) && annotation.verification_state === "unreviewed",
        ) ?? [];
      const matchingProposals =
        updatedImage?.annotations.filter(
          (annotation) =>
            annotation.prompt === requestedPrompt &&
            isCurrentProposal(annotation, health?.inference_provider, health?.inference_model),
        ) ?? [];
      const cached = matchingProposals.find((annotation) => annotation.verification_state === "unreviewed");
      setSelectedAnnotationId(newProposals[0]?.id ?? cached?.id ?? null);
      const selectedProposal = newProposals[0] ?? cached;
      if (selectedProposal) setSelectedLabelId(selectedProposal.label_id);
      if (newProposals.length > 0) {
        setInferenceNotice(`${newProposals.length} proposal${newProposals.length === 1 ? "" : "s"} ready for review.`);
      } else if (matchingProposals.length > 0) {
        setInferenceNotice("Results for this prompt are already saved in the project.");
      } else {
        setInferenceNotice(`No detections found for “${requestedPrompt}”.`);
      }
      const results = await Promise.allSettled([refreshProjectList(), api.getEvaluation(updated.id), api.getHealth()]);
      const evaluationResult = results[1];
      const healthResult = results[2];
      if (evaluationResult.status === "fulfilled") setEvaluation(evaluationResult.value);
      if (healthResult.status === "fulfilled") setHealth(healthResult.value);
      if (results.some((result) => result.status === "rejected")) {
        setError("Proposals were saved, but the view did not fully refresh.");
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Inference could not be completed");
      try {
        setHealth(await api.getHealth());
      } catch {
        // The original inference error is the most useful message to retain.
      }
    } finally {
      setInferenceBusy(false);
    }
  };

  const selectNextUnresolved = useCallback((updated: Project, imageId: UUID, currentAnnotationId: UUID) => {
    const annotations = updated.images.find((image) => image.id === imageId)?.annotations ?? [];
    const currentIndex = annotations.findIndex((annotation) => annotation.id === currentAnnotationId);
    const ordered = [...annotations.slice(currentIndex + 1), ...annotations.slice(0, currentIndex)];
    setSelectedAnnotationId(
      ordered.find((annotation) => annotation.verification_state === "unreviewed")?.id ?? currentAnnotationId,
    );
  }, []);

  const verifySelected = useCallback(
    async (decision: VerificationDecision) => {
      if (!project || !selectedImage || !selectedAnnotation || selectedAnnotation.source !== "ai") {
        return;
      }
      const updated = await applyMutation(
        () => api.verifyAnnotation(project.id, selectedAnnotation.id, decision),
        decision === "accepted"
          ? "Accept proposal"
          : decision === "adjusted"
            ? "Confirm adjustment"
            : "Reject proposal",
      );
      if (updated) selectNextUnresolved(updated, selectedImage.id, selectedAnnotation.id);
    },
    [applyMutation, project, selectNextUnresolved, selectedAnnotation, selectedImage],
  );

  const deleteSelectedAnnotation = useCallback(async () => {
    if (!project || !selectedAnnotation || selectedAnnotation.source !== "human") return;
    const updated = await applyMutation(() => api.deleteAnnotation(project.id, selectedAnnotation.id), "Delete box");
    if (updated) setSelectedAnnotationId(null);
  }, [applyMutation, project, selectedAnnotation]);

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      const target = event.target;
      if (
        busy ||
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        (target instanceof HTMLElement && target.isContentEditable)
      ) {
        return;
      }

      const key = event.key.toLowerCase();
      const modifier = event.ctrlKey || event.metaKey;
      if (modifier && key === "z") {
        event.preventDefault();
        void restoreHistory(event.shiftKey ? "redo" : "undo");
      } else if (modifier && key === "y") {
        event.preventDefault();
        void restoreHistory("redo");
      } else if (
        !modifier &&
        key === "a" &&
        selectedAnnotation?.verification_state === "unreviewed" &&
        !proposalWasChanged(selectedAnnotation)
      ) {
        event.preventDefault();
        void verifySelected("accepted");
      } else if (!modifier && key === "r" && selectedAnnotation?.verification_state === "unreviewed") {
        event.preventDefault();
        void verifySelected("rejected");
      } else if (!modifier && key === "n") {
        event.preventDefault();
        moveImage(1);
      } else if (!modifier && key === "p") {
        event.preventDefault();
        moveImage(-1);
      } else if (
        !modifier &&
        (event.key === "Delete" || event.key === "Backspace") &&
        selectedAnnotation?.source === "human"
      ) {
        event.preventDefault();
        if (window.confirm("Delete this annotation? You can undo this action.")) {
          void deleteSelectedAnnotation();
        }
      }
    };
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [busy, deleteSelectedAnnotation, moveImage, restoreHistory, selectedAnnotation, verifySelected]);

  if (loading) {
    return (
      <main className="loading-screen">
        <div className="brand-mark" aria-hidden="true">
          <span />
        </div>
        <p>Connecting to VerifyVision…</p>
        <small>Checking the local backend and CUDA runtime.</small>
      </main>
    );
  }

  if (!project) {
    return (
      <WelcomeScreen
        onCreate={createProject}
        onRetry={() => setConnectionAttempt((attempt) => attempt + 1)}
        busy={busy}
        error={error}
      />
    );
  }

  return (
    <div className={`app-shell ${activeView === "workspace" ? "has-annotation-library" : ""}`}>
      {activeView === "workspace" && (
        <ImageList
          project={project}
          projects={projects}
          selectedImageId={selectedImageId}
          busy={busy}
          inferenceBusy={inferenceBusy}
          onOpenProject={openProject}
          onCreateProject={createProject}
          onSelectImage={(imageId) => {
            setSelectedImageId(imageId);
            setSelectedAnnotationId(null);
          }}
          onImport={() => fileInput.current?.click()}
        />
      )}

      <GlobalNavigationDrawer
        open={navigationOpen}
        activeView={activeView}
        onClose={() => setNavigationOpen(false)}
        onChange={setActiveView}
      />

      <div className="app-main">
        <header className="topbar">
          <div className="topbar-leading">
            <button
              className="global-menu-button"
              type="button"
              onClick={() => setNavigationOpen(true)}
              aria-label="Open navigation"
              aria-expanded={navigationOpen}
            >
              <Icon name="menu" />
            </button>
            <div className="topbar-context">
              <span>{activeView === "workspace" ? "Annotate" : activeView[0].toUpperCase() + activeView.slice(1)}</span>
              <i aria-hidden="true">/</i>
              <strong>{project.name}</strong>
            </div>
          </div>

          <div
            className={`runtime-context-pill ${capabilities?.features.inference ? "mode-local" : "mode-unavailable"}`}
          >
            <i aria-hidden="true" />
            <span>Local · LocateAnything-3B</span>
          </div>

          <div className="topbar-actions">
            <input
              ref={fileInput}
              className="visually-hidden"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              multiple
              onChange={(event) => {
                void importFiles(Array.from(event.target.files ?? []));
                event.target.value = "";
              }}
            />
            <button
              className="topbar-theme-button"
              type="button"
              onClick={() => setThemePreference(resolvedTheme === "dark" ? "light" : "dark")}
              aria-label={`Switch to ${resolvedTheme === "dark" ? "light" : "dark"} theme`}
              title={`Switch to ${resolvedTheme === "dark" ? "light" : "dark"} theme`}
            >
              <Icon name={resolvedTheme === "dark" ? "sun" : "moon"} />
            </button>
          </div>
        </header>

        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <button onClick={() => setError(null)} aria-label="Dismiss error">
              <Icon name="close" />
            </button>
          </div>
        )}

        {capabilities && !capabilities.features.inference && activeView === "workspace" && (
          <div className="error-banner runtime-warning" role="alert">
            <span>
              {capabilities.inference.disclosure} VerifyVision requires Windows or Linux with an NVIDIA CUDA-capable GPU
              for local AI inference.
            </span>
          </div>
        )}

        {activeView === "analytics" ? (
          <AnalyticsDashboard
            project={project}
            activeSource={{
              label: "Model",
              value: (health?.inference_model ?? "nvidia/LocateAnything-3B").replace("nvidia/", ""),
            }}
            onOpenExport={() => setActiveView("export")}
            onInspectImage={(imageId, annotationId) => {
              setSelectedImageId(imageId);
              setSelectedAnnotationId(annotationId);
              setActiveView("workspace");
            }}
          />
        ) : activeView === "test" ? (
          <TestWorkspace project={project} enabled={Boolean(capabilities?.features.trained_model_prediction)} />
        ) : activeView === "export" ? (
          <ExportWorkspace project={project} />
        ) : activeView === "training" ? (
          <TrainingWorkspace project={project} evaluation={evaluation} onOpenTest={() => setActiveView("test")} />
        ) : activeView === "settings" ? (
          <SettingsWorkspace
            project={project}
            capabilities={capabilities}
            health={health}
            themePreference={themePreference}
            onThemeChange={setThemePreference}
          />
        ) : (
          <AnnotationWorkspace
            project={project}
            image={selectedImage}
            selectedImageId={selectedImageId}
            selectedImageIndex={selectedImageIndex}
            selectedLabelId={selectedLabelId}
            selectedAnnotationId={selectedAnnotationId}
            selectedAnnotation={selectedAnnotation}
            capabilities={capabilities}
            health={health}
            prompt={prompt}
            busy={busy}
            inferenceBusy={inferenceBusy}
            inferenceNotice={inferenceNotice}
            canUndo={history.index > 0}
            canRedo={history.index < history.entries.length}
            undoDescription={history.entries[history.index - 1]?.description ?? null}
            redoDescription={history.entries[history.index]?.description ?? null}
            onSelectImage={(imageId) => {
              setSelectedImageId(imageId);
              setSelectedAnnotationId(null);
            }}
            onMoveImage={moveImage}
            onSelectLabel={setSelectedLabelId}
            onSelectAnnotation={setSelectedAnnotationId}
            onPromptChange={setPrompt}
            onGenerate={generateProposals}
            onDismissInferenceNotice={() => setInferenceNotice(null)}
            onCreateBox={createBox}
            onUpdateBox={updateBox}
            onUndo={() => restoreHistory("undo")}
            onRedo={() => restoreHistory("redo")}
            onMarkReviewed={async () => {
              if (!selectedImage) return;
              await applyMutation(
                () =>
                  api.updateImageReview(
                    project.id,
                    selectedImage.id,
                    selectedImage.review_state === "complete" ? "in_progress" : "complete",
                  ),
                selectedImage.review_state === "complete" ? "Reopen image review" : "Mark image reviewed",
              );
            }}
            onImport={() => fileInput.current?.click()}
            onDropFiles={importFiles}
            onAddLabel={async (name, color) => {
              const existing = project.labels.find(
                (label) => label.name.localeCompare(name, undefined, { sensitivity: "accent" }) === 0,
              );
              if (existing) return existing.id;
              const updated = await applyMutation(() => api.addLabel(project.id, name, color), "Create class");
              return (
                updated?.labels.find(
                  (label) => label.name.localeCompare(name, undefined, { sensitivity: "accent" }) === 0,
                )?.id ?? null
              );
            }}
            onRenameLabel={async (labelId, name) => {
              const updated = await applyMutation(() => api.renameLabel(project.id, labelId, name), "Rename class");
              return updated !== null;
            }}
            onDeleteLabel={async (labelId) => {
              const updated = await applyMutation(() => api.deleteLabel(project.id, labelId), "Delete class");
              return updated !== null;
            }}
            onAssignAnnotationClass={async (annotationId, name, color) => {
              const updated = await applyMutation(
                () => api.assignAnnotationLabel(project.id, annotationId, name, color),
                "Change class",
              );
              if (!updated) return false;
              const annotation = updated.images
                .flatMap((image) => image.annotations)
                .find((item) => item.id === annotationId);
              if (annotation) setSelectedLabelId(annotation.label_id);
              return true;
            }}
            onChangeAnnotationLabel={async (annotationId, labelId) => {
              setSelectedLabelId(labelId);
              await applyMutation(
                () => api.updateAnnotation(project.id, annotationId, { label_id: labelId }),
                "Change class",
              );
            }}
            onDeleteAnnotation={async () => deleteSelectedAnnotation()}
            onClearAnnotations={async () => {
              if (!selectedImage) return;
              const updated = await applyMutation(
                () => api.clearImageAnnotations(project.id, selectedImage.id),
                "Clear annotations",
              );
              if (updated) setSelectedAnnotationId(null);
            }}
            onDeleteImage={async () => {
              if (!selectedImage) return;
              const currentIndex = selectedImageIndex;
              const updated = await applyMutation(() => api.deleteImage(project.id, selectedImage.id));
              if (!updated) return;
              clearHistory();
              const adjacent = updated.images[Math.min(currentIndex, updated.images.length - 1)];
              setSelectedImageId(adjacent?.id ?? null);
              setSelectedAnnotationId(null);
            }}
            onVerify={verifySelected}
          />
        )}
      </div>
    </div>
  );
}

function WelcomeScreen({
  onCreate,
  onRetry,
  busy,
  error,
}: {
  onCreate: (name: string) => Promise<void>;
  onRetry: () => void;
  busy: boolean;
  error: string | null;
}) {
  return (
    <main className="welcome-screen">
      <section className="welcome-copy">
        <div className="brand-lockup welcome-brand">
          <div className="brand-mark">
            <span />
          </div>
          <div>
            <strong>VerifyVision</strong>
          </div>
        </div>
        <h1>AI-assisted annotation with human verification.</h1>
        <p className="welcome-lede">
          Locate objects, correct the boxes, measure the results, and export verified training data.
        </p>
      </section>
      <section className="welcome-card">
        <h2>Create a project</h2>
        <p>Name the dataset you want to annotate.</p>
        <NewProjectForm busy={busy} onCreate={onCreate} />
        {error && (
          <div className="welcome-error" role="alert">
            <p className="form-error">{error}</p>
            <button className="secondary-button" onClick={onRetry}>
              Retry connection
            </button>
          </div>
        )}
        <p className="local-note">
          <span>●</span> Project data and AI inference stay on this machine.
        </p>
      </section>
    </main>
  );
}

function NewProjectForm({ onCreate, busy }: { onCreate: (name: string) => Promise<void>; busy: boolean }) {
  const [name, setName] = useState("");
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (name.trim()) void onCreate(name.trim());
  };
  return (
    <form className="new-project-form" onSubmit={submit}>
      <input
        id="new-project-name"
        value={name}
        onChange={(event) => setName(event.target.value)}
        placeholder="e.g. Urban cyclists"
        maxLength={120}
      />
      <button className="primary-button" type="submit" disabled={busy || !name.trim()}>
        {busy ? "Creating…" : "Create project"}
      </button>
    </form>
  );
}

export default App;
