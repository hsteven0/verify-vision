import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "../api";
import type { DetectionMetrics, PredictionPreview, Project, TrainingRun, UUID } from "../types";
import { PredictionPreviewCanvas } from "./PredictionPreviewCanvas";
import {
  MetricCell,
  SecondaryWorkspace,
  WorkspaceEmptyState,
  WorkspaceHeader,
  WorkspacePanel,
} from "./SecondaryWorkspace";

type TestSource = "project" | "upload";

function formatMetric(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatRunDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(new Date(value));
}

export function TestWorkspace({ project, enabled = true }: { project: Project; enabled?: boolean }) {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<UUID | null>(null);
  const [source, setSource] = useState<TestSource>("project");
  const [selectedImageId, setSelectedImageId] = useState<UUID>(project.images[0]?.id ?? "");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [confidence, setConfidence] = useState(0.25);
  const [preview, setPreview] = useState<PredictionPreview | null>(null);
  const [previewImageUrl, setPreviewImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(enabled);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const uploadInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setPreview(null);
    setPreviewImageUrl(null);
    setUploadFile(null);
    setSelectedImageId(project.images[0]?.id ?? "");
    if (!enabled) {
      setRuns([]);
      setSelectedRunId(null);
      setLoading(false);
      return;
    }

    let current = true;
    setLoading(true);
    setError(null);
    void api
      .listTrainingRuns(project.id)
      .then((result) => {
        if (!current) return;
        const completed = result.filter((run) => run.status === "completed" && run.validation_metrics !== null);
        setRuns(completed);
        setSelectedRunId(completed[0]?.id ?? null);
      })
      .catch((caught: unknown) => {
        if (current) {
          setError(caught instanceof Error ? caught.message : "Trained models could not be loaded");
        }
      })
      .finally(() => {
        if (current) setLoading(false);
      });
    return () => {
      current = false;
    };
  }, [enabled, project.id, project.images]);

  const selectedRun = useMemo(
    () => runs.find((run) => run.id === selectedRunId) ?? runs[0] ?? null,
    [runs, selectedRunId],
  );

  const canRun = Boolean(selectedRun && (source === "project" ? selectedImageId : uploadFile) && !running);

  const runTest = async () => {
    if (!selectedRun || !canRun) return;
    setRunning(true);
    setError(null);
    try {
      if (source === "project") {
        const result = await api.predictProjectImage(project.id, selectedRun.id, selectedImageId, confidence);
        setPreview(result);
        setPreviewImageUrl(api.imageUrl(project.id, selectedImageId));
      } else if (uploadFile) {
        const result = await api.predictUploadedImage(project.id, selectedRun.id, uploadFile, confidence);
        setPreview(result);
        setPreviewImageUrl(api.trainingPredictionImageUrl(project.id, selectedRun.id, result.id));
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Model test could not run");
    } finally {
      setRunning(false);
    }
  };

  if (loading) {
    return (
      <SecondaryWorkspace className="test-workspace test-loading">
        <div className="analytics-loading-mark" />
        <p>Loading trained models…</p>
      </SecondaryWorkspace>
    );
  }

  if (!enabled || runs.length === 0) {
    return (
      <SecondaryWorkspace className="test-workspace">
        <WorkspaceHeader
          title="Test a trained model"
          description="Run a trained model on a project image or a separate image."
        />
        {error && (
          <p className="secondary-inline-error" role="alert">
            {error}
          </p>
        )}
        <WorkspaceEmptyState
          title="No trained model available"
          description={
            enabled ? "Complete a training run before testing." : "Complete a local training run before testing."
          }
        />
      </SecondaryWorkspace>
    );
  }

  const metrics = selectedRun?.validation_metrics ?? null;

  return (
    <SecondaryWorkspace className="test-workspace">
      <WorkspaceHeader
        title="Test a trained model"
        description="Run a trained checkpoint without changing project annotations."
      />

      {error && (
        <p className="secondary-inline-error" role="alert">
          {error}
        </p>
      )}

      <WorkspacePanel
        title="Choose a model and image"
        description="Run one trained checkpoint against one image."
        className="test-configuration-panel"
      >
        <form
          className="test-configuration"
          onSubmit={(event) => {
            event.preventDefault();
            void runTest();
          }}
        >
          <label className="secondary-field test-model-field">
            <span>Model</span>
            <select
              value={selectedRun?.id ?? ""}
              onChange={(event) => {
                setSelectedRunId(event.target.value);
                setPreview(null);
                setPreviewImageUrl(null);
              }}
            >
              {runs.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.config.run_name} · {run.config.checkpoint} · {formatRunDate(run.created_at)}
                </option>
              ))}
            </select>
          </label>

          <fieldset className="test-source-fieldset">
            <legend>Test source</legend>
            <div className="secondary-segmented-control">
              <button
                type="button"
                className={source === "project" ? "active" : ""}
                onClick={() => setSource("project")}
              >
                Project image
              </button>
              <button type="button" className={source === "upload" ? "active" : ""} onClick={() => setSource("upload")}>
                Held-out upload
              </button>
            </div>
          </fieldset>

          {source === "project" ? (
            <label className="secondary-field test-image-field">
              <span>Image</span>
              <select value={selectedImageId} onChange={(event) => setSelectedImageId(event.target.value)}>
                {project.images.map((image) => (
                  <option key={image.id} value={image.id}>
                    {image.filename}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <div className="secondary-field test-upload-field">
              <span>Image</span>
              <button type="button" className="secondary-button" onClick={() => uploadInput.current?.click()}>
                {uploadFile ? uploadFile.name : "Choose held-out image"}
              </button>
            </div>
          )}

          <button className="primary-button test-run-button" type="submit" disabled={!canRun}>
            {running ? "Running test…" : "Run test"}
          </button>

          <details className="secondary-advanced test-advanced">
            <summary>Advanced</summary>
            <label>
              <span>Confidence threshold</span>
              <input
                type="range"
                aria-label="Confidence threshold"
                min="0.05"
                max="0.95"
                step="0.05"
                value={confidence}
                onChange={(event) => setConfidence(Number(event.target.value))}
              />
              <output>{Math.round(confidence * 100)}%</output>
            </label>
          </details>
        </form>
      </WorkspacePanel>

      <input
        ref={uploadInput}
        className="visually-hidden"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={(event) => {
          setUploadFile(event.target.files?.[0] ?? null);
          setPreview(null);
          setPreviewImageUrl(null);
          event.target.value = "";
        }}
      />

      <WorkspacePanel
        title={preview ? "Prediction result" : "Result preview"}
        description="Test results do not change the project."
        className="test-result-panel"
      >
        {preview && previewImageUrl ? (
          <PredictionPreviewCanvas preview={preview} imageUrl={previewImageUrl} className="test-prediction-canvas" />
        ) : (
          <div className="test-preview-empty">
            <strong>No result yet</strong>
            <p>Choose a model and image, then run a test.</p>
          </div>
        )}
      </WorkspacePanel>

      {metrics && <ValidationMetrics metrics={metrics} run={selectedRun} />}
    </SecondaryWorkspace>
  );
}

function ValidationMetrics({ metrics, run }: { metrics: DetectionMetrics; run: TrainingRun }) {
  return (
    <WorkspacePanel
      title="Validation metrics"
      description={`${run.validation_image_count} validation image${run.validation_image_count === 1 ? "" : "s"} from this training run.`}
      className="test-metrics-panel"
    >
      <div className="secondary-kpi-grid">
        <MetricCell
          label="Precision"
          value={formatMetric(metrics.precision)}
          detail="Correct among predicted objects"
        />
        <MetricCell label="Recall" value={formatMetric(metrics.recall)} detail="Found among expected objects" />
        <MetricCell label="mAP50" value={formatMetric(metrics.map50)} detail="Detection quality at 0.50 IoU" />
        <MetricCell
          label="mAP50–95"
          value={formatMetric(metrics.map50_95)}
          detail="Quality across stricter IoU thresholds"
        />
      </div>
    </WorkspacePanel>
  );
}
