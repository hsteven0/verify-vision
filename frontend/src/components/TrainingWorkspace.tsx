import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import type {
  DetectionMetrics,
  EvaluationSummary,
  Project,
  TrainingArtifactKind,
  TrainingAvailability,
  TrainingConfig,
  TrainingProgressPoint,
  TrainingReadiness,
  TrainingRun,
  TrainingStage,
  UUID,
} from "../types";
import {
  MetricCell,
  SecondaryWorkspace,
  WorkspaceEmptyState,
  WorkspaceHeader,
  WorkspacePanel,
} from "./SecondaryWorkspace";

const DEFAULT_CONFIG: TrainingConfig = {
  checkpoint: "yolo26n.pt",
  epochs: 20,
  image_size: 640,
  batch_size: 8,
  train_ratio: 0.8,
  seed: 1337,
  device: "auto",
  run_name: "verified-dataset",
};

const ACTIVE_STATUSES = new Set(["queued", "preparing", "training", "validating", "cancelling"]);

const PREPARATION_STAGES: Array<{ stage: TrainingStage; label: string }> = [
  { stage: "validating_dataset", label: "Validate verified dataset" },
  { stage: "preparing_dataset", label: "Prepare dataset snapshot" },
  { stage: "loading_model", label: "Load pretrained weights" },
  { stage: "initializing_runtime", label: "Initialize CUDA and AMP" },
  { stage: "preparing_dataloader", label: "Prepare data loader" },
];

function formatMetric(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${remainder}s`;
}

function formatBytes(value: number): string {
  if (value < 1024 * 1024) return `${Math.max(1, Math.round(value / 1024))} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function formatRunDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function humanize(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

export function TrainingWorkspace({
  project,
  onOpenTest,
}: {
  project: Project;
  evaluation: EvaluationSummary | null;
  onOpenTest?: () => void;
}) {
  const [availability, setAvailability] = useState<TrainingAvailability | null>(null);
  const [readiness, setReadiness] = useState<TrainingReadiness | null>(null);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<UUID | null>(null);
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());

  const replaceRun = (updated: TrainingRun) => {
    setRuns((current) => {
      const found = current.some((run) => run.id === updated.id);
      return found ? current.map((run) => (run.id === updated.id ? updated : run)) : [updated, ...current];
    });
  };

  useEffect(() => {
    let current = true;
    setLoading(true);
    setError(null);
    Promise.all([
      api.getTrainingAvailability(project.id),
      api.listTrainingRuns(project.id),
      api.getTrainingReadiness(project.id, DEFAULT_CONFIG),
    ])
      .then(([nextAvailability, nextRuns, nextReadiness]) => {
        if (!current) return;
        setAvailability(nextAvailability);
        setRuns(nextRuns);
        setSelectedRunId(nextRuns[0]?.id ?? null);
        setReadiness(nextReadiness);
      })
      .catch((caught: unknown) => {
        if (current) {
          setError(caught instanceof Error ? caught.message : "Training details could not load");
        }
      })
      .finally(() => {
        if (current) setLoading(false);
      });
    return () => {
      current = false;
    };
  }, [project.id]);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      void api
        .getTrainingReadiness(project.id, {
          ...DEFAULT_CONFIG,
          train_ratio: config.train_ratio,
          seed: config.seed,
        })
        .then(setReadiness)
        .catch((caught: unknown) =>
          setError(caught instanceof Error ? caught.message : "Readiness could not be checked"),
        );
    }, 250);
    return () => window.clearTimeout(timeout);
  }, [config.seed, config.train_ratio, project.id]);

  const activeRun = runs.find((run) => ACTIVE_STATUSES.has(run.status));
  const activeRunId = activeRun?.id;
  useEffect(() => {
    if (!activeRunId) return;
    const interval = window.setInterval(() => {
      void api
        .getTrainingRun(project.id, activeRunId)
        .then(replaceRun)
        .catch((caught: unknown) =>
          setError(caught instanceof Error ? caught.message : "Training status could not refresh"),
        );
    }, 1500);
    return () => window.clearInterval(interval);
  }, [activeRunId, project.id]);

  useEffect(() => {
    if (!activeRunId) return;
    const interval = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, [activeRunId]);

  const selectedRun = useMemo(
    () => runs.find((run) => run.id === selectedRunId) ?? runs[0] ?? null,
    [runs, selectedRunId],
  );
  const displayedRun = activeRun ?? selectedRun;
  const selectedModel = availability?.supported_models.find((model) => model.checkpoint === config.checkpoint);

  const updateConfig = <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) =>
    setConfig((current) => ({ ...current, [key]: value }));

  const startTraining = async () => {
    if (!readiness?.ready || !availability?.available || starting || activeRun) return;
    setStarting(true);
    setError(null);
    try {
      const run = await api.createTrainingRun(project.id, config);
      replaceRun(run);
      setSelectedRunId(run.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Training could not start");
    } finally {
      setStarting(false);
    }
  };

  const cancelTraining = async (run: TrainingRun) => {
    if (run.status === "cancelling") return;
    setError(null);
    try {
      replaceRun(await api.cancelTrainingRun(project.id, run.id));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Training could not be cancelled");
    }
  };

  if (loading) {
    return (
      <SecondaryWorkspace className="training-workspace secondary-loading">
        <span className="analytics-loading-mark" />
        <p>Checking dataset and CUDA…</p>
      </SecondaryWorkspace>
    );
  }

  const datasetImages = readiness?.preview.exportable_images ?? 0;
  const datasetBoxes = readiness?.preview.training_boxes ?? 0;
  const datasetClasses = readiness?.preview.classes ?? project.labels.length;
  const trainingAvailable = Boolean(availability?.available);
  const canStart = Boolean(readiness?.ready && trainingAvailable && !activeRun && !starting);

  return (
    <SecondaryWorkspace className="training-workspace">
      <WorkspaceHeader
        title="Train a YOLO model"
        description={
          <>
            Train on reviewed annotations from <strong>{project.name}</strong>.
          </>
        }
      />

      {error && (
        <p className="secondary-inline-error" role="alert">
          {error}
        </p>
      )}

      <section className="secondary-kpi-grid training-summary-grid" aria-label="Training summary">
        <MetricCell
          label="Verified dataset"
          value={`${datasetImages.toLocaleString()} images`}
          detail={`${datasetBoxes.toLocaleString()} annotations · ${datasetClasses.toLocaleString()} classes`}
        />
        <MetricCell
          label="Model"
          value={selectedModel?.label ? `YOLO26 ${selectedModel.label}` : config.checkpoint}
          detail={`${config.checkpoint} · pretrained`}
        />
        <MetricCell
          label="Device"
          value={availability?.gpu_name?.replace("NVIDIA GeForce ", "") ?? "CUDA required"}
          detail={
            availability?.gpu_memory_gb
              ? `CUDA · ${availability.gpu_memory_gb.toFixed(0)} GB VRAM`
              : (availability?.reason ?? "NVIDIA GPU not detected")
          }
        />
      </section>

      <div className="training-main-grid">
        <WorkspacePanel
          title="Training configuration"
          description="Choose the model and core training settings."
          className="training-configuration-panel"
        >
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void startTraining();
            }}
          >
            <div className="training-primary-fields">
              <label className="secondary-field training-model-field">
                <span>Model</span>
                <select
                  value={config.checkpoint}
                  disabled={Boolean(activeRun)}
                  onChange={(event) => updateConfig("checkpoint", event.target.value)}
                >
                  {(availability?.supported_models ?? []).map((model) => (
                    <option key={model.checkpoint} value={model.checkpoint}>
                      YOLO26 {model.label} · {model.checkpoint}
                    </option>
                  ))}
                </select>
              </label>
              <NumberField
                label="Epochs"
                value={config.epochs}
                min={1}
                max={300}
                disabled={Boolean(activeRun)}
                onChange={(value) => updateConfig("epochs", value)}
              />
              <NumberField
                label="Batch size"
                value={config.batch_size}
                min={1}
                max={128}
                disabled={Boolean(activeRun)}
                onChange={(value) => updateConfig("batch_size", value)}
              />
              <label className="secondary-field">
                <span>Image size</span>
                <select
                  value={config.image_size}
                  disabled={Boolean(activeRun)}
                  onChange={(event) => updateConfig("image_size", Number(event.target.value))}
                >
                  {[320, 640, 960].map((size) => (
                    <option key={size} value={size}>
                      {size} px
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <details className="secondary-advanced training-advanced-settings">
              <summary>Advanced settings</summary>
              <div className="training-advanced-fields">
                <label className="secondary-field">
                  <span>Train / validation split</span>
                  <select
                    value={config.train_ratio}
                    disabled={Boolean(activeRun)}
                    onChange={(event) => updateConfig("train_ratio", Number(event.target.value))}
                  >
                    <option value={0.7}>70 / 30</option>
                    <option value={0.8}>80 / 20</option>
                    <option value={0.9}>90 / 10</option>
                  </select>
                </label>
                <NumberField
                  label="Split seed"
                  value={config.seed}
                  min={0}
                  max={2147483647}
                  disabled={Boolean(activeRun)}
                  onChange={(value) => updateConfig("seed", value)}
                />
                <label className="secondary-field training-run-name-field">
                  <span>Run name</span>
                  <input
                    value={config.run_name}
                    maxLength={80}
                    disabled={Boolean(activeRun)}
                    onChange={(event) => updateConfig("run_name", event.target.value)}
                  />
                </label>
                <div className="training-readonly-device">
                  <span>Runtime</span>
                  <strong>{availability?.gpu_name ?? "NVIDIA CUDA required"}</strong>
                  <small>Ultralytics checks AMP when training starts.</small>
                </div>
              </div>
            </details>

            <div className="training-readiness-row">
              <div className={readiness?.ready ? "ready" : "blocked"}>
                <strong>{readiness?.ready ? "Dataset ready" : "Dataset needs attention"}</strong>
                <span>
                  {readiness?.ready
                    ? `${readiness.preview.train_images} train and ${readiness.preview.validation_images} validation images will be used.`
                    : (readiness?.blockers[0] ?? availability?.reason ?? "Readiness is unavailable.")}
                </span>
                {readiness && (readiness.advisories.length > 0 || readiness.blockers.length > 1) && (
                  <details>
                    <summary>Review validation notes</summary>
                    {[...readiness.blockers, ...readiness.advisories].map((message) => (
                      <p key={message}>{message}</p>
                    ))}
                  </details>
                )}
              </div>
              <button className="primary-button training-start-button" type="submit" disabled={!canStart}>
                {starting ? "Starting…" : activeRun ? "Training in progress" : "Start training"}
              </button>
            </div>
          </form>
        </WorkspacePanel>

        {displayedRun ? (
          <TrainingRunPanel
            run={displayedRun}
            now={now}
            onCancel={() => void cancelTraining(displayedRun)}
            onDownload={(kind) => void api.downloadTrainingArtifact(project.id, displayedRun.id, kind)}
            onOpenTest={onOpenTest}
          />
        ) : (
          <WorkspaceEmptyState title="No training run yet" description="Choose settings, then start training." />
        )}
      </div>

      {runs.length > 0 && (
        <WorkspacePanel
          title="Recent runs"
          description="Completed, cancelled, and failed runs are saved here."
          aside={<span className="secondary-panel-count">{runs.length} runs</span>}
          className="training-history-panel"
        >
          <div className="training-history-list">
            {runs.map((run) => (
              <button
                type="button"
                key={run.id}
                className={run.id === displayedRun?.id ? "active" : ""}
                onClick={() => setSelectedRunId(run.id)}
              >
                <i className={`training-run-dot status-${run.status}`} />
                <span>
                  <strong>{run.config.run_name}</strong>
                  <small>
                    {run.config.checkpoint} · {formatRunDate(run.created_at)}
                  </small>
                </span>
                <em>{run.status === "cancelling" ? "Cancelling…" : humanize(run.status)}</em>
                {run.validation_metrics && <b>{formatMetric(run.validation_metrics.map50)}</b>}
              </button>
            ))}
          </div>
        </WorkspacePanel>
      )}
    </SecondaryWorkspace>
  );
}

function TrainingRunPanel({
  run,
  now,
  onCancel,
  onDownload,
  onOpenTest,
}: {
  run: TrainingRun;
  now: number;
  onCancel: () => void;
  onDownload: (kind: TrainingArtifactKind) => void;
  onOpenTest?: () => void;
}) {
  const active = ACTIVE_STATUSES.has(run.status);
  const cancelling = run.status === "cancelling";
  const elapsed = run.started_at
    ? ((run.completed_at ? new Date(run.completed_at).getTime() : now) - new Date(run.started_at).getTime()) / 1000
    : null;
  const completedEpochs = Math.min(run.current_epoch, run.total_epochs);
  const progress = run.total_epochs ? (completedEpochs / run.total_epochs) * 100 : 0;
  const measuredEpochs = run.progress_history.filter((point) => point.duration_seconds !== null);
  const averageEpoch = measuredEpochs.length
    ? measuredEpochs.reduce((total, point) => total + (point.duration_seconds ?? 0), 0) / measuredEpochs.length
    : null;
  const remaining =
    averageEpoch && measuredEpochs.length >= 2 ? averageEpoch * Math.max(0, run.total_epochs - completedEpochs) : null;
  const latestPoint = run.progress_history.at(-1) ?? null;

  return (
    <WorkspacePanel
      title={run.config.run_name}
      description={`${run.config.checkpoint} · ${run.runtime?.gpu_name ?? run.resolved_device ?? "Runtime pending"}`}
      aside={
        <span className={`training-state-pill status-${run.status}`} aria-live="polite">
          {cancelling ? "Cancelling…" : humanize(run.status)}
        </span>
      }
      className="training-progress-panel"
    >
      {active && (
        <div className="training-active-layout">
          <div className="training-stage-column">
            <div className="training-live-status" aria-live="polite">
              <span>{cancelling ? "Cancelling safely" : humanize(run.stage)}</span>
              <strong>{run.stage_message}</strong>
              <small>{formatDuration(elapsed)} elapsed</small>
            </div>

            {run.status === "preparing" || run.status === "queued" ? (
              <PreparationChecklist stage={run.stage} />
            ) : (
              <div className="training-epoch-progress">
                <div>
                  <span>Epoch progress</span>
                  <strong>
                    {completedEpochs} / {run.total_epochs}
                  </strong>
                </div>
                <div
                  className="training-progress-track"
                  role="progressbar"
                  aria-label="Training epoch progress"
                  aria-valuemin={0}
                  aria-valuemax={run.total_epochs}
                  aria-valuenow={completedEpochs}
                >
                  <i style={{ width: `${progress}%` }} />
                </div>
                <div className="training-progress-meta">
                  <span>
                    {latestPoint?.training_loss !== null && latestPoint?.training_loss !== undefined
                      ? `Loss ${latestPoint.training_loss.toFixed(3)}`
                      : "Loss pending"}
                  </span>
                  <span>{remaining ? `About ${formatDuration(remaining)} remaining` : "ETA pending"}</span>
                </div>
              </div>
            )}

            <button
              type="button"
              className="secondary-button training-cancel-button"
              onClick={onCancel}
              disabled={cancelling}
            >
              {cancelling ? "Cancelling…" : "Cancel training"}
            </button>
          </div>

          <div className="training-chart-shell">
            {run.progress_history.length ? (
              <TrainingProgressChart points={run.progress_history} />
            ) : (
              <div className="training-chart-empty">
                <span className="analytics-loading-mark" />
                <strong>Waiting for epoch metrics</strong>
                <small>Waiting for the first epoch.</small>
              </div>
            )}
          </div>
        </div>
      )}

      {run.status === "completed" && run.validation_metrics && (
        <CompletedTrainingRun run={run} onDownload={onDownload} onOpenTest={onOpenTest} />
      )}

      {run.status === "cancelled" && (
        <div className="training-terminal-state cancelled" role="status">
          <span>Cancelled</span>
          <strong>Training stopped.</strong>
          <p>Cancelled after {formatDuration(run.timings.cancellation_seconds)}. The project was not changed.</p>
          {run.artifacts.some((artifact) => artifact.kind === "last_checkpoint") && (
            <small>A last valid checkpoint was preserved, but this run is not marked complete.</small>
          )}
        </div>
      )}

      {run.status === "failed" && (
        <div className="training-terminal-state failed" role="alert">
          <span>Failed</span>
          <strong>Training did not complete.</strong>
          <p>{run.failure_reason ?? "Review the backend log for technical details."}</p>
        </div>
      )}

      <TrainingTechnicalDetails run={run} />
    </WorkspacePanel>
  );
}

function PreparationChecklist({ stage }: { stage: TrainingStage }) {
  const currentIndex = PREPARATION_STAGES.findIndex((item) => item.stage === stage);
  return (
    <ol className="training-preparation-list" aria-label="Training preparation steps">
      {PREPARATION_STAGES.map((item, index) => {
        const state =
          currentIndex < 0 ? "waiting" : index < currentIndex ? "done" : index === currentIndex ? "current" : "waiting";
        return (
          <li className={state} key={item.stage}>
            <i aria-hidden="true" />
            <span>{item.label}</span>
            <small>{state === "done" ? "Complete" : state === "current" ? "In progress" : "Waiting"}</small>
          </li>
        );
      })}
    </ol>
  );
}

function CompletedTrainingRun({
  run,
  onDownload,
  onOpenTest,
}: {
  run: TrainingRun;
  onDownload: (kind: TrainingArtifactKind) => void;
  onOpenTest?: () => void;
}) {
  const metrics = run.validation_metrics as DetectionMetrics;
  return (
    <>
      <section className="secondary-kpi-grid training-results-grid" aria-label="Final validation metrics">
        <MetricCell label="Precision" value={formatMetric(metrics.precision)} detail="Correct predictions" />
        <MetricCell label="Recall" value={formatMetric(metrics.recall)} detail="Objects found" />
        <MetricCell label="mAP50" value={formatMetric(metrics.map50)} detail="IoU 0.50" tone="positive" />
        <MetricCell label="mAP50–95" value={formatMetric(metrics.map50_95)} detail="Across IoU thresholds" />
      </section>

      {run.progress_history.length > 0 && (
        <div className="training-completed-chart">
          <TrainingProgressChart points={run.progress_history} />
        </div>
      )}

      <div className="training-result-actions">
        <div>
          <span>Run completed in {formatDuration(run.timings.total_seconds)}</span>
          <strong>{run.validation_image_count} held-out validation images</strong>
        </div>
        {onOpenTest && (
          <button type="button" className="primary-button" onClick={onOpenTest}>
            Test model
          </button>
        )}
        {run.artifacts
          .filter((artifact) => ["best_checkpoint", "configuration"].includes(artifact.kind))
          .map((artifact) => (
            <button
              type="button"
              className="secondary-button"
              key={artifact.kind}
              onClick={() => onDownload(artifact.kind)}
            >
              {artifact.kind === "best_checkpoint" ? "Download best checkpoint" : "Download results"}
              <small>{formatBytes(artifact.size_bytes)}</small>
            </button>
          ))}
      </div>

      {run.per_class_metrics.length > 0 && (
        <details className="secondary-advanced training-class-details">
          <summary>Per-class validation</summary>
          <div className="analytics-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>mAP50</th>
                  <th>mAP50–95</th>
                </tr>
              </thead>
              <tbody>
                {run.per_class_metrics.map((item) => (
                  <tr key={item.class_id}>
                    <td>{item.class_name}</td>
                    <td>{formatMetric(item.precision)}</td>
                    <td>{formatMetric(item.recall)}</td>
                    <td>{formatMetric(item.map50)}</td>
                    <td>{formatMetric(item.map50_95)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </>
  );
}

function TrainingProgressChart({ points }: { points: TrainingProgressPoint[] }) {
  const ordered = [...points].sort((a, b) => a.epoch - b.epoch);
  const width = 760;
  const height = 240;
  const padding = 34;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  const x = (index: number) =>
    padding + (ordered.length === 1 ? plotWidth / 2 : (index / (ordered.length - 1)) * plotWidth);
  const losses = ordered.map((point) => point.training_loss).filter((value): value is number => value !== null);
  const maxLoss = Math.max(...losses, 1);
  const lossCoordinates = ordered
    .map((point, index) =>
      point.training_loss === null
        ? null
        : {
            x: x(index),
            y: padding + plotHeight - (point.training_loss / maxLoss) * plotHeight,
          },
    )
    .filter((value): value is { x: number; y: number } => value !== null);
  const mapCoordinates = ordered
    .map((point, index) =>
      point.metrics?.map50 === null || point.metrics?.map50 === undefined
        ? null
        : {
            x: x(index),
            y: padding + plotHeight - point.metrics.map50 * plotHeight,
          },
    )
    .filter((value): value is { x: number; y: number } => value !== null);
  const lossPoints = lossCoordinates.map((point) => `${point.x},${point.y}`).join(" ");
  const mapPoints = mapCoordinates.map((point) => `${point.x},${point.y}`).join(" ");

  return (
    <figure className="training-progress-chart">
      <figcaption>
        <div>
          <span>Training curve</span>
          <strong>Loss and validation mAP50</strong>
        </div>
        <div className="training-chart-legend">
          <span className="loss">Training loss</span>
          <span className="map">Validation mAP50</span>
        </div>
      </figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training loss and validation mAP50 by epoch">
        {[0, 0.5, 1].map((ratio) => (
          <line
            key={ratio}
            x1={padding}
            x2={width - padding}
            y1={padding + ratio * plotHeight}
            y2={padding + ratio * plotHeight}
          />
        ))}
        {lossPoints && <polyline className="loss-line" points={lossPoints} />}
        {mapPoints && <polyline className="map-line" points={mapPoints} />}
        {lossCoordinates.map((point, index) => (
          <circle key={`loss-${index}`} className="loss-point" cx={point.x} cy={point.y} r="4" />
        ))}
        {mapCoordinates.map((point, index) => (
          <circle key={`map-${index}`} className="map-point" cx={point.x} cy={point.y} r="4" />
        ))}
      </svg>
      <div className="training-chart-axis">
        <span>Epoch {ordered[0]?.epoch}</span>
        <span>Epoch {ordered.at(-1)?.epoch}</span>
      </div>
    </figure>
  );
}

function TrainingTechnicalDetails({ run }: { run: TrainingRun }) {
  return (
    <details className="secondary-advanced training-technical-details">
      <summary>Technical details</summary>
      <dl>
        <div>
          <dt>Device</dt>
          <dd>{run.runtime?.gpu_name ?? run.resolved_device ?? "Pending"}</dd>
        </div>
        <div>
          <dt>AMP</dt>
          <dd>
            {run.runtime?.amp_enabled === null || !run.runtime
              ? "Pending"
              : run.runtime.amp_enabled
                ? "Enabled"
                : "Disabled by safety check"}
          </dd>
        </div>
        <div>
          <dt>Batch size</dt>
          <dd>{run.runtime?.batch_size ?? run.config.batch_size}</dd>
        </div>
        <div>
          <dt>Workers</dt>
          <dd>{run.runtime?.workers ?? "Pending"}</dd>
        </div>
        <div>
          <dt>Dataset cache</dt>
          <dd>{run.timings.dataset_reused ? "Reused" : "Prepared for run"}</dd>
        </div>
        <div>
          <dt>Startup to training</dt>
          <dd>{formatDuration(run.timings.startup_seconds)}</dd>
        </div>
        <div>
          <dt>Validation time</dt>
          <dd>{formatDuration(run.timings.validation_seconds)}</dd>
        </div>
        <div>
          <dt>Dataset SHA-256</dt>
          <dd>{run.dataset ? `${run.dataset.sha256.slice(0, 12)}…` : "Pending"}</dd>
        </div>
      </dl>
    </details>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label className="secondary-field">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}
