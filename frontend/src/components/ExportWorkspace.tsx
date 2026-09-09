import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import type { ExportFormat, ExportOptions, ExportPreview, Project } from "../types";
import { MetricCell, SecondaryWorkspace, WorkspaceHeader, WorkspacePanel } from "./SecondaryWorkspace";

const FORMAT_COPY: Record<ExportFormat, { name: string; description: string; action: string; training: boolean }> = {
  yolo: {
    name: "YOLO",
    description: "Export the verified dataset for YOLO training.",
    action: "Export YOLO",
    training: true,
  },
  coco: {
    name: "COCO",
    description: "Export annotations in standard COCO JSON format.",
    action: "Export COCO",
    training: true,
  },
  pascal_voc: {
    name: "Pascal VOC",
    description: "Export one Pascal VOC XML annotation file per image.",
    action: "Export Pascal VOC",
    training: true,
  },
  evaluation_csv: {
    name: "Evaluation CSV",
    description: "Export annotation metrics for Excel, pandas, or other analysis tools.",
    action: "Export CSV",
    training: false,
  },
};

export function ExportWorkspace({ project }: { project: Project }) {
  const [split, setSplit] = useState<ExportOptions["split"]>("none");
  const [trainPercent, setTrainPercent] = useState(80);
  const [seed, setSeed] = useState(1337);
  const [preview, setPreview] = useState<ExportPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState<ExportFormat | null>(null);
  const [error, setError] = useState<string | null>(null);

  const options = useMemo<ExportOptions>(
    () => ({ split, train_ratio: trainPercent / 100, seed }),
    [seed, split, trainPercent],
  );

  useEffect(() => {
    let cancelled = false;
    const timeout = window.setTimeout(() => {
      setLoading(true);
      setPreview(null);
      setError(null);
      void api
        .getExportPreview(project.id, options)
        .then((result) => {
          if (!cancelled) setPreview(result);
        })
        .catch((caught) => {
          if (!cancelled) {
            setError(caught instanceof Error ? caught.message : "Dataset validation could not run");
          }
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    }, 180);
    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [options, project.id]);

  const download = async (format: ExportFormat) => {
    setDownloading(format);
    setError(null);
    try {
      await api.downloadExport(project.id, format, options);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Export could not be generated");
    } finally {
      setDownloading(null);
    }
  };

  const trainingBlocked = !preview || preview.blocking_error_count > 0;
  const hasExclusions = Boolean(preview && preview.excluded_unresolved > 0);
  return (
    <SecondaryWorkspace className="export-workspace">
      <WorkspaceHeader
        title="Export verified annotations"
        description={
          <>
            Choose a format for <strong>{project.name}</strong>. YOLO, COCO, and Pascal VOC use reviewed boxes.
          </>
        }
      />

      {error && (
        <p className="secondary-inline-error" role="alert">
          {error}
        </p>
      )}

      <WorkspacePanel
        title={
          loading
            ? "Checking annotations…"
            : trainingBlocked
              ? "Resolve export issues"
              : hasExclusions
                ? "Ready with exclusions"
                : "Dataset ready"
        }
        description="Final boxes and current review status."
        className="export-readiness-panel"
      >
        {preview ? (
          <>
            <div className="secondary-kpi-grid export-readiness-grid">
              <MetricCell
                label="Images"
                value={preview.exportable_images.toLocaleString()}
                detail={`${preview.total_images.toLocaleString()} imported`}
              />
              <MetricCell
                label="Verified annotations"
                value={preview.training_boxes.toLocaleString()}
                detail="Final training boxes"
              />
              <MetricCell label="Classes" value={preview.classes.toLocaleString()} detail="Stable class order" />
              <MetricCell
                label="Rejected excluded"
                value={preview.excluded_rejected.toLocaleString()}
                detail={`${preview.excluded_unresolved} unresolved excluded`}
                tone="danger"
              />
            </div>

            {(preview.findings.length > 0 || preview.warning_count > 0) && (
              <div className="export-validation-summary">
                <div>
                  <strong>
                    {preview.blocking_error_count
                      ? `${preview.blocking_error_count} issue${preview.blocking_error_count === 1 ? "" : "s"} must be fixed`
                      : "Export can proceed with warnings"}
                  </strong>
                  <span>
                    {preview.warning_count} warning{preview.warning_count === 1 ? "" : "s"}
                  </span>
                </div>
                <details>
                  <summary>View validation details</summary>
                  <div>
                    {preview.findings.map((finding, index) => (
                      <p
                        className={`export-validation-finding ${finding.severity}`}
                        key={`${finding.code}-${finding.image_id ?? "project"}-${index}`}
                      >
                        <strong>{finding.severity}</strong>
                        <span>{finding.message}</span>
                      </p>
                    ))}
                  </div>
                </details>
              </div>
            )}
          </>
        ) : (
          <div className="export-readiness-loading" aria-busy="true">
            <i />
            <i />
            <i />
            <i />
          </div>
        )}
      </WorkspacePanel>

      <section className="export-format-grid" aria-label="Export formats">
        {(Object.keys(FORMAT_COPY) as ExportFormat[]).map((format) => {
          const item = FORMAT_COPY[format];
          return (
            <ExportFormatCard
              key={format}
              name={item.name}
              description={item.description}
              action={item.action}
              disabled={loading || Boolean(downloading) || (item.training && trainingBlocked)}
              blocked={item.training && trainingBlocked && !loading}
              working={downloading === format}
              onDownload={() => void download(format)}
            />
          );
        })}
      </section>

      <WorkspacePanel
        title="Export options"
        description="Optional train/validation split for training exports."
        className="export-options-panel"
      >
        <details className="secondary-advanced export-advanced-options">
          <summary>Configure dataset split</summary>
          <div className="export-options-content">
            <fieldset>
              <legend>Dataset split</legend>
              <div className="secondary-segmented-control">
                <button type="button" className={split === "none" ? "active" : ""} onClick={() => setSplit("none")}>
                  No split
                </button>
                <button
                  type="button"
                  className={split === "train_val" ? "active" : ""}
                  onClick={() => setSplit("train_val")}
                >
                  Train / validation
                </button>
              </div>
            </fieldset>
            {split === "train_val" && (
              <div className="export-split-fields">
                <label className="secondary-field">
                  <span>Train percentage</span>
                  <input
                    type="number"
                    min="50"
                    max="95"
                    value={trainPercent}
                    onChange={(event) => setTrainPercent(Number(event.target.value))}
                  />
                </label>
                <label className="secondary-field">
                  <span>Split seed</span>
                  <input type="number" min="0" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
                </label>
                {preview && (
                  <p>
                    {preview.train_images} train · {preview.validation_images} validation images
                  </p>
                )}
              </div>
            )}
          </div>
        </details>
      </WorkspacePanel>
    </SecondaryWorkspace>
  );
}

function ExportFormatCard({
  name,
  description,
  action,
  disabled,
  blocked,
  working,
  onDownload,
}: {
  name: string;
  description: string;
  action: string;
  disabled: boolean;
  blocked: boolean;
  working: boolean;
  onDownload: () => void;
}) {
  return (
    <article className="export-format-card">
      <div className="export-format-copy">
        <h2>{name}</h2>
        <p>{description}</p>
      </div>
      <button className="secondary-button" disabled={disabled} onClick={onDownload}>
        {working ? "Preparing…" : blocked ? "Resolve validation issues" : action}
      </button>
    </article>
  );
}
