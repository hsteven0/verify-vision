import { useEffect, useState } from "react";

import { api } from "../api";
import { APPLICATION_VERSION, REPOSITORY_URL } from "../appMetadata";
import type { ThemePreference } from "../theme";
import type { ApplicationCapabilities, DependencyUpdateStatus, HealthResponse, Project } from "../types";
import { Icon } from "./Icon";

function formatCheckedAt(value: string | null): string {
  if (!value) return "Never";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function statusLabel(status: DependencyUpdateStatus | null): string {
  if (!status) return "Loading…";
  if (status.state === "checking") return "Checking…";
  if (status.state === "updates_available") {
    return `${status.updates.length} update${status.updates.length === 1 ? "" : "s"} available`;
  }
  if (status.state === "up_to_date") return "Up to date";
  if (status.state === "unavailable") return "Check unavailable";
  return "Not checked";
}

export function SettingsWorkspace({
  project,
  capabilities,
  health,
  themePreference,
  onThemeChange,
  repositoryUrl = REPOSITORY_URL,
}: {
  project: Project;
  capabilities: ApplicationCapabilities | null;
  health: HealthResponse | null;
  themePreference: ThemePreference;
  onThemeChange: (preference: ThemePreference) => void;
  repositoryUrl?: string | null;
}) {
  const [dependencyStatus, setDependencyStatus] = useState<DependencyUpdateStatus | null>(null);
  const [checking, setChecking] = useState(false);
  const [updateError, setUpdateError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    api
      .getDependencyUpdates()
      .then((status) => {
        if (active) setDependencyStatus(status);
      })
      .catch(() => {
        if (active) setUpdateError("Dependency status could not be loaded.");
      });
    return () => {
      active = false;
    };
  }, []);

  const checkForUpdates = async () => {
    setChecking(true);
    setUpdateError(null);
    try {
      setDependencyStatus(await api.checkDependencyUpdates());
    } catch (error) {
      setUpdateError(error instanceof Error ? error.message : "Update check failed.");
    } finally {
      setChecking(false);
    }
  };

  return (
    <main className="settings-page app-page">
      <header className="page-heading">
        <div>
          <h1>Settings</h1>
          <p>Theme, local runtime, and dependency updates.</p>
        </div>
      </header>

      <div className="settings-sections">
        <section className="settings-section">
          <div>
            <h2>Appearance</h2>
            <p>Choose a theme or use the system setting.</p>
          </div>
          <label className="settings-field" htmlFor="appearance-setting">
            <span>Theme</span>
            <select
              id="appearance-setting"
              value={themePreference}
              onChange={(event) => onThemeChange(event.target.value as ThemePreference)}
            >
              <option value="system">System</option>
              <option value="dark">Dark</option>
              <option value="light">Light</option>
            </select>
          </label>
        </section>

        <section className="settings-section about-settings">
          <div>
            <h2>About</h2>
          </div>
          <div className="about-panel">
            <div className="about-product">
              <strong>VerifyVision</strong>
              <span>v{APPLICATION_VERSION}</span>
            </div>
            {repositoryUrl ? (
              <a
                className="secondary-button github-repository-link"
                href={repositoryUrl}
                target="_blank"
                rel="noreferrer"
              >
                <Icon name="github" />
                GitHub repository
              </a>
            ) : (
              <button
                className="secondary-button github-repository-link"
                type="button"
                disabled
                title="Repository link unavailable"
              >
                <Icon name="github" />
                GitHub repository
              </button>
            )}
            {!repositoryUrl ? <p className="settings-helper">Repository link unavailable.</p> : null}
          </div>
        </section>

        <section className="settings-section">
          <div>
            <h2>Local AI runtime</h2>
            <p>LocateAnything-3B runs locally with NVIDIA CUDA.</p>
          </div>
          <dl className="settings-facts">
            <div>
              <dt>Model</dt>
              <dd>{capabilities?.inference.model.replace("nvidia/", "") ?? "LocateAnything-3B"}</dd>
            </div>
            <div>
              <dt>Device</dt>
              <dd>{health?.inference_gpu || health?.inference_device || "Checking CUDA…"}</dd>
            </div>
            <div>
              <dt>Status</dt>
              <dd>{health?.inference_available ? "Available" : "Unavailable"}</dd>
            </div>
          </dl>
        </section>

        <section className="settings-section dependency-settings">
          <div>
            <h2>Dependencies</h2>
            <p>Checking does not install updates.</p>
          </div>
          <div className="dependency-status-panel">
            <dl className="settings-facts">
              <div>
                <dt>Status</dt>
                <dd>{statusLabel(dependencyStatus)}</dd>
              </div>
              <div>
                <dt>Last checked</dt>
                <dd>{formatCheckedAt(dependencyStatus?.last_checked_at ?? null)}</dd>
              </div>
            </dl>
            <button
              className="secondary-button"
              type="button"
              onClick={() => void checkForUpdates()}
              disabled={checking}
            >
              {checking ? "Checking…" : "Check for updates"}
            </button>
            {(updateError || dependencyStatus?.detail) && (
              <p className={updateError ? "form-error" : "settings-helper"}>
                {updateError ?? dependencyStatus?.detail}
              </p>
            )}
            {dependencyStatus?.updates.length ? (
              <details className="dependency-review">
                <summary>Review updates</summary>
                <div className="dependency-update-list">
                  {dependencyStatus.updates.map((update) => (
                    <article key={`${update.ecosystem}:${update.package}`}>
                      <div>
                        <strong>{update.package}</strong>
                        <span>
                          {update.current_version} → {update.latest_version}
                        </span>
                      </div>
                      <span className={`dependency-risk ${update.risk}`}>
                        {update.risk === "critical_runtime" ? "ML runtime" : "Application"}
                      </span>
                    </article>
                  ))}
                </div>
                <p>Use the project update script to install reviewed changes.</p>
              </details>
            ) : null}
          </div>
        </section>

        <section className="settings-section">
          <div>
            <h2>Current project</h2>
          </div>
          <dl className="settings-facts">
            <div>
              <dt>Name</dt>
              <dd>{project.name}</dd>
            </div>
            <div>
              <dt>Images</dt>
              <dd>{project.images.length}</dd>
            </div>
            <div>
              <dt>Classes</dt>
              <dd>{project.labels.length}</dd>
            </div>
          </dl>
        </section>
      </div>
    </main>
  );
}
