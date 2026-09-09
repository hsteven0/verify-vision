import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import type {
  AnalyticsExamples,
  AnalyticsFilters,
  AnalyticsSummary,
  ClassAnalytics,
  CountRate,
  ExampleKind,
  Project,
  PromptAnalytics,
  UUID,
} from "../types";
import {
  MetricCell,
  SecondaryWorkspace,
  WorkspaceEmptyState,
  WorkspaceHeader,
  WorkspacePanel,
} from "./SecondaryWorkspace";
import { Icon } from "./Icon";

const EMPTY_FILTERS: AnalyticsFilters = {
  label_id: null,
  provider: null,
  model: null,
  prompt: null,
};

type TableSort = "risk" | "volume" | "name";

interface AnalyticsDashboardProps {
  project: Project;
  activeSource?: { label: "Model" | "Source"; value: string };
  onInspectImage: (imageId: UUID, annotationId: UUID | null) => void;
  onOpenExport: () => void;
}

function formatPercent(metric: CountRate): string {
  return metric.denominator ? `${metric.rate.toFixed(1)}%` : "—";
}

function formatIou(value: number | null): string {
  return value === null ? "—" : value.toFixed(2);
}

function plural(value: number, singular: string, pluralForm = `${singular}s`): string {
  return `${value.toLocaleString()} ${value === 1 ? singular : pluralForm}`;
}

function distinctProviderCohorts(
  cohorts: Array<{ provider: string; model: string }>,
): Array<{ provider: string; model: string }> {
  const distinct = new Map<string, { provider: string; model: string }>();
  for (const cohort of cohorts) {
    const key = `${cohort.provider.trim().toLocaleLowerCase()}\0${cohort.model.trim().toLocaleLowerCase()}`;
    if (!distinct.has(key)) distinct.set(key, cohort);
  }
  return [...distinct.values()];
}

function AnalyticsDashboard({ project, activeSource, onInspectImage, onOpenExport }: AnalyticsDashboardProps) {
  const [filters, setFilters] = useState<AnalyticsFilters>(EMPTY_FILTERS);
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sort, setSort] = useState<TableSort>("risk");
  const [examples, setExamples] = useState<AnalyticsExamples | null>(null);
  const [examplesTitle, setExamplesTitle] = useState("");
  const [examplesLoading, setExamplesLoading] = useState(false);

  useEffect(() => {
    setFilters(EMPTY_FILTERS);
    setExamples(null);
  }, [project.id]);

  useEffect(() => {
    let current = true;
    setLoading(true);
    setError(null);
    void api
      .getAnalytics(project.id, filters)
      .then((result) => {
        if (current) setAnalytics(result);
      })
      .catch((caught: unknown) => {
        if (current) {
          setError(caught instanceof Error ? caught.message : "Analytics could not be loaded");
        }
      })
      .finally(() => {
        if (current) setLoading(false);
      });
    return () => {
      current = false;
    };
  }, [filters, project.id]);

  const sortedClasses = useMemo(() => {
    const items = [...(analytics?.classes ?? [])];
    if (sort === "volume") {
      return items.sort((left, right) => right.total_ai_proposals - left.total_ai_proposals);
    }
    if (sort === "name") {
      return items.sort((left, right) => left.label_name.localeCompare(right.label_name));
    }
    return items.sort(
      (left, right) =>
        right.rejected.rate - left.rejected.rate ||
        right.adjusted.rate - left.adjusted.rate ||
        right.total_ai_proposals - left.total_ai_proposals,
    );
  }, [analytics?.classes, sort]);

  const inspectExamples = async (kind: ExampleKind, title: string, override: Partial<AnalyticsFilters> = {}) => {
    setExamplesLoading(true);
    setExamplesTitle(title);
    setExamples(null);
    try {
      setExamples(await api.getAnalyticsExamples(project.id, kind, { ...filters, ...override }, 25));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Examples could not be loaded");
    } finally {
      setExamplesLoading(false);
    }
  };

  const updateFilters = (update: Partial<AnalyticsFilters>) => {
    setFilters((current) => ({ ...current, ...update }));
    setExamples(null);
  };

  if (loading && !analytics) {
    return (
      <SecondaryWorkspace className="analytics-workspace secondary-loading">
        <div className="analytics-loading-mark" />
        <p>Loading analytics…</p>
      </SecondaryWorkspace>
    );
  }

  if (!analytics) {
    return (
      <SecondaryWorkspace className="analytics-workspace">
        <WorkspaceHeader
          title="Project analytics unavailable"
          description={error ?? "Analytics could not be loaded."}
        />
      </SecondaryWorkspace>
    );
  }

  const { overview, outcomes, adjusted_box_iou: iou, dataset } = analytics;
  const activeFilterCount = Object.values(filters).filter(Boolean).length;
  const hasReviewedProposals = outcomes.reviewed_ai_proposals > 0;
  const providerCohorts = distinctProviderCohorts(analytics.filter_options.provider_models);
  const hasMultipleProviders = providerCohorts.length > 1;
  const sourceContext = activeSource ?? {
    label: "Model" as const,
    value: providerCohorts[0]?.model.replace("nvidia/", "") ?? "LocateAnything-3B",
  };
  const majorCorrectionCount = iou.buckets
    .filter((bucket) => bucket.key === "major" || bucket.key === "significant")
    .reduce((total, bucket) => total + bucket.count, 0);
  const majorCorrectionRate = iou.count ? (majorCorrectionCount / iou.count) * 100 : 0;
  const openWork = dataset.unresolved_annotations + dataset.review_needed_images;
  const readinessLabel = dataset.validation.export_ready
    ? openWork > 0
      ? "Ready with exclusions"
      : "Review complete"
    : "Blocked";

  return (
    <SecondaryWorkspace className="analytics-workspace">
      <WorkspaceHeader
        title="How accurate were the proposals?"
        description={
          <>
            Review agreement, corrections, and missed objects in <strong>{project.name}</strong>.
          </>
        }
        aside={
          <button className="secondary-button" onClick={onOpenExport}>
            Export
          </button>
        }
      />

      <section className="analytics-compact-filters" aria-label="Analytics filters">
        <label className="secondary-field">
          <span>Class</span>
          <select
            value={filters.label_id ?? ""}
            onChange={(event) => updateFilters({ label_id: event.target.value || null })}
          >
            <option value="">All classes</option>
            {analytics.filter_options.labels.map((label) => (
              <option value={label.id} key={label.id}>
                {label.name}
              </option>
            ))}
          </select>
        </label>
        <label className="secondary-field">
          <span>Prompt</span>
          <select
            value={filters.prompt ?? ""}
            onChange={(event) => updateFilters({ prompt: event.target.value || null })}
          >
            <option value="">All prompts</option>
            {analytics.filter_options.prompts.map((prompt) => (
              <option value={prompt} key={prompt}>
                {prompt}
              </option>
            ))}
          </select>
        </label>
        {hasMultipleProviders && (
          <label className="secondary-field">
            <span>Provider cohort</span>
            <select
              value={filters.provider && filters.model ? JSON.stringify([filters.provider, filters.model]) : ""}
              onChange={(event) => {
                const value = event.target.value;
                const [provider, model] = value ? (JSON.parse(value) as [string, string]) : [null, null];
                updateFilters({ provider, model });
              }}
            >
              <option value="">All provider cohorts</option>
              {providerCohorts.map((item) => (
                <option value={JSON.stringify([item.provider, item.model])} key={`${item.provider}:${item.model}`}>
                  {item.provider} · {item.model}
                </option>
              ))}
            </select>
          </label>
        )}
        <div className="analytics-readonly-source">
          <span>{sourceContext.label}</span>
          <strong>{sourceContext.value}</strong>
        </div>
        <button
          className="text-button analytics-clear"
          disabled={!activeFilterCount}
          onClick={() => updateFilters(EMPTY_FILTERS)}
        >
          Clear filters
        </button>
      </section>

      {error && (
        <p className="secondary-inline-error" role="alert">
          {error}
        </p>
      )}

      {outcomes.total_ai_proposals === 0 ? (
        <WorkspaceEmptyState title="No reviewed proposals yet" description="Review proposals to see quality metrics." />
      ) : (
        <>
          <section className="analytics-lead-panel" aria-label="Proposal agreement">
            <div>
              <span>Verified without changes</span>
              <strong>{formatPercent(outcomes.accepted)}</strong>
              <p>
                {outcomes.accepted.count.toLocaleString()} of {outcomes.reviewed_ai_proposals.toLocaleString()} reviewed
                proposals
              </p>
            </div>
            <div className="analytics-verdict">
              <span>Summary</span>
              <strong>
                {outcomes.accepted.rate >= 80
                  ? "High agreement"
                  : outcomes.accepted.rate >= 60
                    ? "Mixed results"
                    : "Needs attention"}
              </strong>
              <p>
                {outcomes.accepted.rate >= 80
                  ? "Most proposals were accepted without changes."
                  : outcomes.accepted.rate >= 60
                    ? "Corrections are common enough to review."
                    : "Human review changed many proposals."}
                {sortedClasses[0] ? ` ${sortedClasses[0].label_name} has the highest rejection rate.` : ""}
              </p>
              <small>{activeFilterCount ? `${activeFilterCount} filters active` : "All annotations"}</small>
            </div>
          </section>

          <section className="metric-strip analytics-metric-strip" aria-label="Verification outcomes">
            <MetricCell
              label="Adjusted"
              value={formatPercent(outcomes.adjusted)}
              detail={`${outcomes.adjusted.count.toLocaleString()} annotations`}
              tone="warning"
            />
            <MetricCell
              label="Rejected"
              value={formatPercent(outcomes.rejected)}
              detail={`${outcomes.rejected.count.toLocaleString()} annotations`}
              tone="danger"
            />
            <MetricCell
              label="Human added"
              value={overview.human_added_annotations.toLocaleString()}
              detail={`${overview.attributable_ai_misses.toLocaleString()} AI misses`}
              tone="human"
            />
            <MetricCell
              label="Mean adjusted IoU"
              value={formatIou(iou.mean)}
              detail={`${iou.count.toLocaleString()} adjusted boxes`}
            />
          </section>

          <div className="analytics-main-grid">
            <WorkspacePanel
              title="Review outcomes"
              description="Accepted, adjusted, and rejected proposals from the same reviewed set."
              aside={<span className="secondary-panel-count">{plural(outcomes.reviewed_ai_proposals, "review")}</span>}
              className="analytics-primary-chart"
            >
              {hasReviewedProposals ? (
                <OutcomeDistribution outcomes={outcomes} />
              ) : (
                <div className="analytics-panel-empty">
                  <strong>Reviews required</strong>
                  <span>{plural(outcomes.total_ai_proposals, "proposal")} await human decisions.</span>
                </div>
              )}
            </WorkspacePanel>
            <WorkspacePanel
              title="Classes to review"
              description="Highest rejection rates first."
              className="analytics-risk-panel"
            >
              {sortedClasses.length ? (
                <div className="analytics-risk-list">
                  {sortedClasses.slice(0, 5).map((item) => (
                    <button
                      key={item.label_id}
                      onClick={() =>
                        void inspectExamples("intervention", `${item.label_name} interventions`, {
                          label_id: item.label_id,
                        })
                      }
                    >
                      <span>{item.label_name}</span>
                      <strong>{formatPercent(item.rejected)}</strong>
                    </button>
                  ))}
                </div>
              ) : (
                <p className="analytics-panel-empty">No class data yet.</p>
              )}
            </WorkspacePanel>
          </div>

          <div className="analytics-support-grid analytics-quality-grid">
            <WorkspacePanel
              title="Box placement"
              description="Higher IoU means the final box stayed closer to the original proposal."
              aside={<span className="secondary-panel-count">{iou.count} adjusted</span>}
              className="analytics-correction-panel"
            >
              {iou.count ? (
                <>
                  <div className="analytics-correction-metrics">
                    <div>
                      <span>Mean IoU</span>
                      <strong>{formatIou(iou.mean)}</strong>
                    </div>
                    <div>
                      <span>Median IoU</span>
                      <strong>{formatIou(iou.median)}</strong>
                    </div>
                    <div>
                      <span>Major corrections</span>
                      <strong>{majorCorrectionRate.toFixed(1)}%</strong>
                    </div>
                  </div>
                  <div className="analytics-iou-distribution" aria-label="Adjusted box IoU distribution">
                    {iou.buckets.map((bucket) => (
                      <div key={bucket.key}>
                        <span>{bucket.interpretation}</span>
                        <div>
                          <i style={{ width: `${bucket.rate}%` }} />
                        </div>
                        <strong>{bucket.count}</strong>
                        <small>{bucket.range_label}</small>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <p className="analytics-panel-empty">No adjusted boxes are available yet.</p>
              )}
            </WorkspacePanel>

            <WorkspacePanel
              title="Export readiness"
              description="Uses the same validation rules as YOLO and COCO export."
              aside={
                <span className={`secondary-status-pill ${dataset.validation.export_ready ? "ready" : "blocked"}`}>
                  {readinessLabel}
                </span>
              }
              className="analytics-readiness-panel"
            >
              <div className="analytics-readiness-stats">
                <div>
                  <span>Images</span>
                  <strong>{dataset.exportable_images}</strong>
                </div>
                <div>
                  <span>Final boxes</span>
                  <strong>{dataset.total_final_training_boxes}</strong>
                </div>
                <div>
                  <span>Classes</span>
                  <strong>{dataset.total_classes}</strong>
                </div>
                <div>
                  <span>Open work</span>
                  <strong>{openWork}</strong>
                </div>
              </div>
              <div className="analytics-readiness-action">
                <span>
                  {dataset.validation.export_ready
                    ? openWork > 0
                      ? "Unresolved work will be excluded"
                      : "Ready for training export"
                    : `${dataset.validation.blocking_error_count} blocking errors`}
                </span>
                <button className="secondary-button" onClick={onOpenExport}>
                  Open export
                </button>
              </div>
            </WorkspacePanel>
          </div>

          <div className="analytics-support-grid">
            <WorkspacePanel
              title="Class results"
              aside={
                <label className="analytics-sort-control">
                  <span>Sort</span>
                  <select value={sort} onChange={(event) => setSort(event.target.value as TableSort)}>
                    <option value="risk">Highest rejection</option>
                    <option value="volume">Most proposals</option>
                    <option value="name">Class name</option>
                  </select>
                </label>
              }
              className="analytics-performance-panel"
            >
              {sortedClasses.length ? (
                <div className="analytics-performance-list">
                  {sortedClasses.map((item) => (
                    <ClassPerformanceRow
                      key={item.label_id}
                      item={item}
                      onInspect={() =>
                        void inspectExamples("intervention", `${item.label_name} interventions`, {
                          label_id: item.label_id,
                        })
                      }
                    />
                  ))}
                </div>
              ) : (
                <p className="analytics-panel-empty">No classes match the current filters.</p>
              )}
            </WorkspacePanel>

            <WorkspacePanel
              title="Prompt results"
              aside={<span className="secondary-panel-count">{analytics.prompts.length} prompts</span>}
              className="analytics-performance-panel"
            >
              {analytics.prompts.length ? (
                <div className="analytics-performance-list">
                  {analytics.prompts.map((item) => (
                    <PromptPerformanceRow
                      key={item.prompt}
                      item={item}
                      onInspect={() =>
                        void inspectExamples("intervention", `Prompt: “${item.prompt}”`, {
                          prompt: item.prompt,
                        })
                      }
                    />
                  ))}
                </div>
              ) : (
                <p className="analytics-panel-empty">No prompts match the current filters.</p>
              )}
            </WorkspacePanel>
          </div>

          <WorkspacePanel
            title="Review the annotations"
            description="Open related annotations in Annotate."
            className="analytics-issues-panel"
          >
            <div className="analytics-issue-grid">
              <IssueButton
                label="Rejected predictions"
                count={outcomes.rejected.count}
                onClick={() => void inspectExamples("rejected", "Rejected predictions")}
              />
              <IssueButton
                label="Major box corrections"
                count={majorCorrectionCount}
                onClick={() => void inspectExamples("low_iou", "Major box corrections")}
              />
              <IssueButton
                label="Human-added"
                count={overview.human_added_annotations}
                onClick={() => void inspectExamples("human_added", "Human-added annotations")}
              />
              <IssueButton
                label="Unresolved"
                count={outcomes.unresolved.count}
                onClick={() => void inspectExamples("unresolved", "Unresolved proposals")}
              />
              <IssueButton
                label="Needs review"
                count={overview.images_needing_review}
                onClick={() => void inspectExamples("needs_review", "Images needing review")}
              />
            </div>
          </WorkspacePanel>
        </>
      )}

      {(examplesLoading || examples) && (
        <aside className="examples-drawer" aria-label="Problem examples">
          <div className="examples-drawer-head">
            <div>
              <span>Evidence drilldown</span>
              <strong>{examplesTitle}</strong>
            </div>
            <button className="icon-button" aria-label="Close examples" onClick={() => setExamples(null)}>
              <Icon name="close" />
            </button>
          </div>
          {examplesLoading ? (
            <p className="examples-loading">Finding relevant images…</p>
          ) : examples?.items.length ? (
            <>
              <p className="examples-count">
                Showing {examples.items.length} of {examples.total} matching annotations or images.
              </p>
              <div className="example-list">
                {examples.items.map((item, index) => (
                  <button
                    key={`${item.image_id}:${item.annotation_id}:${index}`}
                    onClick={() => onInspectImage(item.image_id, item.annotation_id)}
                  >
                    <span className={`example-kind kind-${item.kind}`}>{item.kind.replaceAll("_", " ")}</span>
                    <strong>{item.image_filename}</strong>
                    <small>
                      {item.label_name ?? "Image-level review"}
                      {item.prompt ? ` · “${item.prompt}”` : ""}
                    </small>
                    <em>{item.iou === null ? "Open workspace →" : `IoU ${item.iou.toFixed(2)} · Open →`}</em>
                  </button>
                ))}
              </div>
            </>
          ) : (
            <p className="examples-loading">No examples match the current filters.</p>
          )}
        </aside>
      )}
    </SecondaryWorkspace>
  );
}

function OutcomeDistribution({ outcomes }: { outcomes: AnalyticsSummary["outcomes"] }) {
  const metrics = [
    { label: "Accepted", metric: outcomes.accepted, tone: "accepted" },
    { label: "Adjusted", metric: outcomes.adjusted, tone: "adjusted" },
    { label: "Rejected", metric: outcomes.rejected, tone: "rejected" },
  ];
  return (
    <div className="analytics-outcome-visual">
      <div className="analytics-stacked-bar" aria-label="Verification outcome distribution">
        {metrics.map(({ label, metric, tone }) => (
          <i
            key={label}
            className={`state-${tone}`}
            style={{ width: `${metric.rate}%` }}
            title={`${label}: ${formatPercent(metric)}`}
          />
        ))}
      </div>
      <div className="analytics-outcome-legend">
        {metrics.map(({ label, metric, tone }) => (
          <div key={label} className={`state-${tone}`}>
            <span>
              <i />
              {label}
            </span>
            <strong>{formatPercent(metric)}</strong>
            <small>
              {metric.count.toLocaleString()} of {metric.denominator.toLocaleString()}
            </small>
          </div>
        ))}
      </div>
    </div>
  );
}

function ClassPerformanceRow({ item, onInspect }: { item: ClassAnalytics; onInspect: () => void }) {
  return (
    <button className="analytics-performance-row" onClick={onInspect}>
      <span className="analytics-performance-name">
        <strong>{item.label_name}</strong>
        <small>
          {item.reviewed_ai_proposals} reviewed{item.small_sample ? " · small sample" : ""}
        </small>
      </span>
      <span className="analytics-performance-bar">
        <i style={{ width: `${item.accepted.rate}%` }} />
      </span>
      <span className="analytics-performance-rate">
        <strong>{formatPercent(item.accepted)}</strong>
        <small>accepted</small>
      </span>
      <span className="analytics-performance-risk">{formatPercent(item.rejected)} rejected</span>
    </button>
  );
}

function PromptPerformanceRow({ item, onInspect }: { item: PromptAnalytics; onInspect: () => void }) {
  return (
    <button className="analytics-performance-row" onClick={onInspect}>
      <span className="analytics-performance-name">
        <strong>“{item.prompt}”</strong>
        <small>
          {item.inference_runs} runs · {item.reviewed_ai_proposals} reviewed
        </small>
      </span>
      <span className="analytics-performance-bar">
        <i style={{ width: `${item.accepted.rate}%` }} />
      </span>
      <span className="analytics-performance-rate">
        <strong>{formatPercent(item.accepted)}</strong>
        <small>accepted</small>
      </span>
      <span className="analytics-performance-risk">{item.attributable_ai_misses} misses</span>
    </button>
  );
}

function IssueButton({ label, count, onClick }: { label: string; count: number; onClick: () => void }) {
  return (
    <button onClick={onClick} disabled={count === 0}>
      <strong>{count.toLocaleString()}</strong>
      <span>{label}</span>
      <small>View examples →</small>
    </button>
  );
}

export { AnalyticsDashboard };
