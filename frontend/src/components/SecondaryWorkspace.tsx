import type { ReactNode } from "react";

type Tone = "neutral" | "positive" | "warning" | "danger" | "human";

export function SecondaryWorkspace({ className = "", children }: { className?: string; children: ReactNode }) {
  return (
    <main className={`secondary-workspace ${className}`.trim()}>
      <div className="secondary-workspace-content">{children}</div>
    </main>
  );
}

export function WorkspaceHeader({
  title,
  description,
  aside,
}: {
  title: string;
  description: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <header className="secondary-workspace-header">
      <div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {aside && <div className="secondary-header-aside">{aside}</div>}
    </header>
  );
}

export function MetricCell({
  label,
  value,
  detail,
  tone = "neutral",
}: {
  label: string;
  value: string;
  detail?: ReactNode;
  tone?: Tone;
}) {
  return (
    <article className={`metric-cell tone-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </article>
  );
}

export function WorkspacePanel({
  title,
  description,
  aside,
  className = "",
  children,
}: {
  title: string;
  description?: ReactNode;
  aside?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`secondary-panel ${className}`.trim()}>
      <div className="secondary-panel-heading">
        <div>
          <h2>{title}</h2>
          {description && <p>{description}</p>}
        </div>
        {aside && <div className="secondary-panel-aside">{aside}</div>}
      </div>
      {children}
    </section>
  );
}

export function WorkspaceEmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description: ReactNode;
  action?: ReactNode;
}) {
  return (
    <section className="secondary-empty-state">
      <div>
        <h2>{title}</h2>
        <p>{description}</p>
        {action && <div className="secondary-empty-action">{action}</div>}
      </div>
    </section>
  );
}
