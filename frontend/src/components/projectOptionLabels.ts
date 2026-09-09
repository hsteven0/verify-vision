import type { ProjectSummary, UUID } from "../types";

export function disambiguateProjectNames(projects: ProjectSummary[]): Map<UUID, string> {
  const groups = new Map<string, ProjectSummary[]>();
  for (const project of projects) {
    const key = project.name.trim().toLocaleLowerCase();
    groups.set(key, [...(groups.get(key) ?? []), project]);
  }

  const labels = new Map<UUID, string>();
  for (const group of groups.values()) {
    if (group.length === 1) {
      labels.set(group[0].id, group[0].name);
      continue;
    }
    const ordered = [...group].sort((left, right) => left.id.localeCompare(right.id));
    const baseLabels = ordered.map((project) => {
      const date = new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
        timeZone: "UTC",
      }).format(new Date(project.updated_at));
      const images = `${project.image_count} ${project.image_count === 1 ? "image" : "images"}`;
      return `${project.name} · ${date} · ${images}`;
    });
    baseLabels.forEach((label, index) => {
      const repeated = baseLabels.filter((candidate) => candidate === label).length > 1;
      labels.set(ordered[index].id, repeated ? `${label} · ${index + 1}` : label);
    });
  }
  return labels;
}
