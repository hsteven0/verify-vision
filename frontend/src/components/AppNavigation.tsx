import { Icon, type IconName } from "./Icon";

export type AppView = "workspace" | "analytics" | "test" | "export" | "training" | "settings";

const destinations: Array<{ id: AppView; label: string; icon: IconName }> = [
  { id: "workspace", label: "Annotate", icon: "annotate" },
  { id: "analytics", label: "Analytics", icon: "analytics" },
  { id: "export", label: "Export", icon: "export" },
  { id: "training", label: "Training", icon: "training" },
  { id: "test", label: "Test", icon: "test" },
];

export function AppNavigation({ activeView, onChange }: { activeView: AppView; onChange: (view: AppView) => void }) {
  return (
    <nav className="app-navigation" aria-label="Project sections">
      <div className="navigation-primary">
        {destinations.map((destination) => (
          <button
            key={destination.id}
            className={activeView === destination.id ? "active" : ""}
            onClick={() => onChange(destination.id)}
            aria-current={activeView === destination.id ? "page" : undefined}
          >
            <Icon name={destination.icon} />
            <span>{destination.label}</span>
          </button>
        ))}
      </div>
      <button
        className={`navigation-settings ${activeView === "settings" ? "active" : ""}`}
        onClick={() => onChange("settings")}
        aria-current={activeView === "settings" ? "page" : undefined}
      >
        <Icon name="settings" />
        <span>Settings</span>
      </button>
    </nav>
  );
}
