import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AppNavigation } from "./AppNavigation";

describe("AppNavigation", () => {
  it("shows workflows and Settings in global navigation", () => {
    render(<AppNavigation activeView="workspace" onChange={() => undefined} />);

    expect(screen.getAllByRole("button")).toHaveLength(6);
    expect(screen.getByRole("button", { name: "Annotate" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "Analytics" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Test" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Export" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Training" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Settings" })).toBeVisible();
    expect(screen.getAllByRole("button").map((button) => button.textContent)).toEqual([
      "Annotate",
      "Analytics",
      "Export",
      "Training",
      "Test",
      "Settings",
    ]);
  });

  it("selects workflows without implementation settings", () => {
    const onChange = vi.fn();
    const view = render(<AppNavigation activeView="workspace" onChange={onChange} />);
    const navigation = within(view.container);

    fireEvent.click(navigation.getByRole("button", { name: "Export" }));

    expect(onChange).toHaveBeenCalledWith("export");
    expect(navigation.queryByText(/provider/i)).not.toBeInTheDocument();
    expect(navigation.queryByText(/cuda/i)).not.toBeInTheDocument();
  });
});
