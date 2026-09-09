import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GlobalNavigationDrawer } from "./GlobalNavigationDrawer";

describe("GlobalNavigationDrawer", () => {
  afterEach(cleanup);

  it("is absent while closed", () => {
    render(<GlobalNavigationDrawer open={false} activeView="workspace" onClose={vi.fn()} onChange={vi.fn()} />);
    expect(screen.queryByLabelText("Global navigation")).not.toBeInTheDocument();
  });

  it("closes on backdrop click and Escape", () => {
    const onClose = vi.fn();
    const { rerender } = render(
      <GlobalNavigationDrawer open activeView="workspace" onClose={onClose} onChange={vi.fn()} />,
    );

    fireEvent.mouseDown(screen.getByRole("dialog", { name: "Navigation" }));
    expect(onClose).toHaveBeenCalledOnce();
    rerender(<GlobalNavigationDrawer open={false} activeView="workspace" onClose={onClose} onChange={vi.fn()} />);
    rerender(<GlobalNavigationDrawer open activeView="workspace" onClose={onClose} onChange={vi.fn()} />);
    fireEvent.keyDown(screen.getByRole("dialog", { name: "Navigation" }), { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it.each(["Analytics", "Export", "Training", "Settings"])("navigates to %s and closes", (label) => {
    const onClose = vi.fn();
    const onChange = vi.fn();
    render(<GlobalNavigationDrawer open activeView="workspace" onClose={onClose} onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: label }));
    expect(onChange).toHaveBeenCalledOnce();
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("focuses the drawer and restores focus after closing", () => {
    const trigger = document.createElement("button");
    document.body.append(trigger);
    trigger.focus();
    const { rerender } = render(
      <GlobalNavigationDrawer open activeView="workspace" onClose={vi.fn()} onChange={vi.fn()} />,
    );

    const close = screen.getByRole("button", { name: "Close navigation" });
    expect(close).toHaveFocus();

    rerender(<GlobalNavigationDrawer open={false} activeView="workspace" onClose={vi.fn()} onChange={vi.fn()} />);
    expect(trigger).toHaveFocus();
    trigger.remove();
  });
});
