import { useEffect, useRef } from "react";

import { AppNavigation, type AppView } from "./AppNavigation";
import { Icon } from "./Icon";

export function GlobalNavigationDrawer({
  open,
  activeView,
  onClose,
  onChange,
}: {
  open: boolean;
  activeView: AppView;
  onClose: () => void;
  onChange: (view: AppView) => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    if (!open) return;
    const dialog = dialogRef.current;
    dialog?.showModal();
    return () => {
      if (dialog?.open) dialog.close();
    };
  }, [open]);

  if (!open) return null;

  const close = () => {
    dialogRef.current?.close();
    onClose();
  };

  const navigate = (view: AppView) => {
    onChange(view);
    close();
  };

  return (
    <dialog
      ref={dialogRef}
      className="global-nav-layer"
      aria-label="Navigation"
      onCancel={(event) => {
        event.preventDefault();
        close();
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          close();
        }
      }}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) close();
      }}
    >
      <aside className="global-nav-drawer">
        <header>
          <div className="global-nav-brand">
            <div className="brand-mark small" aria-hidden="true">
              <span />
            </div>
            <div>
              <strong>VerifyVision</strong>
            </div>
          </div>
          <button className="global-nav-close" type="button" onClick={close} aria-label="Close navigation">
            <Icon name="close" />
          </button>
        </header>
        <AppNavigation activeView={activeView} onChange={navigate} />
      </aside>
    </dialog>
  );
}
