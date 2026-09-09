import "@testing-library/jest-dom/vitest";

const dialogFocus = new WeakMap<HTMLDialogElement, HTMLElement | null>();

HTMLDialogElement.prototype.showModal = function showModal() {
  dialogFocus.set(this, document.activeElement instanceof HTMLElement ? document.activeElement : null);
  this.setAttribute("open", "");
  const initial =
    this.querySelector<HTMLElement>("[autofocus]") ??
    this.querySelector<HTMLElement>(
      "button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled])",
    );
  initial?.focus();
};

HTMLDialogElement.prototype.close = function close() {
  this.removeAttribute("open");
  dialogFocus.get(this)?.focus();
  dialogFocus.delete(this);
};

if (!window.localStorage) {
  const values = new Map<string, string>();
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      get length() {
        return values.size;
      },
      clear: () => values.clear(),
      getItem: (key: string) => values.get(key) ?? null,
      key: (index: number) => [...values.keys()][index] ?? null,
      removeItem: (key: string) => values.delete(key),
      setItem: (key: string, value: string) => values.set(key, value),
    } satisfies Storage,
  });
}
