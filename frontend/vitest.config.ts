import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

import { applicationVersion } from "./buildMetadata";

export default defineConfig({
  define: {
    __VERIFYVISION_VERSION__: JSON.stringify(applicationVersion),
  },
  plugins: [react()],
  test: {
    environment: "jsdom",
    environmentOptions: {
      jsdom: { url: "http://127.0.0.1:5173" },
    },
    setupFiles: ["./src/test/setup.ts"],
    restoreMocks: true,
  },
});
