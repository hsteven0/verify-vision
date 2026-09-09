import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

import { applicationVersion } from "./buildMetadata";

export default defineConfig({
  define: {
    __VERIFYVISION_VERSION__: JSON.stringify(applicationVersion),
  },
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
  },
});
