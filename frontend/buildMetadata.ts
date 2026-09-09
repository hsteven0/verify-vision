import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = dirname(fileURLToPath(import.meta.url));

export const applicationVersion = readFileSync(resolve(frontendRoot, "../backend/app/VERSION"), "utf8").trim();
