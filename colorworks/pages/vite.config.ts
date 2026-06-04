import { defineConfig } from "vitest/config";

// `base` is controlled by PAGES_BASE in CI (project pages live at /colorworks/).
// Locally it defaults to "/" so `npm run dev` and `npm run preview` just work.
export default defineConfig({
  base: process.env.PAGES_BASE || "/",
  build: {
    target: "es2021",
    outDir: "dist",
    assetsDir: "assets",
  },
  test: {
    globals: true,
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
