import tailwindcss from "@tailwindcss/vite";
import { copyFile } from "node:fs/promises";
import { resolve } from "node:path";
import { defineConfig, type Plugin, type ResolvedConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages is a static file host: it serves a file only when that file
// exists at the requested path. Our app is a single-page application, so Vite
// emits one HTML entrypoint (`index.html`) and the React code decides at
// runtime whether `/` should show the main app or `/benchmark` should show the
// benchmark app. That works in Vite dev/preview because those servers include
// history fallback middleware, but it does not work on GitHub Pages for direct
// visits or hard refreshes of nested routes.
//
// Example without this plugin on a project page deployment:
//   1. The user opens https://owner.github.io/web-style-transfer/benchmark.
//   2. GitHub Pages looks for a physical `/web-style-transfer/benchmark` file.
//   3. No such file exists, so Pages serves its own 404 page instead of our app.
//
// GitHub Pages has one useful escape hatch for SPAs: if the deployed artifact
// contains a `404.html`, Pages serves that file for missing paths while keeping
// the browser URL unchanged. Copying the built `index.html` to `404.html` means
// direct `/benchmark` requests still load the same JS/CSS bundles, and then
// `src/RouteApp.tsx` can inspect `window.location.pathname` and render the
// benchmark route.
//
// This is implemented as a tiny Vite plugin instead of a separate shell script
// so every production build produces a complete Pages-compatible artifact. It
// also avoids accidentally copying the source `index.html` before Vite rewrites
// asset URLs. The generated fallback must be copied *after* Vite has injected
// hashed JS/CSS filenames and applied the configured `base` path.
const githubPagesSpaFallback = (): Plugin => {
  // Vite computes values like the project root and output directory after it
  // resolves the config. Store that resolved config so `closeBundle` can find
  // the final `dist` directory without duplicating Vite defaults here.
  let resolvedConfig: ResolvedConfig | null = null;

  return {
    name: "github-pages-spa-fallback",
    // Only run during `vite build`. Dev and preview servers already know how to
    // serve `index.html` for SPA routes, and writing fallback files in dev would
    // be noisy and unnecessary.
    apply: "build",
    configResolved(config) {
      resolvedConfig = config;
    },
    async closeBundle() {
      if (resolvedConfig === null) {
        throw new Error("Vite config was not resolved before closeBundle.");
      }

      // `build.outDir` can be relative to the Vite root, so resolve it before
      // copying files. This keeps the plugin correct even if a future developer
      // changes `root` or `build.outDir` in this config.
      const outputDirectory = resolve(
        resolvedConfig.root,
        resolvedConfig.build.outDir,
      );

      // Copy the finalized HTML, not the source template. At this point
      // `index.html` already contains the hashed asset filenames and honors the
      // `base` value below, including the `/web-style-transfer/` prefix used by
      // the GitHub Pages workflow.
      await copyFile(
        resolve(outputDirectory, "index.html"),
        resolve(outputDirectory, "404.html"),
      );
    },
  };
};

export default defineConfig({
  // Vite prefixes built asset URLs with `base`. Local builds default to `/`.
  // `.github/workflows/deploy-pages.yml` sets `BASE_PATH` to
  // `/${repository-name}/`, which is required for GitHub project pages because
  // the site is hosted under that path rather than at the domain root.
  base: process.env.BASE_PATH ?? "/",
  plugins: [react(), tailwindcss(), githubPagesSpaFallback()],
  server: {
    // Playwright runs its own browser automation. Disabling HMR in that mode
    // removes websocket/reload noise from tests while leaving normal local
    // development behavior unchanged.
    hmr: process.env.PLAYWRIGHT_TEST === "1" ? false : undefined,
  },
});
