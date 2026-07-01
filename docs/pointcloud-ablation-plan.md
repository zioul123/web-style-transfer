# Point-Cloud Ablation Browser Plan

This document tracks the staged implementation of the point-cloud ablation
browser on the `/pointcloud-preview` route. The feature is route-local and
intentionally isolated from the worker protocol, WebGPU optimization pipeline,
and point-cloud JSON schema.

## Current State

The branch currently includes phases 1 through 5:

1. **Parser and data model:** `experimentFilenames.ts` parses experiment
   filenames produced by the Python `name_expt` format, including pooling mode
   and optional `_step<number>` output-step suffixes, and exposes dimension
   summaries and stable value sorting.
2. **Ablation tab shell:** `/pointcloud-preview` now has `Preview` and
   `Ablation` tabs. The ablation tab imports folders or direct JSON file
   selections, parses filenames only, and shows parsed/unparsed counts plus
   dimension summaries.
3. **Filters and matrix:** the ablation tab provides X/Y axis selectors,
   fixed-value selectors for non-axis dimensions, available/missing/ambiguous
   matrix cells, and unique-cell click-to-preview. The fixed `outputStep`
   filter supports selecting multiple steps so late snapshots can be included
   together while earlier checkpoints remain excluded. Preview loading is
   transient: it reads the selected file only when clicked, switches back to the
   Preview tab on success, preserves the camera when possible, and does not add
   the experiment file to the manual upload queue.
4. **Grid PNG export:** the ablation tab exports the current matrix as one
   labelled PNG for a selected saved viewpoint. Export reads unique-cell JSON
   files only during capture, reuses the route's synchronized camera/render
   wait path, renders missing cells as placeholders, restores the previous
   preview state, and blocks while ambiguous cells remain.
5. **Docs, polish, and final review:** the ablation browser has focused UI
   coverage for import, filtering, transient preview, PNG export, and sticky
   selector behavior. Last-used X/Y axes, fixed filters, and export viewpoint
   choices are stored in localStorage; remembered choices are reused when the
   current imported folder still offers them, otherwise the browser falls back
   through the normal available options.

The current matrix defaults are:

- X axis: `contentSamplesPerFace` when it is present and varying.
- Y axis: `distanceMeasure` when it is present and varying.
- Fallback axes: first two varying dimensions in parser-definition order.
- Fixed filters: first sorted value, except `outputStep` defaults to the
  highest numeric step.

## Remaining Follow-Ups

- No route-level implementation phases remain in this branch. Future work should
  be driven by concrete user feedback from ablation-folder usage, such as
  additional export formats or denser matrix navigation controls.

## Guardrails

- Keep experiment-folder files separate from the manual upload queue.
- Do not recreate absent end-to-end point-cloud style-transfer orchestration.
- Do not change worker protocol types, model/fixture formats, or optimization
  code for this browser-only tool.
- Keep JSON contents unread during import/filter/matrix construction; read a
  file only for transient preview or export capture.
- Prefer new route-local helpers under `src/features/pointcloud-preview/ablation/`.

## Useful Verification

Focused checks for the current implementation:

```bash
nvm use 22
npm run test -- tests/pointcloud-preview-ui.spec.ts
npm run build
```

The Playwright command starts a local Vite server and may require sandbox
escalation in Codex when binding `127.0.0.1:4173`.
