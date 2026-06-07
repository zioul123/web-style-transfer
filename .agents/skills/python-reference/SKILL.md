---
name: python-reference
description: Work with this repository's PyTorch reference implementation, fixture exporters, quantization evaluation, generated parity assets, or Python-based numerical validation. Use only when a task touches `python-reference/`, fixture schemas, model-pack generation, numerical parity, or missing optional reference assets.
---

# Python Reference

Load Python/reference context only when the task requires it.

## Route The Task

Read [fixture-routing.md](references/fixture-routing.md), then inspect only the
relevant exporter, its README, the consuming test helper, and the target fixture
or model-pack documentation.

## Run Python Safely

1. Activate the repository environment before every Python command:
   `source .venv/bin/activate`.
2. Prefer existing exporters and deterministic small fixtures.
3. Do not install or upgrade dependencies without approval.
4. Do not generate large fixtures or model packs unless the task requires them.
5. Do not commit newly generated large assets unless explicitly requested.
6. Preserve checksum, shape, dtype, format, and tolerance validation.

## Coordinate With Browser Tests

Trace each fixture from exporter to committed asset, loader/helper, and
Playwright parity test. If a fixture format changes, update all producers,
consumers, skip behavior, tests, and the relevant README in the same change.

Record generated files and their tracked/untracked status in the task artifacts.
