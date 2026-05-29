# VGG19 first-pool fixtures

This directory contains committed fixtures for the first-pool VGG19 parity and optimization tests.

The fixture covers this truncated path:

```text
conv1_1 -> relu1_1 -> conv1_2 -> relu1_2 -> pool1
```

## Generate

From the repository root:

```bash
python python-reference/export_vgg19_first_pool.py
```

## Files

- `vgg19_first_pool_weights.json`: conv weights and biases for the truncated path.
- `vgg19_first_pool_case_madeira16.json`: deterministic 16x16 Madeira input/case data and expected outputs.

The weights fixture is resolution-agnostic. Input shape and image values live in the case fixture.

## Consumers

- `tests/phase2-vgg19-first-pool.spec.ts`
- `tests/phase5-vgg19-first-pool-optimization.spec.ts`
- `tests/phase5-vgg19-first-pool-benchmark.spec.ts`
- `/benchmark` first-pool tab

If these files are absent, related Playwright tests skip or fail depending on whether the test path is optional.
