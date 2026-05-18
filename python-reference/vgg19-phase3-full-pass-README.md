# VGG19 Phase 3 Full-Pass Fixture Generator

This script generates on-demand fixtures for **full phase-3 forward/loss parity** up to `conv5_1`.

## Script

```bash
python python-reference/export_vgg19_phase3_full_pass.py
```

## What it exports

To `public/vgg19-phase3-full-pass/`:

- `vgg19_conv0_to_conv28_weights.json`
  - Conv weights+biases for all conv layers encountered from feature layer indices 0..28.
- `vgg19_phase3_full_pass_fixture.json`
  - Content/style/input tensors (16x16), normalization params,
  - expected per-style-layer losses for `conv1_1, conv2_1, conv3_1, conv4_1, conv5_1`,
  - expected content loss at `conv4_2`,
  - expected totals.

## Notes

- Artifacts are intentionally **not committed** due to size.
- Playwright full-pass test will skip when these artifacts are absent.
