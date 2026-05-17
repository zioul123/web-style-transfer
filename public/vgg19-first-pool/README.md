# VGG19 first-pool fixtures

Generate the truncated VGG19 (conv1_1 -> relu1_1 -> conv1_2 -> relu1_2 -> pool1) fixtures with:

```bash
python python-reference/export_vgg19_first_pool.py
```

This writes:
- `vgg19_first_pool_weights.json`
- `vgg19_first_pool_case_madeira16.json`

The Playwright parity test `tests/phase2-vgg19-first-pool.spec.ts` will auto-skip until these files exist.

Note: the weights fixture is resolution-agnostic and does not include input shape; input shape lives in the case fixture.
