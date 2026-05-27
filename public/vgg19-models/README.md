# VGG19 Weight Manifest (Draft v1)

## File layout

- `fp32/manifest.json`, `fp16/manifest.json`, `int8-per-channel/manifest.json`, `int8log-per-channel/manifest.json`, `int4-experimental/manifest.json`, `int4log-experimental/manifest.json`: pack-specific metadata.
- `*/shard-000.bin`: pack payload shards.
- Optional transition artifact per-pack: `*/vgg19_conv0_to_conv28_weights.json` (legacy).

## Manifest schema

```json
{
  "modelId": "vgg19-features-conv0-28",
  "version": "1.0.0",
  "format": "int8-per-channel-le",
  "quantization": { "scheme": "per-channel-symmetric", "dtype": "int8-per-channel" },
  "layers": {
    "conv0": {
      "weight": { "shape": [64, 3, 3, 3], "dtype": "int8", "shard": "shard-000.bin", "offset": 0, "length": 1728, "quantization": { "scheme": "per-channel-symmetric", "axis": 0, "scale": ["..."], "zeroPoint": [0] } },
      "bias": { "shape": [64], "dtype": "float32", "shard": "shard-000.bin", "offset": 6912, "length": 256 }
    }
  },
  "shards": [
    { "name": "shard-000.bin", "byteLength": 1234567 }
  ],
  "checksums": {
    "algorithm": "sha256",
    "files": {
      "manifest.json": "<hex>",
      "shard-000.bin": "<hex>"
    }
  }
}
```

## Offset/length conventions

- `offset` and `length` are byte offsets/byte lengths within the target shard.
- tensors are contiguous and tightly packed, little-endian numeric payloads.
- `length` must equal `product(shape) * dtypeSize`.
- layer names are `conv{index}` for VGG feature layers.

## Quantization evaluation

- Run: `python python-reference/evaluate_vgg19_quantization.py`.
- Report output: `public/vgg19-phase3-full-pass/quantization_eval_report.json`.
- Default acceptance thresholds:
  - `fp16`: style/content loss deltas near baseline (`|delta| < 1e-3`).
  - `int8-per-channel`: acceptable if style loss delta stays under `5e-2` and visual QA confirms no severe artifacts.
  - `int8log-per-channel`: log-domain variant with exploratory threshold (`|delta| < 7.5e-2`).
  - `int4-experimental`: exploratory-only pack with relaxed style-loss delta threshold (`|delta| < 1e-1`) until visual QA gates are finalized.
  - `int4log-experimental`: log-domain exploratory int4 threshold (`|delta| < 1.5e-1`).
