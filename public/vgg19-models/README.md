# VGG19 Weight Manifest (Draft v1)

## File layout

- `manifest.json`: top-level metadata and layer->byte range mapping.
- `shard-000.bin`, `shard-001.bin`, ...: binary payload shards containing concatenated float32 tensors.
- Optional transition artifact: `vgg19_conv0_to_conv28_weights.json` (legacy).

## Manifest schema

```json
{
  "modelId": "vgg19-features-conv0-28",
  "version": "1.0.0",
  "format": "float32-le",
  "quantization": { "scheme": "none", "dtype": "float32", "scale": null, "zeroPoint": null },
  "layers": {
    "conv0": {
      "weight": { "shape": [64, 3, 3, 3], "dtype": "float32", "shard": "shard-000.bin", "offset": 0, "length": 6912 },
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
- tensors are contiguous and tightly packed, little-endian `float32`.
- `length` must equal `product(shape) * 4` for `float32`.
- layer names are `conv{index}` for VGG feature layers.
