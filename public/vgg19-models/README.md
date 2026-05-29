# VGG19 model packs

This directory stores manifest-backed VGG19 feature-weight packs used by the browser app. The app loads packs from this directory by default, or from `VITE_VGG19_MODEL_BASE_URL` when that environment variable is set at build time.

## Currently committed packs

- `int8-per-channel/`
- `int4log-experimental/`

The TypeScript parser and UI know about additional pack names (`fp32`, `fp16`, `int8log-per-channel`, `int4-experimental`), but those packs are not committed in this checkout. They can be generated locally or hosted externally under the same directory layout.

## File layout

Each pack directory contains:

- `manifest.json`: metadata for layers, tensors, shards, quantization, and checksums.
- `shard-*.bin`: binary tensor payloads referenced by the manifest.

Example:

```text
public/vgg19-models/
  int8-per-channel/
    manifest.json
    shard-000.bin
    shard-001.bin
  int4log-experimental/
    manifest.json
    shard-000.bin
```

## Manifest schema

The manifest shape consumed by `src/ml/worker/models/vgg19/weights.ts` is:

```json
{
  "modelId": "vgg19-features-conv0-28",
  "version": "1.0.0",
  "format": "int8-per-channel-le",
  "quantization": {
    "scheme": "per-channel-symmetric",
    "dtype": "int8-per-channel"
  },
  "layers": {
    "conv0": {
      "weight": {
        "shape": [64, 3, 3, 3],
        "dtype": "int8",
        "shard": "shard-000.bin",
        "offset": 0,
        "length": 1728,
        "quantization": {
          "scheme": "per-channel-symmetric",
          "axis": 0,
          "scale": [0.01],
          "zeroPoint": [0]
        }
      },
      "bias": {
        "shape": [64],
        "dtype": "float32",
        "shard": "shard-000.bin",
        "offset": 6912,
        "length": 256
      }
    }
  },
  "shards": [{ "name": "shard-000.bin", "byteLength": 1234567 }],
  "checksums": {
    "algorithm": "sha256",
    "files": {
      "shard-000.bin": "<hex>"
    }
  }
}
```

## Supported formats

The parser currently accepts these manifest `format` values:

- `fp32-le`
- `fp16-le`
- `int8-per-channel-le`
- `int8log-per-channel-le`
- `int4-experimental-le`
- `int4log-experimental-le`

Supported tensor dtypes are `float32`, `float16`, `int8`, and packed `int4-packed` with per-channel quantization metadata where required.

## Offset and length rules

- `offset` and `length` are byte offsets and byte lengths within the referenced shard.
- Tensor payloads are contiguous and little-endian.
- `length` must match the tensor's shape and dtype.
- Layer names use torch `vgg19.features` convolution indices as `conv{index}`.
- Shard byte lengths and SHA-256 checksums are validated before decoding.

## Runtime behavior

Model packs are decoded to float32 layer caches before they are passed into the worker optimization pipeline. The app caches downloaded manifests and shards in IndexedDB by model ID, version, and pack tier.

For deployments that should not bundle large packs into the Pages artifact, host this directory elsewhere and build with:

```bash
VITE_VGG19_MODEL_BASE_URL=https://raw.githubusercontent.com/<owner>/<repo>/main/public/vgg19-models npm run build
```

## Quantization evaluation

Run the quantization report script from the repository root:

```bash
python python-reference/evaluate_vgg19_quantization.py
```

The report is written under `public/vgg19-phase3-full-pass/` when the required generated fixtures are present. That directory is intentionally not committed by default because the full-pass fixtures are large.
