from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import VGG19_Weights, vgg19
from torchvision.transforms import functional as TF

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSET_CONTENT_PATH = REPO_ROOT / 'assets' / 'madeira_128x128.jpg'
ASSET_STYLE_PATH = REPO_ROOT / 'assets' / 'starry_night_768x970.jpg'
OUTPUT_DIR = REPO_ROOT / 'public' / 'vgg19-phase3-full-pass'
MODEL_OUTPUT_DIR = REPO_ROOT / 'public' / 'vgg19-models'

STYLE_LAYER_INDICES = [1, 6, 11, 20, 29]
CONTENT_LAYER_INDEX = 22
LAST_LAYER_INDEX = 29


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().contiguous().view(-1).tolist()


def gram_matrix(input_tensor: torch.Tensor) -> torch.Tensor:
    a, b, c, d = input_tensor.size()
    features = input_tensor.view(a * b, c * d)
    gram = torch.mm(features, features.t())
    return gram.div(a * b * c * d)


def load_image(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
    return TF.to_tensor(image).unsqueeze(0)


def sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


PACKS = ('fp32', 'fp16', 'int8-per-channel')


def quantize_int8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, list[float], list[int]]:
    out_channels = weight.shape[0]
    quantized = torch.empty_like(weight, dtype=torch.int8)
    scales: list[float] = []
    zero_points: list[int] = []
    for channel_index in range(out_channels):
        channel = weight[channel_index]
        max_abs = channel.abs().max().item()
        scale = max(max_abs / 127.0, 1e-8)
        q = torch.clamp(torch.round(channel / scale), -127, 127).to(torch.int8)
        quantized[channel_index] = q
        scales.append(float(scale))
        zero_points.append(0)
    return quantized, scales, zero_points


def export_weights_manifest(model: nn.Sequential, emit_legacy_json: bool, pack: str) -> None:
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pack_dir = MODEL_OUTPUT_DIR / pack
    pack_dir.mkdir(parents=True, exist_ok=True)
    shard_name = 'shard-000.bin'
    shard_path = pack_dir / shard_name

    layers: dict[str, dict[str, object]] = {}
    legacy_payload: dict[str, object] = {}
    cursor = 0

    with shard_path.open('wb') as shard_file:
        for layer_index in range(0, LAST_LAYER_INDEX + 1):
            layer = model[layer_index]
            if not isinstance(layer, nn.Conv2d):
                continue
            if layer.bias is None:
                raise RuntimeError(f'Expected bias for conv layer {layer_index}')

            weight = layer.weight.detach().cpu().contiguous().to(torch.float32)
            bias = layer.bias.detach().cpu().contiguous().to(torch.float32)
            layer_quantization: dict[str, object] | None = None
            if pack == 'fp32':
                weight_dtype = 'float32'
                weight_bytes = weight.numpy().tobytes(order='C')
            elif pack == 'fp16':
                weight_dtype = 'float16'
                weight_bytes = weight.to(torch.float16).numpy().tobytes(order='C')
            elif pack == 'int8-per-channel':
                weight_dtype = 'int8'
                quantized, scales, zero_points = quantize_int8_per_channel(weight)
                weight_bytes = quantized.numpy().tobytes(order='C')
                layer_quantization = {
                    'scheme': 'per-channel-symmetric',
                    'axis': 0,
                    'scale': scales,
                    'zeroPoint': zero_points,
                }
            else:
                raise RuntimeError(f'Unsupported pack: {pack}')
            bias_bytes = bias.numpy().tobytes(order='C')
            weight_offset = cursor
            shard_file.write(weight_bytes)
            cursor += len(weight_bytes)
            bias_offset = cursor
            shard_file.write(bias_bytes)
            cursor += len(bias_bytes)

            layers[f'conv{layer_index}'] = {
                'weight': {
                    'shape': list(weight.shape),
                    'dtype': weight_dtype,
                    'shard': shard_name,
                    'offset': weight_offset,
                    'length': len(weight_bytes),
                },
                'bias': {
                    'shape': [bias.numel()],
                    'dtype': 'float32',
                    'shard': shard_name,
                    'offset': bias_offset,
                    'length': len(bias_bytes),
                },
            }
            if layer_quantization is not None:
                layers[f'conv{layer_index}']['weight']['quantization'] = layer_quantization
            legacy_payload[f'conv{layer_index}.weightShape'] = list(weight.shape)
            legacy_payload[f'conv{layer_index}.weightValues'] = tensor_to_list(weight)
            legacy_payload[f'conv{layer_index}.biasValues'] = tensor_to_list(bias)

    manifest = {
        'modelId': 'vgg19-features-conv0-28',
        'version': '1.0.0',
        'format': f'{pack}-le',
        'layers': layers,
        'shards': [{
            'name': shard_name,
            'byteLength': cursor,
        }],
        'checksums': {
            'algorithm': 'sha256',
            'files': {
                shard_name: sha256_hex(shard_path),
            },
        },
        'quantization': {'scheme': 'none' if pack != 'int8-per-channel' else 'per-channel-symmetric', 'dtype': pack},
    }

    manifest_path = pack_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    manifest = json.loads(manifest_path.read_text())
    manifest['checksums']['files']['manifest.json'] = sha256_hex(manifest_path)
    manifest_path.write_text(json.dumps(manifest))

    if emit_legacy_json:
        (pack_dir / 'vgg19_conv0_to_conv28_weights.json').write_text(json.dumps(legacy_payload))


def export(emit_legacy_json: bool = True, packs: tuple[str, ...] = PACKS) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    for pack in packs:
        export_weights_manifest(model, emit_legacy_json, pack)
    truncated_layers = nn.Sequential(*list(model.children())[: LAST_LAYER_INDEX + 1])

    torch.manual_seed(13)
    content = load_image(ASSET_CONTENT_PATH, 16)
    style = load_image(ASSET_STYLE_PATH, 16)
    noise = (torch.rand_like(content) * 2.0 - 1.0) * 0.05
    input_image = torch.clamp(content + noise, 0.0, 1.0).clone().detach().requires_grad_(True)

    mean_vec = [0.485, 0.456, 0.406]
    std_vec = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean_vec, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std_vec, dtype=torch.float32).view(1, 3, 1, 1)

    content_norm = (content - mean) / std
    style_norm = (style - mean) / std
    input_norm = (input_image - mean) / std

    content_features = content_norm
    style_features = style_norm
    input_features = input_norm
    style_losses: dict[str, float] = {}
    style_loss_tensors: dict[int, torch.Tensor] = {}
    content_loss: float = 0.0
    content_loss_tensor: torch.Tensor | None = None

    for layer_index, layer in enumerate(truncated_layers):
      content_features = layer(content_features)
      style_features = layer(style_features)
      input_features = layer(input_features)

      if layer_index in STYLE_LAYER_INDICES:
        layer_style_loss = torch.nn.functional.mse_loss(gram_matrix(input_features), gram_matrix(style_features))
        style_losses[f'relu{layer_index}'] = layer_style_loss.item()
        style_loss_tensors[layer_index] = layer_style_loss
      if layer_index == CONTENT_LAYER_INDEX:
        content_loss_tensor = torch.nn.functional.mse_loss(input_features, content_features)
        content_loss = content_loss_tensor.item()

    if content_loss_tensor is None:
      raise RuntimeError('Content loss tensor was not computed.')

    total_style_loss_tensor = sum(style_loss_tensors[layer] for layer in STYLE_LAYER_INDICES)
    total_loss_tensor = content_loss_tensor + total_style_loss_tensor

    total_loss_tensor.backward(retain_graph=True)
    total_grad_input = input_image.grad.detach().clone()
    input_image.grad.zero_()

    content_loss_tensor.backward(retain_graph=True)
    content_grad_input = input_image.grad.detach().clone()
    input_image.grad.zero_()

    style_grad_by_layer: dict[str, list[float]] = {}
    for layer in STYLE_LAYER_INDICES:
      style_loss_tensors[layer].backward(retain_graph=True)
      style_grad_by_layer[f'relu{layer}'] = tensor_to_list(input_image.grad.detach().clone())
      input_image.grad.zero_()

    fixture_payload = {
      'imageSize': 16,
      'inputShape': [1, 3, 16, 16],
      'contentImageValues': tensor_to_list(content),
      'styleImageValues': tensor_to_list(style),
      'inputImageValues': tensor_to_list(input_image),
      'mean': mean_vec,
      'std': std_vec,
      'styleLayerIndices': STYLE_LAYER_INDICES,
      'contentLayerIndex': CONTENT_LAYER_INDEX,
      'expectedStyleLossByLayer': style_losses,
      'expectedStyleLossTotal': sum(style_losses.values()),
      'expectedContentLoss': content_loss,
      'expectedTotalLoss': content_loss + sum(style_losses.values()),
      'expectedGradients': {
        'content': tensor_to_list(content_grad_input),
        'styleByLayer': style_grad_by_layer,
        'total': tensor_to_list(total_grad_input),
      },
    }

    (OUTPUT_DIR / 'vgg19_phase3_full_pass_fixture.json').write_text(json.dumps(fixture_payload))
    if emit_legacy_json:
      (OUTPUT_DIR / 'vgg19_conv0_to_conv28_weights.json').write_text(
          (MODEL_OUTPUT_DIR / 'vgg19_conv0_to_conv28_weights.json').read_text()
      )
    print(f'Wrote artifacts to: {OUTPUT_DIR} and {MODEL_OUTPUT_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--packs', type=str, default=','.join(PACKS))
    parser.add_argument('--no-legacy-json', action='store_true')
    args = parser.parse_args()
    selected_packs = tuple(item.strip() for item in args.packs.split(',') if item.strip() != '')
    export(emit_legacy_json=not args.no_legacy_json, packs=selected_packs)
