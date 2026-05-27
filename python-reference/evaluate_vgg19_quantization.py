from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / 'public' / 'vgg19-models'
STYLE_LAYER_INDICES = [1, 6, 11, 20, 29]
CONTENT_LAYER_INDEX = 22
LAST_LAYER_INDEX = 29


def _load_pack_model(pack: str) -> nn.Sequential:
    model = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    manifest = json.loads((MODEL_ROOT / pack / 'manifest.json').read_text())
    shard = (MODEL_ROOT / pack / 'shard-000.bin').read_bytes()
    for layer_index in range(0, LAST_LAYER_INDEX + 1):
        layer = model[layer_index]
        if not isinstance(layer, nn.Conv2d):
            continue
        entry = manifest['layers'][f'conv{layer_index}']
        w = entry['weight']
        b = entry['bias']
        w_bytes = shard[w['offset']:w['offset'] + w['length']]
        b_bytes = shard[b['offset']:b['offset'] + b['length']]
        if w['dtype'] == 'float32':
            w_t = torch.frombuffer(bytearray(w_bytes), dtype=torch.float32).clone().view(w['shape'])
        elif w['dtype'] == 'float16':
            w_t = torch.frombuffer(bytearray(w_bytes), dtype=torch.float16).clone().to(torch.float32).view(w['shape'])
        elif w['dtype'] == 'int8':
            q = torch.frombuffer(bytearray(w_bytes), dtype=torch.int8).clone().to(torch.float32).view(w['shape'])
            quant = w['quantization']
            scale = torch.tensor(quant['scale'], dtype=torch.float32).view(-1, 1, 1, 1)
            zp = torch.tensor(quant['zeroPoint'], dtype=torch.float32).view(-1, 1, 1, 1)
            w_t = (q - zp) * scale
        else:
            raise RuntimeError('unsupported dtype')
        b_t = torch.frombuffer(bytearray(b_bytes), dtype=torch.float32).clone().view(-1)
        with torch.no_grad():
            layer.weight.copy_(w_t)
            layer.bias.copy_(b_t)
    return nn.Sequential(*list(model.children())[: LAST_LAYER_INDEX + 1]).eval()


def _forward_layers(model: nn.Sequential, image: torch.Tensor) -> dict[int, torch.Tensor]:
    out = image
    result: dict[int, torch.Tensor] = {}
    for idx, layer in enumerate(model):
        out = layer(out)
        result[idx] = out.detach().clone()
    return result


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    a, b, c, d = x.size()
    f = x.view(a * b, c * d)
    return torch.mm(f, f.t()) / (a * b * c * d)


def main() -> None:
    torch.manual_seed(13)
    input_image = torch.rand((1, 3, 16, 16), dtype=torch.float32)
    content_image = torch.rand((1, 3, 16, 16), dtype=torch.float32)
    style_image = torch.rand((1, 3, 16, 16), dtype=torch.float32)

    baseline = _load_pack_model('fp32')
    baseline_layers = _forward_layers(baseline, input_image)

    report: dict[str, object] = {'packs': {}}
    for pack in ('fp16', 'int8-per-channel'):
        model = _load_pack_model(pack)
        layers = _forward_layers(model, input_image)
        layer_error = {
            f'relu{idx}': float(torch.mean(torch.abs(layers[idx] - baseline_layers[idx])).item())
            for idx in STYLE_LAYER_INDICES + [CONTENT_LAYER_INDEX]
        }
        style_loss = sum(torch.nn.functional.mse_loss(gram_matrix(layers[idx]), gram_matrix(_forward_layers(model, style_image)[idx])).item() for idx in STYLE_LAYER_INDICES)
        style_loss_base = sum(torch.nn.functional.mse_loss(gram_matrix(baseline_layers[idx]), gram_matrix(_forward_layers(baseline, style_image)[idx])).item() for idx in STYLE_LAYER_INDICES)
        content_loss = torch.nn.functional.mse_loss(layers[CONTENT_LAYER_INDEX], _forward_layers(model, content_image)[CONTENT_LAYER_INDEX]).item()
        content_loss_base = torch.nn.functional.mse_loss(baseline_layers[CONTENT_LAYER_INDEX], _forward_layers(baseline, content_image)[CONTENT_LAYER_INDEX]).item()
        report['packs'][pack] = {
            'perLayerMae': layer_error,
            'styleLossDelta': style_loss - style_loss_base,
            'contentLossDelta': content_loss - content_loss_base,
            'acceptance': {
                'fp16NearBaseline': pack != 'fp16' or abs(style_loss - style_loss_base) < 1e-3,
                'int8Acceptable': pack != 'int8-per-channel' or abs(style_loss - style_loss_base) < 5e-2,
            },
        }

    out_path = REPO_ROOT / 'public' / 'vgg19-phase3-full-pass' / 'quantization_eval_report.json'
    out_path.write_text(json.dumps(report, indent=2))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
