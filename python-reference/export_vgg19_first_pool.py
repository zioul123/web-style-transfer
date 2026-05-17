from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import VGG19_Weights, vgg19
from torchvision.transforms import functional as TF

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSET_PATH = REPO_ROOT / 'assets' / 'madeira_128x128.jpg'
OUTPUT_DIR = REPO_ROOT / 'public' / 'vgg19-first-pool'


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().contiguous().view(-1).tolist()


def export() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    first_block = nn.Sequential(*list(model.children())[:5])  # conv/relu/conv/relu/pool

    image = Image.open(ASSET_PATH).convert('RGB').resize((16, 16), Image.Resampling.BILINEAR)
    x = TF.to_tensor(image).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    x_norm = (x - mean) / std

    out = first_block(x_norm)

    conv1 = model[0]
    conv2 = model[2]
    if not isinstance(conv1, nn.Conv2d) or not isinstance(conv2, nn.Conv2d):
        raise RuntimeError('Unexpected VGG19 layout for first two convolution layers.')

    weights_payload = {
        'inputShape': [1, 3, 16, 16],
        'conv1WeightShape': list(conv1.weight.shape),
        'conv1WeightValues': tensor_to_list(conv1.weight),
        'conv1BiasValues': tensor_to_list(conv1.bias),
        'conv2WeightShape': list(conv2.weight.shape),
        'conv2WeightValues': tensor_to_list(conv2.weight),
        'conv2BiasValues': tensor_to_list(conv2.bias),
    }

    input_payload = {
        'inputShape': [1, 3, 16, 16],
        'inputValues': tensor_to_list(x),
        'normalizedValues': tensor_to_list(x_norm),
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'expectedShape': [1, 64, 8, 8],
        'expectedValues': tensor_to_list(out),
    }

    (OUTPUT_DIR / 'vgg19_first_pool_weights.json').write_text(json.dumps(weights_payload))
    (OUTPUT_DIR / 'vgg19_first_pool_case_madeira16.json').write_text(json.dumps(input_payload))
    print(f'Wrote artifacts to: {OUTPUT_DIR}')


if __name__ == '__main__':
    export()
