from __future__ import annotations

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

STYLE_LAYER_INDICES = [1, 6, 11, 20, 29]  # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
CONTENT_LAYER_INDEX = 22  # relu4_2
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


def export() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
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

    weights_payload: dict[str, object] = {}
    for layer_index in range(0, LAST_LAYER_INDEX + 1):
      layer = model[layer_index]
      if isinstance(layer, nn.Conv2d):
        weights_payload[f'conv{layer_index}.weightShape'] = list(layer.weight.shape)
        weights_payload[f'conv{layer_index}.weightValues'] = tensor_to_list(layer.weight)
        weights_payload[f'conv{layer_index}.biasValues'] = tensor_to_list(layer.bias)

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

    (OUTPUT_DIR / 'vgg19_conv0_to_conv28_weights.json').write_text(json.dumps(weights_payload))
    (OUTPUT_DIR / 'vgg19_phase3_full_pass_fixture.json').write_text(json.dumps(fixture_payload))
    print(f'Wrote artifacts to: {OUTPUT_DIR}')


if __name__ == '__main__':
    export()
