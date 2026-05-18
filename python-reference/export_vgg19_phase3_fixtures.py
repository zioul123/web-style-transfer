import json
from pathlib import Path

import torch
from torchvision.models import VGG19_Weights, vgg19

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / 'public' / 'vgg19-phase3'


def gram_matrix(input_tensor: torch.Tensor) -> torch.Tensor:
    a, b, c, d = input_tensor.size()
    features = input_tensor.view(a * b, c * d)
    gram = torch.mm(features, features.t())
    return gram.div(a * b * c * d)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    weights = model.state_dict()

    layer_names = ['0', '5', '10', '19', '28']  # conv1_1..conv5_1
    selected = {}
    for layer_id in layer_names:
      selected[f'conv{layer_id}.weight'] = weights[f'{layer_id}.weight'].cpu().tolist()
      selected[f'conv{layer_id}.bias'] = weights[f'{layer_id}.bias'].cpu().tolist()

    torch.manual_seed(7)
    input_tensor = torch.rand((1, 3, 16, 16), dtype=torch.float32)
    target_tensor = torch.rand((1, 3, 16, 16), dtype=torch.float32)

    fixture = {
        'inputShape': [1, 3, 16, 16],
        'inputValues': input_tensor.flatten().tolist(),
        'targetShape': [1, 3, 16, 16],
        'targetValues': target_tensor.flatten().tolist(),
        'expectedGram': gram_matrix(input_tensor).flatten().tolist(),
        'expectedContentLoss': torch.nn.functional.mse_loss(input_tensor, target_tensor).item(),
        'expectedStyleLoss': torch.nn.functional.mse_loss(gram_matrix(input_tensor), gram_matrix(target_tensor)).item(),
    }

    (OUTPUT_DIR / 'vgg19_conv1_1_to_conv5_1_weights.json').write_text(json.dumps(selected))
    (OUTPUT_DIR / 'phase3_loss_fixture.json').write_text(json.dumps(fixture))


if __name__ == '__main__':
    main()
