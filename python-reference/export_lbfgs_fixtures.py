from __future__ import annotations

import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "public" / "lbfgs"


def run_case(
    name: str,
    initial: list[float],
    target: list[float],
    matrix: list[list[float]],
    lr: float,
    steps: int,
    history_size: int,
) -> dict[str, object]:
    x = torch.tensor(initial, dtype=torch.float32, requires_grad=True)
    target_tensor = torch.tensor(target, dtype=torch.float32)
    matrix_tensor = torch.tensor(matrix, dtype=torch.float32)
    optimizer = torch.optim.LBFGS(
        [x],
        lr=lr,
        max_iter=1,
        history_size=history_size,
        line_search_fn=None,
    )
    trajectory: list[dict[str, object]] = []

    for step in range(steps):
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            delta = x - target_tensor
            loss = 0.5 * delta.dot(matrix_tensor.mv(delta))
            loss.backward()
            return loss

        loss_before = optimizer.step(closure)
        trajectory.append(
            {
                "step": step,
                "lossBefore": float(loss_before.detach()),
                "valuesAfter": x.detach().tolist(),
            }
        )

    return {
        "name": name,
        "initial": initial,
        "target": target,
        "matrix": matrix,
        "lr": lr,
        "steps": steps,
        "historySize": history_size,
        "trajectory": trajectory,
    }


def export() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "torchVersion": torch.__version__,
        "cases": [
            run_case(
                name="diagonal_3d",
                initial=[0.2, 0.7, 0.4],
                target=[0.8, 0.1, 0.6],
                matrix=[
                    [1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.5],
                ],
                lr=1.0,
                steps=6,
                history_size=4,
            ),
            run_case(
                name="coupled_4d",
                initial=[0.15, 0.35, 0.65, 0.85],
                target=[0.75, 0.2, 0.5, 0.1],
                matrix=[
                    [2.0, 0.2, 0.0, 0.0],
                    [0.2, 1.5, 0.1, 0.0],
                    [0.0, 0.1, 1.0, 0.15],
                    [0.0, 0.0, 0.15, 1.25],
                ],
                lr=0.8,
                steps=7,
                history_size=5,
            ),
        ],
    }
    (OUTPUT_DIR / "lbfgs_fixture.json").write_text(json.dumps(payload))
    print(f"Wrote {OUTPUT_DIR / 'lbfgs_fixture.json'}")


if __name__ == "__main__":
    export()
