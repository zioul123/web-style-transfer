from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from math import prod
from typing import Protocol

from .schemas import (
    HealthResponse,
    StyleTransferRunRequest,
    StyleTransferRunSuccessResponse,
    WorkerRunStats,
)

MAX_TENSOR_VALUES = 3 * 1024 * 1024


class StyleTransferEngine(Protocol):
    def health(self) -> HealthResponse:
        ...

    def run(self, request: StyleTransferRunRequest) -> StyleTransferRunSuccessResponse:
        ...

    def clear_session(self, session_id: str) -> None:
        ...


@dataclass
class _SessionState:
    signature: tuple[object, ...]
    output: object
    optimizer: object


class PyTorchStyleTransferEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._torch = None
        self._nn = None
        self._optim = None
        self._features = None
        self._device = "unknown"
        self._model_ready = False
        self._sessions: dict[str, _SessionState] = {}

    def health(self) -> HealthResponse:
        try:
            torch = self._import_torch()
            device = self._select_device(torch)
            return HealthResponse(
                ok=True,
                device=str(device),
                modelReady=self._model_ready,
                message=(
                    "PyTorch backend ready."
                    if self._model_ready
                    else "PyTorch backend ready; VGG19 will load on first run."
                ),
            )
        except Exception as error:
            return HealthResponse(
                ok=False,
                device=self._device,
                modelReady=False,
                message=str(error),
            )

    def run(self, request: StyleTransferRunRequest) -> StyleTransferRunSuccessResponse:
        if request.steps < 1:
            raise ValueError("steps must be at least 1.")
        with self._lock:
            torch = self._import_torch()
            nn = self._import_nn()
            optim = self._import_optim()
            model = self._ensure_model()

            input_shape = self._validate_shape(
                "inputShape", request.inputShape, request.inputImageValues
            )
            content_shape = self._validate_shape(
                "contentShape",
                request.contentShape or request.inputShape,
                request.contentImageValues,
            )
            style_shape = self._validate_shape(
                "styleShape",
                request.styleShape or request.inputShape,
                request.styleImageValues,
            )

            content = self._to_tensor(request.contentImageValues, content_shape)
            style = self._to_tensor(request.styleImageValues, style_shape)

            signature = self._session_signature(request, input_shape)
            session = self._get_or_create_session(request, signature, input_shape, optim)
            output = session.output
            optimizer = session.optimizer

            timings = _TimingBuckets()
            losses: list[float] = []
            started_at = time.perf_counter()

            with torch.no_grad():
                target_start = time.perf_counter()
                content_targets = self._features_for(
                    model,
                    content,
                    request.mean,
                    request.std,
                    [request.contentLayerIndex],
                )
                style_features = self._features_for(
                    model, style, request.mean, request.std, request.styleLayerIndices
                )
                style_targets = {
                    layer_index: self._gram_matrix(feature).detach()
                    for layer_index, feature in style_features.items()
                }
                timings.forward += time.perf_counter() - target_start

            for _ in range(request.steps):
                if request.optimizer == "lbfgs":
                    last_loss: float | None = None

                    def closure():
                        nonlocal last_loss
                        with torch.no_grad():
                            output.clamp_(0, 1)
                        optimizer.zero_grad()
                        loss, step_timings = self._compute_loss(
                            model,
                            output,
                            content_targets,
                            style_targets,
                            request,
                            nn,
                        )
                        timings.add(step_timings)
                        loss.backward()
                        last_loss = float(loss.detach().cpu().item())
                        return loss

                    update_start = time.perf_counter()
                    optimizer.step(closure)
                    timings.update += time.perf_counter() - update_start
                    losses.append(last_loss if last_loss is not None else 0.0)
                else:
                    with torch.no_grad():
                        output.clamp_(0, 1)
                    optimizer.zero_grad()
                    loss, step_timings = self._compute_loss(
                        model,
                        output,
                        content_targets,
                        style_targets,
                        request,
                        nn,
                    )
                    timings.add(step_timings)
                    backward_start = time.perf_counter()
                    loss.backward()
                    timings.backward += time.perf_counter() - backward_start
                    update_start = time.perf_counter()
                    optimizer.step()
                    timings.update += time.perf_counter() - update_start
                    losses.append(float(loss.detach().cpu().item()))

            with torch.no_grad():
                output.clamp_(0, 1)

            elapsed = time.perf_counter() - started_at
            final_values = output.detach().cpu().reshape(-1).tolist()
            return StyleTransferRunSuccessResponse(
                ok=True,
                losses=losses,
                finalValues=[float(value) for value in final_values],
                stats=WorkerRunStats(
                    elapsedMs=elapsed * 1000,
                    avgStepMs=(elapsed * 1000) / request.steps,
                    forwardMs=timings.forward * 1000,
                    backwardMs=timings.backward * 1000,
                    lossMs=timings.loss * 1000,
                    updateMs=timings.update * 1000,
                    steps=request.steps,
                ),
            )

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _import_torch(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _import_nn(self):
        if self._nn is None:
            import torch.nn as nn

            self._nn = nn
        return self._nn

    def _import_optim(self):
        if self._optim is None:
            import torch.optim as optim

            self._optim = optim
        return self._optim

    def _select_device(self, torch):
        if torch.cuda.is_available():
            self._device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        return torch.device(self._device)

    def _ensure_model(self):
        if self._features is not None:
            return self._features
        torch = self._import_torch()
        nn = self._import_nn()
        from torchvision.models import VGG19_Weights, vgg19

        device = self._select_device(torch)
        features = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
        features.requires_grad_(False)
        for layer in features:
            if isinstance(layer, nn.ReLU):
                layer.inplace = False
        self._features = features
        self._model_ready = True
        return features

    def _validate_shape(
        self, name: str, shape: tuple[int, int, int, int], values: list[float]
    ) -> tuple[int, int, int, int]:
        if shape[0] != 1 or shape[1] != 3:
            raise ValueError(f"{name} must be [1, 3, height, width].")
        if shape[2] <= 0 or shape[3] <= 0:
            raise ValueError(f"{name} height and width must be positive.")
        expected = prod(shape)
        if expected > MAX_TENSOR_VALUES:
            raise ValueError(f"{name} exceeds the maximum supported tensor size.")
        if len(values) != expected:
            raise ValueError(
                f"{name} expects {expected} values, received {len(values)}."
            )
        return shape

    def _to_tensor(self, values: list[float], shape: tuple[int, int, int, int]):
        torch = self._import_torch()
        device = self._select_device(torch)
        return torch.tensor(values, dtype=torch.float32, device=device).reshape(shape)

    def _get_or_create_session(
        self,
        request: StyleTransferRunRequest,
        signature: tuple[object, ...],
        input_shape: tuple[int, int, int, int],
        optim,
    ) -> _SessionState:
        session_id = request.sessionId or "__ephemeral__"
        existing = self._sessions.get(session_id)
        if existing is not None and existing.signature == signature:
            return existing

        output = self._to_tensor(request.inputImageValues, input_shape)
        output = output.clone().detach().requires_grad_(True)
        optimizer = self._make_optimizer(request, output, optim)
        session = _SessionState(signature=signature, output=output, optimizer=optimizer)
        if request.sessionId is not None:
            self._sessions[session_id] = session
        return session

    def _make_optimizer(self, request: StyleTransferRunRequest, output, optim):
        if request.optimizer == "sgd":
            return optim.SGD([output], lr=request.learningRate)
        if request.optimizer == "adam":
            beta1 = request.adamBeta1 if request.adamBeta1 is not None else 0.9
            beta2 = request.adamBeta2 if request.adamBeta2 is not None else 0.999
            epsilon = request.adamEpsilon if request.adamEpsilon is not None else 1e-8
            return optim.Adam(
                [output], lr=request.learningRate, betas=(beta1, beta2), eps=epsilon
            )
        history_size = request.lbfgsMemory if request.lbfgsMemory is not None else 10
        tolerance_change = (
            request.lbfgsEpsilon if request.lbfgsEpsilon is not None else 1e-9
        )
        return optim.LBFGS(
            [output],
            lr=request.learningRate,
            max_iter=1,
            history_size=max(1, history_size),
            tolerance_change=tolerance_change,
        )

    def _session_signature(
        self, request: StyleTransferRunRequest, input_shape: tuple[int, int, int, int]
    ) -> tuple[object, ...]:
        return (
            input_shape,
            request.optimizer,
            request.learningRate,
            request.adamBeta1,
            request.adamBeta2,
            request.adamEpsilon,
            request.lbfgsMemory,
            request.lbfgsEpsilon,
        )

    def _features_for(
        self,
        model,
        image,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        layer_indices: list[int],
    ):
        torch = self._import_torch()
        wanted = set(layer_indices)
        stop_at = max(wanted)
        device = image.device
        mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device).view(
            1, 3, 1, 1
        )
        std_tensor = torch.tensor(std, dtype=torch.float32, device=device).view(
            1, 3, 1, 1
        )
        x = (image - mean_tensor) / std_tensor
        outputs = {}
        for index, layer in enumerate(model):
            x = layer(x)
            if index in wanted:
                outputs[index] = x
            if index >= stop_at and len(outputs) == len(wanted):
                break
        missing = wanted - set(outputs)
        if missing:
            raise ValueError(f"VGG graph did not produce layers: {sorted(missing)}")
        return outputs

    def _compute_loss(
        self,
        model,
        output,
        content_targets,
        style_targets,
        request: StyleTransferRunRequest,
        nn,
    ):
        timings = _TimingBuckets()
        forward_start = time.perf_counter()
        layer_indices = [request.contentLayerIndex, *request.styleLayerIndices]
        features = self._features_for(
            model, output, request.mean, request.std, layer_indices
        )
        timings.forward += time.perf_counter() - forward_start

        loss_start = time.perf_counter()
        content_loss = nn.functional.mse_loss(
            features[request.contentLayerIndex], content_targets[request.contentLayerIndex]
        )
        style_loss = sum(
            nn.functional.mse_loss(
                self._gram_matrix(features[layer_index]), style_targets[layer_index]
            )
            for layer_index in request.styleLayerIndices
        )
        total = content_loss * request.contentWeight + style_loss * request.styleWeight
        timings.loss += time.perf_counter() - loss_start
        return total, timings

    def _gram_matrix(self, feature):
        batch, channels, height, width = feature.shape
        flattened = feature.view(batch * channels, height * width)
        gram = flattened @ flattened.t()
        return gram.div(batch * channels * height * width)


@dataclass
class _TimingBuckets:
    forward: float = 0.0
    backward: float = 0.0
    loss: float = 0.0
    update: float = 0.0

    def add(self, other: "_TimingBuckets") -> None:
        self.forward += other.forward
        self.backward += other.backward
        self.loss += other.loss
        self.update += other.update
