from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TensorShape = tuple[int, int, int, int]
RgbTriplet = tuple[float, float, float]


class HealthResponse(BaseModel):
    ok: bool
    backend: Literal["fastapi"] = "fastapi"
    engine: Literal["pytorch"] = "pytorch"
    device: str
    modelReady: bool
    message: str


class StyleTransferRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionId: str | None = None
    optimizer: Literal["sgd", "adam", "lbfgs"]
    adamBeta1: float | None = None
    adamBeta2: float | None = None
    adamEpsilon: float | None = None
    lbfgsMemory: int | None = None
    lbfgsEpsilon: float | None = None
    inputShape: TensorShape
    contentShape: TensorShape | None = None
    styleShape: TensorShape | None = None
    inputImageValues: list[float] = Field(min_length=1)
    contentImageValues: list[float] = Field(min_length=1)
    styleImageValues: list[float] = Field(min_length=1)
    mean: RgbTriplet
    std: RgbTriplet
    styleLayerIndices: list[int] = Field(min_length=1)
    contentLayerIndex: int
    contentWeight: float
    styleWeight: float
    learningRate: float
    steps: int = Field(ge=1, le=100)
    lossReadbackInterval: int | None = None


class WorkerRunStats(BaseModel):
    elapsedMs: float
    avgStepMs: float
    forwardMs: float
    backwardMs: float
    lossMs: float
    updateMs: float
    steps: int


class StyleTransferRunSuccessResponse(BaseModel):
    ok: Literal[True]
    losses: list[float]
    finalValues: list[float]
    stats: WorkerRunStats


class ErrorResponse(BaseModel):
    ok: Literal[False]
    message: str


StyleTransferRunResponse = StyleTransferRunSuccessResponse | ErrorResponse


class ClearSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionId: str


class ClearSessionSuccessResponse(BaseModel):
    ok: Literal[True]


ClearSessionResponse = ClearSessionSuccessResponse | ErrorResponse
