from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.style_transfer_backend.api import create_app
from backend.style_transfer_backend.schemas import (
    HealthResponse,
    StyleTransferRunRequest,
    StyleTransferRunSuccessResponse,
    WorkerRunStats,
)


class FakeEngine:
    def __init__(self) -> None:
        self.cleared_sessions: list[str] = []
        self.last_request: StyleTransferRunRequest | None = None

    def health(self) -> HealthResponse:
        return HealthResponse(
            ok=True,
            device="fake-device",
            modelReady=True,
            message="fake engine ready",
        )

    def run(self, request: StyleTransferRunRequest) -> StyleTransferRunSuccessResponse:
        self.last_request = request
        return StyleTransferRunSuccessResponse(
            ok=True,
            losses=[12.5],
            finalValues=[min(1.0, value + 0.25) for value in request.inputImageValues],
            stats=WorkerRunStats(
                elapsedMs=4.0,
                avgStepMs=4.0,
                forwardMs=1.0,
                backwardMs=1.0,
                lossMs=1.0,
                updateMs=1.0,
                steps=request.steps,
            ),
        )

    def clear_session(self, session_id: str) -> None:
        self.cleared_sessions.append(session_id)


class FastApiBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FakeEngine()
        self.client = TestClient(create_app(self.engine))

    def test_health_reports_backend_and_model_readiness(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "ok": True,
                "backend": "fastapi",
                "engine": "pytorch",
                "device": "fake-device",
                "modelReady": True,
                "message": "fake engine ready",
            },
        )

    def test_run_accepts_server_owned_weights_request(self) -> None:
        values = [0.1, 0.2, 0.3]
        response = self.client.post(
            "/style-transfer/run",
            json={
                "sessionId": "session-1",
                "optimizer": "sgd",
                "inputShape": [1, 3, 1, 1],
                "contentShape": [1, 3, 1, 1],
                "styleShape": [1, 3, 1, 1],
                "inputImageValues": values,
                "contentImageValues": values,
                "styleImageValues": values,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "styleLayerIndices": [1],
                "contentLayerIndex": 1,
                "contentWeight": 1,
                "styleWeight": 1,
                "learningRate": 0.1,
                "steps": 1,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["losses"], [12.5])
        self.assertEqual(body["finalValues"], [0.35, 0.45, 0.55])
        self.assertEqual(body["stats"]["steps"], 1)
        self.assertIsNotNone(self.engine.last_request)
        self.assertEqual(self.engine.last_request.sessionId, "session-1")

    def test_run_rejects_browser_weight_payloads(self) -> None:
        response = self.client.post(
            "/style-transfer/run",
            json={
                "sessionId": "session-1",
                "optimizer": "sgd",
                "inputShape": [1, 3, 1, 1],
                "inputImageValues": [0.1, 0.2, 0.3],
                "contentImageValues": [0.1, 0.2, 0.3],
                "styleImageValues": [0.1, 0.2, 0.3],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "styleLayerIndices": [1],
                "contentLayerIndex": 1,
                "contentWeight": 1,
                "styleWeight": 1,
                "learningRate": 0.1,
                "steps": 1,
                "weights": {"conv0.weightValues": [1]},
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_clear_session_routes_to_engine(self) -> None:
        response = self.client.post(
            "/style-transfer/session/clear", json={"sessionId": "session-1"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(self.engine.cleared_sessions, ["session-1"])


if __name__ == "__main__":
    unittest.main()
