from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .engine import PyTorchStyleTransferEngine, StyleTransferEngine
from .schemas import (
    ClearSessionRequest,
    ClearSessionResponse,
    ClearSessionSuccessResponse,
    ErrorResponse,
    HealthResponse,
    StyleTransferRunRequest,
    StyleTransferRunResponse,
)


def create_app(engine: StyleTransferEngine | None = None) -> FastAPI:
    backend_engine = engine or PyTorchStyleTransferEngine()
    app = FastAPI(title="Web Style Transfer Backend")
    app.state.style_transfer_engine = backend_engine
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return backend_engine.health()

    @app.post("/style-transfer/run", response_model=StyleTransferRunResponse)
    def run_style_transfer(
        request: StyleTransferRunRequest,
    ) -> StyleTransferRunResponse:
        try:
            return backend_engine.run(request)
        except Exception as error:
            return ErrorResponse(ok=False, message=str(error))

    @app.post("/style-transfer/session/clear", response_model=ClearSessionResponse)
    def clear_style_transfer_session(
        request: ClearSessionRequest,
    ) -> ClearSessionResponse:
        try:
            backend_engine.clear_session(request.sessionId)
            return ClearSessionSuccessResponse(ok=True)
        except Exception as error:
            return ErrorResponse(ok=False, message=str(error))

    return app


app = create_app()
