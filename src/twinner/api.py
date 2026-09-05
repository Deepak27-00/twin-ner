"""REST API and browser demo.

Design notes:
  * The model is loaded once at startup, not per request -- loading BERT on
    every call would dominate the latency.
  * Requests are size-capped before they reach the model, so a single oversized
    payload cannot pin the process.
  * CORS defaults to same-origin. Widen it in config deliberately; never "*".
  * The server binds 127.0.0.1 by default. Binding 0.0.0.0 exposes the service
    on every interface, which should be an explicit choice, not a default.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Config

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        description="The utterance to analyse",
        examples=["Show me motion sensors in Building003"],
    )


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., description="Several utterances to analyse in one call")


def create_app(config: Config) -> FastAPI:
    """Build the FastAPI application for a given configuration."""
    state: dict = {"predictor": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from .predict import Predictor

        logger.info("Loading checkpoint %s", config.checkpoint_path)
        state["predictor"] = Predictor.from_config(config)
        logger.info("Model ready")
        yield
        state.clear()

    app = FastAPI(
        title="Twin-NER API",
        version=__version__,
        description=(
            "Joint intent classification and entity extraction for digital-twin "
            "assistants. Interactive demo at `/`, OpenAPI docs at `/docs`."
        ),
        lifespan=lifespan,
    )

    if config.serve.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.serve.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    def get_predictor():
        predictor = state.get("predictor")
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model is still loading")
        return predictor

    def validate_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=422, detail="Text must be a non-empty string")
        if len(text) > config.serve.max_text_length:
            raise HTTPException(
                status_code=413,
                detail=f"Text exceeds {config.serve.max_text_length} characters",
            )
        return text.strip()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def demo() -> str:
        page = WEB_DIR / "index.html"
        if not page.exists():
            return "<h1>Twin-NER</h1><p>Demo page missing. API docs at /docs</p>"
        return page.read_text(encoding="utf-8")

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        """Liveness and readiness in one call, including checkpoint provenance."""
        predictor = state.get("predictor")
        return {
            "status": "ok" if predictor is not None else "loading",
            "version": __version__,
            "checkpoint": str(config.checkpoint_path),
            "device": str(predictor.device) if predictor else None,
            "trained_at": predictor.metadata.get("trained_at") if predictor else None,
        }

    @app.get("/labels", tags=["meta"])
    async def labels() -> dict:
        """The label space this model was trained on."""
        predictor = get_predictor()
        return {
            "intents": predictor.scheme.intents,
            "entity_types": predictor.scheme.entity_types,
            "tags": predictor.scheme.tags,
        }

    @app.post("/predict", tags=["inference"])
    async def predict(request: PredictRequest) -> dict:
        """Extract the intent and entities from a single utterance."""
        predictor = get_predictor()
        text = validate_text(request.text)
        started = time.perf_counter()
        prediction = predictor.predict(text)
        return {
            **prediction.to_dict(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    @app.post("/predict/batch", tags=["inference"])
    async def predict_batch(request: BatchPredictRequest) -> dict:
        """Analyse up to `serve.max_batch_size` utterances in one forward pass."""
        predictor = get_predictor()
        if not request.texts:
            raise HTTPException(status_code=422, detail="'texts' must not be empty")
        if len(request.texts) > config.serve.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"Batch exceeds {config.serve.max_batch_size} items",
            )
        texts = [validate_text(text) for text in request.texts]
        started = time.perf_counter()
        predictions = predictor.predict_batch(texts)
        return {
            "results": [prediction.to_dict() for prediction in predictions],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    return app


def serve(config: Config, reload: bool = False) -> None:
    """Run the API with uvicorn."""
    import uvicorn

    if not Path(config.checkpoint_path).exists():
        raise FileNotFoundError(
            f"No checkpoint at {config.checkpoint_path}. Train one first: twinner train"
        )

    print(f"\n  Twin-NER {__version__}")
    print(f"  demo   http://{config.serve.host}:{config.serve.port}/")
    print(f"  docs   http://{config.serve.host}:{config.serve.port}/docs\n")

    uvicorn.run(
        "twinner.api:app_from_env" if reload else create_app(config),
        host=config.serve.host,
        port=config.serve.port,
        reload=reload,
        factory=reload,
    )


def app_from_env() -> FastAPI:
    """Factory used by uvicorn's reloader, which needs an importable target."""
    return create_app(Config.load())
