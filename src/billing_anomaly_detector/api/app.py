from contextlib import asynccontextmanager

from fastapi import FastAPI

from billing_anomaly_detector.api.routes import anomalies, invoices
from billing_anomaly_detector.infrastructure.ai.embedding_adapter import (
    build_embedding_adapter,
)
from billing_anomaly_detector.infrastructure.ai.explanation_chain import (
    build_explanation_chain,
)
from billing_anomaly_detector.infrastructure.config import get_settings
from billing_anomaly_detector.infrastructure.persistence.database import (
    build_engine,
    build_session_factory,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages startup and shutdown of all application-level resources.
    Everything before yield runs at startup.
    Everything after yield runs at shutdown.
    """
    settings = get_settings()
    engine = build_engine(settings.database_url)

    app.state.settings = settings
    app.state.session_factory = build_session_factory(engine)
    app.state.embedding_adapter = build_embedding_adapter(settings)
    app.state.explanation_chain = build_explanation_chain(settings)

    yield

    await engine.dispose()


app = FastAPI(
    title="Billing Anomaly Detector",
    description=(
        "AI-powered healthcare billing anomaly detection. "
        "Uses embeddings and cosine similarity to flag unusual claims, "
        "with GPT-5 explanations on demand."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(invoices.router, prefix="/invoices", tags=["Invoices"])
app.include_router(anomalies.router, prefix="/anomalies", tags=["Anomalies"])


@app.get("/health", tags=["Health"])
async def health() -> dict[str, str]:
    """Liveness check — returns 200 if the API is running."""
    return {"status": "ok"}
