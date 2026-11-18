"""Run with python -m uvicorn api.server:app --host 0.0.0.0 --port 8000."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from api.service import ROOT, PredictionService

log = logging.getLogger('uvicorn.error')


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    trip_id: str = Field(pattern=r'^[0-9]{8}$')
    use_hmm: bool = True
    generate_map: bool = True


class MapData(BaseModel):
    gps: List[List[float]]
    route_segments: List[List[List[float]]]


class PredictResponse(BaseModel):
    trip_id: str
    status: str
    num_gps_observations: int
    execution_time_ms: float
    execution_time_seconds: Optional[float] = None
    metrics: Dict[str, Any]
    predicted_route: List[Optional[int]]
    matched_sequence: Optional[List[Optional[int]]] = None
    map_data: Optional[MapData] = None


class HealthResponse(BaseModel):
    status: str
    service: str = "PathSnap"
    ready: bool = True
    model_loaded: bool = True
    mode: str
    trip_count: int
    trips_count: int = 0
    timestamp: str


class TripsResponse(BaseModel):
    evaluation_trips: List[str]
    trips: List[str] = Field(default_factory=list)
    mode: str
    trip_count: int


def create_app(service: Optional[Any] = None) -> FastAPI:
    """FastAPI application factory for Laptop 1 Model Server."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.service = service
        else:
            try:
                app.state.service = PredictionService()
                log.info(
                    "PathSnap Model Server online. Mode: %s | Trips: %d",
                    app.state.service.mode,
                    len(app.state.service.trips()),
                )
            except Exception as exc:
                log.critical("Failed to initialize PredictionService on startup: %s", exc)
                raise
        yield

    app = FastAPI(
        title="PathSnap Map-Matching API",
        description="Laptop 1 Backend for distributed map-matching, route reconstruction, and server-side evaluation.",
        version="1.0.0",
        lifespan=lifespan,
    )
    lock = Lock()

    @app.get("/health", response_model=HealthResponse)
    def health():
        srv = getattr(app.state, "service", None)
        if srv is None:
            return HealthResponse(
                status="unhealthy",
                service="PathSnap",
                ready=False,
                model_loaded=False,
                mode="unknown",
                trip_count=0,
                trips_count=0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        is_loaded = getattr(srv, "is_ready", lambda: True)()
        mode = getattr(srv, "mode", "unknown")
        trips_list = srv.trips()
        count = len(trips_list)
        return HealthResponse(
            status="healthy" if is_loaded else "degraded",
            service="PathSnap",
            ready=bool(is_loaded),
            model_loaded=bool(is_loaded),
            mode=mode,
            trip_count=count,
            trips_count=count,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @app.get("/api/trips", response_model=TripsResponse)
    def trips():
        srv = getattr(app.state, "service", None)
        if srv is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service is not initialized",
            )
        trip_ids = srv.trips()
        mode = getattr(srv, "mode", "unknown")
        count = len(trip_ids)
        return TripsResponse(
            evaluation_trips=trip_ids,
            trips=trip_ids,
            mode=mode,
            trip_count=count,
        )

    @app.post("/api/predict_trip", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request):
        srv = getattr(app.state, "service", None)
        if srv is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service is not initialized",
            )

        # Enforce single-request concurrency lock (CPU-bound inference protection)
        if not lock.acquire(blocking=False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is processing another trip; retry shortly",
            )

        client_ip = request.client.host if request.client else "unknown"
        now_ts = datetime.now(timezone.utc).isoformat()

        try:
            result = srv.predict(**payload.model_dump())
            metrics = result.get("metrics", {})
            acc = metrics.get("pointwise_accuracy")
            cont = metrics.get("continuity")

            # Structured API Access Log
            log.info(
                "[%s] client=%s trip=%s status=SUCCESS elapsed_ms=%.1f acc=%s cont=%s hmm=%s",
                now_ts,
                client_ip,
                payload.trip_id,
                result["execution_time_ms"],
                f"{acc:.2%}" if acc is not None else "N/A",
                f"{cont:.2%}" if cont is not None else "N/A",
                payload.use_hmm,
            )
            return result
        except FileNotFoundError as fnf:
            log.warning(
                "[%s] client=%s trip=%s status=NOT_FOUND reason=%s",
                now_ts,
                client_ip,
                payload.trip_id,
                str(fnf),
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trip '{payload.trip_id}' or required input file is unavailable",
            )
        except RuntimeError as rerr:
            log.error(
                "[%s] client=%s trip=%s status=RUNTIME_ERROR reason=%s",
                now_ts,
                client_ip,
                payload.trip_id,
                str(rerr),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(rerr),
            )
        except Exception as exc:
            log.exception(
                "[%s] client=%s trip=%s status=FAILED reason=%s",
                now_ts,
                client_ip,
                payload.trip_id,
                str(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed; see server logs",
            )
        finally:
            lock.release()

    return app


app = create_app()
