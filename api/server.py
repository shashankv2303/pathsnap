"""Run with python -m uvicorn api.server:app --host 0.0.0.0 --port 8000."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from api.service import ROOT, PredictionService

log = logging.getLogger('uvicorn.error')


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    trip_id: str = Field(pattern=r'^[0-9]{8}$')
    use_hmm: bool = True
    generate_map: bool = True


class MapData(BaseModel):
    gps: list[list[float]]
    route_segments: list[list[list[float]]]


class PredictResponse(BaseModel):
    trip_id: str
    status: str
    num_gps_observations: int
    execution_time_ms: float
    metrics: dict[str, float | int | bool | None]
    predicted_route: list[int | None]
    map_data: MapData | None


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service or PredictionService(
            os.getenv('PATHSNAP_TRIP_ROOT', str(ROOT / 'Student folder/student_test')),
            os.getenv('PATHSNAP_MODEL', str(ROOT / 'pathsnap/artifacts/deployment_model/model.joblib')))
        yield

    app = FastAPI(title='PathSnap', lifespan=lifespan)
    lock = Lock()

    @app.get('/health')
    def health():
        return dict(status='healthy', service='PathSnap', model_loaded=True,
                    timestamp=datetime.now(timezone.utc).isoformat())

    @app.get('/api/trips')
    def trips():
        return dict(evaluation_trips=app.state.service.trips())

    @app.post('/api/predict_trip', response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request):
        if not lock.acquire(blocking=False):
            raise HTTPException(503, 'Server is processing another trip; retry shortly')
        try:
            result = app.state.service.predict(**payload.model_dump())
            log.info('client=%s trip=%s elapsed_ms=%.1f', request.client.host,
                     payload.trip_id, result['execution_time_ms'])
            return result
        except FileNotFoundError:
            raise HTTPException(404, 'Trip or required input file is unavailable')
        except Exception:
            log.exception('Prediction failed for %s', payload.trip_id)
            raise HTTPException(500, 'Prediction failed; see server logs')
        finally:
            lock.release()

    return app


app = create_app()
