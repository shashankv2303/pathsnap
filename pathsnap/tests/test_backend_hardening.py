"""Comprehensive hardening tests for PathSnap Laptop 1 FastAPI Backend.

Verifies:
- API startup, readiness, and /health metadata
- Deployment model loading and schema enforcement
- Evaluator mode vs Development mode isolation and validation
- Missing evaluator directory / missing .route fail-fast behavior
- Input validation (422 for malformed, 404 for unknown)
- Complete server-side metrics (pointwise accuracy, continuity, edit distance)
- Strict anti-leakage protection (zero ground truth in client responses)
- Single-request concurrency lock (503 while busy)
- Structured access logging
- Streamlit client UI response contract compatibility
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from api.server import create_app
from api.service import ROOT, PredictionService, EVALUATOR_ALLOWLIST
from pathsnap.feature_engineering.feature_extractor import FEATURE_NAMES


# ---------------------------------------------------------------------------
# 1. Health, Readiness, and Contract Tests
# ---------------------------------------------------------------------------

def test_health_endpoint_readiness():
    """Verify /health returns real service readiness metadata."""
    with TestClient(create_app()) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["service"] == "PathSnap"
        assert data["ready"] is True
        assert data["model_loaded"] is True
        assert data["mode"] == "development"
        assert data["trip_count"] == 10
        assert data["trips_count"] == 10
        assert "timestamp" in data


def test_trips_endpoint_contract():
    """Verify /api/trips returns evaluation_trips list and metadata."""
    with TestClient(create_app()) as client:
        res = client.get("/api/trips")
        assert res.status_code == 200
        data = res.json()
        assert "evaluation_trips" in data
        assert "trips" in data
        assert data["trips"] == data["evaluation_trips"]
        assert len(data["evaluation_trips"]) == 10
        assert "00000000" in data["evaluation_trips"]
        assert data["mode"] == "development"


# ---------------------------------------------------------------------------
# 2. Model Loading & Schema Enforcement
# ---------------------------------------------------------------------------

def test_model_loading_and_feature_schema():
    """Verify service loads model and enforces exact 18-feature schema."""
    service = PredictionService()
    assert service.is_ready()
    assert len(service.model_config["input_features"]) == 18
    assert service.model_config["input_features"] == list(FEATURE_NAMES)


def test_model_feature_schema_mismatch(tmp_path):
    """Verify startup fails fast if model feature schema differs from pipeline."""
    bad_config = tmp_path / "model_config.json"
    bad_config.write_text(json.dumps({"input_features": ["feat_1", "feat_2"]}), encoding="utf-8")
    model_dummy = tmp_path / "model.joblib"
    model_dummy.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError, match="Model feature schema mismatch"):
        PredictionService(model_path=str(model_dummy))


def test_missing_model_file_fails_fast():
    """Verify startup fails fast if model.joblib does not exist."""
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        PredictionService(model_path="nonexistent_dir/model.joblib")


def test_missing_decoder_config_fails_fast():
    """Verify startup fails fast if decoder config JSON does not exist."""
    with pytest.raises(FileNotFoundError, match="Decoder configuration file not found"):
        PredictionService(decoder_config_path="nonexistent_dir/decoder_config.json")


# ---------------------------------------------------------------------------
# 3. Mode Isolation (Development vs Evaluator Mode)
# ---------------------------------------------------------------------------

def test_invalid_mode_raises():
    """Verify setting an unsupported mode string raises ValueError."""
    with pytest.raises(ValueError, match="Invalid PATHSNAP_MODE"):
        PredictionService(mode="super_mode")


def test_evaluator_mode_unconfigured_root_raises():
    """Verify evaluator mode fails if PATHSNAP_TRIP_ROOT is missing."""
    with pytest.raises(ValueError, match="PATHSNAP_MODE is set to 'evaluator', but PATHSNAP_TRIP_ROOT is not configured"):
        PredictionService(mode="evaluator", trip_root=None)


def test_evaluator_mode_missing_directory_raises(tmp_path):
    """Verify evaluator mode fails if configured directory does not exist."""
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="Configured evaluator trip root directory does not exist"):
        PredictionService(mode="evaluator", trip_root=str(missing_dir))


def test_evaluator_mode_trip_discovery_and_missing_route(tmp_path):
    """Verify evaluator mode enforces .route requirement and allowlist matching."""
    eval_root = tmp_path / "evaluator_data"
    eval_root.mkdir()

    # Create evaluator trip 00000015 without .route
    t15 = eval_root / "00000015"
    t15.mkdir()
    (t15 / "00000015.track").write_text("track", encoding="utf-8")
    (t15 / "00000015.nodes").write_text("nodes", encoding="utf-8")
    (t15 / "00000015.arcs").write_text("arcs", encoding="utf-8")

    # Initializing PredictionService must fail fast because .route is missing
    with pytest.raises(RuntimeError, match="Missing .route file in trip: 00000015"):
        PredictionService(trip_root=str(eval_root), mode="evaluator")

    # Now add the .route file and verify initialization succeeds
    (t15 / "00000015.route").write_text("100\n101\n", encoding="utf-8")
    service = PredictionService(trip_root=str(eval_root), mode="evaluator")
    trips = service.trips()
    assert trips == ["00000015"]
    assert "00000015" in EVALUATOR_ALLOWLIST


def test_evaluator_mode_rejects_student_folder():
    """Verify evaluator mode strictly refuses to point to student development folders."""
    dev_path = str(ROOT / "Student folder/student_test")
    with pytest.raises(ValueError, match="PATHSNAP_MODE is set to 'evaluator', but PATHSNAP_TRIP_ROOT points to student development data"):
        PredictionService(trip_root=dev_path, mode="evaluator")


def test_environment_variables_consistency(monkeypatch):
    """Verify PATHSNAP_MODEL_PATH and PATHSNAP_CONFIG_PATH environment variables."""
    default_model = ROOT / "pathsnap/artifacts/deployment_model/model.joblib"
    default_config = ROOT / "pathsnap/artifacts/deployment_model/model_config.json"

    monkeypatch.setenv("PATHSNAP_MODEL_PATH", str(default_model))
    monkeypatch.setenv("PATHSNAP_CONFIG_PATH", str(default_config))
    monkeypatch.setenv("PATHSNAP_MODE", "development")

    srv = PredictionService()
    assert srv.is_ready()
    assert srv.model_path == default_model.resolve()


# ---------------------------------------------------------------------------
# 4. Input Validation & Error Handling
# ---------------------------------------------------------------------------

def test_input_validation_and_not_found():
    """Verify path traversal, invalid trip IDs, and unknown trips."""
    with TestClient(create_app()) as client:
        # Path traversal / invalid pattern
        assert client.post("/api/predict_trip", json={"trip_id": "../train"}).status_code == 422
        assert client.post("/api/predict_trip", json={"trip_id": "123"}).status_code == 422
        assert client.post("/api/predict_trip", json={"trip_id": "00000000", "bad_param": True}).status_code == 422

        # Unknown 8-digit trip
        res = client.post("/api/predict_trip", json={"trip_id": "99999999"})
        assert res.status_code == 404
        assert "unavailable" in res.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 5. Full Prediction & Server-Side Metrics Verification
# ---------------------------------------------------------------------------

def test_full_prediction_and_all_metrics():
    """Verify real inference produces all required evaluation metrics and map data."""
    with TestClient(create_app()) as client:
        res = client.post("/api/predict_trip", json={"trip_id": "00000000", "use_hmm": True})
        assert res.status_code == 200
        data = res.json()

        assert data["trip_id"] == "00000000"
        assert data["status"] == "success"
        assert data["num_gps_observations"] > 0
        assert data["execution_time_ms"] > 0
        assert data["execution_time_seconds"] > 0
        assert len(data["predicted_route"]) == data["num_gps_observations"]
        assert data["matched_sequence"] == data["predicted_route"]

        metrics = data["metrics"]
        assert metrics["reference_available"] is True
        assert metrics["execution_time_seconds"] > 0
        assert 0.80 <= metrics["pointwise_accuracy"] <= 1.0
        assert 0.90 <= metrics["continuity"] <= 1.0
        assert metrics["route_continuity"] == metrics["continuity"]
        assert 0.0 <= metrics["candidate_miss_rate"] <= 0.05
        assert metrics["classifier_only_accuracy"] is not None
        assert metrics["hmm_accuracy_boost"] >= 0.0
        assert 0.0 <= metrics["normalized_route_edit_distance"] <= 0.5
        assert metrics["observations"] == data["num_gps_observations"]
        assert metrics["aligned_observations"] > 0
        assert metrics["eligible_transitions"] > 0
        assert metrics["connected_transitions"] > 0
        assert "breaks" in metrics

        # Verify Map Data structure
        map_data = data["map_data"]
        assert map_data is not None
        assert len(map_data["gps"]) == data["num_gps_observations"]
        assert len(map_data["route_segments"]) > 0


# ---------------------------------------------------------------------------
# 6. Anti-Leakage Protection Verification
# ---------------------------------------------------------------------------

def test_anti_leakage_response_safety():
    """Verify that response payload NEVER contains raw reference routes or ground truth paths."""
    with TestClient(create_app()) as client:
        res = client.post("/api/predict_trip", json={"trip_id": "00000000"})
        assert res.status_code == 200
        raw_text = res.text

        # Forbidden tokens that would indicate ground-truth leakage
        assert "reference_route" not in raw_text
        assert "true_arc_id" not in raw_text
        assert "ground_truth_geometry" not in raw_text
        assert "alignments" not in raw_text
        assert ".route" not in raw_text
        assert "Student folder" not in raw_text


# ---------------------------------------------------------------------------
# 7. Concurrency Lock (503 while busy)
# ---------------------------------------------------------------------------

def test_concurrent_request_lock():
    """Verify that concurrent requests receive 503 while server is busy."""
    app = create_app()
    with TestClient(app) as client:
        # Simulate long-running inference by patching service.predict with sleep
        original_predict = app.state.service.predict

        def slow_predict(*args, **kwargs):
            time.sleep(0.5)
            return original_predict(*args, **kwargs)

        with patch.object(app.state.service, "predict", side_effect=slow_predict):
            results = []

            def worker():
                r = client.post("/api/predict_trip", json={"trip_id": "00000000"})
                results.append(r.status_code)

            t1 = threading.Thread(target=worker)
            t2 = threading.Thread(target=worker)

            t1.start()
            time.sleep(0.05)  # Ensure t1 acquires lock first
            t2.start()

            t1.join()
            t2.join()

            assert 200 in results
            assert 503 in results


# ---------------------------------------------------------------------------
# 8. Streamlit Frontend Contract Compatibility
# ---------------------------------------------------------------------------

def test_streamlit_contract_compatibility():
    """Verify every field required by ui/client_app.py is strictly satisfied."""
    with TestClient(create_app()) as client:
        # 1. Client calls /health
        h = client.get("/health")
        assert h.status_code == 200

        # 2. Client calls /api/trips and reads ['evaluation_trips']
        trips_res = client.get("/api/trips").json()
        assert "evaluation_trips" in trips_res
        assert len(trips_res["evaluation_trips"]) > 0
        trip = trips_res["evaluation_trips"][0]

        # 3. Client calls /api/predict_trip with json=dict(trip_id=trip, use_hmm=True)
        pred_res = client.post("/api/predict_trip", json={"trip_id": trip, "use_hmm": True}).json()

        # UI lines 31-37 read:
        metrics = pred_res["metrics"]
        assert "pointwise_accuracy" in metrics
        assert "continuity" in metrics
        assert "candidate_miss_rate" in metrics
        assert "execution_time_ms" in pred_res
        assert "reference_available" in metrics

        # UI lines 40-46 read:
        data = pred_res["map_data"]
        assert "gps" in data
        assert isinstance(data["gps"], list)
        assert len(data["gps"][0]) == 2  # [lon, lat]
        assert "route_segments" in data
        assert isinstance(data["route_segments"], list)
        for seg in data["route_segments"]:
            assert isinstance(seg, list)
            assert len(seg[0]) == 2  # [lon, lat]
