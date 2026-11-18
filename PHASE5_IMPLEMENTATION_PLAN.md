# PathSnap Phase 5: Distributed API and UI Implementation Plan

## Executive assessment

Phase 5 can be implemented now. Phases 1-4 provide the required inference foundation: processed trip data, candidate generation, feature extraction, a serialized production model, and the HMM/Viterbi decoder with evaluation results. The repository also contains the Phase 5 dependency declarations and a detailed architecture/data-contract design.

Phase 5 is not implemented yet. The `api/` and `ui/` directories contain only `.gitkeep` files, so there is currently no FastAPI application, no Streamlit client, no shared response schema, and no integration test. The remaining work is application integration and deployment hardening rather than a new ML research phase.

## Evidence reviewed

- `README.md`: marks Phase 5 as pending and defines the intended two-laptop topology.
- `pathsnap_analysis/08_system_architecture.md`: specifies responsibilities, endpoints, payloads, anti-leakage rules, and launch commands.
- `pathsnap/decoding/pipeline.py` and `pathsnap/decoding/evaluate.py`: provide reusable prediction and evaluation functions.
- `pathsnap/artifacts/final_model/`: contains `model.joblib`, feature schema, model configuration, and tuned hyperparameters.
- `results/hmm-viterbi/`: contains Phase 4 comparison and per-trip outputs.
- The supplied PDFs in `docs/`, including `Main Guidelines.pdf`, `Deliverables.pdf`, `Dataset Exploration.pdf`, `Models and Training Guidelines.pdf`, and `Assessment and Plagiarism Policy.pdf`, were treated as requirements and integrity constraints. The attached problem statement video was retained in `docs/` as project reference material.

## What is already ready

1. `predict_trip(trip_dir, model)` loads a `.track` file, road nodes/arcs, builds candidates, extracts features, scores them, and calls the decoder.
2. `decode_prepared(..., sequential=True)` supports HMM/Viterbi decoding; `sequential=False` supports the classifier-only comparison.
3. The production model artifact and feature schema are present.
4. Student-test data and reference routes are present server-side in the repository's data layout.
5. `requirements.txt` already includes FastAPI, Uvicorn, requests, Folium, and the scientific dependencies.

## Gaps that must be closed

- Implement and configure `api/server.py`; load the model once at startup, validate paths and trip IDs, and expose `/health`, `/api/trips`, and `/api/predict_trip`.
- Define typed request/response models and stable JSON serialization for NumPy values, route segments, metrics, and map geometry.
- Keep evaluator `.route` files and all accuracy calculations on Laptop 1. Laptop 2 must never receive raw reference routes.
- Add map geometry generation from predicted arcs and raw GPS observations. Return compact GeoJSON or Folium HTML; avoid sending an unbounded HTML payload by default.
- Implement `ui/client_app.py` with a server URL setting, trip selector, run button, metric cards, error state, and interactive map.
- Add CORS/network configuration, request timeouts, structured logs, and a clear startup health check.
- Add integration tests using a small fixture trip and a mocked model, plus one live local API/UI smoke test.
- Verify the documented evaluator trip IDs against the actual `data/student_test` directory before hard-coding them. The current design document lists IDs that must be checked against the checkout.
- Document Windows firewall/LAN binding (`0.0.0.0`), Laptop 1's private IP, and the Laptop 2 `SERVER_URL` configuration.

## Proposed implementation sequence

### 1. Freeze the server contract

Create Pydantic models:

- `HealthResponse`: `status`, `service`, `model_loaded`, `timestamp`.
- `TripsResponse`: `evaluation_trips`.
- `PredictRequest`: `trip_id`, `use_hmm` (default `true`), `generate_map` (default `true`).
- `PredictResponse`: `trip_id`, `status`, observation count, execution time, metrics, predicted route, and map data.

Reject unknown or unsafe trip IDs and return HTTP 404/422 with useful messages. Do not accept arbitrary filesystem paths from the client.

### 2. Build the inference service

At startup, resolve the repository root, load `HistGradientBoostingPipeline` from `pathsnap/artifacts/final_model/model.joblib`, and validate the feature schema. Implement a service function that calls `prepare_trip`, then `decode_prepared` with the configured decoder settings. Use the existing evaluation helpers only on Laptop 1, where the reference `.route` files remain available.

Keep the endpoint synchronous in the first version because inference is CPU-bound and the existing pipeline is synchronous. Run it behind Uvicorn with one worker for predictable model memory use; add a bounded thread pool only if concurrent requests are required.

### 3. Add server-side metrics and map output

Return pointwise accuracy, continuity, candidate miss rate, classifier-only accuracy, HMM accuracy change, processing time, and counts. Build map layers from raw GPS coordinates, road geometry, and predicted route geometry. Prefer a JSON/GeoJSON representation so the UI controls rendering and the server response stays testable.

### 4. Build the Streamlit client

The client should call `/health`, populate the selector from `/api/trips`, submit `/api/predict_trip`, show progress and timeout errors, render metric cards, and draw raw observations plus predicted route. It should not load model files or reference routes locally. Put the API base URL in an environment variable or a Streamlit configuration setting.

### 5. Test in one laptop before LAN deployment

Run the API locally, exercise OpenAPI and all three endpoints, then start Streamlit against `127.0.0.1`. Confirm that a selected trip produces a route, metrics, and map without reading ground truth in the client. Add a test that asserts the response contains no `.route` contents.

### 6. Validate the two-laptop deployment

On Laptop 1, bind Uvicorn to `0.0.0.0:8000` and allow the private-network firewall rule. On Laptop 2, set `SERVER_URL` to Laptop 1's private IP and run Streamlit on its local port. Verify `/health`, one complete prediction, server request logs, timeout behavior, and recovery after an invalid trip ID.

## Acceptance criteria

- API starts from a clean environment using the documented commands.
- `/health` reports a loaded model; `/api/trips` lists only the reserved evaluator trips.
- `/api/predict_trip` produces a valid prediction and metrics for every reserved trip.
- The UI can select a trip, invoke the API, display metrics, and render the map.
- The client receives no raw reference route or training artifact.
- API and UI tests pass, including malformed requests, missing trips, model-load failure, and network timeout handling.
- A complete LAN smoke test is recorded with the exact host/IP and commands used.

## Recommended file layout

```text
api/
  __init__.py
  schemas.py
  service.py
  server.py
ui/
  client_app.py
tests/
  test_api.py
  test_client_contract.py
```

## Final decision

Proceed with Phase 5 implementation. Nothing in the reviewed materials blocks the work, but the current checkout is not Phase 5-complete until the API, UI, contract tests, map serialization, and LAN deployment verification above are added.
