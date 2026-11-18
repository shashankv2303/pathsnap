# PathSnap Phase 5: Distributed API and UI Implementation Plan & Status

## Executive Assessment

Phase 5 implementation has been completed, hardened, and verified locally. The full distributed map-matching architecture—comprising the Laptop 1 FastAPI inference backend, deployment model artifact loader, server-side evaluation engine, and Laptop 2 Streamlit dashboard—is operational and backed by automated integration tests.

---

## 1. Component Implementation Status

| Component | Status | Implementation Details |
| :--- | :---: | :--- |
| **FastAPI Backend (`api/server.py`)** | **Implemented & Tested** | Non-blocking mutex concurrency lock, structured logging, `/health`, `/api/trips`, `/api/predict_trip` |
| **Prediction Service (`api/service.py`)** | **Implemented & Tested** | Dual-mode operation (`development` vs `evaluator`), schema verification, HMM/Viterbi inference, server-side scoring |
| **Deployment Model (`pathsnap/artifacts/deployment_model/`)** | **Implemented & Tested** | Serialized `model.joblib` and `model_config.json` fitted on 80 trips ($1.2\text{M}$ rows) with verified 18-D feature schema |
| **Server-Side Evaluator (`pathsnap/decoding/evaluate.py`)** | **Implemented & Tested** | Pointwise accuracy, continuity, candidate miss rate, normalized route edit distance |
| **Streamlit Client UI (`ui/client_app.py`)** | **Implemented & Tested** | Server URL selector, trip dropdown, metric scorecards, Leaflet/Folium map rendering |
| **Automated Test Suite (`pathsnap/tests/`)** | **Implemented & Tested** | 29 passing tests across unit, integration, API contract, hardening, and UI |
| **Dual-Mode Isolation (`PATHSNAP_MODE`)** | **Implemented & Tested** | Prevents development data from substituting for evaluation data |
| **Anti-Leakage Protection** | **Implemented & Tested** | Verifies zero reference `.route` data or ground-truth geometry is exposed |
| **Physical 2-Laptop LAN Verification** | *Pending Demonstration* | Requires physical event network setup between Laptop 1 and Laptop 2 |
| **Reserved Evaluator Data Delivery** | *Pending Evaluator Input* | Awaiting hackathon organizers to provide the reserved evaluation directory |

---

## 2. Completed Implementation Tasks

### 1. Robust Server-Side Inference & Scoring
- Preprocessing, candidate generation, 18-D feature extraction, model scoring, and Viterbi sequence decoding execute on Laptop 1.
- Complete server-side evaluation against reference `.route` files computes:
  - Pointwise accuracy ($89.95\%$ mean on student test)
  - Route continuity ($99.81\%$ mean on student test)
  - Normalized route edit distance ($0.1810$ mean on student test)
  - Candidate miss rate ($0.09\%$ mean on student test)
  - Classifier-only baseline and HMM accuracy improvement ($+6.07\%$)

### 2. Dual-Mode Evaluation Isolation (`PATHSNAP_MODE`)
- **Development Mode**: Defaults to `Student folder/student_test` for local regression testing.
- **Evaluator Mode**: Requires `PATHSNAP_TRIP_ROOT` pointing to the private evaluation directory. Enforces presence of `.route` server-side and checks against the official allowlist (`00000015` ... `00000094`). Fails fast if files are missing; never silently falls back to development data.

### 3. Anti-Leakage Guards
- Evaluator reference routes remain confidential on Laptop 1.
- Response payloads contain strictly:
  - Observation count and execution time
  - Aggregated evaluation metrics
  - Predicted route arc IDs
  - Map geometry: raw GPS points and predicted route polylines.
- Forbidden fields (`.route` contents, true arc IDs, ground truth geometry) are strictly omitted.

### 4. Structured Observable Logging
- Every request logs to console:
  `[timestamp] client=<IP> trip=<ID> status=<STATUS> elapsed_ms=<TIME> acc=<ACC> cont=<CONT> hmm=<BOOL>`
- Evaluators on Laptop 1 can observe real-time requests from Laptop 2 directly in standard output.

---

## 3. Verification & Testing Summary

1. `test_api.py`: Validates contract schemas, path traversal rejection (422), and startup failure handling.
2. `test_backend_hardening.py` (15 tests):
   - `/health` readiness and metadata reporting.
   - 18-feature schema mismatch rejection.
   - Missing model and decoder configuration fail-fast.
   - Evaluator mode missing directory / missing `.route` exceptions.
   - Anti-leakage response assertions.
   - Single-request concurrency locking (HTTP 503).
   - Streamlit frontend contract compatibility.
3. `test_phase5_integration.py`: End-to-end inference and evaluation across all 10 student-test trips via FastAPI TestClient.
4. `test_phase5_ui.py`: Streamlit `AppTest` simulation verifying UI rendering, metric cards, and network timeout error handling.
5. All 29 unit, integration, and UI tests pass: `pytest pathsnap/tests -v` (100% passing).

---

## 4. Pending Deliverables & Next Action

- **Physical LAN Handshake**: Connect Laptop 1 and Laptop 2 to the same local Wi-Fi or Ethernet router, bind Uvicorn to `0.0.0.0:8000`, and set `$env:SERVER_URL` on Laptop 2.
- **Evaluator Trip Loading**: Upon receiving the evaluator folder from hackathon judges, set `PATHSNAP_MODE=evaluator` and `PATHSNAP_TRIP_ROOT="<path_to_evaluator_trips>"` on Laptop 1.
