# PathSnap Phase 5: Distributed Deployment Guide

## 1. System Architecture & Topology

PathSnap deploys in a two-tier distributed setup across two physical laptops:

```mermaid
flowchart LR
    subgraph Laptop1 ["LAPTOP 1: Model Server & Evaluation Authority"]
        Uvicorn["Uvicorn / FastAPI (:8000)"]
        HGBModel["Deployment Model (HistGradientBoosting)"]
        ViterbiEngine["HMM / Viterbi Decoder Engine"]
        PrivateRoutes["Protected Server-Side .route Store"]
        ServerEval["Server-Side Accuracy & Continuity Scorer"]
        
        Uvicorn --> HGBModel
        Uvicorn --> ViterbiEngine
        ViterbiEngine --> ServerEval
        PrivateRoutes --> ServerEval
    end

    subgraph Laptop2 ["LAPTOP 2: Evaluator-Facing Client"]
        Streamlit["Streamlit Dashboard (:8501)"]
        FoliumMaps["Interactive Folium Map View"]
        MetricCards["Evaluation Metric Scorecards"]
        
        Streamlit --> FoliumMaps
        Streamlit --> MetricCards
    end

    Streamlit <-->|"HTTP REST API (LAN TCP 8000)"| Uvicorn
```

- **Laptop 1 (Model Server & Evaluation Authority)**:
  - Binds to `0.0.0.0:8000` on the local private network.
  - Loads serialized deployment model and Phase 4 HMM/Viterbi decoder configuration once at startup.
  - Houses the confidential evaluator ground truth (`.route`) files server-side.
  - Computes all evaluation metrics internally and displays live structured console logs.
- **Laptop 2 (Evaluator Client Interface)**:
  - Runs Streamlit UI (`ui/client_app.py`).
  - Calls `GET /health`, `GET /api/trips`, and `POST /api/predict_trip`.
  - Displays interactive Leaflet/Folium maps (raw GPS points and predicted route polylines) and metric scorecards.
  - **Zero ground truth access**: Never receives `.route` files, ground-truth geometry, or training labels.

---

## 2. Laptop 1 Configuration & Execution

### Prerequisites
- Python 3.9+ with `requirements.txt` installed.
- Windows 11 / Linux with private network connectivity.

### Environment Modes (`PATHSNAP_MODE`)

| Mode | Environment Variables | Behavior & Data Source |
| :--- | :--- | :--- |
| **Development** *(Default)* | `PATHSNAP_MODE=development`<br>`PATHSNAP_TRIP_ROOT="Student folder/student_test"` | Discovers 10 development trips (`00000000`, `00000007`, etc.). Calculates accuracy if local `.route` exists. |
| **Evaluator** | `PATHSNAP_MODE=evaluator`<br>`PATHSNAP_TRIP_ROOT="C:\PathSnap\private_evaluator_data"` | Strictly requires `PATHSNAP_TRIP_ROOT` containing the 10 reserved evaluator trips (`00000015`...`00000094`). Requires `.route` server-side. Fails fast with clear configuration error if directory or `.route` is missing. **Never falls back to development data.** |

### Step-by-Step Laptop 1 Deployment

1. **Install Dependencies**:
   ```bash
   python -m pip install -r requirements.txt
   ```

2. **Discover Laptop 1 Private IP Address**:
   - In PowerShell:
     ```powershell
     (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -notlike "169.254*" }).IPAddress
     ```
   - Example output: `192.168.1.105`

3. **Configure Windows Firewall for Port 8000** (Private Network):
   - Run in Administrator PowerShell:
     ```powershell
     New-NetFirewallRule -DisplayName "PathSnap Laptop 1 Model Server" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow -Profile Private
     ```

4. **Launch Model Server**:
   - **For Development Mode**:
     ```powershell
     $env:PATHSNAP_MODE="development"
     python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
     ```
   - **For Evaluator Demonstration Mode**:
     ```powershell
     $env:PATHSNAP_MODE="evaluator"
     $env:PATHSNAP_TRIP_ROOT="C:\path\to\reserved_evaluator_trips"
     python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
     ```

5. **Verify Local Readiness**:
   - Health check: `curl http://127.0.0.1:8000/health`
   - Swagger Documentation: open `http://127.0.0.1:8000/docs` in a browser.

---

## 3. Laptop 2 Configuration & Execution

1. **Install Client Dependencies**:
   ```bash
   pip install streamlit requests folium
   ```

2. **Configure Server URL**:
   - Point `$env:SERVER_URL` to Laptop 1's private LAN IP:
     ```powershell
     $env:SERVER_URL="http://192.168.1.105:8000"
     ```

3. **Launch Streamlit Client**:
   ```powershell
   python -m streamlit run ui/client_app.py
   ```
   Open `http://localhost:8501` in a browser.

---

## 4. API Endpoints & Contract Specification

### `GET /health`
Returns server readiness, operating mode, and discovered trip counts:
```json
{
  "status": "healthy",
  "service": "PathSnap",
  "model_loaded": true,
  "mode": "evaluator",
  "trip_count": 10,
  "timestamp": "2026-11-18T16:00:00.000000Z"
}
```

### `GET /api/trips`
Returns the allowlisted evaluation trips:
```json
{
  "evaluation_trips": [
    "00000015", "00000017", "00000023", "00000030", "00000049",
    "00000059", "00000067", "00000073", "00000087", "00000094"
  ],
  "mode": "evaluator",
  "trip_count": 10
}
```

### `POST /api/predict_trip`
- **Request Payload**:
  ```json
  {
    "trip_id": "00000015",
    "use_hmm": true,
    "generate_map": true
  }
  ```
- **Response Payload**:
  ```json
  {
    "trip_id": "00000015",
    "status": "success",
    "num_gps_observations": 1840,
    "execution_time_ms": 342.5,
    "metrics": {
      "reference_available": true,
      "pointwise_accuracy": 0.9412,
      "continuity": 0.9885,
      "route_continuity": 0.9885,
      "candidate_miss_rate": 0.0000,
      "classifier_only_accuracy": 0.8920,
      "hmm_accuracy_boost": 0.0492,
      "hmm_accuracy_improvement": 0.0492,
      "normalized_route_edit_distance": 0.1250,
      "observations": 1840,
      "aligned_observations": 1836,
      "eligible_transitions": 1839,
      "connected_transitions": 1818,
      "breaks": 3,
      "backend_processing_time_ms": 342.5
    },
    "predicted_route": [1420, 1420, 1420, 3105, 3105, 8912],
    "map_data": {
      "gps": [[77.5946, 12.9716], [77.5947, 12.9718]],
      "route_segments": [
        [[77.5945, 12.9715], [77.5948, 12.9720]]
      ]
    }
  }
  ```

---

## 5. Security & Anti-Leakage Verification

- **Strict Server-Side Scoring**: Evaluator reference routes (`.route`) are never transmitted over the network or exposed via API payloads.
- **Single-Request Protection**: CPU-bound inference is guarded by a non-blocking mutex lock (`threading.Lock()`); concurrent requests receive HTTP 503 (`"Server is processing another trip; retry shortly"`).
- **Input Validation**: `trip_id` strictly validated via Pydantic regex `^[0-9]{8}$`. Non-existent trips return HTTP 404. Malformed requests return HTTP 422.
- **Structured Logging**: Every request outputs client IP, trip ID, execution time, accuracy, continuity, and decoding method to `stdout` for evaluator review on Laptop 1.

---

## 6. Verification Status

| Milestone / Component | Status | Evidence & Test Coverage |
| :--- | :---: | :--- |
| **FastAPI Backend (`api/server.py`)** | **VERIFIED** | Model loaded at startup, `/health`, `/api/trips`, `/api/predict_trip` |
| **Inference Service (`api/service.py`)** | **VERIFIED** | Dual-mode operation (`development` vs `evaluator`), schema enforcement |
| **Server-Side Scoring (`evaluate.py`)** | **VERIFIED** | Pointwise accuracy, continuity, candidate miss rate, edit distance |
| **Anti-Leakage Protection** | **VERIFIED** | Zero reference routes or ground truth paths in response payload |
| **Streamlit Client Compatibility** | **VERIFIED** | Tested via `AppTest`; displays metrics, map, and error states |
| **Automated Test Suite** | **VERIFIED** | **29/29 tests passed** across API, hardening, integration, and UI |
| **Physical LAN Verification** | *Pending* | Requires connecting two physical laptops to the same Wi-Fi network |
| **Reserved Evaluator Data Delivery** | *Pending* | Requires hackathon judges to provide the confidential evaluation folder |
