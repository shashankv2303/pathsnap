# PathSnap Official Evaluator Dataset — Final Evaluation Report

**Evaluation Date:** November 18, 2026
**Execution Environment:** Windows 11, Python 3.9.13, Uvicorn 0.34.0, FastAPI 0.115.11
**Deployment Model:** `pathsnap/artifacts/deployment_model/model.joblib` (Tuned HistGradientBoostingClassifier)
**Feature Schema:** 18 continuous and spatial features (`pathsnap/artifacts/deployment_model/model_config.json`)
**Decoder Engine:** Phase 4 HMM / Viterbi Decoder (`results/hmm-viterbi/decoder_config.json`)
**Evaluator Dataset Location:** `C:\PathSnapData\evaluator_test` (Isolated outside Git repository)
**Backend Mode:** `PATHSNAP_MODE="evaluator"` (Strict server-side scoring & anti-leakage isolation)

---

## 1. Executive Summary

The official held-out evaluation dataset consisting of **ten reserved trips** (`00000015` through `00000094`) was integrated into the Laptop 1 FastAPI Model Server and evaluated end-to-end.

* **Trip Evaluation Success Rate:** **10 / 10 (100.0%)**
* **Failed Trips:** **0**
* **Total GPS Observations Processed:** **19,674**
* **Total Ground-Truth Aligned Observations:** **17,708**
* **Arithmetic Mean Pointwise Accuracy (across trips):** **91.10%** *(vs 83.01% classifier-only; +8.09% HMM boost)*
* **Observation-Weighted Pooled Accuracy:** **86.52%**
* **Mean Route Continuity:** **99.86%**
* **Mean Candidate Miss Rate:** **0.04%** *(99.96% candidate recall)*
* **Mean Normalized Route Edit Distance:** **0.2004**
* **Mean Inference Latency:** **2.32 seconds / trip** *(Total runtime: 23.22 seconds)*
* **Ground-Truth Data Isolation:** **100% Verified** *(Zero reference routes or true geometries leaked)*

---

## 2. Evaluation Methodology

1. **Two-Stage Map-Matching Architecture**:
   - **Stage 1 (Candidate Generation & Emission Scoring)**: For each GPS observation, candidates within a 50-meter bounding radius are extracted and represented as 18-dimensional spatial and kinematic feature vectors. The deployment model (`HistGradientBoostingClassifier`) scores candidate probabilities.
   - **Stage 2 (Global Sequence Optimization via Viterbi)**: A Hidden Markov Model links successive candidate states. Transition penalties penalize differences between road-network graph distance and GPS displacement. The Viterbi algorithm identifies the globally optimal trajectory.
2. **Strict Server-Side Scoring**:
   - Evaluator ground-truth `.route` files reside exclusively on Laptop 1.
   - Predictions are evaluated against the reference routes on the server using `pathsnap.ground_truth.route_aligner.RouteAligner` and `pathsnap.decoding.evaluate.evaluate`.
   - The client API response strictly strips all reference labels, alignments, and true arc geometries.

---

## 3. Per-Trip Official Evaluation Results

The table below reports the verified server-side evaluation metrics for each of the ten official evaluator trips processed via `POST /api/predict_trip` against the live Laptop 1 Model Server (reconciled directly from `evaluator_results_raw.json`):

| Trip ID | GPS Obs | Aligned Obs | Pointwise Accuracy | Route Continuity | Candidate Miss Rate | Classifier-Only Accuracy | HMM Accuracy Boost | Normalized Edit Distance | Execution Latency | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000015` | 6,854 | 6,833 | **74.33%** | 98.89% | 0.03% | 51.09% | **+23.24%** | 0.8038 | 4.84s | SUCCESS |
| `00000017` | 1,346 | 1,346 | **94.58%** | 100.00% | 0.00% | 88.41% | **+6.17%** | 0.1196 | 5.05s | SUCCESS |
| `00000023` | 458 | 458 | **94.98%** | 100.00% | 0.00% | 91.05% | **+3.93%** | 0.2716 | 0.62s | SUCCESS |
| `00000030` | 654 | 654 | **86.70%** | 100.00% | 0.00% | 78.13% | **+8.56%** | 0.2000 | 1.35s | SUCCESS |
| `00000049` | 2,642 | 2,642 | **99.74%** | 100.00% | 0.00% | 95.57% | **+4.16%** | 0.0073 | 1.72s | SUCCESS |
| `00000059` | 1,031 | 1,031 | **97.48%** | 100.00% | 0.00% | 91.27% | **+6.21%** | 0.0990 | 1.73s | SUCCESS |
| `00000067` | 916 | 916 | **85.26%** | 99.78% | 0.00% | 72.49% | **+12.77%** | 0.1474 | 0.54s | SUCCESS |
| `00000073` | 1,144 | 1,144 | **98.25%** | 100.00% | 0.00% | 88.64% | **+9.62%** | 0.0846 | 1.45s | SUCCESS |
| `00000087` | 1,399 | 1,399 | **99.43%** | 100.00% | 0.00% | 93.64% | **+5.79%** | 0.0110 | 2.87s | SUCCESS |
| `00000094` | 3,230 | 1,285 | **80.23%** | 99.91% | 0.39% | 79.77% | **+0.47%** | 0.2598 | 3.07s | SUCCESS |

---

## 4. Aggregate Performance Statistics

All aggregate statistics are calculated over the complete set of ten successfully evaluated trips ($N = 10$):

| Metric | Aggregate Mean | Minimum | Maximum | Standard Deviation |
| :--- | :---: | :---: | :---: | :---: |
| **Pointwise Map-Matching Accuracy (Trip Mean)** | **91.10%** | 74.33% (`00000015`) | 99.74% (`00000049`) | $\pm 8.46\%$ |
| **Observation-Weighted Pooled Accuracy** | **86.52%** | — | — | — |
| **Route Continuity** | **99.86%** | 98.89% (`00000015`) | 100.00% (7 trips) | $\pm 0.33\%$ |
| **Candidate Miss Rate** | **0.04%** | 0.00% (8 trips) | 0.39% (`00000094`) | $\pm 0.12\%$ |
| **Classifier-Only Accuracy** | **83.01%** | 51.09% (`00000015`) | 95.57% (`00000049`) | $\pm 12.76\%$ |
| **HMM Accuracy Improvement (Boost)** | **+8.09%** | +0.47% (`00000094`) | +23.24% (`00000015`) | $\pm 5.98\%$ |
| **Normalized Route Edit Distance** | **0.2004** | 0.0073 (`00000049`) | 0.8038 (`00000015`) | $\pm 0.2188$ |
| **Inference Processing Latency** | **2.32s** | 0.54s (`00000067`) | 5.05s (`00000017`) | $\pm 1.55s$ |

---

## 5. Key Engineering Insights

1. **Massive Sequence Optimization via HMM/Viterbi**:
   - On challenging noisy trajectories such as trip `00000015` (6,854 GPS points), the independent point classifier scored only 51.09%. Global Viterbi sequence optimization with transition distance penalties boosted accuracy to **74.33% (+23.24 percentage points)**.
   - For trip `00000067`, Viterbi improved accuracy from 72.49% to **85.26% (+12.77 percentage points)**.
   - Overall, Viterbi provided an average accuracy lift of **+8.09 percentage points** across the entire evaluator partition.
2. **Exceptional Spatial Continuity**:
   - 7 out of 10 trips achieved **100.00% route continuity** (zero disconnected road transitions). The remaining 3 trips achieved $> 98.89\%$ continuity, yielding a cohort mean of **99.86%**.
3. **High Candidate Recall**:
   - The spatial candidate generator missed true road segments at only **0.04%** of aligned observations, demonstrating that the candidate generation bounding radius and metric projections capture the true trajectory geometry across unseen road networks.
4. **Computational Efficiency**:
   - The entire 10-trip evaluation completed in **23.22 seconds** of CPU processing time ($\approx 2.32$ seconds per trip), comfortably meeting real-time evaluation requirements.

---

## 6. Security & Anti-Leakage Audit

* **File Storage:** Evaluator reference files (`.route`) reside exclusively in `C:\PathSnapData\evaluator_test` and are completely excluded from Git via `.gitignore`.
* **API Payloads:** Every API response was verified:
  - No `reference_route`, `true_arc_id`, or `ground_truth_geometry` exists in any `/api/predict_trip` response payload.
  - The client UI receives only raw GPS observations (`map_data.gps`) and model-predicted road segment polylines (`map_data.route_segments`).
* **Console Logs:** Laptop 1 logs output only aggregate scalars (`accuracy`, `continuity`, `latency`, `status`); no road segment sequences are printed to stdout or written to network sockets.
