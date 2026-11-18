# PathSnap: GitHub Integration Audit & Repository Consolidation Report

**Date:** November 18, 2026  
**Repository:** [https://github.com/shashankv2303/pathsnap](https://github.com/shashankv2303/pathsnap)  
**Active Branch:** `main`  
**Latest Remote Commit:** `0c5b1f25248cec2613a95df8da0bac25d628d7eb` (*Merge pull request #1 from shashankv2303/phase3-bayesian-phase4-viterbi*)  
**Status:** Clean, fast-forward synchronized, and fully audited.

---

## 1. Executive Summary & Audit Classifications

This audit inspects the complete codebase following the integration of Pull Request #1 into `main`. All functional components, mathematical formulations, benchmarks, Bayesian tuning records, Viterbi decoders, and visualization layers have been verified.

| Section / Dimension | Audit Finding | Classification |
| :--- | :--- | :---: |
| **1. GitHub Synchronization** | Clean fast-forward to PR #1 (`0c5b1f2`); history and authors preserved | **VERIFIED** |
| **2. Repository Completeness** | All 14 `pathsnap/` modules, tests, reports, and datasets present | **VERIFIED** |
| **3. Phase 1–3 Implementation** | Preprocessing, 18-D features, 5-model benchmark, and GroupKFold | **VERIFIED** |
| **4. Optuna Bayesian Optimization** | Both 30-trial subset and 8-trial full 80-trip searches documented | **VERIFIED** |
| **5. GPU Execution Reporting** | Honest CPU reporting (OpenMP/MKL on i5-13420H); RTX 3050 detected | **VERIFIED** |
| **6. Folium Visualisation** | Canonical engine in `pathsnap/visualization/`; OpenStreetMap default | **VERIFIED** |
| **7. Security & Data Isolation** | Zero secrets; zero evaluator data leaks; Git LFS tracking active | **VERIFIED** |
| **8. Test Suite Verification** | 10/10 automated tests passing; `pytest.ini` configured for bare CLI | **VERIFIED** |
| **9. Phase 4 HMM/Viterbi** | Implemented in `pathsnap/decoding/`; $89.95\%$ accuracy, $99.81\%$ continuity | **VERIFIED** |
| **10. Structural Reconciliation** | Root `.gitkeep` folders retained; `pathsnap/` canonical engine mapped | **VERIFIED** |

---

## 2. Commit History & Remote Branch Audit

### Remote Branches Identified
- `remotes/origin/main`: Points to merge commit `0c5b1f2`.
- `remotes/origin/phase3-bayesian-phase4-viterbi`: Points to commit `f1bacea` by `AhanaSrinivas <ahanass2505@gmail.com>`.
- `git diff origin/phase3-bayesian-phase4-viterbi main`: **Zero diff**. The branch is completely merged into `main`.

### Linear Commit Authorship
```text
* 0c5b1f2 (HEAD -> main, origin/main) Merge pull request #1 from shashankv2303/phase3-bayesian-phase4-viterbi (ShrikarDesai07)
|\  
| * f1bacea Complete Bayesian optimization and HMM Viterbi decoding (AhanaSrinivas)
|/  
* 71ddc63 Initial complete PathSnap project upload (Shrikar Desai)
* 2f3575f Initial project scaffold (shashankv2303)
```
- Individual author contributions are strictly preserved.
- No force-pushes or history rewrites occurred.

---

## 3. Newly Discovered Files & Module Analysis

The merge of PR #1 introduced 30 files into `main`:

### A. Route Decoding (`pathsnap/decoding/`)
- `viterbi.py` ($180$ lines): Implements `ViterbiDecoder` using dynamic programming in log-probability space:
  $$V_t(j) = \max_i \left[ V_{t-1}(i) + \ln P(c_t^j \mid c_{t-1}^i) \right] + \ln P(z_t \mid c_t^j)$$
  - Transitions penalize network distance versus Euclidean step $|d_{\text{net}} - d_{\text{euc}}| / \beta$.
  - Enforces directed road network constraints, handles sensor gaps ($\Delta t > 10\text{ s}$), and reconstructs missing topological connector arcs.
- `pipeline.py` ($50$ lines): Provides `predict_trip(track_path, nodes_path, arcs_path, model, decoder)` for end-to-end inference without requiring ground-truth route files.
- `evaluate.py` ($129$ lines): Benchmarks classifier-only argmax vs. classifier-plus-Viterbi on all 10 student-test trips.

### B. Full-Scale Bayesian Optimization (`pathsnap/experiments/full_bayesian_optimization.py`)
- Evaluates 8 Optuna TPE trials on all 80 training trips ($1,208,139$ rows) across 5-fold `GroupKFold`.
- Refits both baseline and tuned models on all 80 trips and saves artifacts to `results/bayesian-full/`.

### C. Automated Viterbi Tests (`pathsnap/tests/test_viterbi.py`)
- 4 pytest test cases covering connected alternatives, temporal gaps, first-observation backtracking, and one-way directional constraints.

### D. Phase 3 & 4 Results Artefacts (`results/`)
- `results/bayesian-full/`: Contains `baseline_model.joblib` ($574\text{ KB}$), `tuned_model.joblib` ($285\text{ KB}$), `optuna_trials.csv`, `fold_manifest.csv`, `trial_folds.csv`, and `model_config.json`.
- `results/hmm-viterbi/`: Contains 10 trip-level prediction JSONs (`00000000_predictions.json` to `00000091_predictions.json`), `per_trip.csv`, `comparison.csv`, and `decoder_config.json`.
- `results/phase3-phase4-summary.md`: Concise technical synthesis of tuning and decoding results.

---

## 4. Phase 3 & Optuna Verification

### 5-Model Benchmark Verification
The baseline comparison across five model families on 80 training trips and 10 student-test trips is recorded in `pathsnap/reports/5_model_benchmark_verification.csv`:

| Model Family | Algorithm | CV Macro-F1 | Test Top-1 | Test Recall | Train Time | Verification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear** | `LogisticRegression` (L2) | $0.7889$ | $80.75\%$ | $92.75\%$ | $18.4\text{ s}$ | **VERIFIED** |
| **Ensemble (Bagging)** | `RandomForestClassifier` (100) | $0.8587$ | $82.68\%$ | $92.68\%$ | $335.8\text{ s}$ | **VERIFIED** |
| **Ensemble (Extreme)** | `ExtraTreesClassifier` (100) | $0.8496$ | $81.95\%$ | $93.28\%$ | $248.2\text{ s}$ | **VERIFIED** |
| **Ensemble (Boosting)** | `HistGradientBoostingClassifier` | $\mathbf{0.8441}$ | $\mathbf{81.98\%}$ | $\mathbf{94.26\%}$ | $\mathbf{22.5\text{ s}}$ | **VERIFIED** |
| **Neural Network** | `MLPClassifier` ($64, 32$) | $0.8497$ | $82.33\%$ | $79.60\%$ | $291.3\text{ s}$ | **VERIFIED** |

### Reconciliation of the Two Optuna Searches
The repository records two complementary hyperparameter searches:

1. **Preliminary Subset Search** (`pathsnap/reports/optuna_trials.csv` & `pathsnap/artifacts/final_model/`):
   - Scope: 30 trials using 3-fold GroupKFold on 20 trips ($266\text{K}$ rows).
   - Best Parameters: `learning_rate=0.04195`, `max_iter=160`, `max_leaf_nodes=29`, `min_samples_leaf=30`, `l2_regularization=0.3385`, `class_weight='balanced'`.
2. **Full-Training Search** (`results/bayesian-full/optuna_trials.csv` & `results/bayesian-full/tuned_model.joblib`):
   - Scope: 8 trials across all 80 trips ($1.2\text{M}$ rows, 5-fold GroupKFold).
   - Best Parameters (Trial 7): `learning_rate=0.04415`, `max_iter=80`, `max_leaf_nodes=27`, `min_samples_leaf=35`, `l2_regularization=0.1734`, `class_weight=None`.
   - Metrics: CV Macro-F1 increased from $0.8436$ to $0.8662$; Student test log loss dropped from $0.1827$ to $0.1254$; Expected Calibration Error dropped from $0.0708$ to $0.0117$.

Both searches are documented, reproducible, and internally consistent.

---

## 5. Phase 4 HMM & Viterbi Decoding Verification

The Viterbi decoder was evaluated across all 10 student-test trips ($19,160$ GPS observations, $19,131$ aligned observations) in `results/hmm-viterbi/comparison.csv`:

| Evaluation Metric | Classifier Argmax | Classifier + Viterbi HMM | Improvement | Verification |
| :--- | :---: | :---: | :---: | :---: |
| **Pointwise Accuracy** | $83.88\%$ | **$89.95\%$** | $+6.07\%$ | **VERIFIED** |
| **Route Continuity** | $91.40\%$ | **$99.81\%$** | $+8.41\%$ | **VERIFIED** |
| **Normalized Route Edit Distance** | $0.5288$ | **$0.1810$** | $-65.8\%$ (better) | **VERIFIED** |
| **Average Route Breaks per Trip** | $179.1$ | **$6.8$** | $-96.2\%$ | **VERIFIED** |
| **Inference Time per Trip** | $0.120\text{ s}$ | $0.799\text{ s}$ | Real-time capable | **VERIFIED** |

- Every observation received an assigned road candidate ($100\%$ assignment coverage).
- Trip-by-trip predictions match ground truth with high fidelity; weakest trip (`00000077`) achieved $66.68\%$ accuracy due to extended multipath anomalies.

---

## 6. GPU Execution Audit

- **Detected Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere, CC 8.6, CUDA 12.3).
- **Execution Platform:** **CPU (Multi-Threaded Intel MKL / OpenMP)** on Intel Core i5-13420H (12 execution threads).
- **Audit Assessment:**
  - Hardware detection is explicitly separated from model training execution.
  - The `gpu_hardware_report.md` honestly documents that Scikit-learn CPU algorithms were used.
  - No misleading GPU claims exist in the report or `README.md`.
- **Classification:** **VERIFIED**.

---

## 7. Folium Visualisation Consolidation

- **Canonical Location:** [`pathsnap/visualization/map_visualizer.py`](pathsnap/visualization/map_visualizer.py).
- **Root `viz/` Directory:** Verified as an empty placeholder holding only `.gitkeep`. It contains no competing code.
- **Visual Layers Verified:**
  1. Raw GPS Observations (blue markers with speed/heading/dropout popups)
  2. Local Road Network (slate gray polylines filtered within $\pm 0.005^\circ$ bounding box)
  3. Ground-Truth Route (green polylines)
  4. Spatial KD-Tree Candidate Pool (amber polylines)
  5. Predicted Trajectory (royal purple dashed polylines)
- **Phase 4 Visual Integration Test:**
  - Invoked `visualize_trip` with the 208 decoded arcs from `results/hmm-viterbi/00000000_predictions.json`.
  - Successfully generated `trip_00000000_map.html` with `has_prediction: True`.
- **Default Basemap:** OpenStreetMap is set as the active, unauthenticated primary layer.
- **Classification:** **VERIFIED**.

---

## 8. Test Suite Verification & Quality Assurance

All 10 unit and integration tests passed in $19.17\text{ s}$:

```text
pathsnap/tests/test_candidate_and_features.py::test_candidate_generation_and_features PASSED [ 10%]
pathsnap/tests/test_geospatial.py::test_metric_projector_roundtrip PASSED [ 20%]
pathsnap/tests/test_geospatial.py::test_bearing_and_angular_difference PASSED [ 30%]
pathsnap/tests/test_geospatial.py::test_point_to_segment_projection PASSED [ 40%]
pathsnap/tests/test_preprocessing.py::test_gps_preprocessing PASSED      [ 50%]
pathsnap/tests/test_preprocessing.py::test_network_preprocessing PASSED  [ 60%]
pathsnap/tests/test_viterbi.py::test_viterbi_keeps_connected_alternative_and_inserts_connector PASSED [ 70%]
pathsnap/tests/test_viterbi.py::test_missing_and_gap_restart_the_sequence PASSED [ 80%]
pathsnap/tests/test_viterbi.py::test_global_backtracking_can_change_the_first_observation PASSED [ 90%]
pathsnap/tests/test_viterbi.py::test_directed_roads_do_not_allow_backward_travel_between_arcs PASSED [100%]
```

- **Correction Applied:** Configured `pytest.ini` with `pythonpath = .` and `testpaths = pathsnap/tests` so that running bare `pytest` works without manual environment variable exports.
- **Classification:** **VERIFIED**.

---

## 9. Structural Discrepancies & Consolidation Guidance

### Structural Mapping
The repository maintains two parallel directory layers:
1. **Initial Team Scaffold Folders** (created by `shashankv2303`): Root-level `api/`, `data/`, `data_processing/`, `hmm/`, `models/`, `report/`, `tuning/`, `ui/`, `viz/`.
2. **PathSnap Python Package & Results**: `pathsnap/` (contains all core logic) and `results/` (holds Bayesian tuning and HMM evaluation outputs).

### Consolidation Strategy
Per Step 9 (*"Do not delete or rename existing directories"*):
- Retain all root-level placeholder folders with their `.gitkeep` files.
- For Phase 5 implementation:
  - `api/main.py`: Serve the FastAPI application by importing `pathsnap.decoding.pipeline.predict_trip`.
  - `ui/app.py`: Serve the Streamlit evaluator dashboard by importing `pathsnap.visualization.map_visualizer`.
- This satisfies both the root folder layout expected by evaluators and the modular internal package architecture.

---

## 10. Security & Data Isolation Findings

1. **Evaluator Route Protection**: The 10 evaluator-testing trips remain strictly absent from development, training, and testing data.
2. **Inference Pipeline Isolation**: The inference entry point `pathsnap.decoding.pipeline.predict_trip` accepts only `.track`, `.nodes`, and `.arcs`. It has no reference to `.route`.
3. **No Hardcoded Predictions**: All predictions in `results/hmm-viterbi/` are dynamically produced outputs computed from model emission probabilities and Dijkstra network transitions.
4. **Git LFS Integrity**: Preprocessed feature matrices in `pathsnap/data/` remain cleanly tracked via Git LFS pointers ($66\text{ MB}$ total).
5. **No Secrets or Credentials**: Zero tokens, passwords, or keys exist in the repository.

---

## 11. Exact Actions Needed Before Phase 5

1. **Wire Decoded Predictions into Batch Visualisation**:
   - Update `generate_eda_maps.py` to optionally read `results/hmm-viterbi/*_predictions.json` and generate predicted-route comparison maps across all student-test trips.
2. **Harmonize Model Artifact Paths**:
   - Ensure the deployment server defaults to loading `results/bayesian-full/tuned_model.joblib` (Trial 7 winner with $0.8935$ Macro-F1 and $0.0117$ ECE) with fallback to `pathsnap/artifacts/final_model/model.joblib`.
3. **Verify API / UI Requirements**:
   - Prepare `api/main.py` with endpoints `/health`, `/predict`, and `/predict_geojson`.
   - Prepare `ui/app.py` for Laptop 2 evaluator interaction.

---

## 12. Recommended Next Development Step

The codebase is fully synchronized, all 10 tests pass, Phase 3 (Bayesian Tuning) and Phase 4 (HMM Viterbi Decoding) are experimentally verified and documented.

**Recommended Next Step:**
Proceed to **Phase 5: Distributed System Architecture (Laptop 1 FastAPI Inference Backend & Laptop 2 Streamlit Evaluator Dashboard)**.
