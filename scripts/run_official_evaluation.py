"""Official Evaluator Dataset Execution Script.
Runs all 10 held-out evaluator trips against the running Laptop 1 FastAPI Model Server,
verifies data isolation and anti-leakage, records exact metrics, and generates summary report.
"""
import json
import time
from pathlib import Path
import requests

EVALUATOR_TRIPS = [
    "00000015", "00000017", "00000023", "00000030", "00000049",
    "00000059", "00000067", "00000073", "00000087", "00000094"
]

SERVER_URL = "http://127.0.0.1:8000"

def main():
    # 1. Health Verification
    h_res = requests.get(f"{SERVER_URL}/health", timeout=10)
    assert h_res.status_code == 200, f"Health check failed: {h_res.text}"
    h_data = h_res.json()
    print("Health Status:", h_data)
    assert h_data.get("ready") is True
    assert h_data.get("mode") == "evaluator"
    assert h_data.get("trip_count") == 10

    # 2. Trips Verification
    t_res = requests.get(f"{SERVER_URL}/api/trips", timeout=10)
    assert t_res.status_code == 200, f"Trips list failed: {t_res.text}"
    t_data = t_res.json()
    trips_list = t_data.get("evaluation_trips", [])
    print("Available Trips:", trips_list)
    assert trips_list == EVALUATOR_TRIPS

    # 3. Sequential Inference
    results = []
    print("\nStarting sequential evaluation across all 10 trips...")
    for tid in EVALUATOR_TRIPS:
        t0 = time.time()
        payload = {"trip_id": tid, "use_hmm": True, "generate_map": True}
        res = requests.post(f"{SERVER_URL}/api/predict_trip", json=payload, timeout=300)
        dur = time.time() - t0

        assert res.status_code == 200, f"Trip {tid} failed with {res.status_code}: {res.text}"
        raw_text = res.text

        # Anti-leakage checks
        for forbidden in ["reference_route", "true_arc_id", "ground_truth_geometry", "alignments"]:
            assert forbidden not in raw_text, f"Leakage detected: {forbidden} in response"

        data = res.json()
        m = data["metrics"]

        rec = {
            "trip_id": tid,
            "status": data["status"],
            "observations": m.get("observations"),
            "aligned_observations": m.get("aligned_observations"),
            "pointwise_accuracy": m.get("pointwise_accuracy"),
            "continuity": m.get("continuity"),
            "candidate_miss_rate": m.get("candidate_miss_rate"),
            "classifier_only_accuracy": m.get("classifier_only_accuracy"),
            "hmm_accuracy_boost": m.get("hmm_accuracy_boost"),
            "normalized_route_edit_distance": m.get("normalized_route_edit_distance"),
            "eligible_transitions": m.get("eligible_transitions"),
            "connected_transitions": m.get("connected_transitions"),
            "breaks": m.get("breaks"),
            "execution_time_seconds": data.get("execution_time_seconds", dur),
            "execution_time_ms": data.get("execution_time_ms", dur * 1000.0),
            "num_gps_observations": data.get("num_gps_observations"),
            "has_map_data": bool(data.get("map_data")),
            "has_predicted_route": bool(data.get("predicted_route")),
        }
        results.append(rec)
        print(
            f"Trip {tid} | Observations: {rec['observations']:4d} | "
            f"Acc: {rec['pointwise_accuracy']:.2%} | "
            f"Cont: {rec['continuity']:.2%} | "
            f"Miss: {rec['candidate_miss_rate']:.2%} | "
            f"BaseAcc: {rec['classifier_only_accuracy']:.2%} | "
            f"Boost: +{rec['hmm_accuracy_boost']:.2%} | "
            f"EditDist: {rec['normalized_route_edit_distance']:.4f} | "
            f"Time: {rec['execution_time_seconds']:.2f}s"
        )

    # Save JSON raw results
    out_json = Path("pathsnap/reports/evaluator_results_raw.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nRaw results saved to {out_json}")

if __name__ == "__main__":
    main()
