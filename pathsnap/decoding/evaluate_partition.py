"""Score a frozen classifier and decoder on one trip directory.

This only runs inference. The model and decoder configuration are loaded from
disk unchanged, every trip is predicted before its reference route is read, and
results must not be used to change the model, features or decoder settings.

Example:
    python -m pathsnap.decoding.evaluate_partition \
        --trip-root C:\\PathSnapPrivate\\evaluator \
        --model results/bayesian-full/tuned_model.joblib \
        --out results/evaluator-test/tuned_model
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from pathsnap.decoding.evaluate import evaluate, load_reference
from pathsnap.decoding.pipeline import decode_prepared, prepare_trip
from pathsnap.decoding.viterbi import DecoderConfig
from pathsnap.models.hist_gradient_boosting import HistGradientBoostingPipeline

METHODS = (("classifier_only", False), ("classifier_plus_hmm", True))


def trip_dirs(root):
    trips = sorted(p for p in root.iterdir() if p.is_dir() and (p / f"{p.name}.track").is_file())
    if not trips:
        raise ValueError(f"No trips found under {root}")
    missing = [p.name for p in trips if not (p / f"{p.name}.route").is_file()]
    if missing:
        raise ValueError(f"Reference routes are missing for {missing}")
    return trips


def summarise(frame):
    rows = []
    for method, group in frame.groupby("method", sort=False):
        aligned = group.aligned_observations
        rows.append(dict(
            method=method, trips=len(group), observations=int(group.observations.sum()),
            aligned_observations=int(aligned.sum()),
            mean_per_trip_accuracy=group.pointwise_accuracy.mean(),
            pooled_accuracy=(group.pointwise_accuracy * aligned).sum() / aligned.sum(),
            mean_continuity=group.continuity.mean(),
            mean_normalized_route_edit_distance=group.normalized_route_edit_distance.mean(),
            total_breaks=int(group.breaks.sum()),
            mean_seconds_per_trip=(group.prepare_seconds + group.decode_seconds).mean(),
        ))
    return pd.DataFrame(rows)


def run(trip_root, model_path, decoder_config, output):
    output.mkdir(parents=True, exist_ok=False)
    model = HistGradientBoostingPipeline.load(str(model_path))
    config = DecoderConfig(**json.loads(decoder_config.read_text(encoding="utf-8")))
    rows = []
    for trip in trip_dirs(trip_root):
        start = perf_counter()
        prepared = prepare_trip(trip, model)
        prepare_seconds = perf_counter() - start
        predictions = {}
        for name, sequential in METHODS:
            start = perf_counter()
            predictions[name] = (decode_prepared(prepared, config, sequential), perf_counter() - start)
        # Reference data is read only after both predictions are complete.
        reference = load_reference(trip, prepared)
        for name, (predicted, seconds) in predictions.items():
            rows.append(dict(trip_id=trip.name, method=name, prepare_seconds=prepare_seconds,
                             decode_seconds=seconds, **evaluate(predicted, reference)))
        (output / f"{trip.name}_predictions.json").write_text(
            json.dumps({name: asdict(route) for name, (route, _) in predictions.items()}),
            encoding="utf-8",
        )
        print(f"Scored trip {trip.name}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_trip.csv", index=False)
    summary = summarise(frame)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "run_config.json").write_text(json.dumps(dict(
        model=Path(model_path).as_posix(), decoder_config=Path(decoder_config).as_posix(),
        decoder=asdict(config), trips=sorted(frame.trip_id.unique()),
    ), indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trip-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--decoder-config", type=Path, default=Path("results/hmm-viterbi/decoder_config.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.trip_root, args.model, args.decoder_config, args.out)
