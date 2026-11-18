"""Inference and optional reference scoring stay on the model server."""
import json
from pathlib import Path
from time import perf_counter

from pathsnap.decoding.pipeline import prepare_trip, decode_prepared
from pathsnap.decoding.evaluate import evaluate, load_reference
from pathsnap.decoding.viterbi import DecoderConfig
from pathsnap.feature_engineering.feature_extractor import FEATURE_NAMES
from pathsnap.models.hist_gradient_boosting import HistGradientBoostingPipeline

ROOT = Path(__file__).resolve().parents[1]


class PredictionService:
    def __init__(self, trip_root, model_path):
        self.trip_root = Path(trip_root).resolve()
        schema = json.loads(Path(model_path).with_name('model_config.json').read_text())
        if schema['input_features'] != list(FEATURE_NAMES):
            raise ValueError('Model feature schema does not match the inference pipeline')
        self.model = HistGradientBoostingPipeline.load(str(model_path))
        self.config = DecoderConfig(**json.loads(
            (ROOT / 'results/hmm-viterbi/decoder_config.json').read_text()))

    def trips(self):
        return sorted(p.name for p in self.trip_root.iterdir()
                      if p.is_dir() and len(p.name) == 8 and p.name.isascii()
                      and p.name.isdigit() and not p.is_symlink()
                      and all((p / (p.name + ext)).is_file()
                              for ext in ('.track', '.nodes', '.arcs')))

    def predict(self, trip_id, use_hmm=True, generate_map=True):
        if trip_id not in self.trips():
            raise FileNotFoundError('Unknown trip')
        start = perf_counter()
        trip = self.trip_root / trip_id
        prepared = prepare_trip(trip, self.model)
        trajectory, network, candidates, _ = prepared
        baseline = decode_prepared(prepared, self.config, sequential=False)
        result = decode_prepared(prepared, self.config) if use_hmm else baseline
        metrics = dict(continuity=(result.connected_transitions / result.eligible_transitions
                                  if result.eligible_transitions else None),
                       pointwise_accuracy=None, classifier_only_accuracy=None,
                       hmm_accuracy_boost=None, candidate_miss_rate=None,
                       reference_available=False)
        if (trip / (trip_id + '.route')).is_file():
            reference = load_reference(trip, prepared)
            metrics.update(evaluate(result, reference, measure_sequence=False))
            base = evaluate(baseline, reference, measure_sequence=False)['pointwise_accuracy']
            reliable = [a for a in reference[0] if a.is_reliable]
            misses = sum(not any(c.arc_id == a.true_arc_id for c in candidates[a.obs_idx])
                         for a in reliable)
            accuracy = metrics['pointwise_accuracy']
            metrics.update(reference_available=True, classifier_only_accuracy=base,
                           hmm_accuracy_boost=(accuracy - base if accuracy is not None
                                               and base is not None else None),
                           candidate_miss_rate=misses / len(reliable) if reliable else None)
        geometry = None
        if generate_map:
            segments = []
            for segment in result.route_segments:
                # Each decoder segment is continuous; keep breaks as separate polylines.
                coords = []
                for arc in segment:
                    u, v = network.arcs[arc]
                    coords.extend(network.nodes_lonlat[[u, v]].tolist())
                if coords:
                    segments.append(coords)
            geometry = {'gps': trajectory.df[['lon', 'lat']].values.tolist(),
                        'route_segments': segments}
        return dict(trip_id=trip_id, status='success', num_gps_observations=len(trajectory),
                    execution_time_ms=(perf_counter() - start) * 1000,
                    metrics=metrics, predicted_route=result.observation_arc_ids,
                    map_data=geometry)
