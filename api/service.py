"""Inference and server-side evaluation service for PathSnap Laptop 1 Backend."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Set, Union

import sklearn

from pathsnap.decoding.pipeline import prepare_trip, decode_prepared
from pathsnap.decoding.evaluate import evaluate, load_reference
from pathsnap.decoding.viterbi import DecoderConfig
from pathsnap.feature_engineering.feature_extractor import FEATURE_NAMES
from pathsnap.models.hist_gradient_boosting import HistGradientBoostingPipeline

log = logging.getLogger('uvicorn.error')

ROOT = Path(__file__).resolve().parents[1]

# Official 10 reserved evaluator trip IDs
EVALUATOR_ALLOWLIST: Set[str] = {
    "00000015", "00000017", "00000023", "00000030", "00000049",
    "00000059", "00000067", "00000073", "00000087", "00000094",
}


class PredictionService:
    """Manages model loading, trip discovery, inference, and server-side scoring."""

    def __init__(
        self,
        trip_root: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        decoder_config_path: Optional[Union[str, Path]] = None,
        mode: Optional[str] = None,
    ):
        # 1. Mode Configuration
        raw_mode = (mode or os.getenv('PATHSNAP_MODE', 'development')).lower().strip()
        if raw_mode in ('evaluator', 'eval'):
            self.mode = 'evaluator'
        elif raw_mode in ('development', 'dev'):
            self.mode = 'development'
        else:
            raise ValueError(
                f"Invalid PATHSNAP_MODE: '{raw_mode}'. Expected 'development' or 'evaluator'."
            )

        # 2. Trip Root Directory Configuration
        if self.mode == 'evaluator':
            configured_root = trip_root or os.getenv('PATHSNAP_TRIP_ROOT')
            if not configured_root:
                raise ValueError(
                    "PATHSNAP_MODE is set to 'evaluator', but PATHSNAP_TRIP_ROOT is not configured. "
                    "Evaluator mode requires an explicit path to the reserved evaluation trips."
                )
            self.trip_root = Path(configured_root).resolve()
            if not self.trip_root.exists() or not self.trip_root.is_dir():
                raise FileNotFoundError(
                    f"Configured evaluator trip root directory does not exist: {self.trip_root}"
                )
            # Never allow evaluator mode to silently point to student_test or train
            normalized_str = str(self.trip_root).replace('\\', '/').lower()
            if 'student_test' in normalized_str or (
                'student folder' in normalized_str and 'train' in normalized_str
            ):
                raise ValueError(
                    f"PATHSNAP_MODE is set to 'evaluator', but PATHSNAP_TRIP_ROOT points to student development data: {self.trip_root}. "
                    "Evaluator mode requires an explicit path to the reserved evaluation trips."
                )
        else:
            default_dev_root = str(ROOT / 'Student folder/student_test')
            configured_root = trip_root or os.getenv('PATHSNAP_TRIP_ROOT', default_dev_root)
            self.trip_root = Path(configured_root).resolve()
            if not self.trip_root.exists() or not self.trip_root.is_dir():
                raise FileNotFoundError(
                    f"Configured development trip root directory does not exist: {self.trip_root}"
                )

        # 3. Model Loading & Verification
        default_model = str(ROOT / 'pathsnap/artifacts/deployment_model/model.joblib')
        env_model = os.getenv('PATHSNAP_MODEL_PATH') or os.getenv('PATHSNAP_MODEL')
        model_file = Path(model_path or env_model or default_model).resolve()
        if not model_file.is_file():
            raise FileNotFoundError(f"Model artifact not found at: {model_file}")

        env_config = os.getenv('PATHSNAP_CONFIG_PATH')
        config_file = Path(
            config_path or env_config or model_file.with_name('model_config.json')
        ).resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"Model configuration file not found at: {config_file}")

        self.model_config = json.loads(config_file.read_text(encoding='utf-8'))
        configured_features = self.model_config.get('input_features', [])
        expected_features = list(FEATURE_NAMES)
        if configured_features != expected_features:
            raise ValueError(
                f"Model feature schema mismatch: expected {expected_features}, got {configured_features}"
            )

        # Check scikit-learn version compatibility
        model_sklearn_ver = self.model_config.get('sklearn_version', 'unknown')
        installed_sklearn_ver = sklearn.__version__
        log.info(
            "Loading model artifact from %s (model sklearn=%s, installed sklearn=%s)",
            model_file.name,
            model_sklearn_ver,
            installed_sklearn_ver,
        )

        self.model = HistGradientBoostingPipeline.load(str(model_file))
        self.model_path = model_file

        # 4. Decoder Configuration
        default_decoder_config = str(ROOT / 'results/hmm-viterbi/decoder_config.json')
        decoder_file = Path(
            decoder_config_path or os.getenv('PATHSNAP_DECODER_CONFIG', default_decoder_config)
        ).resolve()
        if not decoder_file.is_file():
            raise FileNotFoundError(f"Decoder configuration file not found at: {decoder_file}")

        self.decoder_config_data = json.loads(decoder_file.read_text(encoding='utf-8'))
        self.config = DecoderConfig(**self.decoder_config_data)
        self.decoder_config_path = decoder_file

        log.info(
            "PredictionService initialized in [%s] mode with root: %s (%d trips discovered)",
            self.mode,
            self.trip_root.name,
            len(self.trips()),
        )

    def is_ready(self) -> bool:
        """Returns True if model and configuration are loaded and ready to serve."""
        return self.model is not None and self.config is not None

    def trips(self) -> List[str]:
        """Discovers valid trip directories based on mode and file availability."""
        if not self.trip_root.exists():
            return []

        candidates = [
            p.name
            for p in self.trip_root.iterdir()
            if p.is_dir()
            and len(p.name) == 8
            and p.name.isascii()
            and p.name.isdigit()
            and not p.is_symlink()
            and all((p / (p.name + ext)).is_file() for ext in ('.track', '.nodes', '.arcs'))
        ]

        if self.mode == 'evaluator':
            # In evaluator mode, all valid evaluator trips must have server-side .route files
            evaluator_trips = []
            for tid in sorted(candidates):
                p = self.trip_root / tid
                route_file = p / (tid + '.route')
                if not route_file.is_file():
                    raise RuntimeError(
                        f"Evaluator mode requires server-side ground truth (.route) for all evaluator trips. "
                        f"Missing .route file in trip: {tid}"
                    )
                evaluator_trips.append(tid)

            # If matching allowlist, restrict to official evaluator IDs
            matching_allowlist = [t for t in evaluator_trips if t in EVALUATOR_ALLOWLIST]
            if matching_allowlist:
                return sorted(matching_allowlist)
            return sorted(evaluator_trips)

        return sorted(candidates)

    def predict(
        self, trip_id: str, use_hmm: bool = True, generate_map: bool = True
    ) -> Dict[str, Any]:
        """Executes full pipeline: preprocessing, candidate extraction, scoring, decoding, and evaluation."""
        available_trips = self.trips()
        if trip_id not in available_trips:
            raise FileNotFoundError(
                f"Unknown or unauthorized trip ID '{trip_id}'. Available in {self.mode} mode: {available_trips}"
            )

        start = perf_counter()
        trip_dir = self.trip_root / trip_id

        # 1. Feature preparation & inference
        prepared = prepare_trip(trip_dir, self.model)
        trajectory, network, candidates, _ = prepared

        # 2. Decoding (Baseline vs HMM)
        baseline = decode_prepared(prepared, self.config, sequential=False)
        result = decode_prepared(prepared, self.config, sequential=True) if use_hmm else baseline

        # 3. Server-Side Ground Truth Evaluation
        route_file = trip_dir / f"{trip_id}.route"
        has_route = route_file.is_file()

        if self.mode == 'evaluator' and not has_route:
            raise RuntimeError(
                f"Missing required server-side reference route for evaluator trip '{trip_id}'"
            )

        if has_route:
            reference = load_reference(trip_dir, prepared)
            eval_metrics = evaluate(result, reference, measure_sequence=True)
            base_metrics = evaluate(baseline, reference, measure_sequence=True)

            reliable = [a for a in reference[0] if a.is_reliable]
            misses = sum(
                not any(c.arc_id == a.true_arc_id for c in candidates[a.obs_idx])
                for a in reliable
            )

            acc = eval_metrics.get('pointwise_accuracy')
            base_acc = base_metrics.get('pointwise_accuracy')
            continuity = eval_metrics.get('continuity')
            edit_dist = eval_metrics.get('normalized_route_edit_distance')
            miss_rate = misses / len(reliable) if reliable else 0.0
            boost = (acc - base_acc) if (acc is not None and base_acc is not None) else 0.0

            metrics: Dict[str, Any] = {
                'reference_available': True,
                'pointwise_accuracy': acc,
                'continuity': continuity,
                'route_continuity': continuity,  # canonical alias
                'candidate_miss_rate': miss_rate,
                'classifier_only_accuracy': base_acc,
                'hmm_accuracy_boost': boost,
                'hmm_accuracy_improvement': boost,  # canonical alias
                'normalized_route_edit_distance': edit_dist,
                'observations': eval_metrics.get('observations', len(trajectory)),
                'aligned_observations': eval_metrics.get('aligned_observations', len(reliable)),
                'eligible_transitions': eval_metrics.get('eligible_transitions'),
                'connected_transitions': eval_metrics.get('connected_transitions'),
                'breaks': eval_metrics.get('breaks', 0),
            }
        else:
            continuity = (
                result.connected_transitions / result.eligible_transitions
                if result.eligible_transitions
                else None
            )
            metrics = {
                'reference_available': False,
                'pointwise_accuracy': None,
                'continuity': continuity,
                'route_continuity': continuity,
                'candidate_miss_rate': None,
                'classifier_only_accuracy': None,
                'hmm_accuracy_boost': None,
                'hmm_accuracy_improvement': None,
                'normalized_route_edit_distance': None,
                'observations': len(trajectory),
                'aligned_observations': None,
                'eligible_transitions': result.eligible_transitions,
                'connected_transitions': result.connected_transitions,
                'breaks': len(result.breaks),
            }

        # 4. Map Geometry Construction (Strictly Predicted Coordinates & Raw GPS only)
        geometry = None
        if generate_map:
            segments: List[List[List[float]]] = []
            for segment in result.route_segments:
                coords: List[List[float]] = []
                for arc in segment:
                    u, v = network.arcs[arc]
                    coords.extend(network.nodes_lonlat[[u, v]].tolist())
                if coords:
                    segments.append(coords)
            geometry = {
                'gps': trajectory.df[['lon', 'lat']].values.tolist(),
                'route_segments': segments,
            }

        elapsed_ms = (perf_counter() - start) * 1000.0
        elapsed_sec = elapsed_ms / 1000.0
        metrics['backend_processing_time_ms'] = elapsed_ms
        metrics['execution_time_seconds'] = elapsed_sec

        # 5. Return Clean API Contract (Zero Ground-Truth Leaks)
        return {
            'trip_id': trip_id,
            'status': 'success',
            'num_gps_observations': len(trajectory),
            'execution_time_ms': elapsed_ms,
            'execution_time_seconds': elapsed_sec,
            'metrics': metrics,
            'predicted_route': result.observation_arc_ids,
            'matched_sequence': result.observation_arc_ids,
            'map_data': geometry,
        }
