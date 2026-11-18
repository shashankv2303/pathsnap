"""Rebuild a portable deployment artifact from all 80 training trips only."""
import json
import pandas as pd
import sklearn
from api.service import ROOT
from pathsnap.dataset.dataset_builder import DatasetBuilder
from pathsnap.feature_engineering.feature_extractor import FEATURE_NAMES
from pathsnap.models.hist_gradient_boosting import HistGradientBoostingPipeline


def main():
    output = ROOT / 'pathsnap/artifacts/deployment_model'
    output.mkdir(exist_ok=False)
    builder = DatasetBuilder()
    frames = []
    trips = sorted((ROOT / 'Student folder/train').iterdir())
    assert len(trips) == 80
    for trip in trips:
        frame, _ = builder.process_single_trip(str(trip), trip.name)
        frames.append(frame[frame.target >= 0])
        print('Prepared', trip.name, flush=True)
    data = pd.concat(frames, ignore_index=True)
    original = json.loads((ROOT / 'pathsnap/artifacts/final_model/model_config.json').read_text())
    model = HistGradientBoostingPipeline(**original['hyperparameters'])
    model.fit(data[FEATURE_NAMES].values, data.target.values)
    model.save(str(output / 'model.joblib'))
    (output / 'model_config.json').write_text(json.dumps(dict(
        input_features=FEATURE_NAMES, training_trips=80, training_samples=len(data),
        sklearn_version=sklearn.__version__, hyperparameters=original['hyperparameters']), indent=2))


if __name__ == '__main__':
    main()
