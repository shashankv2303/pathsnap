from fastapi.testclient import TestClient
import pytest
from api.server import create_app


class StubService:
    def trips(self):
        return ['00000000']

    def predict(self, trip_id, **kwargs):
        if trip_id not in self.trips():
            raise FileNotFoundError()
        return dict(trip_id=trip_id, execution_time_ms=1, status='success',
                    num_gps_observations=0, metrics={}, predicted_route=[], map_data=None)


def test_contract_and_validation():
    with TestClient(create_app(StubService())) as client:
        assert client.get('/health').json()['model_loaded']
        assert client.get('/api/trips').json()['evaluation_trips'] == ['00000000']
        assert client.post('/api/predict_trip', json={'trip_id': '../train'}).status_code == 422
        assert client.post('/api/predict_trip', json={'trip_id': '99999999'}).status_code == 404
        assert client.post('/api/predict_trip', json={'trip_id': '00000000', 'path': '/'}).status_code == 422
        assert client.post('/api/predict_trip', json={'trip_id': '00000000'}).status_code == 200


def test_startup_failure(monkeypatch):
    monkeypatch.setenv('PATHSNAP_MODEL', 'missing-model/model.joblib')
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app()):
            pass
