"""Real model, trip inputs, JSON serialization and server scoring."""
from fastapi.testclient import TestClient
from api.server import create_app


def test_real_predictions():
    with TestClient(create_app()) as client:
        trips = client.get('/api/trips').json()['evaluation_trips']
        assert len(trips) == 10
        for trip in trips:
            response = client.post('/api/predict_trip', json={'trip_id': trip})
            assert response.status_code == 200, response.text
            result = response.json()
            assert len(result['predicted_route']) == result['num_gps_observations']
            assert result['map_data']['gps']
            assert 'reference_route' not in response.text
            assert 0 <= result['metrics']['pointwise_accuracy'] <= 1
