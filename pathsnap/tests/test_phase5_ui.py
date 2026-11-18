from pathlib import Path
from unittest.mock import patch
import requests
import pytest
from streamlit.testing.v1 import AppTest
from ui.client_app import call, coordinates

SCRIPT = Path(__file__).resolve().parents[2] / 'ui/client_app.py'
HEALTH = dict(status='healthy', ready=True, model_loaded=True, mode='development', trip_count=2)


def reply(data, status=200):
    response = requests.Response()
    response.status_code = status
    response.json = lambda: data
    return response


def run_ui(health=None, trips=None, prediction=None, error=None):
    def request(method, url, **kwargs):
        if error:
            raise error
        if url.endswith('/health'):
            return reply(HEALTH if health is None else health)
        if url.endswith('/api/trips'):
            return reply(dict(evaluation_trips=['00000000', '00000007']) if trips is None else trips)
        return prediction
    return patch('requests.request', side_effect=request)


def payload(**kwargs):
    return dict(trip_id='00000000', status='success', matched_sequence=[1], metrics={},
                execution_time_seconds=0., map_data=kwargs.get('map_data'))


def test_health_and_dynamic_trip_fallback():
    with run_ui(trips={'trips': ['00000007', '00000009']}):
        app = AppTest.from_file(str(SCRIPT)).run()
        assert not app.exception
        assert app.selectbox[0].options == ['00000007', '00000009']
        assert any('Model: ready' in item.value for item in app.markdown)


@pytest.mark.parametrize('health,trips', [
    ({**HEALTH, 'ready': False}, None),
    (None, {'evaluation_trips': []}),
    (None, {'evaluation_trips': 'invalid'}),
])
def test_not_ready_empty_or_invalid_trips(health, trips):
    with run_ui(health=health, trips=trips):
        app = AppTest.from_file(str(SCRIPT)).run()
        assert not app.exception
        assert not app.selectbox
        assert app.warning or app.error


@pytest.mark.parametrize('error', [requests.ConnectionError(), requests.Timeout()] )
def test_network_errors(error):
    with run_ui(error=error):
        app = AppTest.from_file(str(SCRIPT)).run()
        assert not app.exception
        assert len(app.error) == 1


@pytest.mark.parametrize('geometry', [None, {'gps': [], 'route_segments': []},
    {'gps': [[77., 12.]], 'route_segments': [[[77., 12.], [77.001, 12.001]]]},
    {'gps': [], 'route_segments': [[[77., 12.], [77.001, 12.001]]]}])
def test_prediction_missing_optional_metrics_and_empty_geometry(geometry):
    with run_ui(prediction=reply(payload(map_data=geometry))):
        app = AppTest.from_file(str(SCRIPT)).run()
        app.button[0].click().run()
        assert not app.exception
        assert len(app.metric) == 5
        assert app.metric[0].value == 'Unavailable'
        assert app.metric[4].value == '0 ms'


@pytest.mark.parametrize('status,fragment', [(404, 'Refresh'), (422, 'rejected'), (503, 'three attempts')])
def test_prediction_http_errors(status, fragment):
    with run_ui(prediction=reply({}, status)), patch('ui.client_app.time.sleep'):
        app = AppTest.from_file(str(SCRIPT)).run()
        app.button[0].click().run()
        assert not app.exception
        assert fragment in app.error[0].value
        assert not app.metric


def test_busy_retries_are_bounded():
    with patch('requests.request', return_value=reply({}, 503)) as request, patch('time.sleep') as sleep:
        with pytest.raises(requests.HTTPError):
            call('http://backend', 'POST', '/api/predict_trip')
        assert request.call_count == 3
        assert sleep.call_count == 2
        assert request.call_args.kwargs['timeout'] == (5, 300)


def test_busy_retry_recovers():
    with patch('requests.request', side_effect=[reply({}, 503), reply({'status': 'success'})]), patch('time.sleep'):
        assert call('http://backend', 'POST', '/api/predict_trip')['status'] == 'success'


def test_coordinates_and_ground_truth_isolation():
    assert coordinates([[77.5, 12.9]]) == [[12.9, 77.5]]
    assert coordinates([]) == []
    with pytest.raises(ValueError):
        coordinates([[float('nan'), 12.]])
    source = SCRIPT.read_text(encoding='utf-8')
    assert 'api.service' not in source
    assert 'read_text' not in source
    assert 'open(' not in source
    assert 'true_arc_id' not in source
