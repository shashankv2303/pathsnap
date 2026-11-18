from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest


def test_ui_prediction_and_network_failure():
    script = Path(__file__).resolve().parents[2] / 'ui/client_app.py'
    payload = dict(metrics=dict(pointwise_accuracy=.8, continuity=1.,
                               candidate_miss_rate=0., reference_available=True),
                   execution_time_ms=50,
                   map_data=dict(gps=[[77., 12.], [77.001, 12.001]],
                                 route_segments=[[[77., 12.], [77.001, 12.001]]]))
    def response(method, url, **kwargs):
        from unittest.mock import Mock
        result = Mock()
        result.json.return_value = (payload if method == 'POST' else
                                    dict(evaluation_trips=['00000000']))
        return result
    with patch('requests.request', side_effect=response):
        app = AppTest.from_file(str(script)).run()
        app.button[0].click().run()
        assert not app.exception
        assert len(app.metric) == 4
    import requests
    with patch('requests.request', side_effect=requests.Timeout('timed out')):
        app = AppTest.from_file(str(script)).run()
        assert not app.exception
        assert len(app.error) == 1
