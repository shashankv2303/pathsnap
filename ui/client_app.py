"""HTTP-only evaluator client; reference scoring stays on Laptop 1."""
import math
import os
import time
import folium
import requests
import streamlit as st
import streamlit.components.v1 as components


def call(server, method, path, timeout=300, **kwargs):
    for attempt in range(3):
        response = requests.request(method, server + path, timeout=(5, timeout), **kwargs)
        if response.status_code == 503 and method == 'POST' and attempt < 2:
            time.sleep(1.5)
            continue
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError('Unexpected backend response.')
        return data


def coordinates(points):
    if not isinstance(points, list):
        raise ValueError('Invalid map coordinates.')
    output = []
    for p in points:
        if (not isinstance(p, list) or len(p) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in p)
                or not -180 <= p[0] <= 180 or not -90 <= p[1] <= 90):
            raise ValueError('Invalid map coordinates.')
        output.append([p[1], p[0]])
    return output


def render_result(result):
    metrics = result.get('metrics') or {}
    route = result.get('predicted_route')
    if route is None:
        route = result.get('matched_sequence')
    if result.get('status') != 'success' or not isinstance(route, list) or not isinstance(metrics, dict):
        raise ValueError('Invalid successful prediction response.')
    elapsed = result.get('execution_time_ms')
    if elapsed is None and result.get('execution_time_seconds') is not None:
        elapsed = result['execution_time_seconds'] * 1000
    values = [metrics.get('pointwise_accuracy'), metrics.get('continuity', metrics.get('route_continuity')),
              metrics.get('candidate_miss_rate'), metrics.get('normalized_route_edit_distance'), elapsed]
    labels = ['Accuracy', 'Continuity', 'Candidate miss rate', 'Route edit distance', 'Processing time']
    for i, (col, label, value) in enumerate(zip(st.columns(5), labels, values)):
        text = 'Unavailable' if value is None else (f'{value:.2%}' if i < 3 else f'{value:.3f}' if i == 3 else f'{value:.0f} ms')
        col.metric(label, text)
    st.caption('Route edit distance: lower is better.')
    if metrics.get('reference_available') is False:
        st.info('Reference labels are unavailable. Accuracy cannot be calculated.')
    with st.expander('Execution diagnostics'):
        st.write(f'Matched observations: {len(route)}')
        st.json(metrics)
    data = result.get('map_data') or {}
    gps = coordinates(data.get('gps', []))
    segments = [coordinates(s) for s in data.get('route_segments', [])]
    bounds = gps + [p for s in segments for p in s]
    if not bounds:
        st.info('No map geometry was returned for this trip.')
        return
    view = folium.Map(location=bounds[0], zoom_start=13)
    observations = folium.FeatureGroup(name='GPS observations').add_to(view)
    for p in gps:
        folium.CircleMarker(p, radius=2, color='blue', fill=True).add_to(observations)
    predicted = folium.FeatureGroup(name='Predicted route').add_to(view)
    for s in segments:
        if len(s) > 1:
            folium.PolyLine(s, color='red', weight=5).add_to(predicted)
    folium.LayerControl().add_to(view)
    view.fit_bounds(bounds)
    components.html(view.get_root().render(), height=600)
    st.caption('Blue: GPS observations. Red: predicted route. Basemap requires internet.')


def main():
    st.set_page_config(page_title='PathSnap', layout='wide')
    st.title('PathSnap')
    st.caption('Map matching and trajectory reconstruction')
    server = st.sidebar.text_input('Model server URL', os.getenv('SERVER_URL', 'http://127.0.0.1:8000')).rstrip('/')
    timeout = st.sidebar.number_input('Prediction timeout (seconds)', min_value=300, value=300)
    try:
        health = call(server, 'GET', '/health', timeout=10)
        ready = health.get('ready') is True and health.get('model_loaded') is True
        st.write(f"Backend: {health.get('status', 'unknown')} | Model: {'ready' if ready else 'not ready'} | "
                 f"Mode: {health.get('mode', 'unknown')} | Trips: {health.get('trip_count', health.get('trips_count', 'unknown'))}")
        if not ready or health.get('status') != 'healthy':
            st.warning('Laptop 1 is connected but not ready. Check its model and server logs.')
            return
        listing = call(server, 'GET', '/api/trips', timeout=10)
        trips = listing.get('evaluation_trips')
        if trips is None:
            trips = listing.get('trips')
        if not isinstance(trips, list) or any(not isinstance(t, str) or len(t) != 8 or not t.isascii() or not t.isdigit() for t in trips):
            raise ValueError('Invalid trip list.')
        if not trips:
            st.warning('No trips are available on the server.')
            return
        trip = st.selectbox('Evaluation trip', trips)
        hmm = st.checkbox('Use HMM / Viterbi decoding', value=True)
        # The form batches input changes; Streamlit runs each submission sequentially.
        with st.form('reconstruction'):
            submitted = st.form_submit_button('Reconstruct Route', type='primary')
        if submitted:
            with st.spinner('Reconstructing trajectory…'):
                result = call(server, 'POST', '/api/predict_trip', timeout=timeout,
                              json=dict(trip_id=trip, use_hmm=hmm, generate_map=True))
            if result.get('trip_id') != trip:
                raise ValueError('Backend returned a different trip.')
            render_result(result)
    except requests.Timeout:
        st.error('Request timed out. Check Laptop 1 backend logs before submitting again.')
    except requests.HTTPError as exc:
        code = exc.response.status_code
        messages = {503: 'Laptop 1 is busy after three attempts. Wait and try again.',
                    404: 'Trip is no longer available. Refresh the page to reload the trip list.',
                    422: 'Laptop 1 rejected the request (422). Check the selected trip and settings.'}
        st.error(messages.get(code, f'Laptop 1 returned HTTP {code}. Check its server logs.'))
    except requests.RequestException:
        st.error('Laptop 1 is unreachable. Check SERVER_URL, its running backend, shared Wi-Fi and firewall port 8000.')
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f'Invalid backend response: {exc}')


if __name__ == '__main__':
    main()
