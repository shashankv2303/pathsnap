"""Evaluator client: only HTTP access to the model server is required."""
import os
import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title='PathSnap', layout='wide')
st.title('PathSnap')
st.caption('Map matching and trajectory reconstruction')
server = st.sidebar.text_input('Model server URL', os.getenv('SERVER_URL', 'http://127.0.0.1:8000')).rstrip('/')


def call(method, path, **kwargs):
    response = requests.request(method, server + path, timeout=(5, 300), **kwargs)
    response.raise_for_status()
    return response.json()


try:
    call('GET', '/health')
    trips = call('GET', '/api/trips')['evaluation_trips']
    if not trips:
        st.warning('No trips are available on the server.')
        st.stop()
    trip = st.selectbox('Evaluation trip', trips)
    hmm = st.checkbox('Use HMM / Viterbi decoding', value=True)
    if st.button('Reconstruct route', type='primary'):
        with st.spinner('Reconstructing trajectory…'):
            result = call('POST', '/api/predict_trip', json=dict(trip_id=trip, use_hmm=hmm))
        metrics = result['metrics']
        columns = st.columns(4)
        for col, label, value in zip(columns, ['Accuracy', 'Continuity', 'Candidate miss rate', 'Processing time'],
                                    [metrics['pointwise_accuracy'], metrics['continuity'],
                                     metrics['candidate_miss_rate'], result['execution_time_ms']]):
            col.metric(label, ('Unavailable' if value is None else
                              f'{value:.0f} ms' if label == 'Processing time' else f'{value:.2%}'))
        if not metrics['reference_available']:
            st.info('Reference labels are unavailable. Accuracy cannot be calculated.')
        data = result['map_data']
        gps = [[lat, lon] for lon, lat in data['gps']]
        map_view = folium.Map(location=gps[0], zoom_start=13)
        folium.PolyLine(gps, color='blue', tooltip='GPS observations').add_to(map_view)
        for segment in data['route_segments']:
            folium.PolyLine([[lat, lon] for lon, lat in segment], color='red', weight=5,
                            tooltip='Predicted route').add_to(map_view)
        map_view.fit_bounds(gps)
        components.html(map_view.get_root().render(), height=600)
        st.caption('Blue: GPS observations. Red: predicted route. Basemap requires internet.')
        st.json(metrics, expanded=False)
except (requests.RequestException, ValueError, KeyError) as exc:
    st.error(f'Unable to complete the request: {exc}')
