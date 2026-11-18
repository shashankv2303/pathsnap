# Phase 5 deployment

## Laptop 1

From the repository root, install `python -m pip install -r requirements.txt`.
Run `python -m uvicorn api.server:app --host 0.0.0.0 --port 8000`.
Check `http://127.0.0.1:8000/health` and the interactive API at `/docs`.
Use one worker: the service accepts one prediction at a time and returns 503
while busy. Requests use the tuned decoder configuration from Phase 4.

Default trips come from `Student folder/student_test`; IDs are discovered from
the actual directory rather than the outdated architecture document list.
Set `PATHSNAP_TRIP_ROOT` to a server-side evaluation directory when provided.
No reference files are required for prediction; missing references produce null
accuracy metrics. Reference routes are read only for server-side scoring.

## Model compatibility repair

The original artifact failed to deserialize on the installed Python environment
(`ModuleNotFoundError: _loss`). Its feature_schema.json also describes different
features from its model_config.json and the current extractor. The original
files are preserved. `python -m api.train_deployment_model` builds a separate
deployment artifact using all 80 training trips and the saved hyperparameters.
It refuses to overwrite an existing deployment directory. The API validates
the actual ordered input_features in model_config.json at startup.
Rebuilding changes the fitted model; historical Phase 4 scores must not be
presented as fresh deployment-model scores.

## Laptop 2

Install `streamlit requests folium`. Copy only `ui/client_app.py` to Laptop 2;
model files and reference routes stay on Laptop 1.
Set `$env:SERVER_URL='http://LAPTOP1_PRIVATE_IP:8000'` in PowerShell, then run
`python -m streamlit run ui/client_app.py` and open `http://localhost:8501`.
Connect both laptops to the same network and allow TCP port 8000 on Laptop 1
for the private network. The Streamlit process makes HTTP requests server-side,
so browser CORS configuration is unnecessary for this client.
Map tiles require internet; predictions and route coordinates do not.
Timeouts are 5 seconds for connection and 300 seconds for prediction.

## Verification

Run `python -m pytest pathsnap/tests -q`. Run a prediction via `/docs` before
demonstrating on two laptops. Confirm invalid IDs return 422/404, request logs
appear on Laptop 1, and no ground-truth geometry is returned to the client.
Physical two-laptop connectivity must be checked on the actual LAN.

Local verification completed on 18 November 2026: 14 tests passed, including
real HTTP-handler predictions on all ten student-test trips and Streamlit
prediction/error-flow tests. The deployment model was fitted on 1,208,139
candidate rows from 80 training trips using scikit-learn 1.9.0, which is pinned
in requirements.txt for artifact compatibility. These tests use the ASGI test
client; they do not establish physical LAN reachability or browser map-tile
availability. The original Phase 5 plan overstated initial readiness: the
artifact loading failure required the separate rebuild documented above.
