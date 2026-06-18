#!/bin/sh
# Startup command for the demo on Azure App Service (Linux, Python, F1 free plan).
# Set it as the startup command of the web app: sh startup.sh
# The build installs requirements.txt; copy requirements-demo.txt to requirements.txt in the
# deployment package. GEMINI_API_KEY is set in the app settings, never in the repository.
# Without the key, or with DEMO_REPLAY=1, the demo serves recorded answers only.
export DEMO_STATE_DIR="${DEMO_STATE_DIR:-/home/demo_state}"
exec python -m streamlit run app/Home.py \
  --server.port "${PORT:-8000}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
