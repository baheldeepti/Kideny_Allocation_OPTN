#!/bin/bash
# Launch the Kidney Waitlist Risk Explorer locally.
cd "$(dirname "$0")"
exec /Users/deepti.bahel/ckd-ml/bin/streamlit run app.py "$@"
