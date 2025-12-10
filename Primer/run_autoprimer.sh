#!/bin/bash

# Run the AutoPrimer Streamlit app
# Usage: ./run_autoprimer.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to that directory
cd "$SCRIPT_DIR"

# Run the Streamlit app
streamlit run autoprimer.py

