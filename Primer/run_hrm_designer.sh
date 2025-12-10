#!/bin/bash

# Run the HRM Primer Designer Streamlit app
# Usage: ./run_hrm_designer.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to that directory
cd "$SCRIPT_DIR"

# Run the Streamlit app
streamlit run hrm_primer_designer.py

