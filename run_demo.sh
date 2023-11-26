#!/bin/bash

set -e
set -u

cd "$( dirname "${BASH_SOURCE[0]}" )"
export PYTHONPATH='.'
streamlit run demo.py --server.port=8505 --server.address=0.0.0.0 --server.fileWatcherType none
