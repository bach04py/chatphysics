#!/bin/bash

echo "🚀 Starting Physics AI..."

# set port
export PORT=${PORT:-8501}

# tránh crash bitsandbytes CPU
export BNB_CUDA_VERSION=0

# run app
streamlit run app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false