#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env.defaults ]; then
    source .env.defaults
fi

if [ -f .env ]; then
    source .env
fi

# Default paths (override with .env or command line)
LLAMA_SERVER_PATH="${LLAMA_SERVER_PATH:-./llama-server}"
LLAMA_MODELS_DIR="${LLAMA_MODELS_DIR:-./models}"

"$LLAMA_SERVER_PATH" \
    --models-dir "$LLAMA_MODELS_DIR" \
    --parallel 1 \
    --ctx-size ${CTX_SIZE} \
    --n-gpu-layers ${GPU_LAYERS} \
    --host 0.0.0.0 --port 8080 \
    --flash-attn 1 --cache-type-k q4_0 --cache-type-v q4_0
    #-m ${LLAMA_MODELS_DIR}/${CURRENT_MODEL}.gguf

#  \
    
