#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env.defaults ]; then
    source .env.defaults
fi

if [ -f .env ]; then
    source .env
fi

podman build -t agent1 .
podman run -it --rm --env-file .env.defaults --network=host -v ${LLAMA_MODELS_DIR}:/app/models -v ./conversations:/app/conversations agent1
