#!/bin/zsh
# launchd entrypoint for the Wan api_server.
# Reads Runpod credentials from REELFORGE's .env (single source of truth)
# without shell-sourcing it (the DATABASE_URL line breaks `source`).
ENV_FILE="$HOME/Documents/video/.env"
export RUNPOD_API_KEY="$(grep '^RUNPOD_API_KEY=' "$ENV_FILE" | cut -d= -f2-)"
export RUNPOD_POD_NAME="$(grep '^RUNPOD_POD_NAME=' "$ENV_FILE" | cut -d= -f2-)"
export WAN_API_PORT=8787
export WAN_IDLE_STOP_MIN=45
cd "$(dirname "$0")"
exec /usr/bin/env python3 api_server.py
