#!/bin/zsh
# launchd entrypoint for the Wan api_server.
# api_server.py reads WAN_/RUNPOD_ values from local_wrapper/.env, with the
# legacy $HOME/Documents/video/.env path supported as a fallback.
cd "$(dirname "$0")"
exec /usr/bin/env python3 api_server.py
