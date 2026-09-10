#!/bin/zsh
# Launched by the com.exo.ingestion launchd service (see
# ~/Library/LaunchAgents/com.exo.ingestion.plist). Not meant to be run
# interactively, though it's safe to — it just starts main.py's scheduler.
#
# Sources ~/.zshrc for API credentials (KALSHI_*, FRED_API_KEY,
# FINNHUB_API_KEY, EIA_API_KEY) since launchd does not inherit the login
# shell's environment. Loading the whole file (rather than grepping out
# individual vars) is deliberate: KALSHI_PRIVATE_KEY is a multi-line PEM
# value, and line-filtering would truncate it.
set -euo pipefail

source "$HOME/.zshrc"

export DATA_DIR="/Users/kevin/Desktop/ds/exo/data"

cd /Users/kevin/Desktop/ds/exo
exec /Users/kevin/Desktop/ds/environments/exo_env/bin/python main.py
