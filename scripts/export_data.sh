#!/usr/bin/env bash
# Creates a tarball of essential exo data for Railway volume seeding.
# Excludes Kalshi and Polymarket — both repopulate from live APIs on first scheduler run.
# Output: /tmp/exo_data.tar.gz

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
OUT="/tmp/exo_data.tar.gz"

echo "Exporting essential data from $DATA_DIR ..."

tar czf "$OUT" \
  -C "$DATA_DIR" \
  --exclude='./features/source=kalshi' \
  features/ risk_index/

SIZE=$(du -sh "$OUT" | cut -f1)
echo "Done. Archive: $OUT ($SIZE)"
echo ""
echo "Transfer options:"
echo ""
echo "Option A — ngrok (recommended):"
echo "  1. In a new terminal: python3 -m http.server 9000 --directory /tmp"
echo "  2. In another terminal: ngrok http 9000"
echo "  3. curl -X POST 'https://exo.dosi.io/api/admin/seed?url=https://<ngrok-url>/exo_data.tar.gz'"
