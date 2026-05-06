#!/usr/bin/env bash
# Creates a tarball of all exo data for Railway volume seeding.
# Output: /tmp/exo_data.tar.gz

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
OUT="/tmp/exo_data.tar.gz"

echo "Exporting all data from $DATA_DIR ..."

tar czf "$OUT" -C "$DATA_DIR" features/ risk_index/

SIZE=$(du -sh "$OUT" | cut -f1)
echo "Done. Archive: $OUT ($SIZE)"
echo ""
echo "Transfer options:"
echo ""
echo "Option A — ngrok (recommended for large files):"
echo "  1. In a new terminal: python3 -m http.server 9000 --directory /tmp"
echo "  2. In another terminal: ngrok http 9000"
echo "  3. In railway shell:   wget https://<ngrok-url>/exo_data.tar.gz -O /data/tmp.tar.gz"
echo "                         tar xzf /data/tmp.tar.gz -C /data && rm /data/tmp.tar.gz"
echo ""
echo "Option B — base64 pipe (smaller files only):"
echo "  base64 $OUT | railway shell --command \"base64 -d > /data/tmp.tar.gz\""
echo "  Then in railway shell: tar xzf /data/tmp.tar.gz -C /data && rm /data/tmp.tar.gz"
